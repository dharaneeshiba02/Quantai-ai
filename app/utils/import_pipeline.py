"""
app/utils/import_pipeline.py — Upload and import pipeline for CSV/Excel files.
Imports into MySQL quantai_data schema using the import user.
150 cols × 10,000 rows must complete in under 30 seconds.
"""
from __future__ import annotations

import logging
import hashlib
import re
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from sqlalchemy import text

from app.config import get_settings
from app.storage.database import import_connection, app_session

log = logging.getLogger(__name__)

MYSQL_MAX_COL_LEN = 64
BATCH_SIZE = 500  # rows per executemany batch


def _safe_index_name(column_name: str) -> str:
    """Build a unique MySQL index name within MySQL's 64-character limit."""
    digest = hashlib.sha1(column_name.encode("utf-8")).hexdigest()[:8]
    prefix = "idx_"
    return f"{prefix}{column_name[:64 - len(prefix) - len(digest) - 1]}_{digest}"


def _safe_col_name(name: str, existing: set[str]) -> str:
    """Convert column name to MySQL-safe snake_case, max 64 chars, deduplicated."""
    name = str(name).strip()
    name = re.sub(r"[^\w]", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.lower().strip("_")
    name = name[:MYSQL_MAX_COL_LEN]
    if not name or name[0].isdigit():
        name = "col_" + name

    # Deduplicate
    original = name
    counter = 1
    while name in existing:
        suffix = f"_{counter}"
        name = original[:MYSQL_MAX_COL_LEN - len(suffix)] + suffix
        counter += 1

    existing.add(name)
    return name


def _infer_mysql_type(series: Any) -> str:
    """Infer the appropriate MySQL type for a Pandas Series."""
    dtype = series.dtype
    values = series.dropna()

    if pd.api.types.is_bool_dtype(dtype):
        return "TINYINT(1)"

    if pd.api.types.is_integer_dtype(dtype):
        max_val = values.abs().max() if len(values) > 0 else 0
        return "BIGINT" if max_val > 2_147_483_647 else "INT"

    if pd.api.types.is_float_dtype(dtype):
        return "DOUBLE"

    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "DATETIME"

    if pd.api.types.is_object_dtype(dtype):
        # Try to detect dates
        str_sample = values.astype(str).head(10).tolist()
        if _looks_like_date(str_sample):
            return "DATE"
        # Size text columns from every row; sampling truncated real-world values.
        max_len = int(values.astype(str).str.len().max()) if len(values) > 0 else 0
        if max_len > 16_777_215:
            return "LONGTEXT"
        if max_len > 65_535:
            return "MEDIUMTEXT"
        if max_len > 255:
            return "TEXT"
        return f"VARCHAR({max(max_len, 64)})"

    return "TEXT"


def _looks_like_date(samples: list[str]) -> bool:
    if not samples:
        return False
    date_re = re.compile(r"\d{4}[-/]\d{2}[-/]\d{2}")
    return sum(1 for s in samples if date_re.search(s)) >= len(samples) // 2


async def import_file(
    file_path: Path,
    source_name: str,
) -> dict[str, Any]:
    """
    Import a CSV or Excel file into MySQL quantai_data.
    Returns source metadata including table_name.
    """
    t_start = time.perf_counter()
    cfg = get_settings()
    source_id = str(uuid.uuid4())[:8]
    slug = re.sub(r"[^\w]", "_", source_name.lower())[:20]
    table_name = f"upl_{slug}_{source_id}"

    log.info("Importing '%s' as table '%s'...", file_path.name, table_name)

    # ── 1. Read file ─────────────────────────────────────────────────────────
    t_read = time.perf_counter()
    if str(file_path).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path, dtype=str)  # read as str first for flexibility
    else:
        # Try multiple encodings
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                df = pd.read_csv(file_path, dtype=str, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            df = pd.read_csv(file_path, dtype=str, encoding="utf-8", encoding_errors="replace")

    t_read_done = time.perf_counter()
    log.info("File read: %.2fs, %d rows × %d cols", t_read_done - t_read, len(df), len(df.columns))

    # ── 2. Normalise column names ────────────────────────────────────────────
    existing_names: set[str] = set()
    original_to_safe: dict[str, str] = {}
    for orig in df.columns:
        safe = _safe_col_name(str(orig), existing_names)
        original_to_safe[str(orig)] = safe

    df.rename(columns=original_to_safe, inplace=True)

    # ── 3. Infer types and convert ───────────────────────────────────────────
    col_types: dict[str, str] = {}
    for col in df.columns:
        # Convert numeric where possible
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            samples = df[col].dropna().astype(str).head(10).tolist()
            if _looks_like_date(samples):
                parsed_dates = pd.to_datetime(df[col], errors="coerce")
                if parsed_dates.notna().sum() >= max(1, int(df[col].notna().sum() * 0.5)):
                    df[col] = parsed_dates
        col_types[col] = _infer_mysql_type(df[col])

    # ── 4. Build CREATE TABLE ────────────────────────────────────────────────
    col_defs = []
    for col, mysql_type in col_types.items():
        col_defs.append(f"  `{col}` {mysql_type}")

    create_sql = (
        f"CREATE TABLE IF NOT EXISTS `{cfg.mysql_data_schema}`.`{table_name}` (\n"
        f"  `_row_id` BIGINT AUTO_INCREMENT PRIMARY KEY,\n"
        + ",\n".join(col_defs)
        + "\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;"
    )

    # ── 5. Create table and bulk insert ─────────────────────────────────────
    rows_total = len(df)
    try:
        with import_connection() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS `{cfg.mysql_data_schema}`.`{table_name}`"))
            conn.execute(text(create_sql))
            conn.commit()

            cols_quoted = ", ".join(f"`{c}`" for c in df.columns)
            placeholders = ", ".join([f":{c}" for c in df.columns])
            insert_sql = text(
                f"INSERT INTO `{cfg.mysql_data_schema}`.`{table_name}` ({cols_quoted}) VALUES ({placeholders})"
            )

            for i in range(0, rows_total, BATCH_SIZE):
                batch = df.iloc[i : i + BATCH_SIZE]
                records = batch.astype(object).where(pd.notna(batch), None).to_dict("records")
                conn.execute(insert_sql, records)
                if i % (BATCH_SIZE * 10) == 0:
                    conn.commit()
                    log.debug("Inserted %d/%d rows...", min(i + BATCH_SIZE, rows_total), rows_total)

            conn.commit()

            for col, mysql_type in col_types.items():
                is_date = "DATE" in mysql_type.upper()
                is_key = any(kw in col.lower() for kw in ["id", "key", "code", "num"])
                if (is_date or is_key) and "TEXT" not in mysql_type:
                    try:
                        index_name = _safe_index_name(col)
                        conn.execute(text(
                            f"CREATE INDEX `{index_name}` ON `{cfg.mysql_data_schema}`.`{table_name}` (`{col}`)"
                        ))
                        conn.commit()
                    except Exception:
                        pass

            conn.execute(text(f"ANALYZE TABLE `{cfg.mysql_data_schema}`.`{table_name}`"))
            conn.commit()
    except Exception:
        try:
            with import_connection() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS `{cfg.mysql_data_schema}`.`{table_name}`"))
                conn.commit()
        except Exception:
            pass
        raise

    t_done = time.perf_counter()
    elapsed = t_done - t_start
    log.info("Import complete: %d rows in %.2fs (%.0f rows/s)", rows_total, elapsed, rows_total / elapsed)

    # ── 6. Build catalog metadata ────────────────────────────────────────────
    catalog_entries = _build_catalog(df, original_to_safe, col_types)

    return {
        "source_id": None,  # will be set by caller
        "table_name": table_name,
        "schema_name": "quantai_data",
        "row_count": rows_total,
        "col_count": len(df.columns),
        "original_to_safe": original_to_safe,
        "col_types": col_types,
        "catalog_entries": catalog_entries,
        "elapsed_sec": round(elapsed, 2),
    }


def _build_catalog(
    df: pd.DataFrame,
    original_to_safe: dict[str, str],
    col_types: dict[str, str],
) -> list[dict[str, Any]]:
    """Build column catalog entries for the imported table."""
    safe_to_orig = {v: k for k, v in original_to_safe.items()}
    entries = []

    for i, col in enumerate(df.columns):
        series = df[col]
        null_count = series.isna().sum()
        null_pct = round(null_count / max(len(series), 1) * 100, 2)
        distinct = series.nunique()
        mysql_type = col_types.get(col, "TEXT")

        # Type classification
        if pd.api.types.is_numeric_dtype(series):
            col_type = "float" if series.dtype.kind == "f" else "int"
        elif pd.api.types.is_datetime64_any_dtype(series):
            col_type = "datetime"
        elif "DATE" in mysql_type.upper() and "TIME" not in mysql_type.upper():
            col_type = "date"
        else:
            col_type = "str"

        # Stats
        col_min = col_max = None
        top_values = []
        try:
            if col_type in ("int", "float"):
                col_min = str(series.min())
                col_max = str(series.max())
            elif col_type in ("date", "datetime"):
                col_min = str(series.min())
                col_max = str(series.max())
            else:
                if distinct <= 50:
                    top_values = series.value_counts().head(5).index.tolist()
        except Exception:
            pass

        # Detect date / key columns
        is_date_col = col_type in ("date", "datetime")
        is_key_col = any(kw in col.lower() for kw in ["id", "key", "code", "_num"])

        entries.append({
            "original_name": safe_to_orig.get(col, col),
            "safe_name": col,
            "col_type": col_type,
            "mysql_type": mysql_type,
            "null_pct": null_pct,
            "distinct_count": int(distinct),
            "col_min": col_min,
            "col_max": col_max,
            "top_values": [str(v) for v in top_values[:5]],
            "is_date_col": is_date_col,
            "is_key_col": is_key_col,
            "ordinal_position": i,
        })

    return entries
