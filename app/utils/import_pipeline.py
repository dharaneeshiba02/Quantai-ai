"""
app/utils/import_pipeline.py — Upload and import pipeline for CSV/Excel files.
Imports into MySQL quantai_data schema using the import user.
150 cols × 10,000 rows must complete in under 30 seconds.

Supports:
  - CSV with encoding auto-detection
  - XLS/XLSX with optional sheet selection
  - Multi-sheet Excel workbook listing
  - Transformation logging and cleaning summary
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
MAX_FILE_SIZE_BYTES = 200 * 1024 * 1024  # 200 MB


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


def list_excel_sheets(file_path: Path) -> list[str]:
    """Return sheet names from an Excel workbook for UI sheet selection."""
    suffix = file_path.suffix.lower()
    if suffix not in (".xlsx", ".xls"):
        return []
    try:
        xls = pd.ExcelFile(file_path)
        return xls.sheet_names
    except Exception as exc:
        log.warning("Could not list sheets from %s: %s", file_path.name, exc)
        return []


def _detect_csv_encoding(file_path: Path) -> str:
    """Try multiple encodings and return the one that works."""
    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]:
        try:
            with open(file_path, "r", encoding=enc) as f:
                f.read(8192)
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return "utf-8"


def _detect_csv_delimiter(file_path: Path, encoding: str) -> str:
    """Detect the most likely delimiter in a CSV file."""
    try:
        with open(file_path, "r", encoding=encoding) as f:
            sample = f.read(4096)
        # Count common delimiters in sample
        candidates = {",": 0, ";": 0, "\t": 0, "|": 0}
        for char, _ in candidates.items():
            candidates[char] = sample.count(char)
        best = max(candidates, key=candidates.get)
        # Only use non-comma if it clearly dominates
        if best != "," and candidates[best] > candidates[","] * 2:
            return best
    except Exception:
        pass
    return ","


async def import_file(
    file_path: Path,
    source_name: str,
    sheet_name: str | int | None = None,
) -> dict[str, Any]:
    """
    Import a CSV or Excel file into MySQL quantai_data.
    Returns source metadata including table_name.

    Args:
        file_path: Path to the uploaded file.
        source_name: Original filename for display.
        sheet_name: Optional Excel sheet name or index. Ignored for CSV files.
    """
    t_start = time.perf_counter()
    cfg = get_settings()
    source_id = str(uuid.uuid4())[:8]
    slug = re.sub(r"[^\w]", "_", source_name.lower())[:20]
    table_name = f"upl_{slug}_{source_id}"
    transformations: list[str] = []

    log.info("Importing '%s' as table '%s'...", file_path.name, table_name)

    # ── 0. Validate file ─────────────────────────────────────────────────────
    if not file_path.exists():
        raise FileNotFoundError(f"Upload file not found: {file_path}")

    file_size = file_path.stat().st_size
    if file_size == 0:
        raise ValueError("The uploaded file is empty.")
    if file_size > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"File too large ({file_size / 1024 / 1024:.1f} MB). Maximum is {MAX_FILE_SIZE_BYTES / 1024 / 1024:.0f} MB.")

    # ── 1. Read file ─────────────────────────────────────────────────────────
    t_read = time.perf_counter()
    suffix = file_path.suffix.lower()
    sheet_names: list[str] = []

    if suffix in (".xlsx", ".xls"):
        try:
            xls = pd.ExcelFile(file_path)
            sheet_names = xls.sheet_names
        except Exception as exc:
            raise ValueError(f"Could not read Excel file: {exc}") from exc

        if len(sheet_names) == 0:
            raise ValueError("The Excel workbook contains no sheets.")

        # Select sheet
        selected_sheet = sheet_name if sheet_name is not None else 0
        try:
            df = pd.read_excel(file_path, sheet_name=selected_sheet, dtype=str)
        except Exception as exc:
            raise ValueError(f"Could not read sheet '{selected_sheet}': {exc}") from exc

        if sheet_name is not None:
            transformations.append(f"Selected sheet: '{selected_sheet}'")
    else:
        # CSV with encoding and delimiter detection
        encoding = _detect_csv_encoding(file_path)
        delimiter = _detect_csv_delimiter(file_path, encoding)

        if encoding != "utf-8":
            transformations.append(f"Detected encoding: {encoding}")
        if delimiter != ",":
            transformations.append(f"Detected delimiter: '{delimiter}'")

        try:
            df = pd.read_csv(file_path, dtype=str, encoding=encoding, sep=delimiter)
        except Exception:
            # Final fallback
            df = pd.read_csv(file_path, dtype=str, encoding="utf-8", encoding_errors="replace")
            transformations.append("Used fallback encoding with character replacement")

    t_read_done = time.perf_counter()
    log.info("File read: %.2fs, %d rows × %d cols", t_read_done - t_read, len(df), len(df.columns))

    # Validate content
    if df.empty or len(df.columns) == 0:
        raise ValueError("The file contains no data rows or columns.")
    if len(df.columns) > 500:
        raise ValueError(f"The file has {len(df.columns)} columns, which exceeds the 500-column limit.")

    # ── 2. Normalise column names ────────────────────────────────────────────
    existing_names: set[str] = set()
    original_to_safe: dict[str, str] = {}
    renamed_count = 0
    for orig in df.columns:
        safe = _safe_col_name(str(orig), existing_names)
        original_to_safe[str(orig)] = safe
        if safe != str(orig).lower().strip():
            renamed_count += 1

    df.rename(columns=original_to_safe, inplace=True)
    if renamed_count > 0:
        transformations.append(f"Normalized {renamed_count} column name(s) to safe identifiers")

    # ── 3. Infer types and convert ───────────────────────────────────────────
    from app.utils.type_inference import safe_parse_numeric_series, profile_column_semantics
    col_types: dict[str, str] = {}
    col_semantics: dict[str, dict[str, Any]] = {}
    type_conversions = 0
    for col in df.columns:
        # Safe numeric parsing first (handles commas, currencies, percentages, suppression codes)
        numeric_res = safe_parse_numeric_series(df[col])
        if numeric_res["is_numeric"]:
            df[col] = numeric_res["parsed_series"]
            type_conversions += 1
            if numeric_res["suppression_count"] > 0:
                transformations.append(
                    f"Converted `{col}` to numeric ({numeric_res['parsed_count']:,} parsed, {numeric_res['suppression_count']:,} suppression codes stored as null)"
                )
            else:
                transformations.append(f"Converted `{col}` to numeric ({numeric_res['parsed_count']:,} rows)")
        else:
            samples = df[col].dropna().astype(str).head(10).tolist()
            if _looks_like_date(samples):
                parsed_dates = pd.to_datetime(df[col], errors="coerce")
                if parsed_dates.notna().sum() >= max(1, int(df[col].notna().sum() * 0.5)):
                    df[col] = parsed_dates
                    type_conversions += 1
                    transformations.append(f"Parsed `{col}` as date/datetime")

        col_types[col] = _infer_mysql_type(df[col])
        col_semantics[col] = profile_column_semantics(
            col,
            df[col],
            parsed_numeric_series=df[col] if pd.api.types.is_numeric_dtype(df[col]) else None,
            explicit_unit=numeric_res.get("detected_unit"),
        )

    if type_conversions > 0:
        transformations.append(f"Inferred data types for {type_conversions} column(s)")

    # ── 4. Detect data quality issues ────────────────────────────────────────
    quality_notes: list[str] = []
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        quality_notes.append(f"Found {dup_count} duplicate row(s)")

    missing_cols = [(col, df[col].isna().mean() * 100) for col in df.columns if df[col].isna().any()]
    high_missing = [(col, pct) for col, pct in missing_cols if pct > 50]
    if high_missing:
        names = ", ".join(f"{col} ({pct:.0f}%)" for col, pct in high_missing[:5])
        quality_notes.append(f"High missingness: {names}")

    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        quality_notes.append(f"Constant column(s): {', '.join(constant_cols[:5])}")

    # ── 5. Build CREATE TABLE ────────────────────────────────────────────────
    col_defs = []
    for col, mysql_type in col_types.items():
        col_defs.append(f"  `{col}` {mysql_type}")

    create_sql = (
        f"CREATE TABLE IF NOT EXISTS `{cfg.mysql_data_schema}`.`{table_name}` (\n"
        f"  `_row_id` BIGINT AUTO_INCREMENT PRIMARY KEY,\n"
        + ",\n".join(col_defs)
        + "\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;"
    )

    # ── 6. Create table and bulk insert ─────────────────────────────────────
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

    # ── 7. Build catalog metadata ────────────────────────────────────────────
    catalog_entries = _build_catalog(df, original_to_safe, col_types, col_semantics)

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
        "sheet_names": sheet_names,
        "transformations": transformations,
        "quality_notes": quality_notes,
    }


def _build_catalog(
    df: pd.DataFrame,
    original_to_safe: dict[str, str],
    col_types: dict[str, str],
    col_semantics: dict[str, dict[str, Any]] | None = None,
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
        elif pd.api.types.is_bool_dtype(series):
            col_type = "bool"
        else:
            col_type = "str"

        # Semantic roles and flags
        sem = (col_semantics or {}).get(col, {})
        sem_role = sem.get("role")
        is_date_col = sem.get("is_temporal", False) or col_type in ("date", "datetime")
        is_key_col = sem.get("is_identifier", False) or any(kw in col.lower() for kw in ["id", "key", "code", "_num"])

        col_unit = sem.get("unit")
        if sem_role == "temporal_year":
            is_date_col = True
            col_unit = "year"

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

        # Auto-detect column semantic type for description
        description = sem.get("description") or _infer_column_description(col, col_type, distinct, len(df))

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
            "unit": col_unit,
            "is_date_col": is_date_col,
            "is_key_col": is_key_col,
            "ordinal_position": i,
            "description": description,
        })

    return entries


def _infer_column_description(col_name: str, col_type: str, distinct: int, row_count: int) -> str:
    """Generate a basic semantic description based on column name and stats."""
    name = col_name.lower()

    # Identifiers
    if any(kw in name for kw in ["_id", "id_", "uuid", "serial"]) or name == "id":
        return "Identifier / primary key"
    if name.endswith("_key") or name.endswith("_code"):
        return "Reference key or code"

    # Dates
    if any(kw in name for kw in ["date", "time", "created", "updated", "timestamp"]):
        return "Date or timestamp field"

    # Numeric measures
    if col_type in ("int", "float"):
        if any(kw in name for kw in ["price", "cost", "amount", "revenue", "salary", "income", "profit"]):
            return "Monetary value"
        if any(kw in name for kw in ["count", "quantity", "qty", "num", "total"]):
            return "Count or quantity"
        if any(kw in name for kw in ["score", "rating", "grade", "rank"]):
            return "Score or rating measure"
        if any(kw in name for kw in ["percent", "pct", "ratio", "rate"]):
            return "Percentage or rate"
        if any(kw in name for kw in ["weight", "height", "length", "width", "size", "area", "volume"]):
            return "Physical measurement"

    # Categorical
    if col_type == "str":
        if distinct == 2:
            return "Binary category"
        if distinct <= 20:
            return "Low-cardinality category"
        if any(kw in name for kw in ["name", "title", "label"]):
            return "Name or label"
        if any(kw in name for kw in ["description", "comment", "note", "text"]):
            return "Free-text field"
        if any(kw in name for kw in ["email", "phone", "address"]):
            return "Contact information (PII)"
        if any(kw in name for kw in ["url", "link", "website"]):
            return "URL or link"

    return ""
