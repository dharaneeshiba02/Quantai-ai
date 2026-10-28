"""
app/agents/cleaner.py — Comprehensive domain-agnostic data cleaning pipeline.
Vectorised Pandas only, no LLM. Works on any dataset (Sales, Healthcare, Education, HR, etc.).

Features:
  - Whitespace trimming and string normalization
  - Currency conversion ($1,200.50, €45.00, ₹10,000 -> float)
  - Percentage conversion (45.5% -> 45.5 or 0.455)
  - Boolean standardization ('yes'/'no', 'true'/'false' -> bool)
  - Robust multi-format date parsing
  - Duplicate row detection and removal
  - Constant (zero-variance) column detection
  - Outlier detection using IQR (1.5*IQR) and Z-score (|z| > 3.0)
  - Configurable missing value handling (none, drop, mean, median, mode, constant)
  - Structured transformation logging and cleaning summary
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Literal

import numpy as np
import pandas as pd

from app.graph.state import AgentState, cache_get, cache_put

log = logging.getLogger(__name__)

_CURRENCY_SYMBOLS = ("$", "€", "£", "¥", "₹", "₱", "₩", "₿", "C$", "A$")
_BOOLEAN_TRUE = {"true", "yes", "y", "t", "1"}
_BOOLEAN_FALSE = {"false", "no", "n", "f", "0"}


async def cleaner_node(state: AgentState) -> dict[str, Any]:
    """
    Clean the query result DataFrame.
    Operates on query result df and records detailed summary and warnings.
    """
    t0 = time.perf_counter()
    step = "cleaner"
    run_id = state["run_id"]

    try:
        df: pd.DataFrame | None = cache_get(run_id, "result_df")
        if df is None or df.empty:
            elapsed = int((time.perf_counter() - t0) * 1000)
            return {
                "current_step": step,
                "trace": state.get("trace", []) + [{"step": step, "status": "skipped_empty", "duration_ms": elapsed}],
            }

        df_cleaned, summary = clean_dataset(df, drop_duplicates=True, missing_strategy="none")
        cache_put(run_id, "result_df", df_cleaned)

        elapsed = int((time.perf_counter() - t0) * 1000)
        return {
            "current_step": step,
            "warnings": state.get("warnings", []) + [f"Cleaning: {c}" for c in summary["changes"]],
            "trace": state.get("trace", []) + [{
                "step": step,
                "status": "completed",
                "duration_ms": elapsed,
                "changes": summary["changes"],
                "cleaning_summary": summary,
            }],
        }

    except Exception as exc:
        log.warning("Cleaner error: %s", exc)
        elapsed = int((time.perf_counter() - t0) * 1000)
        return {
            "current_step": step,
            "trace": state.get("trace", []) + [{"step": step, "status": "warning", "duration_ms": elapsed}],
        }


def clean_dataset(
    df: pd.DataFrame,
    *,
    drop_duplicates: bool = True,
    missing_strategy: Literal["none", "drop", "mean", "median", "mode", "constant"] = "none",
    constant_value: Any = None,
    outlier_threshold_iqr: float = 1.5,
    outlier_threshold_z: float = 3.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Clean a DataFrame using domain-agnostic heuristics.

    Returns:
        (cleaned_df, cleaning_summary_dict)
    """
    changes: list[str] = []
    conversions: dict[str, str] = {}
    outliers: dict[str, dict[str, Any]] = {}
    constant_cols: list[str] = []
    missing_stats: dict[str, int] = {}

    rows_orig = len(df)
    cols_orig = len(df.columns)
    df_clean = df.copy()

    # 1. Duplicate Row Detection & Removal
    dup_count = int(df_clean.duplicated().sum())
    if dup_count > 0:
        if drop_duplicates:
            df_clean = df_clean.drop_duplicates().reset_index(drop=True)
            changes.append(f"Removed {dup_count} duplicate row(s)")
        else:
            changes.append(f"Detected {dup_count} duplicate row(s)")

    # 2. String Cleaning (Whitespace & Control Characters)
    str_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()
    for col in str_cols:
        series = df_clean[col].dropna()
        if len(series) == 0:
            continue
        try:
            # Strip outer whitespace and collapse multiple inner spaces
            cleaned_series = df_clean[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
            # Restore genuine NaNs
            mask = df_clean[col].isna()
            cleaned_series[mask] = np.nan
            df_clean[col] = cleaned_series
        except Exception:
            pass

    # 3. Intelligent Type Casting for String Columns
    # Re-evaluate string columns after whitespace trimming
    str_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()
    for col in str_cols:
        series = df_clean[col].dropna()
        if len(series) == 0:
            continue

        sample = series.head(20).tolist()

        # 3a. Currency String Conversion ($1,200.50 -> 1200.50)
        if _looks_like_currency(sample):
            try:
                numeric_vals = (
                    df_clean[col]
                    .astype(str)
                    .str.replace(r"[\$,€,£,¥,₹,₱,₩,₿\s]", "", regex=True)
                    .str.replace(",", "", regex=False)
                )
                converted = pd.to_numeric(numeric_vals, errors="coerce")
                if converted.notna().sum() >= max(1, int(len(series) * 0.7)):
                    df_clean[col] = converted
                    conversions[col] = "currency -> float"
                    changes.append(f"Converted currency column `{col}` to numeric")
                    continue
            except Exception:
                pass

        # 3b. Percentage String Conversion (45.5% -> 45.5)
        if _looks_like_percentage(sample):
            try:
                numeric_vals = (
                    df_clean[col]
                    .astype(str)
                    .str.replace("%", "", regex=False)
                    .str.strip()
                )
                converted = pd.to_numeric(numeric_vals, errors="coerce")
                if converted.notna().sum() >= max(1, int(len(series) * 0.7)):
                    df_clean[col] = converted
                    conversions[col] = "percentage -> float"
                    changes.append(f"Converted percentage column `{col}` to numeric")
                    continue
            except Exception:
                pass

        # 3c. Boolean String Conversion ('yes'/'no', 'true'/'false')
        if _looks_like_boolean(sample):
            try:
                lowered = df_clean[col].astype(str).str.lower().str.strip()
                bool_map = {**{k: True for k in _BOOLEAN_TRUE}, **{k: False for k in _BOOLEAN_FALSE}}
                mapped = lowered.map(bool_map)
                if mapped.notna().sum() >= max(1, int(len(series) * 0.8)):
                    df_clean[col] = mapped
                    conversions[col] = "boolean-string -> bool"
                    changes.append(f"Converted boolean strings in `{col}` to bool")
                    continue
            except Exception:
                pass

        # 3d. Date String Parsing
        if _looks_like_date(sample):
            try:
                n_before = df_clean[col].isna().sum()
                parsed = pd.to_datetime(df_clean[col], errors="coerce")
                n_after = parsed.isna().sum()
                success_count = int(parsed.notna().sum())
                if success_count >= max(1, int(len(series) * 0.6)):
                    df_clean[col] = parsed
                    conversions[col] = "date-string -> datetime"
                    changes.append(f"Parsed {success_count} datetime values in `{col}`")
                    failed = int(n_after - n_before)
                    if failed > 0:
                        changes.append(f"⚠️ {failed} values could not be parsed as datetime in `{col}`")
                    continue
            except Exception:
                pass

        # 3e. General Thousands-Separated / Suppressed Numeric Strings
        try:
            from app.utils.type_inference import safe_parse_numeric_series
            parse_res = safe_parse_numeric_series(df_clean[col])
            if parse_res["is_numeric"]:
                df_clean[col] = parse_res["parsed_series"]
                conversions[col] = "formatted-string -> float"
                changes.append(
                    f"Converted formatted numeric column `{col}` ({parse_res['parsed_count']:,} values parsed"
                    + (f", {parse_res['suppression_count']:,} suppressed" if parse_res['suppression_count'] > 0 else "")
                    + ")"
                )
                continue
        except Exception:
            pass

    # 4. Numeric Downcasting & Optimization
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        try:
            df_clean[col] = pd.to_numeric(df_clean[col], downcast="float")
        except Exception:
            pass

    # 5. Constant / Zero-Variance Column Detection
    for col in df_clean.columns:
        n_unique = df_clean[col].nunique(dropna=True)
        if n_unique <= 1 and len(df_clean) > 1:
            constant_cols.append(col)
            val = df_clean[col].dropna().iloc[0] if n_unique == 1 else "ALL_NULL"
            changes.append(f"⚠️ Column `{col}` is constant (unique value: {val!r})")

    # 6. Missing Value Imputation / Handling
    for col in df_clean.columns:
        n_null = int(df_clean[col].isna().sum())
        if n_null > 0:
            missing_stats[col] = n_null
            null_pct = n_null / max(len(df_clean), 1) * 100
            if null_pct > 50:
                changes.append(f"⚠️ Column `{col}` has {null_pct:.1f}% missing values")

    if missing_strategy == "drop":
        n_before_drop = len(df_clean)
        df_clean = df_clean.dropna().reset_index(drop=True)
        dropped_missing = n_before_drop - len(df_clean)
        if dropped_missing > 0:
            changes.append(f"Dropped {dropped_missing} rows with missing values")
    elif missing_strategy in ("mean", "median"):
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            if df_clean[col].isna().sum() > 0:
                fill_val = df_clean[col].mean() if missing_strategy == "mean" else df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(fill_val)
                changes.append(f"Imputed missing `{col}` with {missing_strategy} ({fill_val:.3g})")
    elif missing_strategy == "mode":
        for col in df_clean.columns:
            if df_clean[col].isna().sum() > 0:
                mode_vals = df_clean[col].mode()
                if not mode_vals.empty:
                    df_clean[col] = df_clean[col].fillna(mode_vals.iloc[0])
                    changes.append(f"Imputed missing `{col}` with mode ({mode_vals.iloc[0]})")
    elif missing_strategy == "constant" and constant_value is not None:
        df_clean = df_clean.fillna(constant_value)
        changes.append(f"Imputed all missing values with constant {constant_value!r}")

    # 7. Outlier Detection (IQR & Z-score)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        series = df_clean[col].dropna()
        if len(series) < 8:
            continue
        try:
            q25, q75 = np.percentile(series, [25, 75])
            iqr = q75 - q25
            if iqr > 0:
                iqr_lower = float(q25 - outlier_threshold_iqr * iqr)
                iqr_upper = float(q75 + outlier_threshold_iqr * iqr)
                iqr_outliers = series[(series < iqr_lower) | (series > iqr_upper)]

                # Z-Score
                mean_val = series.mean()
                std_val = series.std()
                z_count = 0
                if std_val > 0:
                    z_scores = np.abs((series - mean_val) / std_val)
                    z_count = int((z_scores > outlier_threshold_z).sum())

                if len(iqr_outliers) > 0:
                    outliers[col] = {
                        "iqr_outlier_count": int(len(iqr_outliers)),
                        "iqr_lower_bound": iqr_lower,
                        "iqr_upper_bound": iqr_upper,
                        "z_score_outlier_count": z_count,
                        "pct_outliers": round(len(iqr_outliers) / len(series) * 100, 2),
                    }
                    if len(iqr_outliers) / len(series) >= 0.01:
                        changes.append(
                            f"Detected {len(iqr_outliers)} outlier(s) in `{col}` "
                            f"(bounds: [{iqr_lower:.2f}, {iqr_upper:.2f}])"
                        )
        except Exception:
            pass

    summary = {
        "rows_original": rows_orig,
        "rows_cleaned": len(df_clean),
        "rows_dropped": rows_orig - len(df_clean),
        "duplicates_removed": dup_count if drop_duplicates else 0,
        "columns_original": cols_orig,
        "columns_converted": conversions,
        "constant_columns": constant_cols,
        "outliers_detected": outliers,
        "missing_values_by_col": missing_stats,
        "changes": changes,
    }

    return df_clean, summary


def _clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Legacy helper maintained for backward compatibility."""
    cleaned_df, summary = clean_dataset(df, drop_duplicates=False)
    return cleaned_df, summary["changes"]


def _looks_like_currency(sample: list) -> bool:
    """Check if strings appear to represent monetary amounts."""
    if not sample:
        return False
    hits = 0
    for val in sample:
        s = str(val).strip()
        if any(sym in s for sym in _CURRENCY_SYMBOLS):
            hits += 1
    return hits >= max(1, len(sample) // 2)


def _looks_like_percentage(sample: list) -> bool:
    """Check if strings represent percentage values."""
    if not sample:
        return False
    hits = 0
    for val in sample:
        s = str(val).strip()
        if s.endswith("%"):
            hits += 1
    return hits >= max(1, len(sample) // 2)


def _looks_like_boolean(sample: list) -> bool:
    """Check if strings represent boolean values."""
    if not sample:
        return False
    hits = 0
    valid_set = _BOOLEAN_TRUE | _BOOLEAN_FALSE
    for val in sample:
        s = str(val).lower().strip()
        if s in valid_set:
            hits += 1
    return hits >= max(1, int(len(sample) * 0.8))


def _looks_like_date(sample: list) -> bool:
    """Heuristic: does this column contain date-like strings?"""
    if not sample:
        return False
    date_indicators = ["-", "/", "T", "Z"]
    date_parts = [
        "20", "19", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    hits = 0
    for val in sample:
        s = str(val).strip()
        if any(ind in s for ind in date_indicators) and any(p.lower() in s.lower() for p in date_parts):
            hits += 1
    return hits >= max(1, len(sample) // 2)
