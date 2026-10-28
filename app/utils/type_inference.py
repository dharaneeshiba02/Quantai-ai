"""
app/utils/type_inference.py — General-purpose semantic column detection and safe numeric parsing.
Vectorised, deterministic, domain-agnostic. Works across all industry domains.

Features:
  - Distinguishes storage dtype from analytical semantic role.
  - Safely parses thousands separators, currencies, percentages, negative numbers, parentheses.
  - Detects statistical suppression codes ('C', 'S') and documented missing tokens.
  - Maintains conversion audits (success, suppression, unparseable counts).
  - Categorizes column roles: measure, temporal_year, date, identifier, categorical, boolean, text.
  - Excludes temporal years, IDs, codes, and constant fields from default analytical measures.
"""
from __future__ import annotations

import re
from typing import Any
import numpy as np
import pandas as pd

# Statistical suppression and common missing value representations
SUPPRESSION_CODES = frozenset({"c", "s", "confidential", "suppressed", "..", "...", "-", "--", "nil"})
MISSING_TOKENS = frozenset({"na", "n/a", "nan", "null", "none", "", "empty", "blank", "not available", "nd", "n.d."})

# Currency symbols across global currencies
CURRENCY_SYMBOLS = ("$", "€", "£", "¥", "₹", "₱", "₩", "₿", "A$", "C$", "NZ$", "R$", "kr", "CHF")

# Regex patterns for numeric strings
_RE_PAREN_NEG = re.compile(r"^\s*\(\s*(.*?)\s*\)\s*$")
_RE_PERCENT = re.compile(r"^\s*(.*?)\s*%\s*$")
_RE_THOUSANDS_COMMA = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")
_RE_PLAIN_NUMBER = re.compile(r"^[+-]?\d+(\.\d+)?$")
_RE_YEAR = re.compile(r"^(19\d\d|20\d\d|2100)$")
_RE_IDENTIFIER_NAME = re.compile(r"(^|_)(id|key|code|num|no|serial|uuid|guid|nzsioc|anzsic|ssn)($|_)", re.IGNORECASE)


def parse_numeric_token(val: Any) -> tuple[float | None, str, str | None]:
    """
    Parse a single scalar value into a float, returning (parsed_float, status, unit).
    Status is one of: 'numeric', 'suppression', 'missing', 'unparseable'.
    Unit is one of: 'currency', 'percentage', None.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None, "missing", None

    if isinstance(val, (int, np.integer)):
        return float(val), "numeric", None

    if isinstance(val, (float, np.floating)):
        return float(val), "numeric", None

    s = str(val).strip()
    if not s:
        return None, "missing", None

    lowered = s.lower()
    if lowered in MISSING_TOKENS:
        return None, "missing", None
    if lowered in SUPPRESSION_CODES:
        return None, "suppression", None

    # Check for parentheses negative: e.g. (1,200.50)
    is_neg = False
    paren_match = _RE_PAREN_NEG.match(s)
    if paren_match:
        s = paren_match.group(1).strip()
        is_neg = True

    # Check for percentage: e.g. 15.5%
    unit = None
    pct_match = _RE_PERCENT.match(s)
    if pct_match:
        s = pct_match.group(1).strip()
        unit = "percentage"

    # Check for currency symbols prefix/suffix
    for sym in CURRENCY_SYMBOLS:
        if s.startswith(sym):
            s = s[len(sym):].strip()
            unit = "currency"
            break
        elif s.endswith(sym):
            s = s[:-len(sym)].strip()
            unit = "currency"
            break

    # Strip remaining commas if valid thousands separator pattern
    if _RE_THOUSANDS_COMMA.match(s):
        s = s.replace(",", "")
    elif "," in s and not "." in s:
        # Check if single comma is a decimal comma (e.g. European 12,50)
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) in (1, 2) and parts[0].lstrip("+-").isdigit() and parts[1].isdigit():
            s = f"{parts[0]}.{parts[1]}"
        elif len(parts) == 2 and len(parts[1]) == 3 and parts[0].lstrip("+-").isdigit() and parts[1].isdigit():
            # Could be thousand 1,000
            s = parts[0] + parts[1]

    try:
        num = float(s)
        if is_neg:
            num = -num
        return num, "numeric", unit
    except (ValueError, TypeError):
        return None, "unparseable", None


def safe_parse_numeric_series(
    series: pd.Series,
    min_success_ratio: float = 0.70,
) -> dict[str, Any]:
    """
    Evaluate and safely convert a Pandas Series to numeric if it qualifies.
    Preserves original data and records conversion statistics.

    Returns dict containing:
      - is_numeric: bool
      - parsed_series: pd.Series (float with NaNs)
      - total_count: int
      - non_null_count: int
      - parsed_count: int
      - suppression_count: int
      - missing_count: int
      - unparseable_count: int
      - success_ratio: float
      - detected_unit: str | None
      - conversion_notes: list[str]
    """
    total = len(series)
    if total == 0:
        return {
            "is_numeric": False,
            "parsed_series": series,
            "total_count": 0,
            "non_null_count": 0,
            "parsed_count": 0,
            "suppression_count": 0,
            "missing_count": 0,
            "unparseable_count": 0,
            "success_ratio": 0.0,
            "detected_unit": None,
            "conversion_notes": ["Series is empty"],
        }

    # If already native numeric:
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        non_null = int(series.notna().sum())
        return {
            "is_numeric": True,
            "parsed_series": series.astype(float),
            "total_count": total,
            "non_null_count": non_null,
            "parsed_count": non_null,
            "suppression_count": 0,
            "missing_count": total - non_null,
            "unparseable_count": 0,
            "success_ratio": 1.0 if non_null > 0 else 0.0,
            "detected_unit": None,
            "conversion_notes": ["Native numeric type preserved"],
        }

    # Evaluate each non-null value
    parsed_vals: list[float | None] = []
    statuses: list[str] = []
    units: list[str | None] = []

    for val in series:
        p_val, status, unit = parse_numeric_token(val)
        parsed_vals.append(p_val)
        statuses.append(status)
        units.append(unit)

    parsed_count = statuses.count("numeric")
    suppression_count = statuses.count("suppression")
    missing_count = statuses.count("missing")
    unparseable_count = statuses.count("unparseable")
    non_null_eval = total - missing_count

    success_ratio = (parsed_count / non_null_eval) if non_null_eval > 0 else 0.0
    known_ratio = ((parsed_count + suppression_count) / non_null_eval) if non_null_eval > 0 else 0.0

    # Determine if qualified as numeric column
    is_numeric = (
        parsed_count >= 2
        and (success_ratio >= min_success_ratio or (known_ratio >= 0.90 and parsed_count >= 5))
        and unparseable_count <= max(1, int(non_null_eval * (1 - min_success_ratio)))
    )

    detected_unit = None
    unit_counts = pd.Series([u for u in units if u is not None]).value_counts()
    if not unit_counts.empty and unit_counts.iloc[0] >= parsed_count * 0.5:
        detected_unit = unit_counts.index[0]

    notes = []
    if is_numeric:
        notes.append(f"Parsed {parsed_count:,}/{non_null_eval:,} values as numeric ({success_ratio * 100:.1f}%)")
        if suppression_count > 0:
            notes.append(f"Recorded {suppression_count:,} suppression/confidential code(s) as nulls")
        if unparseable_count > 0:
            notes.append(f"{unparseable_count:,} unparseable value(s) stored as nulls")
        parsed_s = pd.Series(parsed_vals, index=series.index, dtype=float)
    else:
        notes.append(f"Rejected as numeric: only {success_ratio * 100:.1f}% parseable")
        parsed_s = series

    return {
        "is_numeric": is_numeric,
        "parsed_series": parsed_s,
        "total_count": total,
        "non_null_count": non_null_eval,
        "parsed_count": parsed_count,
        "suppression_count": suppression_count,
        "missing_count": missing_count,
        "unparseable_count": unparseable_count,
        "success_ratio": success_ratio,
        "detected_unit": detected_unit,
        "conversion_notes": notes,
    }


def profile_column_semantics(
    col_name: str,
    original_series: pd.Series,
    parsed_numeric_series: pd.Series | None = None,
    explicit_unit: str | None = None,
) -> dict[str, Any]:
    """
    Profile a column's semantic analytical role, distinguishing storage type from analytical role.

    Possible roles:
      - 'temporal_year': Calendar year (e.g. 2013-2025). Ordinal/temporal, NOT an analytical measure.
      - 'date': Date or timestamp.
      - 'identifier': ID, key, code, serial number.
      - 'measure': Quantitative metric suitable for statistics, averages, and charts.
      - 'categorical': Categorical dimension suitable for grouping.
      - 'boolean': Binary flag.
      - 'text': High-cardinality descriptive text.
    """
    clean_name = str(col_name).strip()
    lowered_name = clean_name.lower()
    total = len(original_series)
    non_null_series = original_series.dropna()
    distinct_count = int(original_series.nunique(dropna=True))
    null_count = int(original_series.isna().sum())
    null_pct = (null_count / max(total, 1)) * 100.0

    # Check for boolean
    if pd.api.types.is_bool_dtype(original_series):
        return {
            "name": clean_name,
            "role": "boolean",
            "is_measure": False,
            "is_group_dimension": True,
            "is_temporal": False,
            "is_identifier": False,
            "distinct_count": distinct_count,
            "null_pct": null_pct,
            "unit": None,
            "description": "Binary flag",
        }

    # Check for Date/Datetime
    if pd.api.types.is_datetime64_any_dtype(original_series):
        return {
            "name": clean_name,
            "role": "date",
            "is_measure": False,
            "is_group_dimension": False,
            "is_temporal": True,
            "is_identifier": False,
            "distinct_count": distinct_count,
            "null_pct": null_pct,
            "unit": None,
            "description": "Date or timestamp",
        }

    # Use parsed numeric series if available, otherwise check original
    num_series = parsed_numeric_series if parsed_numeric_series is not None and pd.api.types.is_numeric_dtype(parsed_numeric_series) else None
    if num_series is None and pd.api.types.is_numeric_dtype(original_series) and not pd.api.types.is_bool_dtype(original_series):
        num_series = original_series.astype(float)

    # 1. Check for Year column (Temporal Year)
    if num_series is not None:
        valid_nums = num_series.dropna()
        if len(valid_nums) > 0:
            is_int_like = (valid_nums == valid_nums.round()).all()
            min_val = float(valid_nums.min())
            max_val = float(valid_nums.max())
            # If name contains 'year' or values are bounded in realistic year range [1800, 2100]
            if ("year" in lowered_name or (1800 <= min_val <= 2090 and 1800 <= max_val <= 2100)) and is_int_like and distinct_count <= 250:
                return {
                    "name": clean_name,
                    "role": "temporal_year",
                    "is_measure": False,  # CRITICAL: Year is NOT an analytical measure!
                    "is_group_dimension": True,
                    "is_temporal": True,
                    "is_identifier": False,
                    "distinct_count": distinct_count,
                    "null_pct": null_pct,
                    "unit": "year",
                    "min": min_val,
                    "max": max_val,
                    "description": "Calendar year (temporal dimension)",
                }

    # 2. Check for Identifier / Code
    is_id_name = bool(_RE_IDENTIFIER_NAME.search(lowered_name))
    if is_id_name:
        is_measure = False
        return {
            "name": clean_name,
            "role": "identifier",
            "is_measure": False,
            "is_group_dimension": distinct_count <= 50,
            "is_temporal": False,
            "is_identifier": True,
            "distinct_count": distinct_count,
            "null_pct": null_pct,
            "unit": None,
            "description": "Identifier or reference code",
        }

    # 3. Numeric Measure
    if num_series is not None:
        valid_nums = num_series.dropna()
        if len(valid_nums) > 0:
            min_val = float(valid_nums.min())
            max_val = float(valid_nums.max())
            is_constant = (min_val == max_val)

            # High distinct count or clearly a metric
            is_measure = (not is_constant) and (distinct_count > 5 or any(k in lowered_name for k in ("val", "amount", "cost", "rev", "score", "rate", "pct", "pct", "total", "income", "expense", "tax", "wage", "stock", "ratio", "asset", "liabilit")))

            return {
                "name": clean_name,
                "role": "measure" if is_measure else "categorical",
                "is_measure": is_measure,
                "is_group_dimension": distinct_count <= 25,
                "is_temporal": False,
                "is_identifier": False,
                "distinct_count": distinct_count,
                "null_pct": null_pct,
                "unit": explicit_unit,
                "min": min_val,
                "max": max_val,
                "description": f"Numeric measure (range: {min_val:,.2f} to {max_val:,.2f})" if is_measure else "Discrete numeric code",
            }

    # 4. String / Categorical / Text
    str_vals = non_null_series.astype(str)
    avg_len = str_vals.str.len().mean() if len(str_vals) > 0 else 0
    is_text = avg_len > 60 or distinct_count > 1000

    return {
        "name": clean_name,
        "role": "text" if is_text else "categorical",
        "is_measure": False,
        "is_group_dimension": 2 <= distinct_count <= 100,
        "is_temporal": False,
        "is_identifier": distinct_count == total and total > 5,
        "distinct_count": distinct_count,
        "null_pct": null_pct,
        "unit": None,
        "description": "Free text field" if is_text else f"Categorical dimension ({distinct_count} categories)",
    }
