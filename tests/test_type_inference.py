"""
tests/test_type_inference.py — Comprehensive unit tests for safe numeric parsing and semantic column detection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.utils.type_inference import (
    parse_numeric_token,
    safe_parse_numeric_series,
    profile_column_semantics,
)


def test_parse_numeric_token_scalars():
    # Plain ints and floats
    val, status, unit = parse_numeric_token(42)
    assert val == 42.0 and status == "numeric" and unit is None

    val, status, unit = parse_numeric_token(3.14159)
    assert val == 3.14159 and status == "numeric" and unit is None

    # Strings with thousands separators
    val, status, unit = parse_numeric_token("1,148")
    assert val == 1148.0 and status == "numeric"

    val, status, unit = parse_numeric_token("1,234,567.89")
    assert val == 1234567.89 and status == "numeric"

    # Currency strings
    val, status, unit = parse_numeric_token("$1,200.50")
    assert val == 1200.50 and status == "numeric" and unit == "currency"

    val, status, unit = parse_numeric_token("€45.00")
    assert val == 45.00 and status == "numeric" and unit == "currency"

    # Percentage strings
    val, status, unit = parse_numeric_token("15.5%")
    assert val == 15.5 and status == "numeric" and unit == "percentage"

    # Parentheses negative
    val, status, unit = parse_numeric_token("(500.25)")
    assert val == -500.25 and status == "numeric"

    val, status, unit = parse_numeric_token("($1,500)")
    assert val == -1500.0 and status == "numeric"

    # Suppression codes
    for code in ("C", "S", "c", "s", "..", "-"):
        val, status, unit = parse_numeric_token(code)
        assert val is None and status == "suppression"

    # Missing tokens
    for token in ("NA", "N/A", "null", "NaN", "", "   "):
        val, status, unit = parse_numeric_token(token)
        assert val is None and status == "missing"

    # Non-numeric string
    val, status, unit = parse_numeric_token("ANZSIC06 Division A")
    assert val is None and status == "unparseable"


def test_safe_parse_numeric_series_with_survey_data_sample():
    # Simulate the actual survey dataset Value column: mixed integers, comma-separated numbers, and suppression codes
    values = ["976077", "854561", "1,148", "1,422", "79", "10", "C", "C", "S", "109848"]
    s = pd.Series(values)

    result = safe_parse_numeric_series(s)
    assert result["is_numeric"] is True
    assert result["parsed_count"] == 7
    assert result["suppression_count"] == 3
    assert result["unparseable_count"] == 0
    parsed = result["parsed_series"]
    assert pd.api.types.is_float_dtype(parsed)
    assert parsed.iloc[2] == 1148.0
    assert parsed.iloc[3] == 1422.0
    assert np.isnan(parsed.iloc[6])  # 'C' -> NaN in numeric series


def test_profile_column_semantics_year_is_never_a_measure():
    # Year must be temporal_year and NOT an analytical measure
    years = pd.Series([2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025])
    profile = profile_column_semantics("Year", years)
    assert profile["role"] == "temporal_year"
    assert profile["is_measure"] is False, "Year must NEVER be marked as an analytical measure!"
    assert profile["is_temporal"] is True
    assert profile["is_group_dimension"] is True


def test_profile_column_semantics_value_is_measure():
    values = pd.Series([1000.0, 50000.0, 12000.0, 4500.0, 300.0, 8900.0, 150000.0])
    profile = profile_column_semantics("Value", values)
    assert profile["role"] == "measure"
    assert profile["is_measure"] is True
    assert profile["is_temporal"] is False


def test_profile_column_semantics_identifier_detection():
    ids = pd.Series([f"NZSIOC_{i}" for i in range(100)])
    profile = profile_column_semantics("Industry_code_NZSIOC", ids)
    assert profile["role"] == "identifier"
    assert profile["is_measure"] is False
    assert profile["is_identifier"] is True
