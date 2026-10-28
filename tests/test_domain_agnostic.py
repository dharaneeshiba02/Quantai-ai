"""
tests/test_domain_agnostic.py — Comprehensive test suite validating that QuantAI
is fully domain-agnostic, capable of processing data from any domain (Healthcare,
Education, HR, Sales, Logistics, Real Estate, Sports, etc.).
"""
from __future__ import annotations

import io
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agents.cleaner import clean_dataset, _clean_dataframe
from app.agents.visualizer import _select_chart_type, _build_chart, _apply_theme
from app.agents.data_summary import _METRIC_HINTS, _DIMENSION_HINTS
from app.main import _generate_suggestions
from app.reports.dataset_report import _draw_chart, _chart_interpretation, _chart_context


# ── 1. Data Cleaning Pipeline Tests ──────────────────────────────────────────

def test_currency_string_conversion():
    df = pd.DataFrame({
        "revenue": ["$1,200.50", "$3,450.00", "$500.25", "$12,000.00", "$95.10"],
        "cost_eur": ["€100.00", "€250.50", "€1,100.00", "€45.00", "€800.00"],
    })
    df_clean, summary = clean_dataset(df)

    assert pd.api.types.is_numeric_dtype(df_clean["revenue"])
    assert pd.api.types.is_numeric_dtype(df_clean["cost_eur"])
    assert df_clean["revenue"].iloc[0] == 1200.50
    assert df_clean["cost_eur"].iloc[2] == 1100.00
    assert "revenue" in summary["columns_converted"]
    assert "cost_eur" in summary["columns_converted"]


def test_percentage_string_conversion():
    df = pd.DataFrame({
        "tax_rate": ["5.5%", "12.0%", "20%", "0.0%", "15.2%"],
        "grade_pct": ["85%", "92%", "78%", "95%", "88%"],
    })
    df_clean, summary = clean_dataset(df)

    assert pd.api.types.is_numeric_dtype(df_clean["tax_rate"])
    assert pd.api.types.is_numeric_dtype(df_clean["grade_pct"])
    assert df_clean["tax_rate"].iloc[0] == 5.5
    assert df_clean["grade_pct"].iloc[1] == 92.0


def test_boolean_string_conversion():
    df = pd.DataFrame({
        "is_active": ["yes", "no", "yes", "yes", "no"],
        "readmitted": ["true", "false", "false", "true", "false"],
    })
    df_clean, summary = clean_dataset(df)

    assert pd.api.types.is_bool_dtype(df_clean["is_active"])
    assert pd.api.types.is_bool_dtype(df_clean["readmitted"])
    assert df_clean["is_active"].iloc[0] is True or df_clean["is_active"].iloc[0] == 1
    assert df_clean["readmitted"].iloc[1] is False or df_clean["readmitted"].iloc[1] == 0


def test_duplicate_row_removal():
    df = pd.DataFrame({
        "id": [1, 2, 2, 3, 4, 4],
        "val": ["A", "B", "B", "C", "D", "D"],
    })
    df_clean, summary = clean_dataset(df, drop_duplicates=True)

    assert len(df_clean) == 4
    assert summary["duplicates_removed"] == 2


def test_constant_column_detection():
    df = pd.DataFrame({
        "metric": [10, 20, 30, 40],
        "constant_flag": ["STATIC", "STATIC", "STATIC", "STATIC"],
    })
    _, summary = clean_dataset(df)

    assert "constant_flag" in summary["constant_columns"]


def test_outlier_detection_iqr():
    # 20 normal values around 50, plus 2 extreme outliers
    vals = [50, 52, 49, 51, 48, 53, 50, 49, 51, 50, 52, 48, 50, 51, 49, 50, 52, 49, 51, 50, 500, -200]
    df = pd.DataFrame({"systolic_bp": vals})
    _, summary = clean_dataset(df)

    assert "systolic_bp" in summary["outliers_detected"]
    assert summary["outliers_detected"]["systolic_bp"]["iqr_outlier_count"] >= 2


def test_missing_value_imputation():
    df = pd.DataFrame({
        "score": [10.0, 20.0, np.nan, 30.0, 40.0],
    })
    df_clean, summary = clean_dataset(df, missing_strategy="mean")

    assert df_clean["score"].isna().sum() == 0
    assert df_clean["score"].iloc[2] == 25.0


# ── 2. Visualizer Chart Selection Tests ──────────────────────────────────────

def test_chart_selection_time_series():
    df = pd.DataFrame({
        "order_date": pd.date_range("2024-01-01", periods=10, freq="D"),
        "revenue": np.random.uniform(100, 500, 10),
    })
    chart_type = _select_chart_type(df, {})
    assert chart_type == "line"


def test_chart_selection_category_ranking():
    df = pd.DataFrame({
        "department": ["Engineering", "Product", "Sales", "HR"],
        "avg_salary": [120000, 110000, 95000, 75000],
    })
    chart_type = _select_chart_type(df, {}, question="Which department has the highest salary?")
    assert chart_type in ("bar", "horizontal_bar")


def test_chart_selection_pie_breakdown():
    df = pd.DataFrame({
        "diagnosis": ["Hypertension", "Diabetes", "Asthma", "Other"],
        "patient_count": [120, 85, 45, 30],
    })
    chart_type = _select_chart_type(df, {}, question="What is the proportion of patient diagnoses?")
    assert chart_type == "pie"


def test_chart_selection_box_plot():
    df = pd.DataFrame({
        "test_score": [65, 70, 75, 80, 85, 90, 95],
    })
    chart_type = _select_chart_type(df, {}, question="Show the boxplot spread of test scores")
    assert chart_type == "box"


def test_chart_selection_correlation_heatmap():
    df = pd.DataFrame({
        "ad_spend": [1000, 2000, 3000, 4000],
        "impressions": [10000, 25000, 32000, 48000],
        "conversions": [50, 120, 180, 240],
    })
    chart_type = _select_chart_type(df, {}, question="Show the correlation matrix between metrics")
    assert chart_type == "heatmap"


# ── 3. Domain Agnostic Hints & Suggestions Tests ─────────────────────────────

def test_no_restaurant_bias_in_hints():
    for word in ("cuisine", "booking", "delivery", "restaurant", "menu", "food"):
        assert word not in _DIMENSION_HINTS, f"Found restaurant word '{word}' in _DIMENSION_HINTS"
        assert word not in _METRIC_HINTS, f"Found restaurant word '{word}' in _METRIC_HINTS"


def test_domain_aware_suggestions_healthcare():
    class MockCol:
        def __init__(self, safe_name, col_type, is_date_col=False, distinct_count=10):
            self.safe_name = safe_name
            self.original_name = safe_name.replace("_", " ").title()
            self.col_type = col_type
            self.is_date_col = is_date_col
            self.distinct_count = distinct_count
            self.is_key_col = False

    cols = [
        MockCol("patient_id", "int"),
        MockCol("blood_pressure", "float"),
        MockCol("department", "str", distinct_count=5),
        MockCol("admit_date", "date", is_date_col=True),
    ]
    suggestions = _generate_suggestions(cols, None)

    assert len(suggestions) >= 4
    # Check that questions are relevant to medical measures
    joined = " ".join(suggestions)
    assert "blood pressure" in joined.lower() or "department" in joined.lower()


def test_domain_aware_suggestions_education():
    class MockCol:
        def __init__(self, safe_name, col_type, is_date_col=False, distinct_count=10):
            self.safe_name = safe_name
            self.original_name = safe_name.replace("_", " ").title()
            self.col_type = col_type
            self.is_date_col = is_date_col
            self.distinct_count = distinct_count
            self.is_key_col = False

    cols = [
        MockCol("student_id", "str"),
        MockCol("final_exam_score", "float"),
        MockCol("grade_level", "str", distinct_count=4),
        MockCol("exam_date", "date", is_date_col=True),
    ]
    suggestions = _generate_suggestions(cols, None)

    assert len(suggestions) >= 4
    joined = " ".join(suggestions)
    assert "final exam score" in joined.lower() or "grade level" in joined.lower()


# ── 4. Dataset Report Chart Rendering Tests ──────────────────────────────────

def test_draw_chart_heatmap():
    chart = {
        "kind": "heatmap",
        "labels": ["Revenue", "Cost", "Profit"],
        "matrix": [
            [1.0, 0.85, 0.72],
            [0.85, 1.0, 0.40],
            [0.72, 0.40, 1.0],
        ],
    }
    img_data = _draw_chart(chart)
    assert img_data is not None
    assert isinstance(img_data, io.BytesIO)
    assert len(img_data.getvalue()) > 500


def test_draw_chart_missing_bar():
    chart = {
        "kind": "missing_bar",
        "labels": ["Phone", "Address", "Secondary Email"],
        "values": [45.2, 28.0, 12.5],
    }
    img_data = _draw_chart(chart)
    assert img_data is not None
    assert isinstance(img_data, io.BytesIO)
    assert len(img_data.getvalue()) > 500


def test_draw_chart_bar_and_line():
    chart_bar = {
        "kind": "bar",
        "labels": ["Cardiology", "Neurology", "Pediatrics"],
        "values": [120, 85, 95],
    }
    img_bar = _draw_chart(chart_bar)
    assert img_bar is not None

    chart_line = {
        "kind": "line",
        "labels": ["2024-01", "2024-02", "2024-03"],
        "values": [50000, 62000, 58000],
    }
    img_line = _draw_chart(chart_line)
    assert img_line is not None


def test_chart_context_and_interpretation():
    chart_hm = {"kind": "heatmap", "metric": "test"}
    purpose, use = _chart_context(chart_hm)
    assert "correlation" in purpose.lower()
    interp = _chart_interpretation(chart_hm)
    assert "correlation" in interp.lower()

    chart_missing = {"kind": "missing_bar", "labels": ["col_a", "col_b"], "values": [35.0, 15.0]}
    purpose_m, _ = _chart_context(chart_missing)
    assert "missing" in purpose_m.lower()
    interp_m = _chart_interpretation(chart_missing)
    assert "col_a" in interp_m and "35.0%" in interp_m


# ── 5. Multi-Sheet Excel Ingestion Test ───────────────────────────────────────

def test_multi_sheet_excel_workbook():
    test_excel = Path("data/test_datasets/multi_domain_workbook.xlsx")
    if test_excel.exists():
        xls = pd.ExcelFile(test_excel)
        expected_sheets = {"Sales_Finance", "Healthcare_Patients", "HR_Employees", "Student_Education"}
        assert expected_sheets.issubset(set(xls.sheet_names))

        # Test reading individual sheets
        df_sales = pd.read_excel(test_excel, sheet_name="Sales_Finance")
        assert "revenue" in df_sales.columns
        assert len(df_sales) > 0

        df_health = pd.read_excel(test_excel, sheet_name="Healthcare_Patients")
        assert "patient_id" in df_health.columns or "primary_diagnosis" in df_health.columns
        assert len(df_health) > 0
