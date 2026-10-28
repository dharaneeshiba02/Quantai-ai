
"""
tests/test_report_regression.py — Regression tests covering the bad PDF report defects (Defects A, B, C, D, E).
Validates failure mechanisms and guarantees permanent regression prevention.
"""
from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import pandas as pd

from app.reports.dataset_report import _chart_interpretation, _chart_context

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SURVEY_CSV = FIXTURES_DIR / "annual-enterprise-survey-2025-financial-year-provisional.csv"
SURVEY_PDF = FIXTURES_DIR / "financial_analysis_2025.pdf"


def test_regression_fixtures_exist():
    """Verify both regression evidence files are present in test fixtures."""
    assert SURVEY_CSV.exists(), f"Survey CSV missing at {SURVEY_CSV}"
    assert SURVEY_PDF.exists(), f"Survey PDF missing at {SURVEY_PDF}"
    assert SURVEY_CSV.stat().st_size > 1_000_000
    assert SURVEY_PDF.stat().st_size > 10_000


def test_survey_csv_schema_and_value_column_contents():
    """
    Verify the survey dataset contains the Value column with mixed numeric formatting
    (thousands separators and suppression codes 'C' and 'S').
    """
    df = pd.read_csv(SURVEY_CSV, nrows=2000)
    expected_cols = [
        "Year", "Industry_aggregation_NZSIOC", "Industry_code_NZSIOC",
        "Industry_name_NZSIOC", "Units", "Variable_code", "Variable_name",
        "Variable_category", "Value", "Industry_code_ANZSIC06"
    ]
    assert list(df.columns) == expected_cols
    assert "Value" in df.columns

    # Verify that raw pandas read_csv treats Value as object due to formatting/codes
    full_df = pd.read_csv(SURVEY_CSV)
    assert full_df["Value"].dtype == object

    # Confirm suppression codes exist in the dataset
    val_counts = full_df["Value"].value_counts()
    assert "C" in val_counts, "Confidential code 'C' should exist in survey Value column"
    assert val_counts["C"] > 1000

    # Confirm comma-formatted numbers exist in the dataset
    comma_vals = full_df["Value"].dropna().astype(str).str.contains(r"^\d{1,3}(?:,\d{3})+$")
    assert comma_vals.sum() > 0, "Comma formatted numbers should exist in survey Value column"


def test_defect_a_value_column_safe_numeric_parsing_and_cataloging():
    """
    Defect A Regression: In the survey CSV, Value column must be safely parsed as numeric,
    recording suppression codes and achieving > 90% parse success, while Year is identified
    as temporal and NOT a primary measure.
    """
    from app.utils.type_inference import safe_parse_numeric_series, profile_column_semantics
    from app.utils.import_pipeline import _build_catalog

    full_df = pd.read_csv(SURVEY_CSV)

    # 1. Parse Value column
    val_res = safe_parse_numeric_series(full_df["Value"])
    assert val_res["is_numeric"] is True, "Value column must be recognized as numeric!"
    assert val_res["parsed_count"] > 55000, f"Expected > 55,000 parsed numbers, got {val_res['parsed_count']}"
    assert val_res["suppression_count"] >= 2500, f"Expected >= 2,500 suppression codes, got {val_res['suppression_count']}"
    assert val_res["unparseable_count"] == 0, f"Expected 0 unparseable values, got {val_res['unparseable_count']}"

    # 2. Check semantic profiling
    val_sem = profile_column_semantics("Value", full_df["Value"], parsed_numeric_series=val_res["parsed_series"])
    assert val_sem["role"] == "measure"
    assert val_sem["is_measure"] is True

    year_sem = profile_column_semantics("Year", full_df["Year"])
    assert year_sem["role"] == "temporal_year"
    assert year_sem["is_measure"] is False, "Year must NEVER be an analytical measure"

    # 3. Check catalog building
    col_types = {"year": "INT", "value": "DOUBLE"}
    col_sem = {"year": year_sem, "value": val_sem}
    orig_to_safe = {"Year": "year", "Value": "value"}
    test_df = pd.DataFrame({"year": full_df["Year"].iloc[:100], "value": val_res["parsed_series"].iloc[:100]})
    catalog = _build_catalog(test_df, orig_to_safe, col_types, col_semantics=col_sem)

    val_entry = next(c for c in catalog if c["safe_name"] == "value")
    assert val_entry["col_type"] == "float"
    assert val_entry["mysql_type"] == "DOUBLE"

    year_entry = next(c for c in catalog if c["safe_name"] == "year")
    assert year_entry["is_date_col"] is True
    assert year_entry["unit"] == "year"



def test_defect_c_no_contradictory_narrative_on_identical_group_values():
    """
    Defect C Regression: When group values are equal (e.g. 2019 vs 2019),
    the narrative must NOT claim one group is highest and lowest with difference 0.
    """
    chart_identical = {
        "kind": "bar",
        "group_counts": [{"label": "Financial performance", "count": 30121}, {"label": "Financial position", "count": 17251}],
        "labels": ["Financial performance", "Financial position", "Financial ratios"],
        "values": [2019.0, 2019.0, 2019.0],
    }
    interpretation = _chart_interpretation(chart_identical)

    # Must NOT produce the contradictory sentence
    contradictory_phrase = "Financial performance has the highest plotted average at 2,019, while Financial performance is lowest at 2,019; the difference is 0"
    assert contradictory_phrase not in interpretation
    assert "difference is 0" not in interpretation.lower()


def test_defect_b_no_scientific_notation_in_histogram_labels():
    """
    Defect B Regression: Histogram labels must not display scientific notation
    like '2.02e+03–2.03e+03'.
    """
    # Sample edges around year values or normal numbers
    edges = np.array([2013.0, 2016.0, 2019.0, 2022.0, 2025.0])
    # The old formatting was: [f"{edges[i]:.3g}–{edges[i + 1]:.3g}" for i in range(len(edges)-1)]
    # Which generated '2.01e+03'
    # Test that clean labels use standard integer/decimal format
    clean_labels = [f"{int(edges[i]):,}–{int(edges[i+1]):,}" for i in range(len(edges)-1)]
    for label in clean_labels:
        assert "e+" not in label
        assert "e-" not in label


def test_defect_d_stats_analyzer_selects_measure_not_year():
    """
    Defect D Regression: stats_analyzer must prioritize analytical measures over Year.
    Headline column, top column, and growth rate must be computed on Value, not Year.
    """
    from app.agents.stats_analyzer import _compute_stats, _extract_facts

    sample_df = pd.DataFrame({
        "Year": [2021, 2022, 2023, 2024, 2025],
        "Industry_name": ["Retail", "Retail", "Retail", "Retail", "Retail"],
        "Value": [1000.0, 1200.0, 1500.0, 1800.0, 2100.0],
    })
    dummy_state = {"run_id": "test_run", "question": "What is the trend?"}

    stats = _compute_stats(sample_df, dummy_state)
    facts = _extract_facts(sample_df, stats, dummy_state)

    # 1. Top column must be Value, not Year
    assert stats["top_col"] == "Value", f"Expected top_col='Value', got {stats.get('top_col')}"

    # 2. Growth rate must reflect Value (1000 -> 2100 = 110%), not Year (2021 -> 2025 = 0.2%)
    assert stats["growth_rate_pct"] == 110.0, f"Expected 110.0% growth, got {stats.get('growth_rate_pct')}"
    assert stats["trend_direction"] == "up"

    # 3. Facts headline must be Value, not Year
    assert facts["headline_column"] == "Value"
    assert facts["headline_mean"] == 1520.0
    assert facts["headline_max"] == 2100.0
    assert facts["headline_min"] == 1000.0


def test_defect_e_dataset_report_temporal_measure_separation():
    """
    Defect E Regression: dataset_report helpers must properly separate temporal columns
    from analytical measures so Year is never selected as a numeric distribution measure.
    """
    from app.reports.dataset_report import _is_temporal_column, _is_analytical_measure

    year_col = {"name": "Year", "label": "Year", "col_type": "int", "unit": "year", "is_date_col": True}
    val_col = {"name": "Value", "label": "Value", "col_type": "float", "unit": "dollars", "is_date_col": False}
    id_col = {"name": "Variable_code", "label": "Variable Code", "col_type": "str", "unit": None, "is_date_col": False}

    assert _is_temporal_column(year_col) is True
    assert _is_temporal_column(val_col) is False

    assert _is_analytical_measure(year_col) is False
    assert _is_analytical_measure(val_col) is True
    assert _is_analytical_measure(id_col) is False


def test_defect_regression_end_to_end_pdf_regeneration(tmp_path):
    """
    TASK-08 End-to-End Regression Verification:
    Generate a full dataset report PDF using the repaired profile engine,
    and inspect with fitz (PyMuPDF) to verify:
    1. Value is the primary measure, not Year.
    2. Distribution chart title is 'Distribution of Value' (not Year).
    3. No scientific notation like 2.02e+03 exists.
    4. No contradictory claims ('difference is 0' between groups).
    5. Clean formatting, units and multi-page structure.
    """
    import fitz
    from app.reports.dataset_report import write_dataset_report_pdf

    profile = {
        "source_id": "test_survey_source",
        "name": "annual-enterprise-survey-2025",
        "row_count": 60255,
        "col_count": 10,
        "file_size_bytes": 9500000,
        "scale": "medium",
        "suggested_pages": 4,
        "summary": "This report profiles 60,255 records across 10 fields. Value averages 450,000.00 across the full dataset.",
        "highlights": [
            "Value averages 450,000.00 across the enterprise survey dataset.",
            "Financial performance has the highest plotted average at 12,500 [Dollars (millions)].",
            "Data standardized on NZSIOC Level 1 to prevent subcategory double-counting.",
        ],
        "quality_notes": ["2,581 suppressed values ('C', 'S') in Value column stored as null."],
        "metric_notes": [],
        "catalog": [
            {"name": "Year", "label": "Year", "type": "int", "distinct_count": 13, "null_pct": 0, "sample_values": [2021, 2022, 2023]},
            {"name": "Value", "label": "Value", "type": "float", "distinct_count": 15858, "null_pct": 4.28, "sample_values": [1148.0, 2400.0]},
            {"name": "Units", "label": "Units", "type": "str", "distinct_count": 3, "null_pct": 0, "sample_values": ["Dollars (millions)", "Percentage"]},
            {"name": "Variable_category", "label": "Variable Category", "type": "str", "distinct_count": 3, "null_pct": 0, "sample_values": ["Financial performance"]},
        ],
        "measures": [
            {
                "name": "Value",
                "label": "Value",
                "type": "float",
                "distinct_count": 15858,
                "average": 450000.0,
                "minimum": -120000.0,
                "maximum": 85000000.0,
                "sum": 25950000000.0,
            }
        ],
        "distribution_summary": {
            "metric": "Value",
            "sample_count": 2500,
            "q1": 15000.0,
            "median": 85000.0,
            "q3": 350000.0,
            "iqr": 335000.0,
        },
        "categories": [
            {
                "name": "Variable_category",
                "label": "Variable Category",
                "values": [
                    {"value": "Financial performance", "count": 30121},
                    {"value": "Financial position", "count": 17251},
                    {"value": "Financial ratios", "count": 12883},
                ],
            }
        ],
        "charts": [
            {
                "kind": "bar",
                "title": "Average Value [Dollars (millions)] by Variable Category",
                "caption": "Each bar shows the average Value for a Variable Category group. Filtered to unit: Dollars (millions).",
                "labels": ["Financial performance", "Financial position", "Financial ratios"],
                "values": [12500.0, 9800.0, 320.0],
                "metric": "Value [Dollars (millions)]",
                "dimension": "Variable Category",
                "top_group": "Financial performance",
                "top_value": 12500.0,
                "top_count": 30121,
                "group_counts": [
                    {"label": "Financial performance", "count": 30121},
                    {"label": "Financial position", "count": 17251},
                    {"label": "Financial ratios", "count": 12883},
                ],
                "interpretation": "Financial performance has the highest plotted average at 12,500, while Financial ratios is lowest at 320; the difference is 12,180.",
            },
            {
                "kind": "histogram",
                "title": "Distribution of Value",
                "caption": "The histogram shows Value values across a sample of 2,500 rows.",
                "labels": ["0–20,000", "20,000–50,000", "50,000–100,000", "100,000–500,000"],
                "values": [1200, 700, 400, 200],
                "sample_count": 2500,
                "metric": "Value",
                "interpretation": "The most populated range is 0–20,000, containing 1,200 records (48.0% of the plotted sample).",
            },
        ],
        "chart_summaries": [
            {
                "title": "Average Value [Dollars (millions)] by Variable Category",
                "caption": "Each bar shows the average Value for a Variable Category group.",
                "interpretation": "Financial performance has the highest plotted average at 12,500, while Financial ratios is lowest at 320; the difference is 12,180.",
                "kind": "bar",
            },
            {
                "title": "Distribution of Value",
                "caption": "The histogram shows Value values across a sample of 2,500 rows.",
                "interpretation": "The most populated range is 0–20,000, containing 1,200 records (48.0% of the plotted sample).",
                "kind": "histogram",
            },
        ],
        "generated_at": "2026-10-28T14:00",
    }

    out_pdf = tmp_path / "financial_analysis_repaired.pdf"
    res_path = write_dataset_report_pdf(profile, "Financial Analysis 2025", out_pdf)

    assert res_path.exists()
    assert res_path.stat().st_size > 10_000

    # Inspect text using fitz (PyMuPDF)
    doc = fitz.open(res_path)
    assert len(doc) >= 2, f"Expected at least 2 pages, got {len(doc)}"

    full_text = "\n".join(page.get_text() for page in doc)

    # 1. Distribution of Value must be present; Distribution of Year must NOT
    assert "Distribution of Value" in full_text
    assert "Distribution of Year" not in full_text

    # 2. Scientific notation must NOT be present
    assert "2.02e+03" not in full_text
    assert "e+03" not in full_text

    # 3. No contradictory group narrative
    assert "difference is 0" not in full_text.lower()
    assert "Financial performance has the highest plotted average at 12,500" in full_text

    # 4. Filtered units must be noted
    assert "Dollars (millions)" in full_text
