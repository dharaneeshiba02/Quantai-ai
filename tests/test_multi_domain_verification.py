"""
tests/test_multi_domain_verification.py — Comprehensive multi-domain verification across all test datasets.
"""
from __future__ import annotations

import unittest
from pathlib import Path
import pandas as pd

from app.utils.type_inference import safe_parse_numeric_series, profile_column_semantics
from app.ml.dataset_intelligence import analyze_numeric_features, assess_ml_suitability
from app.rag.summary import detect_text_columns, compute_text_statistics

DATASETS_DIR = Path(__file__).parent.parent / "data" / "test_datasets"


class MultiDomainVerificationTests(unittest.TestCase):
    def test_all_test_datasets_exist(self):
        csv_files = list(DATASETS_DIR.glob("domain_*.csv"))
        self.assertGreaterEqual(len(csv_files), 8, "Expected at least 8 domain test datasets")

    def test_multi_domain_pipeline_execution(self):
        """
        Verify that type inference, semantic column profiling, ML analysis,
        and text detection run cleanly without errors across all domain datasets.
        """
        csv_files = list(DATASETS_DIR.glob("domain_*.csv"))

        for csv_path in csv_files:
            domain_name = csv_path.stem
            df = pd.read_csv(csv_path)

            self.assertGreater(len(df), 10, f"Dataset {domain_name} has too few rows")
            self.assertGreater(len(df.columns), 2, f"Dataset {domain_name} has too few columns")

            # 1. Profile each column
            measures_found = []
            temporals_found = []
            for col in df.columns:
                parsed = safe_parse_numeric_series(df[col])
                sem = profile_column_semantics(
                    str(col),
                    df[col],
                    parsed_numeric_series=parsed["parsed_series"] if parsed["is_numeric"] else None,
                )

                if sem["is_measure"]:
                    measures_found.append(str(col))
                if sem["role"] in ("temporal_year", "date"):
                    temporals_found.append(str(col))

                # Never allow a column named year or id to be classified as a measure
                col_lower = str(col).lower()
                if col_lower in ("year", "yr") or col_lower.endswith("_id") or col_lower == "id":
                    self.assertFalse(
                        sem["is_measure"],
                        f"Column {col} in {domain_name} was incorrectly classified as an analytical measure!",
                    )

            # 2. Check ML suitability & numeric feature analysis
            ml_suit = assess_ml_suitability(df)
            self.assertIn("suitable", ml_suit)

            ml_res = analyze_numeric_features(df, max_features=5, max_clusters=3)
            self.assertIn(ml_res["status"], ("completed", "skipped"))
            if ml_res["status"] == "completed":
                self.assertIn("outliers", ml_res)
                self.assertIn("features", ml_res)

            # 3. Text analytics
            text_cols = detect_text_columns(df)
            if text_cols:
                stats = compute_text_statistics(df[text_cols[0]])
                self.assertGreater(stats["record_count"], 0)

    def test_excel_multi_domain_workbook(self):
        """Verify reading and profiling multi_domain_workbook.xlsx sheets."""
        excel_path = DATASETS_DIR / "multi_domain_workbook.xlsx"
        if not excel_path.exists():
            self.skipTest("multi_domain_workbook.xlsx not present")

        xls = pd.ExcelFile(excel_path)
        self.assertGreater(len(xls.sheet_names), 0)

        for sheet in xls.sheet_names[:3]:
            df = pd.read_excel(xls, sheet_name=sheet)
            self.assertGreater(len(df), 5)
            # Check profiling runs without error
            for col in df.columns[:5]:
                parsed = safe_parse_numeric_series(df[col])
                sem = profile_column_semantics(str(col), df[col])
                self.assertIn("role", sem)


if __name__ == "__main__":
    unittest.main()
