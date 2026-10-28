"""
tests/test_dataset_intelligence.py — Comprehensive tests for dataset ML engine.
"""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from app.ml.dataset_intelligence import (
    analyze_numeric_features,
    assess_ml_suitability,
    summarize_ml_analysis,
    train_baseline_model,
)


class DatasetIntelligenceTests(unittest.TestCase):
    def test_numeric_analysis_is_explainable_and_does_not_mutate_input(self):
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "revenue": np.random.normal(1000, 150, n),
            "orders": np.random.randint(10, 100, n),
            "returns": np.random.randint(0, 15, n),
        })
        # Inject one clear outlier
        df.loc[0, "revenue"] = 85000.0

        original_df = df.copy(deep=True)

        res = analyze_numeric_features(df, max_features=3, max_clusters=3)

        # 1. Output structure
        self.assertEqual(res["status"], "completed")
        self.assertEqual(res["outliers"]["algorithm"], "Isolation Forest")
        self.assertGreater(res["outliers"]["count"], 0)
        self.assertEqual(res["features"], ["revenue", "orders", "returns"])
        self.assertIn("original uploaded dataset remains unchanged", res["methodology"])

        # 2. Immutability check
        pd.testing.assert_frame_equal(df, original_df)

        # 3. Summary check
        summary = summarize_ml_analysis(res)
        self.assertIn("Isolation Forest flagged", summary)

    def test_non_numeric_data_is_skipped_without_error(self):
        df = pd.DataFrame({
            "team": ["A", "B", "C", "A", "B"],
        })
        res = analyze_numeric_features(df)
        self.assertEqual(res["status"], "skipped")
        self.assertEqual(res["features"], [])

    def test_suitability_assessment(self):
        np.random.seed(42)
        n = 50
        df = pd.DataFrame({
            "feature_a": np.random.normal(50, 10, n),
            "feature_b": np.random.normal(100, 20, n),
            "category": np.random.choice(["High", "Medium", "Low"], n),
            "target_num": np.random.normal(200, 30, n),
        })

        suitability = assess_ml_suitability(df)
        self.assertTrue(suitability["suitable"])
        tasks = suitability["tasks"]
        self.assertTrue(tasks["anomaly_detection"]["suitable"])
        self.assertTrue(tasks["clustering"]["suitable"])
        self.assertTrue(tasks["classification"]["suitable"])
        self.assertIn("category", tasks["classification"]["candidate_targets"])
        self.assertTrue(tasks["regression"]["suitable"])
        self.assertIn("target_num", tasks["regression"]["candidate_targets"])

    def test_supervised_baseline_classification(self):
        np.random.seed(42)
        n = 60
        X = np.random.normal(0, 1, (n, 3))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        df = pd.DataFrame(X, columns=["f1", "f2", "f3"])
        df["target"] = y

        res = train_baseline_model(df, target_col="target", task_type="classification")
        self.assertEqual(res["status"], "completed")
        self.assertEqual(res["task_type"], "classification")
        self.assertIn("accuracy", res["metrics"])
        self.assertIn("f1_score_weighted", res["metrics"])
        self.assertGreater(res["metrics"]["accuracy"], 0.5)
        self.assertIn("top_features", res)

    def test_supervised_baseline_regression(self):
        np.random.seed(42)
        n = 60
        X = np.random.normal(10, 2, (n, 3))
        y = X[:, 0] * 2.5 + X[:, 1] * 1.5 + np.random.normal(0, 0.5, n)
        df = pd.DataFrame(X, columns=["f1", "f2", "f3"])
        df["target"] = y

        res = train_baseline_model(df, target_col="target", task_type="regression")
        self.assertEqual(res["status"], "completed")
        self.assertEqual(res["task_type"], "regression")
        self.assertIn("rmse", res["metrics"])
        self.assertIn("r2_score", res["metrics"])
        self.assertGreater(res["metrics"]["r2_score"], 0.7)


if __name__ == "__main__":
    unittest.main()
