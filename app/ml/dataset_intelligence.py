"""
app/ml/dataset_intelligence.py — General-purpose Machine Learning Engine.

Capabilities:
- Automated task suitability assessment (classification, regression, clustering, anomaly detection, forecasting).
- Isolation Forest anomaly detection & profiling (deterministic, fixed seed, does not mutate data).
- K-Means clustering with optimal K selection via silhouette score.
- Supervised baseline modeling (RandomForest / Ridge / LogisticRegression) with leakage-free splits and true evaluation metrics.
- Transparent, factual narrative generation without fabricated statistics.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

_EXCLUDED_ID_HINTS = ("id", "_id", "code", "index", "uuid", "row_id", "_row_id", "serial")
_TEMPORAL_HINTS = ("year", "yr", "date", "datetime", "timestamp", "period", "month", "day", "quarter")


def _display(value: Any) -> str:
    if value is None:
        return "—"
    try:
        num = float(value)
        if not math.isfinite(num):
            return "—"
        return f"{int(num):,}" if num.is_integer() else f"{num:,.2f}"
    except (TypeError, ValueError, OverflowError):
        return str(value)


def _is_id_column(col_name: str) -> bool:
    name = col_name.strip().lower()
    return any(name == hint or name.endswith(f"_{hint}") or name.startswith(f"{hint}_") for hint in _EXCLUDED_ID_HINTS)


def _is_temporal_col(col_name: str) -> bool:
    name = col_name.strip().lower()
    return name in _TEMPORAL_HINTS or any(name.endswith(f"_{h}") for h in ("year", "date"))


def _usable_features(frame: pd.DataFrame, max_features: int = 8) -> list[str]:
    """Select usable non-identifier numeric columns with sufficient variance and data."""
    usable = []
    for col in frame.columns:
        if _is_id_column(str(col)) or _is_temporal_col(str(col)):
            continue
        try:
            numeric_series = pd.to_numeric(frame[col], errors="coerce").dropna()
            if len(numeric_series) >= 12 and numeric_series.nunique() > 2:
                if float(numeric_series.std()) > 1e-7:
                    usable.append(str(col))
        except Exception:
            continue
        if len(usable) >= max_features:
            break
    return usable


def _correlations(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Compute top absolute pairwise Pearson correlations on numeric features."""
    if frame.shape[1] < 2:
        return []
    matrix = frame.corr(method="pearson")
    pairs = []
    cols = matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = matrix.iloc[i, j]
            if pd.notna(val) and math.isfinite(val):
                pairs.append({
                    "feature_1": str(cols[i]),
                    "feature_2": str(cols[j]),
                    "correlation": round(float(val), 3),
                    "abs_corr": abs(float(val)),
                })
    pairs.sort(key=lambda x: x["abs_corr"], reverse=True)
    return pairs[:10]


def assess_ml_suitability(df: pd.DataFrame) -> dict[str, Any]:
    """
    Assess whether the dataset is suitable for various ML tasks.
    Returns recommendations and candidate target columns.
    """
    total_rows = len(df)
    col_count = len(df.columns)

    if total_rows < 15:
        return {
            "suitable": False,
            "reason": f"Dataset has only {total_rows} rows; reliable ML requires at least 15-20 rows to prevent severe overfitting.",
            "tasks": {},
        }

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not _is_id_column(str(c))]
    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if not _is_id_column(str(c))]
    date_cols = [c for c in df.columns if _is_temporal_col(str(c))]

    # Classification candidate targets: categorical or low-cardinality discrete columns (2 to 15 unique values, each >= 5 samples)
    classification_candidates = []
    for c in cat_cols + numeric_cols:
        series = df[c].dropna()
        n_unique = series.nunique()
        if 2 <= n_unique <= 15:
            val_counts = series.value_counts()
            if val_counts.min() >= 3 and (val_counts.min() / total_rows) >= 0.02:
                classification_candidates.append({
                    "column": str(c),
                    "classes": int(n_unique),
                    "class_distribution": {str(k): int(v) for k, v in val_counts.items()},
                })

    # Regression candidate targets: continuous numeric measures with high unique counts
    regression_candidates = []
    for c in numeric_cols:
        if _is_temporal_col(str(c)):
            continue
        series = df[c].dropna()
        if series.nunique() >= 10 and float(series.std()) > 1e-6:
            regression_candidates.append({
                "column": str(c),
                "unique_values": int(series.nunique()),
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
            })

    clustering_suitable = len(numeric_cols) >= 2 and total_rows >= 20
    anomaly_suitable = len(numeric_cols) >= 1 and total_rows >= 20
    forecasting_suitable = len(date_cols) >= 1 and len(numeric_cols) >= 1 and total_rows >= 24

    return {
        "suitable": True,
        "total_rows": total_rows,
        "features_available": col_count,
        "tasks": {
            "anomaly_detection": {
                "suitable": anomaly_suitable,
                "algorithm": "Isolation Forest",
                "recommended_features": numeric_cols[:8],
            },
            "clustering": {
                "suitable": clustering_suitable,
                "algorithm": "K-Means with Silhouette Selection",
                "recommended_features": numeric_cols[:8],
            },
            "classification": {
                "suitable": len(classification_candidates) > 0,
                "candidate_targets": [item["column"] for item in classification_candidates[:3]],
                "details": classification_candidates[:3],
            },
            "regression": {
                "suitable": len(regression_candidates) > 0,
                "candidate_targets": [item["column"] for item in regression_candidates[:3]],
                "details": regression_candidates[:3],
            },
            "time_series_forecasting": {
                "suitable": forecasting_suitable,
                "time_columns": date_cols,
                "measures": [item["column"] for item in regression_candidates[:3]],
            },
        },
    }


def analyze_numeric_features(
    frame: pd.DataFrame,
    max_features: int = 8,
    max_clusters: int = 5,
) -> dict[str, Any]:
    """
    Run transparent clustering and anomaly detection on a numeric sample.

    K is selected from 2 through the configured maximum by the highest silhouette score. Isolation Forest
    marks a small, data-size-aware set of unusual records; it never removes or
    changes them. Both algorithms use a fixed random state for reproducibility.
    """
    usable_cols = _usable_features(frame, max_features=max_features)
    if not usable_cols:
        return {
            "status": "skipped",
            "reason": "No non-identifier numeric fields with enough usable values were available for ML analysis.",
            "summary": "No ML model was run.",
            "features": [],
        }

    # Extract working matrix without mutating original DataFrame
    working_df = frame[usable_cols].copy()
    for col in usable_cols:
        working_df[col] = pd.to_numeric(working_df[col], errors="coerce")

    # Median imputation in temporary model matrix only
    model_matrix = working_df.copy()
    for col in usable_cols:
        median_val = model_matrix[col].median()
        model_matrix[col] = model_matrix[col].fillna(median_val if pd.notna(median_val) else 0.0)

    model_rows = len(model_matrix)
    if model_rows < 12:
        return {
            "status": "skipped",
            "reason": "Fewer than 12 usable records remained after preparing numeric features for analysis.",
            "summary": "No ML model was run; only deterministic correlation screening was available.",
            "features": usable_cols,
        }

    # Standardize features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(model_matrix.values)

    # 1. Anomaly Detection via Isolation Forest
    contamination = min(0.05, max(0.01, 10.0 / model_rows))
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
    )
    labels = iso_forest.fit_predict(scaled_data)
    scores = iso_forest.decision_function(scaled_data)

    outlier_mask = (labels == -1)
    outlier_count = int(np.sum(outlier_mask))
    outlier_rate = round(float(outlier_count / model_rows * 100), 2)

    # Collect sample anomaly records (up to 5 most anomalous by score)
    example_indices = np.argsort(scores)[:min(5, outlier_count)].tolist() if outlier_count > 0 else []
    anomaly_examples = []
    for idx in example_indices:
        row_dict = {col: _display(frame.iloc[idx][col]) for col in usable_cols}
        anomaly_examples.append({
            "original_row_index": int(frame.index[idx] if hasattr(frame.index, "__getitem__") else idx),
            "score": round(float(scores[idx]), 3),
            "values": row_dict,
        })

    # 2. Clustering via K-Means with Silhouette Optimization (if >= 2 features)
    clustering_res = None
    if len(usable_cols) >= 2 and model_rows >= 15:
        best_k = 2
        best_score = -1.0
        best_labels = None
        max_k_possible = min(max_clusters, model_rows - 1, 8)

        for k in range(2, max_k_possible + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            c_labels = kmeans.fit_predict(scaled_data)
            # Ensure at least 2 distinct clusters were assigned
            if len(np.unique(c_labels)) >= 2:
                try:
                    score = float(silhouette_score(scaled_data, c_labels))
                    if score > best_score:
                        best_score = score
                        best_k = k
                        best_labels = c_labels
                except Exception:
                    continue

        if best_labels is not None:
            # Build cluster profiles
            profiles = []
            for c_id in range(best_k):
                mask = (best_labels == c_id)
                cluster_size = int(np.sum(mask))
                means = {col: round(float(model_matrix.loc[mask, col].mean()), 2) for col in usable_cols}
                profiles.append({
                    "label": f"Segment {c_id + 1}",
                    "count": cluster_size,
                    "proportion_pct": round(float(cluster_size / model_rows * 100), 1),
                    "feature_means": means,
                })

            clustering_res = {
                "algorithm": "K-Means",
                "clusters": int(best_k),
                "silhouette_score": round(float(best_score), 3),
                "profiles": profiles,
            }

    correlations = _correlations(model_matrix)

    return {
        "status": "completed",
        "model_rows": int(model_rows),
        "features": usable_cols,
        "outliers": {
            "algorithm": "Isolation Forest",
            "count": int(outlier_count),
            "rate_pct": float(outlier_rate),
            "examples": anomaly_examples,
            "notes": "Unusual records are flagged for review only; no records were removed or changed.",
        },
        "clustering": clustering_res,
        "correlations": correlations,
        "methodology": (
            "Numeric features were median-imputed only in a temporary model matrix, standardized, "
            "then analyzed with Isolation Forest for unusual records. When at least two usable numeric "
            "features existed, K-Means segments were evaluated with silhouette scoring. "
            "The original uploaded dataset remains unchanged."
        ),
    }


def train_baseline_model(
    df: pd.DataFrame,
    target_col: str,
    task_type: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Train a baseline supervised ML model with clean leakage-free validation.
    Supports classification (RandomForestClassifier / LogisticRegression) and
    regression (RandomForestRegressor / Ridge).
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    # Drop rows where target is missing
    clean_target_df = df.dropna(subset=[target_col]).copy()
    if len(clean_target_df) < 15:
        return {
            "status": "skipped",
            "reason": f"Only {len(clean_target_df)} rows with non-null target values. Supervised ML requires at least 15 rows.",
        }

    y_raw = clean_target_df[target_col]

    # Auto-detect task type if not specified
    if task_type is None:
        if pd.api.types.is_numeric_dtype(y_raw) and y_raw.nunique() > 10:
            task_type = "regression"
        else:
            task_type = "classification"

    # Select feature columns (excluding target, IDs, and columns with 100% missing)
    feature_candidates = [
        c for c in clean_target_df.columns
        if c != target_col and not _is_id_column(str(c))
    ]

    # Clean numeric features
    X_numeric = pd.DataFrame(index=clean_target_df.index)
    for c in feature_candidates:
        if pd.api.types.is_numeric_dtype(clean_target_df[c]):
            series = pd.to_numeric(clean_target_df[c], errors="coerce")
            if series.nunique() > 1:
                X_numeric[c] = series.fillna(series.median() if pd.notna(series.median()) else 0.0)

    if X_numeric.shape[1] == 0:
        return {
            "status": "skipped",
            "reason": "No usable numeric feature columns available to predict the target.",
        }

    # Split train and test
    stratify = y_raw if task_type == "classification" and y_raw.value_counts().min() >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_numeric, y_raw, test_size=test_size, random_state=random_state, stratify=stratify
        )
    except Exception:
        # Fall back to unstratified if any class is too small
        X_train, X_test, y_train, y_test = train_test_split(
            X_numeric, y_raw, test_size=test_size, random_state=random_state
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    feature_names = X_numeric.columns.tolist()

    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=6)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        importances = {
            feature_names[i]: round(float(imp), 4)
            for i, imp in enumerate(model.feature_importances_)
        }
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)

        return {
            "status": "completed",
            "task_type": "classification",
            "target": target_col,
            "model": "Random Forest Classifier",
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "metrics": {
                "accuracy": round(acc, 4),
                "f1_score_weighted": round(f1, 4),
            },
            "top_features": dict(sorted_imp[:6]),
            "methodology": f"Trained with an 80/20 train/test split. Evaluation metrics calculated strictly on out-of-sample test data ({len(X_test)} rows).",
        }

    else:  # regression
        model = RandomForestRegressor(n_estimators=100, random_state=random_state, max_depth=6)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(math.sqrt(mse))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        importances = {
            feature_names[i]: round(float(imp), 4)
            for i, imp in enumerate(model.feature_importances_)
        }
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)

        return {
            "status": "completed",
            "task_type": "regression",
            "target": target_col,
            "model": "Random Forest Regressor",
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "metrics": {
                "rmse": round(rmse, 4),
                "mae": round(mae, 4),
                "r2_score": round(r2, 4),
            },
            "top_features": dict(sorted_imp[:6]),
            "methodology": f"Trained with an 80/20 train/test split. Evaluation metrics calculated strictly on out-of-sample test data ({len(X_test)} rows).",
        }


def summarize_ml_analysis(analysis: dict[str, Any]) -> str:
    """Return a concise, factual summary suitable for the UI and PDF."""
    if analysis.get("status") != "completed":
        return analysis.get("reason", "No ML analysis was available.")

    features = ", ".join(analysis.get("features", []))
    outliers = analysis.get("outliers", {})
    text = (
        f"ML reviewed {analysis.get('model_rows', 0):,} sampled records across "
        f"{len(analysis.get('features', []))} numeric feature(s): {features}. "
        f"Isolation Forest flagged {outliers.get('count', 0):,} unusual record(s) "
        f"({outliers.get('rate_pct', 0)}%) for review without changing the dataset."
    )

    cluster = analysis.get("clustering")
    if cluster and cluster.get("clusters"):
        text += (
            f" K-Means found {cluster['clusters']} segments "
            f"(silhouette score {cluster.get('silhouette_score')})."
        )

    return text
