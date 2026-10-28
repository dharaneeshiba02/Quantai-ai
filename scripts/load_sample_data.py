"""
scripts/load_sample_data.py — Generate and upload sample data for any domain:
  --domain [sales|healthcare|education|hr|retail|all] (default: sales)
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from app.utils.import_pipeline import import_file
from app.storage.database import Base, get_app_engine, app_session
from app.storage.models import Source, ColumnCatalog


def _create_domain_dataframe(domain: str, n: int = 800) -> pd.DataFrame:
    np.random.seed(42)
    if domain == "healthcare":
        depts = ["Cardiology", "Neurology", "Orthopedics", "Pediatrics", "Oncology"]
        diagnoses = ["Hypertension", "Diabetes", "Asthma", "Arrhythmia"]
        return pd.DataFrame({
            "patient_id": [f"PAT-{i:05d}" for i in range(1, n + 1)],
            "admit_date": pd.date_range("2024-01-01", periods=n, freq="8h"),
            "department": np.random.choice(depts, n),
            "diagnosis": np.random.choice(diagnoses, n),
            "patient_age": np.random.randint(18, 90, n),
            "systolic_bp": np.random.randint(95, 175, n),
            "cholesterol": np.random.randint(140, 290, n),
            "length_of_stay": np.random.randint(1, 14, n),
            "treatment_cost": np.round(np.random.uniform(500.0, 18000.0, n), 2),
            "is_readmitted": np.random.choice([True, False], n, p=[0.12, 0.88]),
        })
    elif domain == "education":
        grades = ["Grade 9", "Grade 10", "Grade 11", "Grade 12"]
        subjects = ["Mathematics", "Physics", "Literature", "Chemistry"]
        return pd.DataFrame({
            "student_id": [f"STU-{i:04d}" for i in range(1, n + 1)],
            "grade_level": np.random.choice(grades, n),
            "subject": np.random.choice(subjects, n),
            "midterm_score": np.random.randint(45, 100, n),
            "final_score": np.random.randint(50, 100, n),
            "attendance_pct": np.round(np.random.uniform(70.0, 100.0, n), 1),
            "study_hours_week": np.random.randint(2, 25, n),
            "passed": np.random.choice([True, False], n, p=[0.9, 0.1]),
        })
    elif domain == "hr":
        depts = ["Engineering", "Product", "Sales", "Marketing", "Finance", "HR"]
        return pd.DataFrame({
            "employee_id": [f"EMP-{i:04d}" for i in range(1, n + 1)],
            "hire_date": pd.date_range("2020-01-01", periods=n, freq="18h"),
            "department": np.random.choice(depts, n),
            "salary": np.random.randint(48000, 185000, n),
            "tenure_years": np.round(np.random.uniform(0.5, 10.0, n), 1),
            "performance_score": np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.10, 0.50, 0.25, 0.10]),
            "attrition_risk": np.random.choice(["Low", "Medium", "High"], n, p=[0.7, 0.2, 0.1]),
        })
    elif domain == "retail":
        regions = ["North America", "Europe", "Asia", "South America"]
        cats = ["Electronics", "Clothing", "Home", "Sports Equipment"]
        return pd.DataFrame({
            "order_id": [f"ORD-{i:05d}" for i in range(1, n + 1)],
            "order_date": pd.date_range("2024-01-01", periods=n, freq="4h"),
            "region": np.random.choice(regions, n),
            "category": np.random.choice(cats, n),
            "sales_amount": np.round(np.random.uniform(50.0, 1500.0, n), 2),
            "quantity": np.random.randint(1, 10, n),
            "discount_pct": np.random.choice([0, 5, 10, 15, 20], n),
            "is_returned": np.random.choice([True, False], n, p=[0.05, 0.95]),
        })
    else:  # sales / finance
        regions = ["North", "South", "East", "West", "Central"]
        clients = ["Alpha Corp", "Beta LLC", "Gamma Inc", "Delta Global", "Omega Ent"]
        df = pd.DataFrame({
            "invoice_id": [f"INV-{i:05d}" for i in range(1, n + 1)],
            "invoice_date": pd.date_range("2024-01-01", periods=n, freq="6h"),
            "region": np.random.choice(regions, n),
            "client_name": np.random.choice(clients, n),
            "revenue": np.round(np.random.uniform(1200.0, 65000.0, n), 2),
            "cost": np.round(np.random.uniform(600.0, 38000.0, n), 2),
            "payment_status": np.random.choice(["Paid", "Pending", "Overdue"], n, p=[0.75, 0.2, 0.05]),
        })
        df["profit"] = np.round(df["revenue"] - df["cost"], 2)
        return df


async def load_domain(domain: str, uploads_dir: Path):
    print(f"Generating and importing {domain} dataset...")
    df = _create_domain_dataframe(domain)
    filename = f"sample_{domain}.csv"
    file_path = uploads_dir / filename
    df.to_csv(file_path, index=False)

    Base.metadata.create_all(get_app_engine())

    result = await import_file(file_path, filename)
    source_id = str(uuid.uuid4())

    with app_session() as session:
        src = Source(
            id=source_id,
            name=filename,
            source_type="csv",
            table_name=result["table_name"],
            schema_name=result["schema_name"],
            row_count=result["row_count"],
            col_count=result["col_count"],
            file_path=str(file_path),
        )
        session.add(src)

        for entry in result["catalog_entries"]:
            col = ColumnCatalog(
                source_id=source_id,
                **{k: v for k, v in entry.items() if k != "source_id"},
            )
            session.add(col)

    print(f"Success! Uploaded {result['row_count']} rows for {domain}. Source ID: {source_id}")


async def main():
    parser = argparse.ArgumentParser(description="Generate and upload sample domain datasets.")
    parser.add_argument(
        "--domain",
        default="sales",
        choices=["sales", "healthcare", "education", "hr", "retail", "all"],
        help="Domain of sample data to load (default: sales)",
    )
    args = parser.parse_args()

    uploads_dir = Path("data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)

    domains = ["sales", "healthcare", "education", "hr"] if args.domain == "all" else [args.domain]
    for d in domains:
        await load_domain(d, uploads_dir)


if __name__ == "__main__":
    asyncio.run(main())
