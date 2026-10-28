"""
scripts/generate_test_datasets.py — Generate realistic synthetic test datasets
across all 9 business and analytical domains:
  1. Sales & Finance
  2. Education & Student Performance
  3. Healthcare & Clinical
  4. Human Resources & Workforce
  5. E-commerce & Retail
  6. Logistics & Supply Chain
  7. Marketing & Social Media
  8. Real Estate & Housing
  9. Sports & Entertainment
  + Multi-sheet Excel workbook covering multiple domains in one file.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import numpy as np
import pandas as pd


def generate_all_datasets(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files: dict[str, Path] = {}
    np.random.seed(42)

    # 1. Sales & Finance
    n = 600
    dates = pd.date_range("2024-01-01", periods=n, freq="12H")
    regions = ["North", "South", "East", "West", "Central"]
    entities = ["Alpha Corp", "Beta LLC", "Gamma Inc", "Delta Global", "Omega Enterprises"]
    df_sales = pd.DataFrame({
        "invoice_id": [f"INV-{i:05d}" for i in range(1, n + 1)],
        "invoice_date": dates,
        "region": np.random.choice(regions, n),
        "client_name": np.random.choice(entities, n),
        "revenue": np.round(np.random.uniform(1500.0, 75000.0, n), 2),
        "cost": np.round(np.random.uniform(800.0, 45000.0, n), 2),
        "tax_rate_pct": np.random.choice([5.0, 8.5, 12.0, 15.0], n),
        "payment_status": np.random.choice(["Paid", "Pending", "Overdue"], n, p=[0.75, 0.20, 0.05]),
    })
    df_sales["profit"] = np.round(df_sales["revenue"] - df_sales["cost"], 2)
    path = output_dir / "domain_sales_finance.csv"
    df_sales.to_csv(path, index=False)
    generated_files["sales_finance"] = path

    # 2. Education & Student Performance
    n = 500
    grades = ["Grade 9", "Grade 10", "Grade 11", "Grade 12"]
    subjects = ["Mathematics", "Physics", "Literature", "Chemistry", "History"]
    df_edu = pd.DataFrame({
        "student_id": [f"STU-{i:04d}" for i in range(1, n + 1)],
        "grade_level": np.random.choice(grades, n),
        "subject": np.random.choice(subjects, n),
        "midterm_score": np.random.randint(45, 100, n),
        "final_exam_score": np.random.randint(50, 100, n),
        "attendance_rate_pct": np.round(np.random.uniform(70.0, 100.0, n), 1),
        "passed": np.random.choice([True, False], n, p=[0.88, 0.12]),
        "study_hours_weekly": np.random.randint(2, 25, n),
    })
    path = output_dir / "domain_student_education.csv"
    df_edu.to_csv(path, index=False)
    generated_files["education"] = path

    # 3. Healthcare & Clinical
    n = 600
    departments = ["Cardiology", "Neurology", "Orthopedics", "Pediatrics", "Oncology"]
    diagnoses = ["Hypertension", "Type 2 Diabetes", "Asthma", "Arrhythmia", "Migraine"]
    df_health = pd.DataFrame({
        "patient_id": [f"PAT-{i:05d}" for i in range(1, n + 1)],
        "department": np.random.choice(departments, n),
        "primary_diagnosis": np.random.choice(diagnoses, n),
        "age": np.random.randint(18, 88, n),
        "systolic_bp": np.random.randint(100, 180, n),
        "cholesterol_mg_dl": np.random.randint(150, 280, n),
        "length_of_stay_days": np.random.randint(1, 15, n),
        "readmitted_30_days": np.random.choice(["Yes", "No"], n, p=[0.14, 0.86]),
    })
    path = output_dir / "domain_clinical_healthcare.csv"
    df_health.to_csv(path, index=False)
    generated_files["healthcare"] = path

    # 4. Human Resources & Workforce
    n = 450
    dept_hr = ["Engineering", "Product", "Sales", "Marketing", "HR", "Finance"]
    df_hr = pd.DataFrame({
        "employee_id": [f"EMP-{i:04d}" for i in range(1, n + 1)],
        "department": np.random.choice(dept_hr, n),
        "job_title": np.random.choice(["Junior Associate", "Senior Specialist", "Lead", "Manager", "Director"], n),
        "salary": np.random.randint(48000, 195000, n),
        "tenure_years": np.round(np.random.uniform(0.5, 12.0, n), 1),
        "performance_rating": np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.10, 0.50, 0.25, 0.10]),
        "attrition_risk": np.random.choice(["Low", "Medium", "High"], n, p=[0.70, 0.22, 0.08]),
    })
    path = output_dir / "domain_hr_workforce.csv"
    df_hr.to_csv(path, index=False)
    generated_files["hr"] = path

    # 5. E-commerce & Retail
    n = 800
    categories = ["Electronics", "Apparel", "Home Goods", "Beauty", "Sports Equipment"]
    df_ecom = pd.DataFrame({
        "order_id": [f"ECOM-{i:06d}" for i in range(1, n + 1)],
        "category": np.random.choice(categories, n),
        "item_price": np.round(np.random.uniform(12.0, 850.0, n), 2),
        "quantity": np.random.randint(1, 8, n),
        "discount_percent": np.random.choice([0, 5, 10, 15, 20, 30], n),
        "customer_rating": np.random.randint(1, 6, n),
        "order_status": np.random.choice(["Delivered", "In Transit", "Processing", "Cancelled"], n, p=[0.82, 0.10, 0.05, 0.03]),
    })
    path = output_dir / "domain_retail_ecommerce.csv"
    df_ecom.to_csv(path, index=False)
    generated_files["ecommerce"] = path

    # 6. Logistics & Supply Chain
    n = 550
    carriers = ["DHL Express", "FedEx Freight", "UPS Supply", "Maersk Logistics"]
    warehouses = ["WH-Chicago", "WH-Dallas", "WH-Rotterdam", "WH-Singapore"]
    df_log = pd.DataFrame({
        "shipment_id": [f"SHP-{i:05d}" for i in range(1, n + 1)],
        "origin_warehouse": np.random.choice(warehouses, n),
        "carrier": np.random.choice(carriers, n),
        "weight_kg": np.round(np.random.uniform(2.5, 1200.0, n), 1),
        "shipping_cost": np.round(np.random.uniform(45.0, 3400.0, n), 2),
        "delivery_time_days": np.random.randint(1, 21, n),
        "on_time_delivery": np.random.choice([True, False], n, p=[0.85, 0.15]),
    })
    path = output_dir / "domain_logistics_shipping.csv"
    df_log.to_csv(path, index=False)
    generated_files["logistics"] = path

    # 7. Marketing & Social Media
    n = 400
    channels = ["Google Ads", "Meta Ads", "LinkedIn", "YouTube", "TikTok", "Email Newsletter"]
    df_mkt = pd.DataFrame({
        "campaign_id": [f"CMP-{i:04d}" for i in range(1, n + 1)],
        "channel": np.random.choice(channels, n),
        "ad_spend": np.round(np.random.uniform(200.0, 15000.0, n), 2),
        "impressions": np.random.randint(5000, 500000, n),
        "clicks": np.random.randint(150, 25000, n),
        "conversions": np.random.randint(10, 1200, n),
        "roas": np.round(np.random.uniform(1.2, 8.5, n), 2),
    })
    path = output_dir / "domain_marketing_campaigns.csv"
    df_mkt.to_csv(path, index=False)
    generated_files["marketing"] = path

    # 8. Real Estate & Housing
    n = 500
    neighborhoods = ["Downtown", "Suburbs", "Waterfront", "West End", "North Hills"]
    types = ["Single Family", "Condo", "Townhouse", "Multi-Family"]
    df_re = pd.DataFrame({
        "property_id": [f"PROP-{i:04d}" for i in range(1, n + 1)],
        "neighborhood": np.random.choice(neighborhoods, n),
        "property_type": np.random.choice(types, n),
        "square_feet": np.random.randint(650, 4800, n),
        "bedrooms": np.random.randint(1, 6, n),
        "listing_price": np.random.randint(180000, 1850000, n),
        "days_on_market": np.random.randint(3, 120, n),
    })
    path = output_dir / "domain_real_estate.csv"
    df_re.to_csv(path, index=False)
    generated_files["real_estate"] = path

    # 9. Sports & Entertainment
    n = 400
    teams = ["Warriors", "Lions", "Dragons", "Titans", "Phoenix"]
    positions = ["Forward", "Midfielder", "Defender", "Goalkeeper"]
    df_sports = pd.DataFrame({
        "player_id": [f"PLY-{i:04d}" for i in range(1, n + 1)],
        "team": np.random.choice(teams, n),
        "position": np.random.choice(positions, n),
        "games_played": np.random.randint(10, 82, n),
        "points_scored": np.random.randint(50, 1800, n),
        "minutes_per_game": np.round(np.random.uniform(12.0, 38.0, n), 1),
        "efficiency_rating": np.round(np.random.uniform(8.0, 32.0, n), 1),
    })
    path = output_dir / "domain_sports_athletes.csv"
    df_sports.to_csv(path, index=False)
    generated_files["sports"] = path

    # 10. Multi-sheet Excel workbook
    excel_path = output_dir / "multi_domain_workbook.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_sales.head(200).to_excel(writer, sheet_name="Sales_Finance", index=False)
        df_health.head(200).to_excel(writer, sheet_name="Healthcare_Patients", index=False)
        df_hr.head(200).to_excel(writer, sheet_name="HR_Employees", index=False)
        df_edu.head(200).to_excel(writer, sheet_name="Student_Education", index=False)
    generated_files["multi_sheet_excel"] = excel_path

    print(f"Generated {len(generated_files)} synthetic datasets in {output_dir}:")
    for domain, fpath in generated_files.items():
        print(f"  - {domain:20s}: {fpath.name}")

    return generated_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic test datasets for all domains.")
    parser.add_argument("--dir", default="data/test_datasets", help="Directory to save generated datasets")
    args = parser.parse_args()

    out = Path(args.dir)
    generate_all_datasets(out)
