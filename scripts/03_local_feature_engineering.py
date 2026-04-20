"""
Local feature engineering — pure pandas, no Spark required.
Reads data/spending_patterns_5yr.csv, writes frontend/data/baseline.json.

Computes per-user per-category:
  total_spend          5-year historical total
  avg_per_transaction  mean transaction size
  num_transactions     transaction count
  max_30d_spend        maximum 30-day rolling spend window
  spend_velocity       Q4-2024 / Q4-2023 spend ratio (trend signal)
"""

import json
import os
import numpy as np
import pandas as pd

CSV_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "spending_patterns_5yr.csv")
DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "frontend", "data")
OUT_JSON  = os.path.join(DATA_DIR, "baseline.json")

txn = pd.read_csv(CSV_PATH, parse_dates=["Transaction Date"])
txn = txn.rename(columns={
    "Customer ID":      "customer_id",
    "Category":         "category",
    "Total Spent":      "total_spent",
    "Transaction Date": "transaction_date",
})
txn["total_spent"] = txn["total_spent"].astype(float)

#  Core per-user-category stats 
agg = (
    txn.groupby(["customer_id", "category"])["total_spent"]
    .agg(
        total_spend          = "sum",
        avg_per_transaction  = "mean",
        num_transactions     = "count",
    )
    .reset_index()
)
agg["total_spend"]         = agg["total_spend"].round(2)
agg["avg_per_transaction"] = agg["avg_per_transaction"].round(2)

#  Max 30-day rolling spend per user-category 
daily = (
    txn.groupby(["customer_id", "category", "transaction_date"])["total_spent"]
    .sum().reset_index()
)

max30_rows = []
for (cid, cat), grp in daily.groupby(["customer_id", "category"]):
    ts = (
        grp.set_index("transaction_date")["total_spent"]
        .resample("D").sum()
        .rolling(30, min_periods=1).sum()
    )
    max30_rows.append({
        "customer_id": cid,
        "category":    cat,
        "max_30d_spend": round(float(ts.max()), 2),
    })

max30_df = pd.DataFrame(max30_rows)
agg = agg.merge(max30_df, on=["customer_id", "category"], how="left")
agg["max_30d_spend"] = agg["max_30d_spend"].fillna(0.0)

#  Spend velocity: Q4-2024 / Q4-2023 
def q4_spend(year):
    return (
        txn[
            (txn["transaction_date"].dt.year == year) &
            (txn["transaction_date"].dt.month >= 10)
        ]
        .groupby(["customer_id", "category"])["total_spent"]
        .sum()
        .rename(f"q4_{year}")
    )

q4_2024 = q4_spend(2024)
q4_2023 = q4_spend(2023)

vel = (
    pd.concat([q4_2024, q4_2023], axis=1)
    .fillna(0)
    .reset_index()
)
vel["spend_velocity"] = np.where(
    vel["q4_2023"] > 0,
    (vel["q4_2024"] / vel["q4_2023"]).clip(0.1, 5.0).round(4),
    1.0,
)
vel = vel[["customer_id", "category", "spend_velocity"]]

agg = agg.merge(vel, on=["customer_id", "category"], how="left")
agg["spend_velocity"] = agg["spend_velocity"].fillna(1.0)

#  Save 
agg.to_json(OUT_JSON, orient="records", indent=2)
print(f"Saved {len(agg):,} baseline rows → {OUT_JSON}")
print(f"\nPer-category avg transaction amounts (sanity check):")
check = agg.groupby("category")["avg_per_transaction"].mean().sort_values(ascending=False)
for cat, val in check.items():
    print(f"  {cat:<26}  ${val:>8,.0f}")
