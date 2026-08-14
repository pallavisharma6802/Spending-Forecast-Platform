"""Walk-forward backtest -> per-category model config -> per-user-category
forecasts. Replaces the old 03_feature_engineering.py + 04_time_series_forecast.py
(Spark-cluster-on-HDFS versions) with core/, which runs PySpark in local mode -
no HDFS, no separate feature-engineering pass.

Outputs:
  frontend/data/model_config.json   backtest winner + tuned params per category
  frontend/data/forecasts.json      per-user-category forecast, 7/30/365-day horizons
  frontend/data/baseline.json       per-user-category historical summary stats
  frontend/data/users.json
  frontend/data/categories.json

Also prints the old-vs-new backtest comparison used in the README.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import time

import pandas as pd

from core.data_loader import load_csv_path
from core.forecast_engine import DEFAULT_HORIZONS, forecast, run_backtest, save_model_config
from core.spark_session import get_spark, stop_spark

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "data")
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "spending_patterns_5yr.csv")

os.makedirs(DATA_DIR, exist_ok=True)


def build_baseline_summary(sdf) -> pd.DataFrame:
    pdf = sdf.select("customer_id", "category", "total_spent", "transaction_date").toPandas()
    pdf["transaction_date"] = pd.to_datetime(pdf["transaction_date"])

    agg = pdf.groupby(["customer_id", "category"]).agg(
        total_spend=("total_spent", "sum"),
        avg_per_transaction=("total_spent", "mean"),
        num_transactions=("total_spent", "count"),
    ).reset_index()

    max30 = []
    for (cid, cat), g in pdf.groupby(["customer_id", "category"]):
        daily = g.groupby(g["transaction_date"].dt.normalize())["total_spent"].sum()
        daily = daily.asfreq("D").fillna(0.0)
        rolling = daily.rolling(30, min_periods=1).sum()
        max30.append({"customer_id": cid, "category": cat, "max_30d_spend": float(rolling.max()) if len(rolling) else 0.0})

    return agg.merge(pd.DataFrame(max30), on=["customer_id", "category"], how="left")


def main():
    print(f"Loading {CSV_PATH} ...")
    spark = get_spark("backtest-and-forecast")
    sdf, report = load_csv_path(spark, CSV_PATH)
    print(report)
    sdf = sdf.cache()

    categories = sorted([r["category"] for r in sdf.select("category").distinct().collect()])
    print(f"\nRunning walk-forward backtest for {len(categories)} categories...")

    t0 = time.time()
    config = run_backtest(sdf, categories=categories)
    print(f"Backtest done in {time.time() - t0:.0f}s\n")

    print(config[["category", "model", "wape_baseline", "wape_hierarchical", "wape_sarimax"]].to_string(index=False))
    save_model_config(config, os.path.join(DATA_DIR, "model_config.json"))

    print("\nServing forecasts for all users/categories/horizons...")
    fc = forecast(sdf, config, horizons=DEFAULT_HORIZONS)
    fc.to_json(os.path.join(DATA_DIR, "forecasts.json"), orient="records", indent=2)
    print(f"Saved {len(fc):,} forecast rows -> forecasts.json")

    baseline = build_baseline_summary(sdf)
    baseline.round(2).to_json(os.path.join(DATA_DIR, "baseline.json"), orient="records", indent=2)
    print(f"Saved {len(baseline):,} baseline rows -> baseline.json")

    users = sorted(baseline["customer_id"].unique().tolist())
    with open(os.path.join(DATA_DIR, "users.json"), "w") as f:
        json.dump(users, f, indent=2)
    with open(os.path.join(DATA_DIR, "categories.json"), "w") as f:
        json.dump(categories, f, indent=2)
    print(f"Saved {len(users)} users -> users.json, {len(categories)} categories -> categories.json")

    stop_spark()


if __name__ == "__main__":
    main()
