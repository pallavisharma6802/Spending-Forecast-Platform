"""Collaborative-filtering budget caps for the 200-user reference set.
Replaces 05_collaborative_filter.py (Spark-cluster-on-HDFS) with core/.

Output: frontend/data/budget_caps.json
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json

import pandas as pd

from core.data_loader import load_csv_path
from core.recommender import build_feature_matrix, budget_caps
from core.spark_session import get_spark, stop_spark

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "data")
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "spending_patterns_5yr.csv")


def main():
    spark = get_spark("recommend-budgets")
    sdf, _ = load_csv_path(spark, CSV_PATH)

    with open(os.path.join(DATA_DIR, "forecasts.json")) as f:
        fc = pd.DataFrame(json.load(f))

    fm = build_feature_matrix(sdf, fc)
    print(f"Feature matrix: {fm.shape[0]} users x {fm.shape[1]} features")

    caps = budget_caps(fm, fc)
    caps.to_json(os.path.join(DATA_DIR, "budget_caps.json"), orient="records", indent=2)
    print(f"Saved {len(caps):,} budget cap rows -> budget_caps.json")

    fm.to_csv(os.path.join(DATA_DIR, "reference_feature_matrix.csv"))
    stop_spark()


if __name__ == "__main__":
    main()
