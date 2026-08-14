"""Composite 0-100 financial health score for the 200-user reference set.
Replaces 07_financial_health_score.py with core/.

Output: frontend/data/health_scores.json
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json

import pandas as pd

from core.data_loader import load_csv_path
from core.health_score import RAW_COLUMNS, compute_health_scores, compute_raw_dimensions
from core.spark_session import get_spark, stop_spark

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "data")
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "spending_patterns_5yr.csv")
AS_OF = pd.Timestamp("2025-01-13")


def main():
    spark = get_spark("health-scores")
    sdf, _ = load_csv_path(spark, CSV_PATH)

    with open(os.path.join(DATA_DIR, "forecasts.json")) as f:
        fc = pd.DataFrame(json.load(f))

    raw_dims = compute_raw_dimensions(sdf, fc, as_of=AS_OF)
    raw_dims[RAW_COLUMNS].to_csv(os.path.join(DATA_DIR, "reference_health_raw.csv"), index=False)

    scores = compute_health_scores(sdf, fc, as_of=AS_OF, reference_raw=raw_dims)
    scores.to_json(os.path.join(DATA_DIR, "health_scores.json"), orient="records", indent=2)
    print(f"Saved {len(scores):,} health scores -> health_scores.json")
    print(f"Mean score: {scores['score'].mean():.1f}, grade breakdown:\n{scores['grade'].value_counts()}")

    stop_spark()


if __name__ == "__main__":
    main()
