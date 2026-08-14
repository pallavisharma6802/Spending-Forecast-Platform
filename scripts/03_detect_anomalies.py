"""Mid-period spend pace anomalies for the 200-user reference set.
Replaces 08_anomaly_detection.py (hardcoded to Jan 2025) with core/, which
takes an arbitrary as_of/window so the same function works for day/week/
month windows on any dataset, not just this project's fixed calendar month.

Output: frontend/data/anomalies.json
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json

import pandas as pd

from core.anomaly import detect_anomalies
from core.data_loader import load_csv_path
from core.spark_session import get_spark, stop_spark

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "data")
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "spending_patterns_5yr.csv")

# The dataset's actual final 13 days (Jan 1-13 2025) - a natural mid-month
# snapshot, kept for continuity with the original demo.
AS_OF = pd.Timestamp("2025-01-13")
WINDOW_DAYS = 13
TARGET_PERIOD_DAYS = 31


def main():
    spark = get_spark("detect-anomalies")
    sdf, _ = load_csv_path(spark, CSV_PATH)

    with open(os.path.join(DATA_DIR, "budget_caps.json")) as f:
        caps = pd.DataFrame(json.load(f))

    result = detect_anomalies(
        sdf, as_of=AS_OF, window_days=WINDOW_DAYS,
        target_period_days=TARGET_PERIOD_DAYS, budget_caps=caps,
    )
    with open(os.path.join(DATA_DIR, "anomalies.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved {len(result['alerts']):,} alerts -> anomalies.json (IF used: {result['isolation_forest_used']})")
    stop_spark()


if __name__ == "__main__":
    main()
