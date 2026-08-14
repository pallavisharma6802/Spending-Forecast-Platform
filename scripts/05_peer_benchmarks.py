"""Peer comparison (CF delta + category percentile) for the 200-user
reference set. Replaces 09_peer_benchmarking.py with core/.

Output: frontend/data/peer_benchmarks.json
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json

import pandas as pd

from core.data_loader import load_csv_path
from core.peer_bench import category_totals, compute_peer_benchmarks
from core.spark_session import get_spark, stop_spark

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "data")
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "spending_patterns_5yr.csv")


def main():
    spark = get_spark("peer-benchmarks")
    sdf, _ = load_csv_path(spark, CSV_PATH)

    with open(os.path.join(DATA_DIR, "budget_caps.json")) as f:
        caps = pd.DataFrame(json.load(f))

    totals = category_totals(sdf)
    result = compute_peer_benchmarks(caps, totals)

    with open(os.path.join(DATA_DIR, "peer_benchmarks.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved peer benchmarks for {result['total_users']:,} users -> peer_benchmarks.json")

    stop_spark()


if __name__ == "__main__":
    main()
