#!/usr/bin/env python3
"""
Hadoop Streaming mapper: reads raw CSV transactions from stdin,
emits (category TAB total_spent) pairs for per-category aggregation.
"""
import sys
import csv

reader = csv.reader(sys.stdin)
header = next(reader, None)  # skip header row

for row in reader:
    if len(row) < 6:
        continue
    try:
        category = row[1].strip()
        total_spent = float(row[5].strip())
        print(f"{category}\t{total_spent}")
    except (ValueError, IndexError):
        continue
