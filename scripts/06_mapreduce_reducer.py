#!/usr/bin/env python3
"""
Hadoop Streaming reducer: receives sorted (category TAB total_spent) pairs,
aggregates per category, assigns a spend tier (Low / Medium / High).

Output columns: category, total_spend, transaction_count, avg_spend, spend_tier
"""
import sys

current_category = None
total_spend = 0.0
transaction_count = 0


def emit(category, total, count):
    avg = round(total / count, 2) if count > 0 else 0.0
    if total >= 5000:
        tier = "High"
    elif total >= 1000:
        tier = "Medium"
    else:
        tier = "Low"
    print(f"{category}\t{round(total, 2)}\t{count}\t{avg}\t{tier}")


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    parts = line.split("\t", 1)
    if len(parts) != 2:
        continue
    category, value = parts[0], parts[1]
    try:
        spent = float(value)
    except ValueError:
        continue

    if category == current_category:
        total_spend += spent
        transaction_count += 1
    else:
        if current_category is not None:
            emit(current_category, total_spend, transaction_count)
        current_category = category
        total_spend = spent
        transaction_count = 1

if current_category is not None:
    emit(current_category, total_spend, transaction_count)
