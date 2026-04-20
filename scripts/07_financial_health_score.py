"""
Financial Health Score — 0 to 100 composite per user.

Four equal-weight dimensions (25% each):

  Stability        low variance in monthly total spend
                   CV = std / mean of monthly totals; lower CV → higher score

  Essentials ratio Housing & Utilities + Groceries + Medical/Dental as % of total
                   score peaks at 50% (the 50/30/20 rule midpoint); penalises
                   both under-spending on basics and spending nothing on discretionary

  Volatility       month-over-month absolute % swings in total spend
                   avg |Δt / spend_{t-1}|; lower swings → higher score

  Savings potential actual recent monthly spend vs 30-day forecast
                   spending below forecast signals headroom to save

Each raw dimension is min-max normalised across all users before combining,
so one outlier user cannot collapse everyone else's score.
"""

import json
import os
import sys
import pandas as pd
import numpy as np

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "frontend", "data")
CSV_PATH    = os.path.join(os.path.dirname(__file__), "..", "data", "spending_patterns_5yr.csv")
OUTPUT_JSON = os.path.join(DATA_DIR, "health_scores.json")

ESSENTIALS = {"Groceries", "Housing and Utilities", "Medical/Dental"}
WEIGHTS    = {"stability": 0.25, "essentials": 0.25, "volatility": 0.25, "savings": 0.25}

# ── Load inputs ───────────────────────────────────────────────────────────────
baseline_df = pd.DataFrame(json.load(open(os.path.join(DATA_DIR, "baseline.json"))))
forecast_df = pd.DataFrame(json.load(open(os.path.join(DATA_DIR, "forecasts.json"))))

baseline_df["total_spend"]      = baseline_df["total_spend"].astype(float)
forecast_df["forecasted_spend"] = forecast_df["forecasted_spend"].astype(float)
forecast_df["horizon_days"]     = forecast_df["horizon_days"].astype(int)

txn = pd.read_csv(CSV_PATH, parse_dates=["Transaction Date"])
txn = txn.rename(columns={
    "Customer ID": "customer_id", "Category": "category",
    "Total Spent": "total_spent", "Transaction Date": "transaction_date",
})
txn["month"] = txn["transaction_date"].dt.to_period("M")

users = sorted(baseline_df["customer_id"].unique())

# ── Per-user monthly totals (all years — more months = stabler estimates) ─────
monthly_totals = (
    txn.groupby(["customer_id", "month"])["total_spent"]
    .sum()
    .reset_index()
    .rename(columns={"total_spent": "monthly_total"})
)

# ── Per-user 2024 actual monthly average (ground truth for savings gap) ───────
actual_2024 = (
    txn[txn["transaction_date"].dt.year == 2024]
    .groupby(["customer_id", "month"])["total_spent"].sum()
    .reset_index()
    .groupby("customer_id")["total_spent"].mean()
    .rename("actual_avg_monthly")
)

# ── 30-day forecast total per user (all categories) ───────────────────────────
forecast_30d = (
    forecast_df[forecast_df["horizon_days"] == 30]
    .groupby("customer_id")["forecasted_spend"].sum()
    .rename("forecast_30d")
)

# ── Per-user total spend and essentials spend (from baseline) ─────────────────
user_total = baseline_df.groupby("customer_id")["total_spend"].sum().rename("total_spend")
user_essentials = (
    baseline_df[baseline_df["category"].isin(ESSENTIALS)]
    .groupby("customer_id")["total_spend"].sum()
    .rename("essentials_spend")
)

# ── Build scoring frame ───────────────────────────────────────────────────────
scores = pd.DataFrame({"customer_id": users}).set_index("customer_id")
scores = scores.join(user_total).join(user_essentials).join(forecast_30d).join(actual_2024)
scores["essentials_spend"] = scores["essentials_spend"].fillna(0)
scores["forecast_30d"]     = scores["forecast_30d"].fillna(scores["actual_avg_monthly"])
scores["actual_avg_monthly"] = scores["actual_avg_monthly"].fillna(scores["total_spend"] / 24)


# ── Dimension 1: Stability ────────────────────────────────────────────────────
# Coefficient of variation of monthly total spend.  Lower CV = more stable.
def _cv(uid):
    m = monthly_totals[monthly_totals["customer_id"] == uid]["monthly_total"]
    if len(m) < 2:
        return 1.0                  # only 1 data point → treat as high variance
    return float(m.std() / m.mean()) if m.mean() > 0 else 1.0

scores["cv"] = [_cv(u) for u in scores.index]
scores["raw_stability"] = 1 / (1 + scores["cv"])   # range (0, 1]


# ── Dimension 2: Essentials ratio ─────────────────────────────────────────────
# Peaks at 50%.  Below 20% or above 80% both score poorly.
scores["essentials_ratio"] = (
    scores["essentials_spend"] / scores["total_spend"].replace(0, np.nan)
).fillna(0)
scores["raw_essentials"] = (1 - 2 * (scores["essentials_ratio"] - 0.5).abs()).clip(lower=0)


# ── Dimension 3: Volatility ───────────────────────────────────────────────────
# Mean absolute month-over-month % change.  Lower swings = better.
def _mom_volatility(uid):
    m = (
        monthly_totals[monthly_totals["customer_id"] == uid]
        .sort_values("month")["monthly_total"]
        .values
    )
    if len(m) < 2:
        return 1.0
    pct_changes = np.abs(np.diff(m) / np.where(m[:-1] > 0, m[:-1], 1))
    return float(pct_changes.mean())

scores["mom_vol"] = [_mom_volatility(u) for u in scores.index]
scores["raw_volatility"] = 1 / (1 + scores["mom_vol"])   # range (0, 1]


# ── Dimension 4: Savings potential ───────────────────────────────────────────
# Positive gap (actual < forecast) → room to save.
# Capped at ±100% of forecast to prevent outliers dominating.
scores["savings_gap"] = (
    (scores["forecast_30d"] - scores["actual_avg_monthly"]) / scores["forecast_30d"]
).clip(-1, 1)
scores["raw_savings"] = (scores["savings_gap"] + 1) / 2   # map [-1,1] → [0,1]


# ── Min-max normalise each dimension across all users ─────────────────────────
def minmax(col):
    lo, hi = col.min(), col.max()
    if hi == lo:
        return pd.Series(0.5, index=col.index)
    return (col - lo) / (hi - lo)

scores["norm_stability"]  = minmax(scores["raw_stability"])
scores["norm_essentials"] = minmax(scores["raw_essentials"])
scores["norm_volatility"] = minmax(scores["raw_volatility"])
scores["norm_savings"]    = minmax(scores["raw_savings"])


# ── Composite score 0–100 ─────────────────────────────────────────────────────
scores["score"] = (
    WEIGHTS["stability"]  * scores["norm_stability"]  +
    WEIGHTS["essentials"] * scores["norm_essentials"] +
    WEIGHTS["volatility"] * scores["norm_volatility"] +
    WEIGHTS["savings"]    * scores["norm_savings"]
) * 100

scores["score"] = scores["score"].round(1)


# ── Grade and label ───────────────────────────────────────────────────────────
def _grade(s):
    if s >= 80: return "A"
    if s >= 65: return "B"
    if s >= 50: return "C"
    if s >= 35: return "D"
    return "F"

def _label(s):
    if s >= 80: return "Excellent"
    if s >= 65: return "Good"
    if s >= 50: return "Fair"
    if s >= 35: return "Needs Attention"
    return "At Risk"

scores["grade"] = scores["score"].apply(_grade)
scores["label"] = scores["score"].apply(_label)


# ── Dimension breakdown (rounded for readability) ────────────────────────────
detail_cols = [
    "score", "grade", "label",
    "norm_stability", "norm_essentials", "norm_volatility", "norm_savings",
    "essentials_ratio", "cv", "mom_vol", "savings_gap",
    "actual_avg_monthly", "forecast_30d", "total_spend",
]
output = scores[detail_cols].copy()
output.columns = [
    "score", "grade", "label",
    "stability_score", "essentials_score", "volatility_score", "savings_score",
    "essentials_ratio", "spend_cv", "mom_volatility", "savings_gap",
    "actual_avg_monthly", "forecast_30d", "total_spend_5yr",
]
output = output.round(4).reset_index()


# ── Save ──────────────────────────────────────────────────────────────────────
output.to_json(OUTPUT_JSON, orient="records", indent=2)
print(f"Saved {len(output)} health scores → {OUTPUT_JSON}")


# ── Distribution summary ──────────────────────────────────────────────────────
print("\n── Score distribution ──────────────────────────────────────────")
print(f"  Mean:    {scores['score'].mean():.1f}")
print(f"  Median:  {scores['score'].median():.1f}")
print(f"  Std dev: {scores['score'].std():.1f}")
print(f"  Min:     {scores['score'].min():.1f}  Max: {scores['score'].max():.1f}")

print("\n── Grade breakdown ─────────────────────────────────────────────")
grade_counts = scores["grade"].value_counts().sort_index()
for grade, cnt in grade_counts.items():
    bar = "█" * (cnt // 2)
    print(f"  {grade}  {cnt:>4} users  {bar}")

print("\n── Dimension means (normalised, higher = better) ───────────────")
for dim in ["norm_stability", "norm_essentials", "norm_volatility", "norm_savings"]:
    print(f"  {dim:<22}  {scores[dim].mean():.3f}")

print("\n── Top 5 users ─────────────────────────────────────────────────")
print(output.nlargest(5, "score")[["customer_id","score","grade","label"]].to_string(index=False))

print("\n── Bottom 5 users ──────────────────────────────────────────────")
print(output.nsmallest(5, "score")[["customer_id","score","grade","label"]].to_string(index=False))
