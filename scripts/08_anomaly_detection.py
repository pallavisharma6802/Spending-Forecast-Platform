"""
Anomaly-triggered intervention — mid-month spend pace check.

Two complementary signals:
  Z-score    per user-category, projected full-month spend vs historical monthly
             distribution.  Flags statistically unusual months (|Z| > 2.0).
  Isolation Forest  trained on historical monthly feature vectors per user-category.
             Catches structural anomalies that Z-score misses (e.g. unusual
             transaction frequency even when total spend looks normal).

"Current month" = January 2025, which has 13 days of data in the dataset —
a natural mid-month snapshot.  Spend is projected to month-end via pace factor
(31 / 13) and compared against each user's budget cap.

Output: frontend/data/anomalies.json  (consumed by Streamlit + FastAPI)
"""

import json
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "frontend", "data")
CSV_PATH    = os.path.join(os.path.dirname(__file__), "..", "data", "spending_patterns_5yr.csv")
OUTPUT_JSON = os.path.join(DATA_DIR, "anomalies.json")

CURRENT_MONTH  = pd.Period("2025-01", "M")
DAYS_ELAPSED   = 13
DAYS_IN_MONTH  = 31
PACE_FACTOR    = DAYS_IN_MONTH / DAYS_ELAPSED

Z_THRESHOLD      = 2.5    # flag if projected Z-score exceeds this
IF_CONTAMINATION = 0.05  # expected anomaly rate for Isolation Forest
OVERAGE_THRESHOLD = 25   # minimum % overage to raise a pace alert

#  Load data ─
txn = pd.read_csv(CSV_PATH, parse_dates=["Transaction Date"])
txn = txn.rename(columns={
    "Customer ID": "customer_id", "Category": "category",
    "Total Spent": "total_spent", "Transaction Date": "transaction_date",
})
txn["month"] = txn["transaction_date"].dt.to_period("M")

caps_df = pd.DataFrame(json.load(open(os.path.join(DATA_DIR, "budget_caps.json"))))
caps_df["recommended_budget_cap"] = caps_df["recommended_budget_cap"].astype(float)
caps_idx = caps_df.set_index(["customer_id", "category"])["recommended_budget_cap"]

#  Build monthly spend matrix 
monthly = (
    txn.groupby(["customer_id", "category", "month"])["total_spent"]
    .sum().reset_index()
    .rename(columns={"total_spent": "monthly_spend"})
)

historical = monthly[monthly["month"] < CURRENT_MONTH].copy()
current    = monthly[monthly["month"] == CURRENT_MONTH].copy()

# historical stats per user-category
hist_stats = (
    historical.groupby(["customer_id", "category"])["monthly_spend"]
    .agg(
        hist_mean  = "mean",
        hist_std   = "std",
        hist_median= "median",
        n_months   = "count",
    )
    .reset_index()
)
hist_stats["hist_std"]  = hist_stats["hist_std"].fillna(hist_stats["hist_mean"] * 0.3)
hist_stats["hist_cv"]   = hist_stats["hist_std"] / hist_stats["hist_mean"].replace(0, 1)

#  Z-score anomaly 
current = current.merge(hist_stats, on=["customer_id", "category"], how="left")
current["projected_spend"] = (current["monthly_spend"] * PACE_FACTOR).round(2)

current["z_score"] = (
    (current["projected_spend"] - current["hist_mean"])
    / current["hist_std"].replace(0, 1)
).round(3)

#  Isolation Forest ─
# Train on historical feature vectors: mean, std, cv, n_months, trend
# (monthly trend = slope of monthly spend over time per user-category)
def _trend(series):
    if len(series) < 2:
        return 0.0
    x = np.arange(len(series))
    return float(np.polyfit(x, series, 1)[0])

trends = (
    historical.sort_values("month")
    .groupby(["customer_id", "category"])["monthly_spend"]
    .apply(_trend)
    .reset_index()
    .rename(columns={"monthly_spend": "trend"})
)
hist_stats = hist_stats.merge(trends, on=["customer_id", "category"], how="left")
hist_stats["trend"] = hist_stats["trend"].fillna(0)

# feature matrix for IF training
IF_FEATURES = ["hist_mean", "hist_std", "hist_cv", "n_months", "trend"]
X_train = hist_stats[IF_FEATURES].fillna(0).values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

iso = IsolationForest(
    n_estimators=200,
    contamination=IF_CONTAMINATION,
    random_state=42,
)
iso.fit(X_train_scaled)

# score current month rows using their historical features as context
current_with_stats = current.merge(
    hist_stats[["customer_id", "category"] + IF_FEATURES],
    on=["customer_id", "category"], how="left",
    suffixes=("", "_dup"),
)
# drop any duplicate columns from the merge
current_with_stats = current_with_stats.loc[
    :, ~current_with_stats.columns.str.endswith("_dup")
]
# override hist_mean with projected (captures current anomaly strength)
X_score = current_with_stats[IF_FEATURES].fillna(0).copy()
X_score["hist_mean"] = current_with_stats["projected_spend"].fillna(X_score["hist_mean"])
X_score_scaled = scaler.transform(X_score.values)

current_with_stats["if_score"]   = iso.score_samples(X_score_scaled).round(4)
current_with_stats["if_anomaly"] = iso.predict(X_score_scaled)   # -1 = anomaly

#  Budget cap comparison ─
def _get_cap(row):
    try:
        return float(caps_idx.loc[(row["customer_id"], row["category"])])
    except KeyError:
        return None

current_with_stats["budget_cap_raw"] = current_with_stats.apply(_get_cap, axis=1)
# effective cap = max(CF cap, historical mean, $25 floor)
# prevents near-zero caps from triggering thousands-of-percent false alerts
current_with_stats["budget_cap"] = current_with_stats[["budget_cap_raw", "hist_mean"]].max(axis=1).clip(lower=25)
current_with_stats["overage_pct"] = (
    (current_with_stats["projected_spend"] - current_with_stats["budget_cap"])
    / current_with_stats["budget_cap"].replace(0, 1) * 100
).round(1)

#  Build alert list 
alerts = []

for _, row in current_with_stats.iterrows():
    triggers = []

    z        = float(row["z_score"])
    if_anom  = int(row["if_anomaly"])
    overage  = float(row["overage_pct"]) if pd.notna(row["overage_pct"]) else 0
    cap      = float(row["budget_cap"]) if pd.notna(row["budget_cap"]) else None

    if abs(z) >= Z_THRESHOLD:
        triggers.append("z_score")
    if if_anom == -1:
        triggers.append("isolation_forest")
    if cap is not None and overage > OVERAGE_THRESHOLD:
        triggers.append("pace_overage")

    # require at least a statistical signal + budget threat, or an extreme overage
    actionable = (
        ("pace_overage" in triggers and ("z_score" in triggers or "isolation_forest" in triggers))
        or overage > 100
    )
    if not actionable:
        continue

    # severity: high if multiple triggers or extreme overage/Z
    if (len(triggers) >= 2) or (overage > 30) or (abs(z) > 3.0):
        severity = "high"
    elif overage > 10 or abs(z) > 2.5:
        severity = "medium"
    else:
        severity = "low"

    # human-readable message
    parts = []
    if "pace_overage" in triggers and cap:
        parts.append(f"on pace to exceed {row['category']} cap by {overage:.0f}%")
    if "z_score" in triggers:
        direction = "above" if z > 0 else "below"
        parts.append(f"spend {abs(z):.1f}σ {direction} historical average")
    if "isolation_forest" in triggers:
        parts.append("unusual spending pattern detected")

    alerts.append({
        "customer_id":      row["customer_id"],
        "category":         row["category"],
        "severity":         severity,
        "triggers":         triggers,
        "current_spend":    round(float(row["monthly_spend"]), 2),
        "projected_spend":  float(row["projected_spend"]),
        "budget_cap":       round(cap, 2) if cap else None,
        "overage_pct":      overage if cap else None,
        "z_score":          z,
        "if_score":         float(row["if_score"]),
        "hist_mean":        round(float(row["hist_mean"]), 2),
        "hist_std":         round(float(row["hist_std"]), 2),
        "message":          "; ".join(parts).capitalize(),
    })

# sort: high first, then by overage descending
sev_order = {"high": 0, "medium": 1, "low": 2}
alerts.sort(key=lambda a: (sev_order[a["severity"]], -(a["overage_pct"] or 0)))

#  Save 
output = {
    "generated_at":  str(pd.Timestamp.now().date()),
    "current_month": str(CURRENT_MONTH),
    "days_elapsed":  DAYS_ELAPSED,
    "days_in_month": DAYS_IN_MONTH,
    "pace_factor":   round(PACE_FACTOR, 3),
    "alerts":        alerts,
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f, indent=2)

#  Summary ─
df_alerts = pd.DataFrame(alerts)
print(f"Anomaly detection complete — {CURRENT_MONTH} ({DAYS_ELAPSED}/{DAYS_IN_MONTH} days elapsed)")
print(f"Users with current-month data: {current['customer_id'].nunique()}")
print(f"User-category pairs checked:   {len(current_with_stats)}")
print(f"Total alerts:                  {len(alerts)}")
print()

if not df_alerts.empty:
    print(" By severity ─")
    for sev in ["high", "medium", "low"]:
        n = len(df_alerts[df_alerts["severity"] == sev])
        print(f"  {sev:<8}  {n} alerts")

    print()
    print(" High-severity alerts ─")
    high = df_alerts[df_alerts["severity"] == "high"].head(10)
    for _, r in high.iterrows():
        print(f"  {r['customer_id']}  {r['category']:<25}  "
              f"projected=${r['projected_spend']:>8,.0f}  "
              f"cap=${r['budget_cap']:>8,.0f}  "
              f"overage={r['overage_pct']:>+.0f}%  Z={r['z_score']:>+.1f}")

    print()
    print(" Top categories by alert count ")
    print(df_alerts["category"].value_counts().head(6).to_string())

print(f"\nSaved → {OUTPUT_JSON}")
