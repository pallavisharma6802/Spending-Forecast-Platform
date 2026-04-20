"""
Peer benchmarking — how does each user's 30-day forecast compare to similar users?

Uses the CF similarity matrix already embedded in budget_caps.json:
  cf_predicted_spend  = weighted average of top-10 nearest neighbors' 30d forecasts
  own_forecast_30d    = user's own hybrid model forecast

For each user-category we compute:
  vs_peers_pct       = (own - cf) / cf * 100   (positive = spending more than peers)
  category_percentile = user's rank by total_spend among all 200 users in that category
  insight            = natural language summary

Output: frontend/data/peer_benchmarks.json
"""

import json
import os
import numpy as np
import pandas as pd

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "frontend", "data")
OUTPUT_JSON = os.path.join(DATA_DIR, "peer_benchmarks.json")

caps_df     = pd.DataFrame(json.load(open(os.path.join(DATA_DIR, "budget_caps.json"))))
baseline_df = pd.DataFrame(json.load(open(os.path.join(DATA_DIR, "baseline.json"))))

caps_df["cf_predicted_spend"] = caps_df["cf_predicted_spend"].astype(float)
caps_df["own_forecast_30d"]   = caps_df["own_forecast_30d"].astype(float)
baseline_df["total_spend"]    = baseline_df["total_spend"].astype(float)

#  vs-peers delta 
caps_df["vs_peers_pct"] = (
    (caps_df["own_forecast_30d"] - caps_df["cf_predicted_spend"])
    / caps_df["cf_predicted_spend"].replace(0, np.nan)
    * 100
).round(1)

#  category percentile (rank by total_spend across all users) 
pct_ranks = []
for cat, grp in baseline_df.groupby("category"):
    ranked = grp["total_spend"].rank(pct=True) * 100
    for cid, pct in zip(grp["customer_id"], ranked):
        pct_ranks.append({"customer_id": cid, "category": cat, "category_percentile": round(pct, 1)})

pct_df = pd.DataFrame(pct_ranks)

merged = caps_df.merge(pct_df, on=["customer_id", "category"], how="left")
merged = merged.merge(
    baseline_df[["customer_id", "category", "total_spend"]],
    on=["customer_id", "category"], how="left"
)

#  natural language insight 
def _insight(row):
    delta = row["vs_peers_pct"]
    cat   = row["category"]
    pct   = row["category_percentile"]

    if pd.isna(delta) or pd.isna(pct):
        return f"No comparison data available for {cat}."

    if abs(delta) < 5:
        direction = "about the same as"
        delta_str = ""
    elif delta > 0:
        direction = f"{abs(delta):.0f}% more on"
        delta_str = "more"
    else:
        direction = f"{abs(delta):.0f}% less on"
        delta_str = "less"

    pct_int = int(round(pct))
    if pct_int >= 75:
        rank_label = f"top {100 - pct_int}th percentile"
    elif pct_int <= 25:
        rank_label = f"bottom {pct_int}th percentile"
    else:
        rank_label = f"{pct_int}th percentile"

    if abs(delta) < 5:
        return f"You spend about the same on {cat} as similar users ({rank_label} overall)."
    else:
        return f"You spend {direction} {cat} than similar users ({rank_label} overall)."

merged["insight"] = merged.apply(_insight, axis=1)

#  flag direction for UI coloring ─
def _direction(delta):
    if pd.isna(delta) or abs(delta) < 5:
        return "neutral"
    return "above" if delta > 0 else "below"

merged["direction"] = merged["vs_peers_pct"].apply(_direction)

#  per-user summary: top movers 
records = []
for cid, grp in merged.groupby("customer_id"):
    cats = grp.sort_values("vs_peers_pct", ascending=False, key=abs)
    top_above = cats[cats["direction"] == "above"].head(3)
    top_below = cats[cats["direction"] == "below"].tail(3)

    detail = []
    for _, r in cats.iterrows():
        detail.append({
            "category":           r["category"],
            "own_forecast_30d":   round(float(r["own_forecast_30d"]), 2),
            "peer_forecast_30d":  round(float(r["cf_predicted_spend"]), 2),
            "vs_peers_pct":       float(r["vs_peers_pct"]) if pd.notna(r["vs_peers_pct"]) else None,
            "category_percentile": float(r["category_percentile"]) if pd.notna(r["category_percentile"]) else None,
            "direction":          r["direction"],
            "insight":            r["insight"],
        })

    records.append({
        "customer_id": cid,
        "categories":  detail,
    })

output = {
    "generated_at": str(pd.Timestamp.now().date()),
    "total_users":  len(records),
    "users":        records,
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f, indent=2)

#  summary ─
flat = merged.dropna(subset=["vs_peers_pct"])
print(f"Peer benchmarks generated for {merged['customer_id'].nunique()} users, "
      f"{merged['category'].nunique()} categories")
print(f"Saved → {OUTPUT_JSON}\n")

print(" vs-peers delta distribution ")
print(f"  Mean delta:   {flat['vs_peers_pct'].mean():+.1f}%")
print(f"  Median delta: {flat['vs_peers_pct'].median():+.1f}%")
print(f"  Std dev:      {flat['vs_peers_pct'].std():.1f}%")

print("\n Direction breakdown ")
for d in ["above", "below", "neutral"]:
    n = (merged["direction"] == d).sum()
    print(f"  {d:<7}  {n} user-category pairs")

print("\n Top 5 categories where users overspend vs peers ─")
top_cats = (
    flat[flat["direction"] == "above"]
    .groupby("category")["vs_peers_pct"].mean()
    .sort_values(ascending=False)
    .head(5)
)
for cat, val in top_cats.items():
    print(f"  {cat:<30}  +{val:.1f}%")

print("\n Sample insights (CUST_0001) ─")
sample = merged[merged["customer_id"] == "CUST_0001"].sort_values("vs_peers_pct", ascending=False)
for _, r in sample.head(5).iterrows():
    print(f"  {r['insight']}")
