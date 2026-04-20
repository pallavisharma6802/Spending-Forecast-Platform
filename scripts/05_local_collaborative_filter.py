"""
Local collaborative filtering — pure pandas, no Spark required.
Reads frontend/data/{forecasts,baseline}.json, writes frontend/data/budget_caps.json.

65-feature matrix (5 features × 13 categories) per user:
  forecast_30d, avg_per_transaction, num_transactions, max_30d_spend, spend_velocity

User-user cosine similarity → top-10 neighbors
Budget cap = (0.6 × own_30d + 0.4 × CF_30d) × velocity × 1.15
"""

import json
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "data")
OUT_JSON = os.path.join(DATA_DIR, "budget_caps.json")

forecasts_df = pd.DataFrame(json.load(open(os.path.join(DATA_DIR, "forecasts.json"))))
baseline_df  = pd.DataFrame(json.load(open(os.path.join(DATA_DIR, "baseline.json"))))

for col in ["forecasted_spend", "horizon_days"]:
    forecasts_df[col] = pd.to_numeric(forecasts_df[col], errors="coerce")
for col in ["total_spend", "avg_per_transaction", "num_transactions",
            "max_30d_spend", "spend_velocity"]:
    baseline_df[col] = pd.to_numeric(baseline_df[col], errors="coerce").fillna(0.0)

forecasts_30d = forecasts_df[forecasts_df["horizon_days"] == 30][
    ["customer_id", "category", "forecasted_spend"]
]

#  Pivot to feature matrix ─
def pivot(df, value_col, suffix):
    p = df.pivot_table(index="customer_id", columns="category",
                       values=value_col, fill_value=0.0)
    p.columns = [f"{c}__{suffix}" for c in p.columns]
    return p

fc_pivot  = pivot(forecasts_30d, "forecasted_spend",    "forecast")
avg_pivot = pivot(baseline_df,   "avg_per_transaction", "avg_txn")
cnt_pivot = pivot(baseline_df,   "num_transactions",    "n_txn")
max_pivot = pivot(baseline_df,   "max_30d_spend",       "max30d")
vel_pivot = pivot(baseline_df,   "spend_velocity",      "velocity")

all_users = fc_pivot.index
feature_matrix = pd.concat([
    fc_pivot,
    avg_pivot.reindex(all_users, fill_value=0.0),
    cnt_pivot.reindex(all_users, fill_value=0.0),
    max_pivot.reindex(all_users, fill_value=0.0),
    vel_pivot.reindex(all_users, fill_value=1.0),
], axis=1).fillna(0.0)

#  Min-max normalize ─
col_range = (feature_matrix.max() - feature_matrix.min()).replace(0, 1)
feature_norm = (feature_matrix - feature_matrix.min()) / col_range

#  Cosine similarity ─
mat   = feature_norm.values
norms = np.linalg.norm(mat, axis=1, keepdims=True)
norms[norms == 0] = 1.0
mat_norm   = mat / norms
sim_matrix = mat_norm @ mat_norm.T
np.fill_diagonal(sim_matrix, 0.0)

#  Budget caps ─
categories = [c.replace("__forecast", "") for c in fc_pivot.columns]
users      = list(all_users)
K          = 10
results    = []

for i, user in enumerate(users):
    sims       = sim_matrix[i]
    top_k_idx  = np.argsort(sims)[::-1][:K]
    weights    = sims[top_k_idx]
    weight_sum = weights.sum()

    for j, cat in enumerate(categories):
        own_forecast = float(fc_pivot.iloc[i, j])

        cf_pred = (
            float(np.dot(weights, fc_pivot.values[top_k_idx, j]) / weight_sum)
            if weight_sum > 0 else own_forecast
        )

        vel_col  = f"{cat}__velocity"
        velocity = float(
            vel_pivot.loc[user, vel_col]
            if vel_col in vel_pivot.columns else 1.0
        )
        velocity = float(np.clip(velocity, 0.5, 3.0))

        blended    = 0.6 * own_forecast + 0.4 * cf_pred
        budget_cap = round(blended * velocity * 1.15, 2)

        results.append({
            "customer_id":            user,
            "category":               cat,
            "cf_predicted_spend":     round(cf_pred, 2),
            "own_forecast_30d":       round(own_forecast, 2),
            "spend_velocity":         round(velocity, 4),
            "recommended_budget_cap": budget_cap,
        })

caps_df = pd.DataFrame(results)
caps_df.to_json(OUT_JSON, orient="records", indent=2)
print(f"Saved {len(caps_df):,} budget cap rows → {OUT_JSON}")

#  Sanity check 
print(f"\nAvg recommended budget cap by category:")
check = caps_df.groupby("category")["recommended_budget_cap"].mean().sort_values(ascending=False)
for cat, val in check.items():
    print(f"  {cat:<26}  ${val:>8,.0f}")
