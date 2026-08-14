"""Peer benchmarking: how does a user's spend compare to similar users?

  vs_peers_pct         (own_forecast_30d - cf_predicted_spend) / cf_predicted_spend
                        - straight from recommender.budget_caps() output, positive
                        means spending more than the CF-weighted neighbor average
  category_percentile  the user's historical category spend ranked against a
                        peer population (accepts `reference_totals` so a cold-start
                        upload is ranked against the 200-user reference set rather
                        than against just themselves)
"""

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from scipy import stats


def category_totals(sdf: SparkDataFrame) -> pd.DataFrame:
    pdf = sdf.select("customer_id", "category", "total_spent").toPandas()
    return pdf.groupby(["customer_id", "category"])["total_spent"].sum().reset_index(name="total_spend")


def _insight(row) -> str:
    delta, cat, pct = row["vs_peers_pct"], row["category"], row["category_percentile"]
    if pd.isna(delta) or pd.isna(pct):
        return f"No comparison data available for {cat}."

    pct_int = int(round(pct))
    if pct_int >= 75:
        rank_label = f"top {100 - pct_int}th percentile"
    elif pct_int <= 25:
        rank_label = f"bottom {pct_int}th percentile"
    else:
        rank_label = f"{pct_int}th percentile"

    if abs(delta) < 5:
        return f"You spend about the same on {cat} as similar users ({rank_label} overall)."
    direction = f"{abs(delta):.0f}% more on" if delta > 0 else f"{abs(delta):.0f}% less on"
    return f"You spend {direction} {cat} than similar users ({rank_label} overall)."


def _direction(delta: float) -> str:
    if pd.isna(delta) or abs(delta) < 5:
        return "neutral"
    return "above" if delta > 0 else "below"


def compute_peer_benchmarks(
    budget_caps_df: pd.DataFrame,
    own_totals: pd.DataFrame,
    reference_totals: pd.DataFrame | None = None,
) -> dict:
    caps = budget_caps_df.copy()
    caps["vs_peers_pct"] = (
        (caps["own_forecast_30d"] - caps["cf_predicted_spend"]) / caps["cf_predicted_spend"].replace(0, np.nan) * 100
    ).round(1)

    ref = reference_totals if reference_totals is not None else own_totals
    pct_rows = []
    for cat, own_grp in own_totals.groupby("category"):
        ref_values = ref[ref["category"] == cat]["total_spend"].to_numpy()
        if len(ref_values) == 0:
            continue
        for _, r in own_grp.iterrows():
            pct = float(stats.percentileofscore(ref_values, r["total_spend"], kind="rank"))
            pct_rows.append({"customer_id": r["customer_id"], "category": cat, "category_percentile": round(pct, 1)})
    pct_df = pd.DataFrame(pct_rows)

    merged = caps.merge(pct_df, on=["customer_id", "category"], how="left")
    merged = merged.merge(own_totals, on=["customer_id", "category"], how="left")
    merged["insight"] = merged.apply(_insight, axis=1)
    merged["direction"] = merged["vs_peers_pct"].apply(_direction)

    records = []
    for cid, grp in merged.groupby("customer_id"):
        detail = []
        for _, r in grp.sort_values("vs_peers_pct", ascending=False, key=abs).iterrows():
            detail.append({
                "category": r["category"],
                "own_forecast_30d": round(float(r["own_forecast_30d"]), 2),
                "peer_forecast_30d": round(float(r["cf_predicted_spend"]), 2),
                "vs_peers_pct": float(r["vs_peers_pct"]) if pd.notna(r["vs_peers_pct"]) else None,
                "category_percentile": float(r["category_percentile"]) if pd.notna(r["category_percentile"]) else None,
                "direction": r["direction"],
                "insight": r["insight"],
            })
        records.append({"customer_id": cid, "categories": detail})

    return {
        "generated_at": str(pd.Timestamp.now().date()),
        "total_users": len(records),
        "users": records,
    }
