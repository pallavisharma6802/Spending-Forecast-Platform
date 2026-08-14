"""User-based collaborative filtering for budget-cap recommendations.

Feature matrix (5 features x N categories per user): forecast_30d,
avg_per_transaction, num_transactions, max_30d_spend, spend_velocity.
Min-max normalized, cosine similarity, top-10 neighbors.

Cold start: a freshly uploaded user with too little (or no) history to be
part of the similarity computation on their own gets appended to a
`reference_feature_matrix` (built once from the 200-user demo population) and
matched against it - so an uploaded user still gets a sensible budget cap and
peer comparison instead of an empty result.
"""

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame

from core.forecast_engine import velocity_ratio

TOP_K = 10
VELOCITY_CLAMP = (0.5, 3.0)
OWN_WEIGHT = 0.6
CF_WEIGHT = 0.4
SAFETY_BUFFER = 1.15


def build_feature_matrix(sdf: SparkDataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    """One row per customer_id, columns `{category}__{feature}`."""
    pdf = sdf.select("customer_id", "category", "total_spent", "transaction_date").toPandas()
    pdf["transaction_date"] = pd.to_datetime(pdf["transaction_date"])

    agg = pdf.groupby(["customer_id", "category"]).agg(
        avg_per_transaction=("total_spent", "mean"),
        num_transactions=("total_spent", "count"),
    ).reset_index()

    max30 = []
    for (cid, cat), g in pdf.groupby(["customer_id", "category"]):
        daily = g.groupby(g["transaction_date"].dt.normalize())["total_spent"].sum()
        daily = daily.asfreq("D").fillna(0.0)
        rolling = daily.rolling(30, min_periods=1).sum()
        max30.append({"customer_id": cid, "category": cat, "max_30d_spend": float(rolling.max()) if len(rolling) else 0.0})
    max30_df = pd.DataFrame(max30)

    velocity = []
    for (cid, cat), g in pdf.groupby(["customer_id", "category"]):
        v = velocity_ratio(g["transaction_date"], g["total_spent"], pdf["transaction_date"].max())
        velocity.append({"customer_id": cid, "category": cat, "spend_velocity": v})
    velocity_df = pd.DataFrame(velocity)

    fc30 = forecast_df[forecast_df["horizon_days"] == 30][["customer_id", "category", "forecasted_spend"]]
    fc30 = fc30.rename(columns={"forecasted_spend": "forecast_30d"})

    merged = fc30.merge(agg, on=["customer_id", "category"], how="outer")
    merged = merged.merge(max30_df, on=["customer_id", "category"], how="left")
    merged = merged.merge(velocity_df, on=["customer_id", "category"], how="left")
    merged = merged.fillna({
        "forecast_30d": 0.0, "avg_per_transaction": 0.0, "num_transactions": 0,
        "max_30d_spend": 0.0, "spend_velocity": 1.0,
    })

    def pivot(col, suffix):
        p = merged.pivot_table(index="customer_id", columns="category", values=col, fill_value=0.0)
        p.columns = [f"{c}__{suffix}" for c in p.columns]
        return p

    matrix = pd.concat([
        pivot("forecast_30d", "forecast"),
        pivot("avg_per_transaction", "avg_txn"),
        pivot("num_transactions", "n_txn"),
        pivot("max_30d_spend", "max30d"),
        pivot("spend_velocity", "velocity"),
    ], axis=1).fillna(0.0)
    return matrix


def _normalize(matrix: pd.DataFrame, stats_from: pd.DataFrame | None = None) -> pd.DataFrame:
    """Min-max normalize `matrix` using column stats from `stats_from` (or
    from `matrix` itself if not given). Always pass the larger/reference
    population as `stats_from` - normalizing a single new user against only
    their own min/max makes every column collapse to 0 (min == max == the
    one value present), which zeroes out cosine similarity entirely."""
    basis = stats_from if stats_from is not None else matrix
    col_min, col_max = basis.min(), basis.max()
    col_range = (col_max - col_min).replace(0, 1)
    return (matrix - col_min) / col_range


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norms = np.linalg.norm(a, axis=1, keepdims=True)
    a_norms[a_norms == 0] = 1.0
    b_norms = np.linalg.norm(b, axis=1, keepdims=True)
    b_norms[b_norms == 0] = 1.0
    return (a / a_norms) @ (b / b_norms).T


def budget_caps(
    feature_matrix: pd.DataFrame,
    forecast_df: pd.DataFrame,
    reference_matrix: pd.DataFrame | None = None,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """If `reference_matrix` is given, `feature_matrix`'s users are matched
    against it (cold start: new/uploaded users borrowing neighbors from the
    reference population) rather than against each other."""
    own_users = list(feature_matrix.index)
    neighbor_pool = reference_matrix if reference_matrix is not None else feature_matrix
    neighbor_pool = neighbor_pool.reindex(columns=feature_matrix.columns.union(neighbor_pool.columns), fill_value=0.0)
    own_aligned = feature_matrix.reindex(columns=neighbor_pool.columns, fill_value=0.0)

    pool_norm = _normalize(neighbor_pool)
    own_norm = _normalize(own_aligned, stats_from=neighbor_pool)
    sim = _cosine_sim(own_norm.to_numpy(), pool_norm.to_numpy())

    pool_users = list(neighbor_pool.index)
    self_mask_available = reference_matrix is None

    categories = sorted({c.rsplit("__", 1)[0] for c in feature_matrix.columns if c.endswith("__forecast")})
    forecast_pivot = own_aligned[[f"{c}__forecast" for c in categories]]
    forecast_pivot.columns = categories
    pool_forecast_raw = neighbor_pool[[f"{c}__forecast" for c in categories]]
    pool_forecast_raw.columns = categories
    velocity_pivot = own_aligned[[f"{c}__velocity" for c in categories]]
    velocity_pivot.columns = categories

    rows = []
    for i, user in enumerate(own_users):
        sims = sim[i].copy()
        if self_mask_available:
            self_idx = pool_users.index(user) if user in pool_users else -1
            if self_idx >= 0:
                sims[self_idx] = 0.0
        top_idx = np.argsort(sims)[::-1][:top_k]
        weights = sims[top_idx]
        weight_sum = weights.sum()

        for cat in categories:
            own_forecast = float(forecast_pivot.loc[user, cat])
            if weight_sum > 0:
                neighbor_fc = pool_forecast_raw[cat].to_numpy()[top_idx]
                cf_pred = float(np.dot(weights, neighbor_fc) / weight_sum)
            else:
                cf_pred = own_forecast

            velocity = float(np.clip(velocity_pivot.loc[user, cat], *VELOCITY_CLAMP))
            blended = OWN_WEIGHT * own_forecast + CF_WEIGHT * cf_pred
            cap = round(blended * velocity * SAFETY_BUFFER, 2)

            rows.append({
                "customer_id": user, "category": cat,
                "cf_predicted_spend": round(cf_pred, 2),
                "own_forecast_30d": round(own_forecast, 2),
                "spend_velocity": round(velocity, 4),
                "recommended_budget_cap": cap,
            })

    return pd.DataFrame(rows)
