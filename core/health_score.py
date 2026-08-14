"""Financial health score: composite 0-100 across four equal-weight (25%)
dimensions, each min-max normalized across a population before combining so
one outlier user can't collapse everyone else's score.

  stability     low variance in monthly total spend (CV = std/mean)
  essentials    Housing/Utilities + Groceries + Medical as % of total spend,
                peaks at 50% (the 50/30/20 rule midpoint)
  volatility    low month-over-month swings in total spend
  savings       recent actual spend vs 30-day forecast; spending below
                forecast signals headroom to save

`reference_raw` (four raw-dimension columns from a larger population, e.g.
the 200-user demo set) lets a single freshly uploaded user get normalized
against a real distribution instead of degenerating to a flat 0.5 on every
dimension, which is what min-max normalizing a population of one against
itself would otherwise do.
"""

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame

ESSENTIALS = {"Groceries", "Housing and Utilities", "Medical/Dental"}
WEIGHTS = {"stability": 0.25, "essentials": 0.25, "volatility": 0.25, "savings": 0.25}

RAW_COLUMNS = ["raw_stability", "raw_essentials", "raw_volatility", "raw_savings"]


def _cv(monthly: pd.Series) -> float:
    if len(monthly) < 2:
        return 1.0
    return float(monthly.std() / monthly.mean()) if monthly.mean() > 0 else 1.0


def _mom_volatility(monthly_sorted: pd.Series) -> float:
    m = monthly_sorted.to_numpy()
    if len(m) < 2:
        return 1.0
    pct_changes = np.abs(np.diff(m) / np.where(m[:-1] > 0, m[:-1], 1))
    return float(pct_changes.mean())


def _minmax(col: pd.Series, stats_from: pd.Series | None = None) -> pd.Series:
    basis = stats_from if stats_from is not None else col
    lo, hi = basis.min(), basis.max()
    if hi == lo:
        return pd.Series(0.5, index=col.index)
    return ((col - lo) / (hi - lo)).clip(0, 1)


def _grade(score: float) -> str:
    if score >= 80: return "A"
    if score >= 65: return "B"
    if score >= 50: return "C"
    if score >= 35: return "D"
    return "F"


def _label(score: float) -> str:
    if score >= 80: return "Excellent"
    if score >= 65: return "Good"
    if score >= 50: return "Fair"
    if score >= 35: return "Needs Attention"
    return "At Risk"


def compute_raw_dimensions(
    sdf: SparkDataFrame, forecast_df: pd.DataFrame, as_of: pd.Timestamp, lookback_months: int = 12
) -> pd.DataFrame:
    """The four raw (un-normalized) dimensions per user, before the
    population-relative min-max step. Split out so a reference population's
    raw values can be reused as the normalization basis for a cold-start user."""
    pdf = sdf.select("customer_id", "category", "total_spent", "transaction_date").toPandas()
    pdf["transaction_date"] = pd.to_datetime(pdf["transaction_date"])
    pdf = pdf[pdf["transaction_date"] <= as_of]
    pdf["month"] = pdf["transaction_date"].dt.to_period("M")

    monthly_totals = pdf.groupby(["customer_id", "month"])["total_spent"].sum().reset_index(name="monthly_total")

    lookback_start = as_of - pd.DateOffset(months=lookback_months)
    recent = pdf[pdf["transaction_date"] > lookback_start]
    actual_avg_monthly = (
        recent.groupby(["customer_id", "month"])["total_spent"].sum()
        .reset_index().groupby("customer_id")["total_spent"].mean()
        .rename("actual_avg_monthly")
    )

    forecast_30d = (
        forecast_df[forecast_df["horizon_days"] == 30]
        .groupby("customer_id")["forecasted_spend"].sum().rename("forecast_30d")
    )

    total_spend = pdf.groupby("customer_id")["total_spent"].sum().rename("total_spend")
    essentials_spend = (
        pdf[pdf["category"].isin(ESSENTIALS)].groupby("customer_id")["total_spent"].sum().rename("essentials_spend")
    )

    users = sorted(pdf["customer_id"].unique())
    scores = pd.DataFrame({"customer_id": users}).set_index("customer_id")
    scores = scores.join(total_spend).join(essentials_spend).join(forecast_30d).join(actual_avg_monthly)
    scores["essentials_spend"] = scores["essentials_spend"].fillna(0)
    scores["total_spend"] = scores["total_spend"].fillna(0)
    scores["forecast_30d"] = scores["forecast_30d"].fillna(scores["actual_avg_monthly"])
    n_months_span = max((as_of - pdf["transaction_date"].min()).days / 30.0, 1.0)
    scores["actual_avg_monthly"] = scores["actual_avg_monthly"].fillna(scores["total_spend"] / n_months_span)

    scores["cv"] = [
        _cv(monthly_totals[monthly_totals["customer_id"] == u]["monthly_total"]) for u in scores.index
    ]
    scores["raw_stability"] = 1 / (1 + scores["cv"])

    scores["essentials_ratio"] = (scores["essentials_spend"] / scores["total_spend"].replace(0, np.nan)).fillna(0)
    scores["raw_essentials"] = (1 - 2 * (scores["essentials_ratio"] - 0.5).abs()).clip(lower=0)

    scores["mom_vol"] = [
        _mom_volatility(monthly_totals[monthly_totals["customer_id"] == u].sort_values("month")["monthly_total"])
        for u in scores.index
    ]
    scores["raw_volatility"] = 1 / (1 + scores["mom_vol"])

    scores["savings_gap"] = (
        (scores["forecast_30d"] - scores["actual_avg_monthly"]) / scores["forecast_30d"].replace(0, np.nan)
    ).fillna(0).clip(-1, 1)
    scores["raw_savings"] = (scores["savings_gap"] + 1) / 2

    return scores.reset_index()


def compute_health_scores(
    sdf: SparkDataFrame,
    forecast_df: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback_months: int = 12,
    reference_raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    scores = compute_raw_dimensions(sdf, forecast_df, as_of, lookback_months)

    basis = reference_raw if reference_raw is not None else scores
    scores["norm_stability"] = _minmax(scores["raw_stability"], basis["raw_stability"])
    scores["norm_essentials"] = _minmax(scores["raw_essentials"], basis["raw_essentials"])
    scores["norm_volatility"] = _minmax(scores["raw_volatility"], basis["raw_volatility"])
    scores["norm_savings"] = _minmax(scores["raw_savings"], basis["raw_savings"])

    scores["score"] = (
        WEIGHTS["stability"] * scores["norm_stability"]
        + WEIGHTS["essentials"] * scores["norm_essentials"]
        + WEIGHTS["volatility"] * scores["norm_volatility"]
        + WEIGHTS["savings"] * scores["norm_savings"]
    ) * 100
    scores["score"] = scores["score"].round(1)
    scores["grade"] = scores["score"].apply(_grade)
    scores["label"] = scores["score"].apply(_label)

    out_cols = [
        "customer_id", "score", "grade", "label",
        "norm_stability", "norm_essentials", "norm_volatility", "norm_savings",
        "essentials_ratio", "cv", "mom_vol", "savings_gap",
        "actual_avg_monthly", "forecast_30d", "total_spend",
    ]
    output = scores[out_cols].rename(columns={
        "norm_stability": "stability_score", "norm_essentials": "essentials_score",
        "norm_volatility": "volatility_score", "norm_savings": "savings_score",
        "cv": "spend_cv", "mom_vol": "mom_volatility", "total_spend": "total_spend_hist",
    })
    return output.round(4)
