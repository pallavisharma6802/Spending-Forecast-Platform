"""Anomaly detection: is a user's recent spend pace unusual for a given
user-category, checked against an arbitrary trailing window (not a hardcoded
calendar month, so this works for day/week/month/year windows and for
freshly uploaded data with its own date range).

Two signals, same as the original design:
  z-score           projected spend (current window, scaled to a monthly
                     equivalent) vs that user-category's own historical
                     monthly mean/std
  isolation forest  trained once on every user-category's historical feature
                     vector (mean, std, cv, n_months, trend), pooled across
                     the whole population - catches structural anomalies
                     (e.g. unusual frequency) that z-score alone misses
"""

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

Z_THRESHOLD = 2.5
IF_CONTAMINATION = 0.05
OVERAGE_THRESHOLD = 25
MIN_HIST_ROWS_FOR_IF = 10


def _trend(series: pd.Series) -> float:
    if len(series) < 2:
        return 0.0
    x = np.arange(len(series))
    return float(np.polyfit(x, series.to_numpy(), 1)[0])


def detect_anomalies(
    sdf: SparkDataFrame,
    as_of: pd.Timestamp,
    window_days: int = 30,
    target_period_days: int = 30,
    budget_caps: pd.DataFrame | None = None,
) -> dict:
    """`window_days`: how many trailing days of actual data to evaluate.
    `target_period_days`: the period the projection is expressed in (e.g.
    evaluate a 13-day window projected to a 30-day/monthly equivalent)."""
    pdf = sdf.select("customer_id", "category", "total_spent", "transaction_date").toPandas()
    pdf["transaction_date"] = pd.to_datetime(pdf["transaction_date"])

    window_start = as_of - pd.Timedelta(days=window_days - 1)
    pace_factor = target_period_days / window_days

    historical = pdf[pdf["transaction_date"] < window_start].copy()
    current = pdf[(pdf["transaction_date"] >= window_start) & (pdf["transaction_date"] <= as_of)].copy()

    if historical.empty or current.empty:
        return {"generated_at": str(pd.Timestamp.now().date()), "as_of": str(as_of.date()),
                "window_days": window_days, "target_period_days": target_period_days, "alerts": []}

    historical["month"] = historical["transaction_date"].dt.to_period("M")
    monthly_hist = (
        historical.groupby(["customer_id", "category", "month"])["total_spent"]
        .sum().reset_index().rename(columns={"total_spent": "monthly_spend"})
    )

    hist_stats = (
        monthly_hist.groupby(["customer_id", "category"])["monthly_spend"]
        .agg(hist_mean="mean", hist_std="std", hist_median="median", n_months="count")
        .reset_index()
    )
    hist_stats["hist_std"] = hist_stats["hist_std"].fillna(hist_stats["hist_mean"] * 0.3)
    hist_stats["hist_cv"] = hist_stats["hist_std"] / hist_stats["hist_mean"].replace(0, 1)

    trends = (
        monthly_hist.sort_values("month").groupby(["customer_id", "category"])["monthly_spend"]
        .apply(_trend).reset_index().rename(columns={"monthly_spend": "trend"})
    )
    hist_stats = hist_stats.merge(trends, on=["customer_id", "category"], how="left")
    hist_stats["trend"] = hist_stats["trend"].fillna(0)

    cur_spend = (
        current.groupby(["customer_id", "category"])["total_spent"]
        .sum().reset_index().rename(columns={"total_spent": "window_spend"})
    )
    cur = cur_spend.merge(hist_stats, on=["customer_id", "category"], how="left")
    cur["projected_spend"] = (cur["window_spend"] * pace_factor).round(2)
    cur["z_score"] = ((cur["projected_spend"] - cur["hist_mean"]) / cur["hist_std"].replace(0, 1)).round(3)

    if_features = ["hist_mean", "hist_std", "hist_cv", "n_months", "trend"]
    run_if = len(hist_stats) >= MIN_HIST_ROWS_FOR_IF
    if run_if:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(hist_stats[if_features].fillna(0).to_numpy())
        iso = IsolationForest(n_estimators=200, contamination=IF_CONTAMINATION, random_state=42)
        iso.fit(x_train)

        x_score = cur[if_features].fillna(0).copy()
        x_score["hist_mean"] = cur["projected_spend"].fillna(x_score["hist_mean"])
        x_score_scaled = scaler.transform(x_score.to_numpy())
        cur["if_score"] = iso.score_samples(x_score_scaled).round(4)
        cur["if_anomaly"] = iso.predict(x_score_scaled)
    else:
        cur["if_score"] = 0.0
        cur["if_anomaly"] = 1

    if budget_caps is not None and not budget_caps.empty:
        caps_idx = budget_caps.set_index(["customer_id", "category"])["recommended_budget_cap"]
        cur["budget_cap_raw"] = cur.apply(
            lambda r: caps_idx.get((r["customer_id"], r["category"])), axis=1
        )
    else:
        cur["budget_cap_raw"] = np.nan

    cur["budget_cap"] = cur[["budget_cap_raw", "hist_mean"]].max(axis=1).clip(lower=25)
    cur["overage_pct"] = (
        (cur["projected_spend"] - cur["budget_cap"]) / cur["budget_cap"].replace(0, 1) * 100
    ).round(1)

    alerts = []
    for _, row in cur.iterrows():
        triggers = []
        z = float(row["z_score"]) if pd.notna(row["z_score"]) else 0.0
        if_anom = int(row["if_anomaly"])
        overage = float(row["overage_pct"]) if pd.notna(row["overage_pct"]) else 0.0
        cap = float(row["budget_cap"]) if pd.notna(row["budget_cap"]) else None

        if abs(z) >= Z_THRESHOLD:
            triggers.append("z_score")
        if if_anom == -1:
            triggers.append("isolation_forest")
        if cap is not None and overage > OVERAGE_THRESHOLD:
            triggers.append("pace_overage")

        actionable = (
            ("pace_overage" in triggers and ("z_score" in triggers or "isolation_forest" in triggers))
            or overage > 100
        )
        if not actionable:
            continue

        if (len(triggers) >= 2) or (overage > 30) or (abs(z) > 3.0):
            severity = "high"
        elif overage > 10 or abs(z) > 2.5:
            severity = "medium"
        else:
            severity = "low"

        parts = []
        if "pace_overage" in triggers and cap:
            parts.append(f"on pace to exceed {row['category']} cap by {overage:.0f}%")
        if "z_score" in triggers:
            direction = "above" if z > 0 else "below"
            parts.append(f"spend {abs(z):.1f}sigma {direction} historical average")
        if "isolation_forest" in triggers:
            parts.append("unusual spending pattern detected")

        alerts.append({
            "customer_id": row["customer_id"],
            "category": row["category"],
            "severity": severity,
            "triggers": triggers,
            "window_spend": round(float(row["window_spend"]), 2),
            "projected_spend": float(row["projected_spend"]),
            "budget_cap": round(cap, 2) if cap else None,
            "overage_pct": overage if cap else None,
            "z_score": z,
            "if_score": float(row["if_score"]),
            "hist_mean": round(float(row["hist_mean"]), 2),
            "hist_std": round(float(row["hist_std"]), 2),
            "message": "; ".join(parts).capitalize(),
        })

    sev_order = {"high": 0, "medium": 1, "low": 2}
    alerts.sort(key=lambda a: (sev_order[a["severity"]], -(a["overage_pct"] or 0)))

    return {
        "generated_at": str(pd.Timestamp.now().date()),
        "as_of": str(as_of.date()),
        "window_days": window_days,
        "target_period_days": target_period_days,
        "pace_factor": round(pace_factor, 3),
        "isolation_forest_used": run_if,
        "alerts": alerts,
    }
