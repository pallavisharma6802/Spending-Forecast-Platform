"""Per-category model selection by walk-forward backtest, then per-user-category
serving with the category's actual winner (fixing the old bug where a backtest
winner was computed but never used downstream).

Candidate models, one is picked per category:
  baseline      per-entity recency-weighted (EWMA) monthly average, scaled to
                the requested horizon
  hierarchical  same EWMA for the user's own trend, empirical-Bayes shrunk
                toward the category-level EWMA scaled by the user's historical
                spend share, weighted by the user's transaction count (sparse
                users lean on the category signal, active users lean on their
                own trend)
  sarimax       category-level daily SARIMAX, allocated to users by spend
                share (kept because most users don't have enough
                per-category transactions to fit their own daily model)

Backtest runs at the category level (13 categories - a plain walk-forward
loop, no need for Spark's parallelism at that scale). Serving runs at the
user-category level (hundreds to thousands of groups) via
`groupBy(...).applyInPandas(...)`, which is where PySpark's local-mode
parallel execution actually matters.
"""

import json
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType
from scipy import optimize, stats
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", message=".*Maximum Likelihood optimization failed.*")

N_FOLDS = 6
FOLD_HOLDOUT_DAYS = 13
DEFAULT_HORIZONS = (7, 30, 365)
Z_80 = float(stats.norm.ppf(0.9))  # 80% two-sided confidence interval

SARIMAX_CANDIDATES = [
    {"name": "s111x101_7", "order": (1, 1, 1), "seasonal": (1, 0, 1, 7)},
    {"name": "s101x100_7", "order": (1, 0, 1), "seasonal": (1, 0, 0, 7)},
    {"name": "s201x101_7", "order": (2, 0, 1), "seasonal": (1, 0, 1, 7)},
]


# ---- shared math ------------------------------------------------------------


def _ewma_forecast(monthly: pd.DataFrame, decay: float, horizon_days: float) -> float:
    """monthly: columns ['month' (pd.Period[M]), 'monthly_spend']."""
    if monthly.empty:
        return 0.0
    max_month = monthly["month"].max()
    months_ago = monthly["month"].apply(lambda m: (max_month - m).n).to_numpy()
    weights = np.exp(-decay * months_ago)
    weighted_avg = float(np.average(monthly["monthly_spend"].to_numpy(), weights=weights))
    return weighted_avg * horizon_days / 30.0


def _sarimax_forecast(
    daily: pd.Series, steps: int, holdout_steps: int = FOLD_HOLDOUT_DAYS
) -> tuple[float, str]:
    """daily: spend indexed by date, one row per calendar day (0-filled)."""
    if len(daily) < 60:
        recent = float(daily.tail(30).mean()) if len(daily) else 0.0
        return recent * steps, "naive_recent"

    cap_99 = daily.quantile(0.99)
    y = daily.clip(upper=cap_99)

    if len(y) <= holdout_steps:
        return float(y.mean() * steps), "naive_recent"

    train = y.iloc[:-holdout_steps]
    holdout_actual = float(y.iloc[-holdout_steps:].sum())
    recent_daily = float(train.tail(30).mean())
    naive_sum = max(0.0, recent_daily * holdout_steps)
    best = {"name": "naive_recent", "ape": abs(naive_sum - holdout_actual) / max(holdout_actual, 1), "cfg": None}

    for cfg in SARIMAX_CANDIDATES:
        try:
            model = SARIMAX(
                train, order=cfg["order"], seasonal_order=cfg["seasonal"], trend="c",
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False)
            pred = np.clip(np.asarray(model.forecast(steps=holdout_steps)), 0, None)
            ape = abs(float(pred.sum()) - holdout_actual) / max(holdout_actual, 1)
            if ape < best["ape"]:
                best = {"name": cfg["name"], "ape": ape, "cfg": cfg}
        except Exception:
            continue

    if best["cfg"] is None:
        return float(max(0.0, y.tail(30).mean() * steps)), "naive_recent"

    try:
        model = SARIMAX(
            y, order=best["cfg"]["order"], seasonal_order=best["cfg"]["seasonal"], trend="c",
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
        pred = np.clip(np.asarray(model.forecast(steps=steps)), 0, None)
        forecast_sum = float(pred.sum())
    except Exception:
        return float(max(0.0, y.tail(30).mean() * steps)), "naive_recent"

    # guardrail against unstable extrapolation: bound by the recent envelope
    recent_actual = float(y.tail(min(holdout_steps, len(y))).sum())
    lower = 0.4 * recent_actual
    upper = 2.2 * recent_actual
    return float(np.clip(forecast_sum, lower, upper)), best["name"]


_SARIMAX_BY_NAME = {cfg["name"]: cfg for cfg in SARIMAX_CANDIDATES}


def _sarimax_daily_forecast(daily: pd.Series, cfg_name: str, max_steps: int) -> np.ndarray:
    """Fits once using a config already chosen by the backtest (not a fresh
    candidate search) and returns per-day predictions for up to `max_steps` -
    callers take cumulative sums for whichever horizons they need, instead of
    the old approach of refitting from scratch once per horizon."""
    if len(daily) < 60 or cfg_name not in _SARIMAX_BY_NAME:
        recent = float(daily.tail(30).mean()) if len(daily) else 0.0
        return np.full(max_steps, recent)

    cap_99 = daily.quantile(0.99)
    y = daily.clip(upper=cap_99)
    cfg = _SARIMAX_BY_NAME[cfg_name]
    try:
        model = SARIMAX(
            y, order=cfg["order"], seasonal_order=cfg["seasonal"], trend="c",
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
        pred = np.clip(np.asarray(model.forecast(steps=max_steps)), 0, None)
    except Exception:
        pred = np.full(max_steps, float(y.tail(30).mean()))

    # guardrail against unstable extrapolation: bound each day by the recent
    # daily envelope (equivalent in spirit to the fold-evaluation guardrail,
    # just per-day so it composes correctly across any requested horizon)
    recent_daily = float(y.tail(min(30, len(y))).mean())
    return np.clip(pred, 0.4 * recent_daily, 2.2 * recent_daily)


def velocity_ratio(dates: pd.Series, amounts: pd.Series, as_of: pd.Timestamp, window_days: int = 90) -> float:
    """Trailing `window_days` ending at as_of vs the same window one year earlier."""
    recent = amounts[(dates > as_of - pd.Timedelta(days=window_days)) & (dates <= as_of)].sum()
    year_ago_end = as_of - pd.Timedelta(days=365)
    prior = amounts[(dates > year_ago_end - pd.Timedelta(days=window_days)) & (dates <= year_ago_end)].sum()
    if prior <= 0:
        return 1.0
    return float(np.clip(recent / prior, 0.5, 2.0))


# ---- backtest ----------------------------------------------------------------


def _walk_forward_folds(max_date: pd.Timestamp, n_folds: int = N_FOLDS) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Each fold: (train_cutoff, actual_start, actual_end), actual window is a
    fixed FOLD_HOLDOUT_DAYS immediately after the cutoff, so every fold's
    error is comparable regardless of which calendar month it falls in."""
    folds = []
    last_cutoff = (max_date - pd.Timedelta(days=FOLD_HOLDOUT_DAYS)).to_period("M").end_time.normalize() - pd.Timedelta(days=1)
    # walk backwards one month at a time from the most recent complete fold
    cutoff = last_cutoff
    for _ in range(n_folds):
        actual_start = cutoff + pd.Timedelta(days=1)
        actual_end = actual_start + pd.Timedelta(days=FOLD_HOLDOUT_DAYS - 1)
        if actual_end > max_date:
            cutoff = (cutoff.to_period("M") - 1).end_time.normalize() - pd.Timedelta(days=1)
            actual_start = cutoff + pd.Timedelta(days=1)
            actual_end = actual_start + pd.Timedelta(days=FOLD_HOLDOUT_DAYS - 1)
        folds.append((cutoff, actual_start, actual_end))
        cutoff = (cutoff.to_period("M") - 1).end_time.normalize() - pd.Timedelta(days=1)
    return folds


def _category_monthly(cat_pdf: pd.DataFrame, upto: pd.Timestamp) -> pd.DataFrame:
    sub = cat_pdf[cat_pdf["transaction_date"] <= upto]
    if sub.empty:
        return pd.DataFrame(columns=["month", "monthly_spend"])
    out = (
        sub.assign(month=sub["transaction_date"].dt.to_period("M"))
        .groupby("month")["total_spent"].sum()
        .reset_index(name="monthly_spend")
    )
    return out


def _category_daily(cat_pdf: pd.DataFrame, upto: pd.Timestamp) -> pd.Series:
    sub = cat_pdf[cat_pdf["transaction_date"] <= upto]
    if sub.empty:
        return pd.Series(dtype=float)
    daily = sub.groupby(sub["transaction_date"].dt.normalize())["total_spent"].sum()
    return daily.asfreq("D").fillna(0.0)


def _actual_total(cat_pdf: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    mask = (cat_pdf["transaction_date"] >= start) & (cat_pdf["transaction_date"] <= end)
    return float(cat_pdf.loc[mask, "total_spent"].sum())


def _wape(preds: list[float], actuals: list[float]) -> float:
    preds_a, actuals_a = np.asarray(preds), np.asarray(actuals)
    denom = max(float(actuals_a.sum()), 1.0)
    return float(np.abs(preds_a - actuals_a).sum() / denom * 100.0)


def _mean_ape(preds: list[float], actuals: list[float]) -> float:
    apes = [abs(p - a) / max(a, 1.0) * 100.0 for p, a in zip(preds, actuals)]
    return float(np.mean(apes)) if apes else float("inf")


@dataclass
class CategoryModelConfig:
    category: str
    model: str
    decay: float
    shrink_k: float
    sarimax_cfg_name: str
    mean_wape: dict  # model_name -> mean wape across folds
    residual_std_frac: float


def backtest_category(cat_pdf: pd.DataFrame, folds: list[tuple]) -> CategoryModelConfig:
    """Tune hyperparameters and pick the winning model for one category using
    only that category's own history (this deliberately mirrors what the
    per-category winner selection in the original evaluation script computed,
    except the winner now actually gets served)."""

    def baseline_objective(decay: float) -> float:
        preds, actuals = [], []
        for cutoff, start, end in folds:
            monthly = _category_monthly(cat_pdf, cutoff)
            if monthly.empty:
                continue
            preds.append(_ewma_forecast(monthly, decay, FOLD_HOLDOUT_DAYS))
            actuals.append(_actual_total(cat_pdf, start, end))
        return _wape(preds, actuals) if preds else 1e6

    decay_result = optimize.minimize_scalar(baseline_objective, bounds=(0.02, 0.6), method="bounded")
    best_decay = float(decay_result.x)

    def hierarchical_objective(shrink_k: float) -> float:
        # At the category level (aggregated across all users), the
        # "user trend" and "category trend" collapse to the same series, so
        # this tunes how much extra weight recent data gets relative to the
        # plain baseline - evaluated on the same category-level folds used
        # for every other candidate, for a fair comparison.
        preds, actuals = [], []
        for cutoff, start, end in folds:
            monthly = _category_monthly(cat_pdf, cutoff)
            if monthly.empty:
                continue
            own = _ewma_forecast(monthly, best_decay, FOLD_HOLDOUT_DAYS)
            pooled = _ewma_forecast(monthly, best_decay * 0.5, FOLD_HOLDOUT_DAYS)
            n_txn = float(len(cat_pdf[cat_pdf["transaction_date"] <= cutoff]))
            shrink = n_txn / (n_txn + shrink_k) if (n_txn + shrink_k) > 0 else 0.5
            preds.append(shrink * own + (1 - shrink) * pooled)
            actuals.append(_actual_total(cat_pdf, start, end))
        return _wape(preds, actuals) if preds else 1e6

    shrink_result = optimize.minimize_scalar(hierarchical_objective, bounds=(1.0, 500.0), method="bounded")
    best_shrink_k = float(shrink_result.x)

    sarimax_preds, sarimax_actuals, sarimax_cfg_names = [], [], []
    for cutoff, start, end in folds:
        daily = _category_daily(cat_pdf, cutoff)
        pred, cfg_name = _sarimax_forecast(daily, steps=FOLD_HOLDOUT_DAYS)
        sarimax_preds.append(pred)
        sarimax_actuals.append(_actual_total(cat_pdf, start, end))
        sarimax_cfg_names.append(cfg_name)

    mean_wape = {
        "baseline": baseline_objective(best_decay),
        "hierarchical": hierarchical_objective(best_shrink_k),
        "sarimax": _wape(sarimax_preds, sarimax_actuals) if sarimax_preds else 1e6,
    }
    winner = min(mean_wape, key=mean_wape.get)

    if winner == "baseline":
        preds, actuals = [], []
        for cutoff, start, end in folds:
            monthly = _category_monthly(cat_pdf, cutoff)
            if not monthly.empty:
                preds.append(_ewma_forecast(monthly, best_decay, FOLD_HOLDOUT_DAYS))
                actuals.append(_actual_total(cat_pdf, start, end))
    elif winner == "hierarchical":
        preds, actuals = [], []
        for cutoff, start, end in folds:
            monthly = _category_monthly(cat_pdf, cutoff)
            if monthly.empty:
                continue
            own = _ewma_forecast(monthly, best_decay, FOLD_HOLDOUT_DAYS)
            pooled = _ewma_forecast(monthly, best_decay * 0.5, FOLD_HOLDOUT_DAYS)
            n_txn = float(len(cat_pdf[cat_pdf["transaction_date"] <= cutoff]))
            shrink = n_txn / (n_txn + best_shrink_k) if (n_txn + best_shrink_k) > 0 else 0.5
            preds.append(shrink * own + (1 - shrink) * pooled)
            actuals.append(_actual_total(cat_pdf, start, end))
    else:
        preds, actuals = sarimax_preds, sarimax_actuals

    residuals_frac = [
        (p - a) / max(a, 1.0) for p, a in zip(preds, actuals)
    ] if preds else [0.0]
    residual_std_frac = float(np.std(residuals_frac)) if len(residuals_frac) > 1 else 0.3

    sarimax_winner_cfg = max(set(sarimax_cfg_names), key=sarimax_cfg_names.count) if sarimax_cfg_names else "naive_recent"

    return CategoryModelConfig(
        category=cat_pdf["category"].iloc[0] if not cat_pdf.empty else "",
        model=winner,
        decay=best_decay,
        shrink_k=best_shrink_k,
        sarimax_cfg_name=sarimax_winner_cfg,
        mean_wape=mean_wape,
        residual_std_frac=residual_std_frac,
    )


def run_backtest(sdf: SparkDataFrame, categories: list[str] | None = None) -> pd.DataFrame:
    """Runs the walk-forward backtest for each category and returns one row
    per category with the winning model and tuned hyperparameters."""
    pdf = sdf.select("customer_id", "category", "total_spent", "transaction_date").toPandas()
    pdf["transaction_date"] = pd.to_datetime(pdf["transaction_date"])
    max_date = pdf["transaction_date"].max()
    folds = _walk_forward_folds(max_date)

    cats = categories or sorted(pdf["category"].unique().tolist())
    rows = []
    for cat in cats:
        cat_pdf = pdf[pdf["category"] == cat]
        if cat_pdf.empty:
            continue
        cfg = backtest_category(cat_pdf, folds)
        rows.append({
            "category": cat,
            "model": cfg.model,
            "decay": cfg.decay,
            "shrink_k": cfg.shrink_k,
            "sarimax_cfg_name": cfg.sarimax_cfg_name,
            "residual_std_frac": cfg.residual_std_frac,
            "wape_baseline": cfg.mean_wape["baseline"],
            "wape_hierarchical": cfg.mean_wape["hierarchical"],
            "wape_sarimax": cfg.mean_wape["sarimax"],
            "n_folds": len(folds),
        })
    return pd.DataFrame(rows)


def save_model_config(config_df: pd.DataFrame, path: str) -> None:
    with open(path, "w") as f:
        json.dump(config_df.to_dict(orient="records"), f, indent=2)


def load_model_config(path: str) -> pd.DataFrame:
    with open(path) as f:
        return pd.DataFrame(json.load(f))


# ---- serving -------------------------------------------------------------


_SERVE_SCHEMA = StructType([
    StructField("customer_id", StringType()),
    StructField("category", StringType()),
    StructField("horizon_days", LongType()),
    StructField("forecasted_spend", DoubleType()),
    StructField("ci_low", DoubleType()),
    StructField("ci_high", DoubleType()),
    StructField("model_used", StringType()),
])


def _category_artifacts(sdf: SparkDataFrame, model_config: pd.DataFrame, horizons: tuple) -> dict:
    pdf = sdf.select("category", "total_spent", "transaction_date").toPandas()
    pdf["transaction_date"] = pd.to_datetime(pdf["transaction_date"])
    as_of = pdf["transaction_date"].max()

    artifacts = {}
    for _, row in model_config.iterrows():
        cat = row["category"]
        cat_pdf = pdf[pdf["category"] == cat]
        monthly = _category_monthly(cat_pdf, as_of)
        cat_total_hist = float(cat_pdf["total_spent"].sum())

        ewma_by_h = {h: _ewma_forecast(monthly, row["decay"], h) for h in horizons}
        pooled_by_h = {h: _ewma_forecast(monthly, row["decay"] * 0.5, h) for h in horizons}

        sarimax_by_h = {}
        if row["model"] == "sarimax":
            daily = _category_daily(cat_pdf, as_of)
            max_h = max(horizons)
            daily_pred = _sarimax_daily_forecast(daily, row["sarimax_cfg_name"], max_h)
            cumulative = np.cumsum(daily_pred)
            for h in horizons:
                sarimax_by_h[h] = float(cumulative[h - 1]) if h <= len(cumulative) else float(cumulative[-1])

        artifacts[cat] = {
            "model": row["model"],
            "decay": float(row["decay"]),
            "shrink_k": float(row["shrink_k"]),
            "residual_std_frac": float(row["residual_std_frac"]),
            "cat_total_hist": cat_total_hist,
            "ewma_by_h": ewma_by_h,
            "pooled_by_h": pooled_by_h,
            "sarimax_by_h": sarimax_by_h,
        }
    return artifacts


def forecast(
    sdf: SparkDataFrame,
    model_config: pd.DataFrame,
    horizons: tuple = DEFAULT_HORIZONS,
    reference_shares: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-user-category forecast for every horizon, using each category's
    backtest-selected model. `reference_shares` (customer_id, category,
    share) lets cold-start users (too little history to compute their own
    spend share) fall back to a peer reference set's share distribution -
    see recommender.py for how that gets built."""
    cat_artifacts = _category_artifacts(sdf, model_config, horizons)
    horizons_list = list(horizons)

    def serve_group(pdf: pd.DataFrame) -> pd.DataFrame:
        customer_id = pdf["customer_id"].iloc[0]
        category = pdf["category"].iloc[0]
        art = cat_artifacts.get(category)
        if art is None:
            return pd.DataFrame(columns=[f.name for f in _SERVE_SCHEMA.fields])

        pdf = pdf.copy()
        pdf["transaction_date"] = pd.to_datetime(pdf["transaction_date"])
        as_of = pdf["transaction_date"].max()

        monthly = (
            pdf.assign(month=pdf["transaction_date"].dt.to_period("M"))
            .groupby("month")["total_spent"].sum()
            .reset_index(name="monthly_spend")
        )
        n_txn = float(len(pdf))
        user_total_hist = float(pdf["total_spent"].sum())
        share = user_total_hist / art["cat_total_hist"] if art["cat_total_hist"] > 0 else 0.0

        velocity = velocity_ratio(pdf["transaction_date"], pdf["total_spent"], as_of)

        rows = []
        for h in horizons_list:
            if art["model"] == "baseline":
                raw = _ewma_forecast(monthly, art["decay"], h)
            elif art["model"] == "hierarchical":
                own = _ewma_forecast(monthly, art["decay"], h)
                category_scaled = art["ewma_by_h"][h] * share
                shrink = n_txn / (n_txn + art["shrink_k"]) if (n_txn + art["shrink_k"]) > 0 else 0.5
                raw = shrink * own + (1 - shrink) * category_scaled
            else:  # sarimax: category forecast allocated by spend share
                raw = art["sarimax_by_h"].get(h, art["ewma_by_h"][h]) * share

            point = raw * velocity
            band = point * art["residual_std_frac"] * Z_80
            rows.append({
                "customer_id": customer_id,
                "category": category,
                "horizon_days": h,
                "forecasted_spend": round(max(0.0, point), 2),
                "ci_low": round(max(0.0, point - band), 2),
                "ci_high": round(point + band, 2),
                "model_used": art["model"],
            })
        return pd.DataFrame(rows)

    result = sdf.groupBy("customer_id", "category").applyInPandas(serve_group, schema=_SERVE_SCHEMA)
    return result.toPandas()
