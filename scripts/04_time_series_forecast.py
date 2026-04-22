"""
Hybrid time-series forecasting — local (no Spark) version.

Training cutoff: 2024-12-31
Forecast target: Jan 1–15, 2025  (15 days)
Validation:      Jan 1–13, 2025  (13 days of real actuals in the dataset)

Model routing
  Prophet   — 10 seasonal categories (category-level daily, distributed to users
               via spend share × behavior multiplier)
  Baseline  — 3 irregular categories (per-user recency-weighted monthly avg × velocity)

Outputs
  frontend/data/forecasts.json    per-user per-category 7/15/30-day forecasts
                                  (also stored as 15-day for the Jan-2025 window)
  Console                         MAE, RMSE, MAPE per category vs Jan 1-13 actuals
"""

import json
import os
import warnings
import numpy as np
import pandas as pd
from prophet import Prophet

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    SARIMAX_AVAILABLE = True
except Exception:
    SARIMAX_AVAILABLE = False

warnings.filterwarnings("ignore")

CSV_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "spending_patterns_5yr.csv")
DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "frontend", "data")
OUT_JSON  = os.path.join(DATA_DIR, "forecasts.json")

TRAIN_CUTOFF = pd.Timestamp("2024-12-31")
FORECAST_DAYS = 15              # Jan 1–15 2025
ACTUAL_DAYS   = 13              # Jan 1–13 2025 (real data in CSV)

# 2024 holdout showed baseline beats Prophet on every category (33% vs 74% mean MAPE).
# Prophet overfits synthetic seasonal patterns; a recency-weighted monthly average
# is simpler, faster, and more accurate for this dataset.
PROPHET_CATS  = set()
BASELINE_CATS = {
    "Fitness", "Food", "Friend Activities", "Gifts", "Groceries", "Hobbies",
    "Housing and Utilities", "Medical/Dental", "Personal Hygiene", "Shopping",
    "Subscriptions", "Transportation", "Travel",
}
SARIMAX_CATS = set(BASELINE_CATS)
# Evaluation-only set so we can compare all three models without changing
# the currently served baseline forecast output.
PROPHET_COMPARE_CATS = set(BASELINE_CATS)

#  Load 
txn = pd.read_csv(CSV_PATH, parse_dates=["Transaction Date"])
txn = txn.rename(columns={
    "Customer ID":    "customer_id",
    "Category":       "category",
    "Total Spent":    "total_spent",
    "Transaction Date": "transaction_date",
})
txn["total_spent"] = txn["total_spent"].astype(float)

train = txn[txn["transaction_date"] <= TRAIN_CUTOFF].copy()
actual_jan = txn[
    (txn["transaction_date"] >= pd.Timestamp("2025-01-01")) &
    (txn["transaction_date"] <= pd.Timestamp("2025-01-13"))
].copy()

print(f"Training rows : {len(train):,}  (up to {TRAIN_CUTOFF.date()})")
print(f"Jan-2025 rows : {len(actual_jan):,}  (Jan 1–13 actuals for validation)")
print()

#  Category-level daily spend for Prophet 
cat_daily = (
    train.groupby(["category", "transaction_date"])["total_spent"]
    .sum().reset_index()
    .rename(columns={"total_spent": "daily_spend"})
)

#  User spend shares ─
user_cat_total = train.groupby(["customer_id", "category"])["total_spent"].sum().rename("user_total")
cat_total      = train.groupby("category")["total_spent"].sum().rename("cat_total")
shares = (
    user_cat_total.reset_index()
    .merge(cat_total.reset_index(), on="category")
)
shares["share"] = shares["user_total"] / shares["cat_total"]

#  Per-user monthly spend for baseline categories 
user_monthly = (
    train[train["category"].isin(BASELINE_CATS)]
    .assign(month=lambda d: d["transaction_date"].dt.to_period("M"))
    .groupby(["customer_id", "category", "month"])["total_spent"]
    .sum().reset_index()
    .rename(columns={"total_spent": "monthly_spend"})
)
user_monthly_idx = user_monthly.set_index(["customer_id", "category"])

#  Behavior signals 
reference_date = TRAIN_CUTOFF

user_behavior = (
    train.groupby(["customer_id", "category"])
    .agg(last_txn=("transaction_date", "max"),
         user_txn_count=("total_spent", "count"))
    .reset_index()
)
cat_avg_txn = (
    user_behavior.groupby("category")["user_txn_count"].mean()
    .rename("cat_avg_txn_count").reset_index()
)
user_behavior = user_behavior.merge(cat_avg_txn, on="category")

# load velocity from baseline feature data
baseline_json = json.load(open(os.path.join(DATA_DIR, "baseline.json")))
vel_df = pd.DataFrame(baseline_json)[["customer_id", "category"]].copy()
vel_df["spend_velocity"] = 1.0  # default; overwrite from feature data if available
try:
    vel_lookup = pd.DataFrame(baseline_json)
    if "spend_velocity" in vel_lookup.columns:
        vel_df["spend_velocity"] = pd.to_numeric(vel_lookup["spend_velocity"], errors="coerce").fillna(1.0)
except Exception:
    pass

user_behavior = user_behavior.merge(vel_df, on=["customer_id", "category"], how="left")
user_behavior["spend_velocity"] = user_behavior["spend_velocity"].fillna(1.0)
behavior_idx = user_behavior.set_index(["customer_id", "category"])


def behavior_multiplier(cid, cat):
    try:
        row = behavior_idx.loc[(cid, cat)]
    except KeyError:
        return 1.0
    days_since = max((reference_date - pd.Timestamp(row["last_txn"])).days, 0)
    recency    = max(0.3, float(np.exp(-days_since / 365.0)))
    cat_avg    = float(row["cat_avg_txn_count"]) if row["cat_avg_txn_count"] > 0 else 1.0
    freq_ratio = float(np.clip(row["user_txn_count"] / cat_avg, 0.5, 2.0))
    vel        = float(np.clip(row["spend_velocity"], 0.5, 2.0))
    combined   = float((recency * freq_ratio * vel) ** (1.0 / 3.0))
    return float(np.clip(combined, 0.3, 3.0))


def velocity_for(cid, cat):
    try:
        v = behavior_idx.loc[(cid, cat), "spend_velocity"]
        return float(np.clip(v, 0.5, 2.0)) if pd.notna(v) else 1.0
    except KeyError:
        return 1.0


def baseline_forecast_user(cid, cat, horizon_days, decay=0.15):
    try:
        sub = user_monthly_idx.loc[(cid, cat)].reset_index(drop=True)
    except KeyError:
        return 0.0
    if isinstance(sub, pd.Series):
        sub = sub.to_frame().T
    if sub.empty:
        return 0.0
    max_month = sub["month"].max()
    sub = sub.copy()
    sub["months_ago"] = sub["month"].apply(lambda m: (max_month - m).n)
    sub["weight"]     = np.exp(-decay * sub["months_ago"])
    weighted_avg = float(np.average(sub["monthly_spend"], weights=sub["weight"]))
    return round(weighted_avg * horizon_days / 30.0, 2)


def sarimax_forecast_category_13d(category):
    cat_series = (
        cat_daily[cat_daily["category"] == category]
        .set_index("transaction_date")["daily_spend"]
        .asfreq("D")
        .fillna(0.0)
        .astype(float)
    )
    if len(cat_series) < 60:
        return float(cat_series.tail(30).mean() * ACTUAL_DAYS)

    # Use clipped values to reduce outlier influence, mirroring Prophet preprocessing.
    cap_99 = cat_series.quantile(0.99)
    y = cat_series.clip(upper=cap_99)

    def _fit_and_forecast(series, steps, order, seasonal_order):
        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False)
        return np.clip(np.asarray(fitted.forecast(steps=steps)), 0, None)

    # Use a short holdout in late 2024 to choose a better SARIMAX config per category.
    holdout_steps = ACTUAL_DAYS
    train_series = y.iloc[:-holdout_steps]
    holdout_actual = float(y.iloc[-holdout_steps:].sum())

    candidates = [
        {"name": "s111x101_7", "order": (1, 1, 1), "seasonal": (1, 0, 1, 7)},
        {"name": "s101x100_7", "order": (1, 0, 1), "seasonal": (1, 0, 0, 7)},
        {"name": "s201x101_7", "order": (2, 0, 1), "seasonal": (1, 0, 1, 7)},
    ]

    recent_daily = float(train_series.tail(30).mean())
    naive_pred_sum = max(0.0, recent_daily * holdout_steps)

    best = {"name": "naive_recent", "ape": abs(naive_pred_sum - holdout_actual) / max(holdout_actual, 1), "cfg": None}
    for cfg in candidates:
        try:
            pred = _fit_and_forecast(train_series, holdout_steps, cfg["order"], cfg["seasonal"])
            pred_sum = float(pred.sum())
            ape = abs(pred_sum - holdout_actual) / max(holdout_actual, 1)
            if ape < best["ape"]:
                best = {"name": cfg["name"], "ape": ape, "cfg": cfg}
        except Exception:
            continue

    if best["cfg"] is None:
        forecast_sum = float(max(0.0, y.tail(30).mean() * ACTUAL_DAYS))
    else:
        try:
            pred = _fit_and_forecast(y, ACTUAL_DAYS, best["cfg"]["order"], best["cfg"]["seasonal"])
            forecast_sum = float(pred.sum())
        except Exception:
            forecast_sum = float(max(0.0, y.tail(30).mean() * ACTUAL_DAYS))

    # Guardrail against unstable spikes: bound by recent 13-day spend envelope.
    recent_13d = float(y.tail(ACTUAL_DAYS).sum())
    lower = 0.4 * recent_13d
    upper = 2.2 * recent_13d
    return float(np.clip(forecast_sum, lower, upper))


#  Prophet: fit on ≤2024, forecast Jan 1–15 2025 
def _scaled_fallback(pdf, horizon):
    total = float(pdf["y"].sum())
    span  = max((pdf["ds"].max() - pdf["ds"].min()).days + 1, 1)
    return {h: round(total / span * h, 2) for h in [7, 15, 30]}


def forecast_category(category, group):
    pdf = (
        group[["transaction_date", "daily_spend"]]
        .rename(columns={"transaction_date": "ds", "daily_spend": "y"})
        .copy()
    )
    pdf["ds"] = pd.to_datetime(pdf["ds"])
    pdf = pdf.sort_values("ds")

    cap_99   = pdf["y"].quantile(0.99)
    pdf["y"] = pdf["y"].clip(upper=cap_99)

    try:
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.1,
        )
        m.fit(pdf)

        future   = m.make_future_dataframe(periods=30)  # enough for all horizons
        forecast = m.predict(future)
        # rows after training end = 2025-01-01 onwards
        future_rows = forecast[forecast["ds"] > TRAIN_CUTOFF].copy()
        future_rows["yhat"] = future_rows["yhat"].clip(lower=0)

        return {
            7:  round(float(future_rows.head(7)["yhat"].sum()), 2),
            15: round(float(future_rows.head(15)["yhat"].sum()), 2),
            30: round(float(future_rows.head(30)["yhat"].sum()), 2),
        }
    except Exception:
        return _scaled_fallback(pdf, 30)


print("Fitting Prophet on 2020–2024 data, forecasting Jan 1–15 2025...")
category_forecasts = {}
for cat, group in cat_daily.groupby("category"):
    if cat not in PROPHET_CATS:
        continue
    category_forecasts[cat] = forecast_category(cat, group)
    fc = category_forecasts[cat]
    print(f"  {cat:<25}  7d=${fc[7]:>8,.2f}  15d=${fc[15]:>9,.2f}  30d=${fc[30]:>9,.2f}")

#  Build per-user forecast results 
all_results = []

prophet_shares = shares[shares["category"].isin(PROPHET_CATS)]
for _, row in prophet_shares.iterrows():
    cid, cat, share = row["customer_id"], row["category"], float(row["share"])
    if cat not in category_forecasts:
        continue
    mult = behavior_multiplier(cid, cat)
    for h, cat_fc in category_forecasts[cat].items():
        all_results.append((cid, cat, h, round(cat_fc * share * mult, 2)))

baseline_shares = shares[shares["category"].isin(BASELINE_CATS)]
for _, row in baseline_shares.iterrows():
    cid, cat = row["customer_id"], row["category"]
    vel = velocity_for(cid, cat)
    for h in [7, 15, 30]:
        fc = baseline_forecast_user(cid, cat, h)
        all_results.append((cid, cat, h, round(fc * vel, 2)))

results_df = pd.DataFrame(all_results, columns=["customer_id", "category", "horizon_days", "forecasted_spend"])

#  Save forecasts.json ─
results_df.to_json(OUT_JSON, orient="records", indent=2)
print(f"\nSaved {len(results_df):,} forecast rows → {OUT_JSON}")

#  Validation: compare Jan 1–13 forecast vs actual 
print(f"\n{'='*68}")
print(f"VALIDATION: model forecast (Jan 1–13) vs actual Jan 1–13 2025")
print(f"{'='*68}")
if SARIMAX_AVAILABLE:
    print("Including SARIMAX comparison on baseline-routed categories.")
else:
    print("SARIMAX unavailable (statsmodels not installed); skipping SARIMAX comparison.")

# Actual Jan 1–13 category-level spend
actual_cat = (
    actual_jan.groupby("category")["total_spent"].sum()
    .rename("actual_13d")
    .reset_index()
)

eval_rows = []


def append_eval_row(category, model_name, pred_13d, actual_13d):
    ae = abs(pred_13d - actual_13d)
    ape = ae / max(actual_13d, 1) * 100
    eval_rows.append({
        "category": category,
        "model": model_name,
        "actual_13d": round(actual_13d, 2),
        "pred_13d": round(pred_13d, 2),
        "error": round(pred_13d - actual_13d, 2),
        "abs_error": round(ae, 2),
        "ape_pct": round(ape, 1),
    })

for cat, group in cat_daily.groupby("category"):
    actual_row = actual_cat[actual_cat["category"] == cat]
    actual_13d = float(actual_row["actual_13d"].values[0]) if not actual_row.empty else 0.0

    #  Prophet comparison categories 
    if cat in PROPHET_COMPARE_CATS:
        pdf = (
            group[["transaction_date", "daily_spend"]]
            .rename(columns={"transaction_date": "ds", "daily_spend": "y"})
            .copy()
        )
        pdf["ds"] = pd.to_datetime(pdf["ds"])
        pdf = pdf.sort_values("ds")
        cap_99   = pdf["y"].quantile(0.99)
        pdf["y"] = pdf["y"].clip(upper=cap_99)

        try:
            m = Prophet(
                yearly_seasonality=True, weekly_seasonality=False,
                daily_seasonality=False, changepoint_prior_scale=0.1,
            )
            m.fit(pdf)
            future   = m.make_future_dataframe(periods=ACTUAL_DAYS)
            forecast = m.predict(future)
            pred_13d = float(
                forecast[forecast["ds"] > TRAIN_CUTOFF]
                .head(ACTUAL_DAYS)["yhat"].clip(lower=0).sum()
            )
        except Exception:
            total = float(pdf["y"].sum())
            span  = max((pdf["ds"].max() - pdf["ds"].min()).days + 1, 1)
            pred_13d = total / span * ACTUAL_DAYS

        append_eval_row(cat, "prophet", pred_13d, actual_13d)

    #  Baseline categories ─
    if cat in BASELINE_CATS:
        # aggregate to category level (all users) then recency-weight
        cat_monthly = (
            user_monthly[user_monthly["category"] == cat]
            .groupby("month")["monthly_spend"].sum()
            .reset_index()
        )
        if cat_monthly.empty:
            continue
        max_m = cat_monthly["month"].max()
        cat_monthly["months_ago"] = cat_monthly["month"].apply(lambda m: (max_m - m).n)
        cat_monthly["weight"]     = np.exp(-0.15 * cat_monthly["months_ago"])
        weighted_avg = float(np.average(cat_monthly["monthly_spend"], weights=cat_monthly["weight"]))
        pred_13d = weighted_avg * ACTUAL_DAYS / 30.0
        append_eval_row(cat, "baseline", pred_13d, actual_13d)

    if SARIMAX_AVAILABLE and cat in SARIMAX_CATS:
        try:
            pred_13d = sarimax_forecast_category_13d(cat)
            append_eval_row(cat, "sarimax", pred_13d, actual_13d)
        except Exception:
            pass

eval_df = pd.DataFrame(eval_rows).sort_values("ape_pct")

#  Flag categories where 2024 training amounts differ strongly from Jan 2025 ─
# Check per-transaction avg ratio: 2024 vs 2025
suspect = {}
for cat in eval_df["category"].unique():
    avg_2024 = txn[(txn["category"] == cat) & (txn["transaction_date"].dt.year == 2024)]["total_spent"].mean()
    avg_2025 = actual_jan[actual_jan["category"] == cat]["total_spent"].mean()
    if pd.notna(avg_2024) and pd.notna(avg_2025) and avg_2025 > 0:
        ratio = avg_2024 / avg_2025
        suspect[cat] = ratio

eval_df["avg_txn_ratio_2024_vs_2025"] = eval_df["category"].map(suspect).round(1)
# categories where training avg/txn is >2x the Jan-2025 avg/txn are suspect
SUSPECT_CATS = {c for c, r in suspect.items() if r > 2.0}

eval_df["data_quality"] = eval_df["category"].apply(
    lambda c: f"⚠ inflated {suspect.get(c,1):.0f}×" if c in SUSPECT_CATS else "ok"
)

print(f"\n{'Category':<26} {'Model':<9} {'Actual':>10} {'Forecast':>10} {'Error':>10} {'APE':>7}  {'Data quality'}")
print("-" * 88)
for _, r in eval_df.iterrows():
    flag = f"  {r['data_quality']}" if r["data_quality"] != "ok" else ""
    print(f"  {r['category']:<24} {r['model']:<9} "
          f"${r['actual_13d']:>9,.0f} ${r['pred_13d']:>9,.0f} "
          f"{r['error']:>+10,.0f} {r['ape_pct']:>6.1f}%{flag}")

print("-" * 88)

#  Metrics on clean categories only ─
clean = eval_df[~eval_df["category"].isin(SUSPECT_CATS)]
dirty = eval_df[eval_df["category"].isin(SUSPECT_CATS)]


def print_metric_block(label, metric_df):
    print(f"\n {label} (n={len(metric_df)}) ─")
    for model_name in sorted(metric_df["model"].unique()):
        mdf = metric_df[metric_df["model"] == model_name]
        wape = (mdf["abs_error"].sum() / max(mdf["actual_13d"].sum(), 1)) * 100
        rmse = float(np.sqrt((mdf["error"] ** 2).mean()))
        print(f"  {model_name:<8} mean APE : {mdf['ape_pct'].mean():>6.1f}%   WAPE : {wape:>6.1f}%   RMSE : ${rmse:>8,.0f}")

    overall_rmse = float(np.sqrt((metric_df["error"] ** 2).mean()))
    print(f"  overall  mean APE : {metric_df['ape_pct'].mean():>6.1f}%   MAE  : ${metric_df['abs_error'].mean():>8,.0f}   RMSE : ${overall_rmse:>8,.0f}")


print_metric_block("All categories", eval_df)
print_metric_block(f"Clean categories only, excludes {sorted(SUSPECT_CATS)}", clean)

if len(clean) > 0:
    winners = (
        clean.sort_values(["category", "ape_pct"])
        .groupby("category", as_index=False)
        .first()[["category", "model", "ape_pct"]]
    )
    print("\n Per-category winner on clean set (lowest APE) ─")
    print(winners["model"].value_counts().to_string())

if not dirty.empty:
    print(f"\n Suspect categories (training avg/txn vs Jan-2025 avg/txn) ─")
    for _, r in dirty.iterrows():
        print(f"  {r['category']:<26}  {r['data_quality']}  "
              f"(train avg/txn ≠ Jan-2025 avg/txn — model trained on wrong scale)")

print(f"\n  Evaluation window: Jan 1–{ACTUAL_DAYS} 2025 actual vs model forecast of Jan 1–{FORECAST_DAYS} 2025.")
print(f"  Training cutoff  : {TRAIN_CUTOFF.date()}")
