import sys
import os
sys.path.insert(0, '/tmp/pypackages')

os.environ["STAN_NUM_THREADS"] = "1"
os.environ["CMDSTAN_THREADS"] = "1"

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
import pandas as pd
import numpy as np
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# ── Model routing ─────────────────────────────────────────────────────────────
# Prophet: categories with real repeating seasonal patterns
# Baseline: irregular categories where Prophet overfits noise
PROPHET_CATS = {
    "Fitness", "Food", "Friend Activities", "Hobbies", "Medical/Dental",
    "Personal Hygiene", "Shopping", "Subscriptions", "Transportation", "Travel"
}
BASELINE_CATS = {"Gifts", "Groceries", "Housing and Utilities"}

spark = SparkSession.builder \
    .appName("FintechTimeSeriesForecast") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

df = spark.read.csv(
    "hdfs://namenode:8020/user/fintech/transactions/spending_patterns_5yr.csv",
    header=True,
    inferSchema=True
)

df = df.withColumnRenamed("Customer ID", "customer_id") \
       .withColumnRenamed("Category", "category") \
       .withColumnRenamed("Total Spent", "total_spent") \
       .withColumnRenamed("Transaction Date", "transaction_date") \
       .select("customer_id", "category", "total_spent", "transaction_date")

df = df.withColumn("transaction_date", F.to_date("transaction_date"))

# ── Step 1: daily category spend (Prophet input) ──────────────────────────────
category_daily = df.groupBy("category", "transaction_date") \
                   .agg(F.round(F.sum("total_spent"), 2).alias("daily_spend"))

# ── Step 2: user spend shares (for Prophet distribution) ─────────────────────
user_cat_total = df.groupBy("customer_id", "category") \
                   .agg(F.round(F.sum("total_spent"), 2).alias("user_total"))

cat_total = df.groupBy("category") \
              .agg(F.round(F.sum("total_spent"), 2).alias("cat_total"))

user_shares = (
    user_cat_total
    .join(cat_total, on="category")
    .withColumn("share", F.col("user_total") / F.col("cat_total"))
    .select("customer_id", "category", "share")
)

# ── Step 2b: per-user monthly spend for baseline categories ───────────────────
user_monthly = (
    df.filter(F.col("category").isin(list(BASELINE_CATS)))
    .withColumn("month", F.trunc("transaction_date", "MM"))
    .groupBy("customer_id", "category", "month")
    .agg(F.round(F.sum("total_spent"), 2).alias("monthly_spend"))
)

# ── Step 2c: behavior signals for personalized multiplier ─────────────────────
user_behavior = df.groupBy("customer_id", "category") \
    .agg(
        F.max("transaction_date").alias("last_txn_date"),
        F.count("*").alias("user_txn_count")
    )

cat_txn_stats = user_behavior.groupBy("category") \
    .agg(F.round(F.avg("user_txn_count"), 4).alias("cat_avg_txn_count"))

user_behavior = user_behavior.join(cat_txn_stats, on="category")

velocity_df = spark.read.csv(
    "hdfs://namenode:8020/user/fintech/velocity/",
    header=True, inferSchema=True
).select("customer_id", "category", "spend_velocity")

user_behavior = user_behavior.join(velocity_df, on=["customer_id", "category"], how="left")

# ── Collect to driver ─────────────────────────────────────────────────────────
cat_daily_pd    = category_daily.toPandas()
user_shares_pd  = user_shares.toPandas()
user_monthly_pd = user_monthly.toPandas()
behavior_pd     = user_behavior.toPandas()

user_monthly_pd["month"] = pd.to_datetime(user_monthly_pd["month"]).dt.to_period("M")

reference_date = pd.to_datetime(cat_daily_pd["transaction_date"]).max()

behavior_pd["last_txn_date"]   = pd.to_datetime(behavior_pd["last_txn_date"])
behavior_pd["spend_velocity"]  = pd.to_numeric(behavior_pd["spend_velocity"], errors="coerce")
behavior_idx = behavior_pd.set_index(["customer_id", "category"])

# index user_monthly for fast lookup
user_monthly_idx = user_monthly_pd.set_index(["customer_id", "category"])

# ── Behavior multiplier (Prophet categories only) ─────────────────────────────
# Geometric mean of recency × frequency ratio × velocity, clamped [0.3, 3.0]
def behavior_multiplier(cid, cat):
    try:
        row = behavior_idx.loc[(cid, cat)]
    except KeyError:
        return 1.0

    days_since = max((reference_date - row["last_txn_date"]).days, 0)
    recency    = max(0.3, float(np.exp(-days_since / 365.0)))

    cat_avg    = float(row["cat_avg_txn_count"]) if row["cat_avg_txn_count"] > 0 else 1.0
    freq_ratio = float(np.clip(row["user_txn_count"] / cat_avg, 0.5, 2.0))

    vel      = row["spend_velocity"]
    velocity = float(np.clip(vel, 0.5, 2.0)) if pd.notna(vel) else 1.0

    combined = float((recency * freq_ratio * velocity) ** (1.0 / 3.0))
    return float(np.clip(combined, 0.3, 3.0))


# ── Baseline model (irregular categories) ────────────────────────────────────
# Recency-weighted average of per-user monthly spend.
# Lambda=0.15: last month weight=1.0, 6mo ago=0.41, 12mo ago=0.17, 24mo ago=0.03
# Recency and frequency are already embedded in the weighted average, so only
# velocity is applied on top as a trend adjustment.
def baseline_forecast_user(cid, cat, horizon_days, decay=0.15):
    try:
        sub = user_monthly_idx.loc[(cid, cat)].reset_index(drop=True)
    except KeyError:
        return 0.0

    if isinstance(sub, pd.Series):
        sub = sub.to_frame().T

    if sub.empty:
        return 0.0

    max_month        = sub["month"].max()
    sub              = sub.copy()
    sub["months_ago"] = sub["month"].apply(lambda m: (max_month - m).n)
    sub["weight"]    = np.exp(-decay * sub["months_ago"])

    weighted_avg = float(np.average(sub["monthly_spend"], weights=sub["weight"]))
    return round(weighted_avg * horizon_days / 30.0, 2)


def velocity_for(cid, cat):
    try:
        vel = behavior_idx.loc[(cid, cat), "spend_velocity"]
        return float(np.clip(vel, 0.5, 2.0)) if pd.notna(vel) else 1.0
    except KeyError:
        return 1.0


# ── Step 3: Prophet for seasonal categories ───────────────────────────────────
def _scaled_fallback(prophet_df, horizon):
    total = float(prophet_df['y'].sum())
    span  = max((prophet_df['ds'].max() - prophet_df['ds'].min()).days + 1, 1)
    return round(total / span * horizon, 2)


def forecast_category(category, group):
    pdf = group[['transaction_date', 'daily_spend']].rename(
        columns={'transaction_date': 'ds', 'daily_spend': 'y'}
    )
    pdf['ds'] = pd.to_datetime(pdf['ds'])
    pdf = pdf.sort_values('ds')

    cap_99   = pdf['y'].quantile(0.99)
    pdf['y'] = pdf['y'].clip(upper=cap_99)

    try:
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.1
        )
        m.fit(pdf)

        horizons = {}
        for h in [7, 15, 30]:
            future   = m.make_future_dataframe(periods=h)
            forecast = m.predict(future)
            horizons[h] = round(float(forecast.tail(h)['yhat'].clip(lower=0).sum()), 2)
        return horizons

    except Exception:
        return {h: _scaled_fallback(pdf, h) for h in [7, 15, 30]}


print("Running Prophet for seasonal categories...")
category_forecasts = {}
for category, group in cat_daily_pd.groupby('category'):
    if category not in PROPHET_CATS:
        continue
    n  = len(group)
    category_forecasts[category] = forecast_category(category, group)
    fc = category_forecasts[category]
    print(f"  {category:<25}  n={n:>4}  7d=${fc[7]:>8.2f}  15d=${fc[15]:>9.2f}  30d=${fc[30]:>9.2f}")

print(f"\nRunning recency-weighted baseline for irregular categories: {sorted(BASELINE_CATS)}")

# ── Step 4: combine both models into one results list ─────────────────────────
all_results = []

# Prophet categories: category forecast × spend share × behavior multiplier
prophet_rows = user_shares_pd[user_shares_pd["category"].isin(PROPHET_CATS)]
for _, row in prophet_rows.iterrows():
    cid, cat, share = row['customer_id'], row['category'], float(row['share'])
    if cat not in category_forecasts:
        continue
    mult = behavior_multiplier(cid, cat)
    for h, cat_fc in category_forecasts[cat].items():
        all_results.append((cid, cat, h, round(cat_fc * share * mult, 2)))

# Baseline categories: per-user recency-weighted average × velocity only
baseline_rows = user_shares_pd[user_shares_pd["category"].isin(BASELINE_CATS)]
for _, row in baseline_rows.iterrows():
    cid, cat = row['customer_id'], row['category']
    vel = velocity_for(cid, cat)
    for h in [7, 15, 30]:
        fc = baseline_forecast_user(cid, cat, h)
        all_results.append((cid, cat, h, round(fc * vel, 2)))

results_pd = pd.DataFrame(
    all_results,
    columns=["customer_id", "category", "horizon_days", "forecasted_spend"]
)

forecast_schema = StructType([
    StructField("customer_id",      StringType()),
    StructField("category",         StringType()),
    StructField("horizon_days",     IntegerType()),
    StructField("forecasted_spend", DoubleType()),
])

spark.createDataFrame(results_pd, schema=forecast_schema) \
     .write.mode("overwrite") \
     .csv("hdfs://namenode:8020/user/fintech/forecasts/", header=True)

print("Forecasting complete.")

# ── Step 5: evaluation (train ≤2023, test=2024) ───────────────────────────────
print("\nEvaluating forecast accuracy (train=≤2023, test=2024)...")

eval_results = []

# Prophet eval: category-level monthly MAPE
for category, group in cat_daily_pd.groupby("category"):
    if category not in PROPHET_CATS:
        continue

    pdf = group[["transaction_date", "daily_spend"]].rename(
        columns={"transaction_date": "ds", "daily_spend": "y"}
    )
    pdf["ds"] = pd.to_datetime(pdf["ds"])
    train  = pdf[pdf["ds"].dt.year < 2024].sort_values("ds")
    actual = pdf[pdf["ds"].dt.year == 2024].copy()

    if len(train) < 10 or len(actual) == 0:
        continue

    cap_99        = train['y'].quantile(0.99)
    train['y']    = train['y'].clip(upper=cap_99)

    try:
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, changepoint_prior_scale=0.1)
        m.fit(train)

        future    = m.make_future_dataframe(periods=366)
        forecast  = m.predict(future)
        pred_2024 = forecast[forecast["ds"].dt.year == 2024][["ds", "yhat"]].copy()
        pred_2024["yhat"] = pred_2024["yhat"].clip(lower=0)

        actual["month"]    = actual["ds"].dt.to_period("M")
        pred_2024["month"] = pred_2024["ds"].dt.to_period("M")

        actual_m = actual.groupby("month")["y"].sum().reset_index()
        pred_m   = pred_2024.groupby("month")["yhat"].sum().reset_index()
        merged   = actual_m.merge(pred_m, on="month")

        mape = float(((merged["y"] - merged["yhat"]).abs() /
                      merged["y"].clip(lower=1)).mean() * 100)

        eval_results.append((category, "prophet", round(mape, 2), len(merged)))
        print(f"  [prophet]  {category:<25}  MAPE={mape:>6.1f}%")

    except Exception as e:
        print(f"  [prophet]  {category:<25}  eval failed: {e}")

# Baseline eval: category-aggregate monthly MAPE (apples-to-apples with Prophet eval)
# Per-user MAPE would always look worse due to individual noise — not a fair comparison.
for cat in sorted(BASELINE_CATS):
    sub = user_monthly_pd[user_monthly_pd["category"] == cat].copy()
    # aggregate to category level
    cat_monthly = sub.groupby("month")["monthly_spend"].sum().reset_index()
    cat_monthly.columns = ["month", "spend"]

    train_monthly  = cat_monthly[cat_monthly["month"] < pd.Period("2024-01", "M")].copy()
    actual_monthly = cat_monthly[cat_monthly["month"].apply(lambda m: m.year) == 2024].copy()

    if train_monthly.empty or actual_monthly.empty:
        continue

    max_m = train_monthly["month"].max()
    train_monthly["months_ago"] = train_monthly["month"].apply(lambda m: (max_m - m).n)
    train_monthly["weight"]     = np.exp(-0.15 * train_monthly["months_ago"])
    pred_monthly = float(np.average(train_monthly["spend"],
                                    weights=train_monthly["weight"]))

    mape = float(((actual_monthly["spend"] - pred_monthly).abs() /
                  actual_monthly["spend"].clip(lower=1)).mean() * 100)

    eval_results.append((cat, "baseline", round(mape, 2), len(actual_monthly)))
    print(f"  [baseline] {cat:<25}  MAPE={mape:>6.1f}%")

eval_pd = pd.DataFrame(eval_results,
                        columns=["category", "model", "mape_pct", "months_tested"])
print(f"\n  Prophet  mean MAPE: {eval_pd[eval_pd.model=='prophet']['mape_pct'].mean():.1f}%")
print(f"  Baseline mean MAPE: {eval_pd[eval_pd.model=='baseline']['mape_pct'].mean():.1f}%")
print(f"  Overall  mean MAPE: {eval_pd['mape_pct'].mean():.1f}%")

eval_schema = StructType([
    StructField("category",      StringType()),
    StructField("model",         StringType()),
    StructField("mape_pct",      DoubleType()),
    StructField("months_tested", IntegerType()),
])
spark.createDataFrame(eval_pd, schema=eval_schema) \
     .write.mode("overwrite") \
     .csv("hdfs://namenode:8020/user/fintech/forecast_eval/", header=True)

spark.stop()
