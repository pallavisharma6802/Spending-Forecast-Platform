import sys
import os
sys.path.insert(0, '/tmp/pypackages')

os.environ["STAN_NUM_THREADS"] = "1"
os.environ["CMDSTAN_THREADS"] = "1"

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
import pandas as pd
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

spark = SparkSession.builder \
    .appName("FintechTimeSeriesForecast") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

df = spark.read.csv(
    "hdfs://namenode:8020/user/fintech/transactions/spending_patterns_detailed.csv",
    header=True,
    inferSchema=True
)

df = df.withColumnRenamed("Customer ID", "customer_id") \
       .withColumnRenamed("Category", "category") \
       .withColumnRenamed("Total Spent", "total_spent") \
       .withColumnRenamed("Transaction Date", "transaction_date") \
       .select("customer_id", "category", "total_spent", "transaction_date")

df = df.withColumn("transaction_date", F.to_date("transaction_date"))

# ── Step 1: daily spend per CATEGORY across all users ────────────────────────
# ~13 categories × ~730 days → rich enough for Prophet to learn real trends
category_daily = df.groupBy("category", "transaction_date") \
                   .agg(F.round(F.sum("total_spent"), 2).alias("daily_spend"))

# ── Step 2: each user's historical share per category ────────────────────────
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

# collect both to driver — 13 category time series + 2600 user shares
cat_daily_pd   = category_daily.toPandas()
user_shares_pd = user_shares.toPandas()


# ── Step 3: run Prophet once per category ────────────────────────────────────
def _scaled_fallback(prophet_df, horizon):
    total    = float(prophet_df['y'].sum())
    span     = max((prophet_df['ds'].max() - prophet_df['ds'].min()).days + 1, 1)
    return round(total / span * horizon, 2)


def forecast_category(category, group):
    pdf = group[['transaction_date', 'daily_spend']].rename(
        columns={'transaction_date': 'ds', 'daily_spend': 'y'}
    )
    pdf['ds'] = pd.to_datetime(pdf['ds'])
    pdf = pdf.sort_values('ds')

    try:
        m = Prophet(
            yearly_seasonality=True,   # 2 yrs of data — real seasonal signal
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


print("Running Prophet per category...")
category_forecasts = {}
for category, group in cat_daily_pd.groupby('category'):
    n = len(group)
    category_forecasts[category] = forecast_category(category, group)
    fc = category_forecasts[category]
    print(f"  {category:<25}  n={n:>3}  7d=${fc[7]:>8.2f}  15d=${fc[15]:>9.2f}  30d=${fc[30]:>9.2f}")


# ── Step 4: distribute category forecast to each user by their spend share ───
all_results = []
for _, row in user_shares_pd.iterrows():
    cid, cat, share = row['customer_id'], row['category'], float(row['share'])
    if cat not in category_forecasts:
        continue
    for h, cat_fc in category_forecasts[cat].items():
        all_results.append((cid, cat, h, round(cat_fc * share, 2)))

results_pd = pd.DataFrame(
    all_results,
    columns=["customer_id", "category", "horizon_days", "forecasted_spend"]
)

forecast_schema = StructType([
    StructField("customer_id",     StringType()),
    StructField("category",        StringType()),
    StructField("horizon_days",    IntegerType()),
    StructField("forecasted_spend", DoubleType()),
])

spark.createDataFrame(results_pd, schema=forecast_schema) \
     .write.mode("overwrite") \
     .csv("hdfs://namenode:8020/user/fintech/forecasts/", header=True)

print("Forecasting complete.")

# ── Forecast evaluation: train 2023, test 2024, MAPE per category ────────────
print("\nEvaluating forecast accuracy (train=2023, test=2024)...")

eval_results = []

for category, group in cat_daily_pd.groupby("category"):
    pdf = group[["transaction_date", "daily_spend"]].rename(
        columns={"transaction_date": "ds", "daily_spend": "y"}
    )
    pdf["ds"] = pd.to_datetime(pdf["ds"])

    train  = pdf[pdf["ds"].dt.year == 2023].sort_values("ds")
    actual = pdf[pdf["ds"].dt.year == 2024].copy()

    if len(train) < 10 or len(actual) == 0:
        continue

    try:
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.1
        )
        m.fit(train)

        future   = m.make_future_dataframe(periods=366)
        forecast = m.predict(future)
        pred_2024 = forecast[forecast["ds"].dt.year == 2024][["ds", "yhat"]].copy()
        pred_2024["yhat"] = pred_2024["yhat"].clip(lower=0)

        # monthly aggregation — stabler than daily MAPE
        actual["month"]   = actual["ds"].dt.to_period("M")
        pred_2024["month"] = pred_2024["ds"].dt.to_period("M")

        actual_monthly = actual.groupby("month")["y"].sum().reset_index()
        pred_monthly   = pred_2024.groupby("month")["yhat"].sum().reset_index()
        merged = actual_monthly.merge(pred_monthly, on="month")

        # MAPE — clip actual to 1 to avoid divide-by-zero on sparse months
        mape = float((
            (merged["y"] - merged["yhat"]).abs() / merged["y"].clip(lower=1)
        ).mean() * 100)

        eval_results.append((category, round(mape, 2), len(merged)))
        print(f"  {category:<25}  MAPE={mape:>6.1f}%  months_tested={len(merged)}")

    except Exception as e:
        print(f"  {category:<25}  eval failed: {e}")

eval_pd = pd.DataFrame(eval_results, columns=["category", "mape_pct", "months_tested"])
overall_mape = round(float(eval_pd["mape_pct"].mean()), 2)
print(f"\n  Overall mean MAPE: {overall_mape}%")

eval_schema = StructType([
    StructField("category",      StringType()),
    StructField("mape_pct",      DoubleType()),
    StructField("months_tested", IntegerType()),
])
spark.createDataFrame(eval_pd, schema=eval_schema) \
     .write.mode("overwrite") \
     .csv("hdfs://namenode:8020/user/fintech/forecast_eval/", header=True)

spark.stop()
