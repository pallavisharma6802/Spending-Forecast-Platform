import sys
sys.path.insert(0, '/tmp/pypackages')

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
import pandas as pd
import numpy as np

spark = SparkSession.builder \
    .appName("FintechCollaborativeFilter") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# ── Load all feature sources ──────────────────────────────────────────────────
forecasts_pdf = spark.read.csv(
    "hdfs://namenode:8020/user/fintech/forecasts/",
    header=True, inferSchema=True
).filter(F.col("horizon_days") == 30) \
 .select("customer_id", "category", "forecasted_spend").toPandas()

baseline_pdf = spark.read.csv(
    "hdfs://namenode:8020/user/fintech/baseline/",
    header=True, inferSchema=True
).select("customer_id", "category",
         "avg_per_transaction", "num_transactions", "max_30d_spend").toPandas()

velocity_pdf = spark.read.csv(
    "hdfs://namenode:8020/user/fintech/velocity/",
    header=True, inferSchema=True
).select("customer_id", "category", "spend_velocity").toPandas()

# coerce numerics
for col in ["forecasted_spend"]:
    forecasts_pdf[col] = pd.to_numeric(forecasts_pdf[col], errors="coerce").fillna(0.0)
for col in ["avg_per_transaction", "num_transactions", "max_30d_spend"]:
    baseline_pdf[col] = pd.to_numeric(baseline_pdf[col], errors="coerce").fillna(0.0)
velocity_pdf["spend_velocity"] = pd.to_numeric(
    velocity_pdf["spend_velocity"], errors="coerce"
)

# ── Build per-user feature matrix ─────────────────────────────────────────────
# Features per user-category: forecast_30d, avg_per_transaction,
#                              num_transactions, max_30d_spend, spend_velocity
# Final matrix shape: users × (categories × 5 features)

def pivot(df, value_col, suffix):
    p = df.pivot_table(index="customer_id", columns="category",
                       values=value_col, fill_value=0.0)
    p.columns = [f"{c}__{suffix}" for c in p.columns]
    return p

fc_pivot   = pivot(forecasts_pdf, "forecasted_spend",     "forecast")
avg_pivot  = pivot(baseline_pdf,  "avg_per_transaction",  "avg_txn")
cnt_pivot  = pivot(baseline_pdf,  "num_transactions",     "n_txn")
max_pivot  = pivot(baseline_pdf,  "max_30d_spend",        "max30d")

# velocity: fill missing with 1.0 (neutral — no YoY change observed)
velocity_pdf["spend_velocity"] = velocity_pdf["spend_velocity"].fillna(1.0)
vel_pivot  = pivot(velocity_pdf,  "spend_velocity",       "velocity")

# align all pivots to the same user index
all_users = fc_pivot.index
for p in [avg_pivot, cnt_pivot, max_pivot, vel_pivot]:
    p = p.reindex(all_users, fill_value=0.0)

feature_matrix = pd.concat(
    [fc_pivot,
     avg_pivot.reindex(all_users, fill_value=0.0),
     cnt_pivot.reindex(all_users, fill_value=0.0),
     max_pivot.reindex(all_users, fill_value=0.0),
     vel_pivot.reindex(all_users, fill_value=1.0)],
    axis=1
).fillna(0.0)

# min-max normalize each feature column so no single feature dominates
col_min = feature_matrix.min()
col_max = feature_matrix.max()
col_range = (col_max - col_min).replace(0, 1)
feature_norm = (feature_matrix - col_min) / col_range

# ── User-user cosine similarity on rich feature space ────────────────────────
mat = feature_norm.values
norms = np.linalg.norm(mat, axis=1, keepdims=True)
norms[norms == 0] = 1.0
mat_norm = mat / norms

sim_matrix = mat_norm @ mat_norm.T
np.fill_diagonal(sim_matrix, 0.0)

# ── Generate budget caps ──────────────────────────────────────────────────────
# Forecast column names in fc_pivot correspond to categories
categories = [c.replace("__forecast", "") for c in fc_pivot.columns]
users      = list(all_users)

K = 10
results = []

for i, user in enumerate(users):
    sims        = sim_matrix[i]
    top_k_idx   = np.argsort(sims)[::-1][:K]
    weights     = sims[top_k_idx]
    weight_sum  = weights.sum()

    for j, category in enumerate(categories):
        own_forecast = float(fc_pivot.iloc[i, j])

        if weight_sum > 0:
            neighbor_fc = fc_pivot.values[top_k_idx, j]
            cf_pred     = float(np.dot(weights, neighbor_fc) / weight_sum)
        else:
            cf_pred = own_forecast

        # velocity adjustment: if user is accelerating in this category, raise cap
        vel_col = f"{category}__velocity"
        velocity = float(
            vel_pivot.loc[user, vel_col]
            if vel_col in vel_pivot.columns and user in vel_pivot.index
            else 1.0
        )
        velocity = max(0.5, min(velocity, 3.0))   # clamp to [0.5, 3.0]

        blended    = 0.6 * own_forecast + 0.4 * cf_pred
        budget_cap = round(blended * velocity * 1.15, 2)

        results.append((user, category, round(cf_pred, 2), round(own_forecast, 2),
                        round(velocity, 4), budget_cap))

caps_pdf = pd.DataFrame(results, columns=[
    "customer_id", "category", "cf_predicted_spend",
    "own_forecast_30d", "spend_velocity", "recommended_budget_cap"
])

schema = StructType([
    StructField("customer_id",           StringType()),
    StructField("category",              StringType()),
    StructField("cf_predicted_spend",    DoubleType()),
    StructField("own_forecast_30d",      DoubleType()),
    StructField("spend_velocity",        DoubleType()),
    StructField("recommended_budget_cap", DoubleType()),
])

spark.createDataFrame(caps_pdf, schema=schema) \
     .write.mode("overwrite") \
     .csv("hdfs://namenode:8020/user/fintech/recommendations/", header=True)

print("Collaborative filtering complete. Budget caps saved to HDFS.")
spark.stop()
