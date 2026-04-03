from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder \
    .appName("FintechFeatureEngineering") \
    .getOrCreate()

df = spark.read.csv(
    "hdfs://namenode:8020/user/fintech/transactions/spending_patterns_detailed.csv",
    header=True,
    inferSchema=True
)

df = df.withColumnRenamed("Customer ID", "customer_id") \
       .withColumnRenamed("Category", "category") \
       .withColumnRenamed("Item", "item") \
       .withColumnRenamed("Quantity", "quantity") \
       .withColumnRenamed("Price Per Unit", "price_per_unit") \
       .withColumnRenamed("Total Spent", "total_spent") \
       .withColumnRenamed("Payment Method", "payment_method") \
       .withColumnRenamed("Location", "location") \
       .withColumnRenamed("Transaction Date", "transaction_date")

df = df.withColumn("transaction_date", F.to_timestamp("transaction_date"))

window_7d  = Window.partitionBy("customer_id", "category").orderBy(F.col("transaction_date").cast("long")).rangeBetween(-7 * 86400, 0)
window_10d = Window.partitionBy("customer_id", "category").orderBy(F.col("transaction_date").cast("long")).rangeBetween(-10 * 86400, 0)
window_30d = Window.partitionBy("customer_id", "category").orderBy(F.col("transaction_date").cast("long")).rangeBetween(-30 * 86400, 0)

df_features = df.withColumn("rolling_7d_spend", F.sum("total_spent").over(window_7d)) \
                .withColumn("rolling_10d_spend", F.sum("total_spent").over(window_10d)) \
                .withColumn("rolling_30d_spend", F.sum("total_spent").over(window_30d))

baseline = df_features.groupBy("customer_id", "category") \
                      .agg(
                          F.round(F.sum("total_spent"), 2).alias("total_spend"),
                          F.round(F.avg("total_spent"), 2).alias("avg_per_transaction"),
                          F.count("*").alias("num_transactions"),
                          F.round(F.max("rolling_30d_spend"), 2).alias("max_30d_spend")
                      )

baseline.write.mode("overwrite").csv(
    "hdfs://namenode:8020/user/fintech/baseline/",
    header=True
)

print("Feature engineering complete. Baseline saved to HDFS.")
spark.stop()
