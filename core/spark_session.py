"""Shared local-mode SparkSession. No HDFS, no YARN, no cluster - just a JVM
library call inside whatever process imports this (dash_app, FastAPI, scripts, tests).
"""

import os

from pyspark.sql import SparkSession

_spark: SparkSession | None = None


def get_spark(app_name: str = "fintech-core") -> SparkSession:
    global _spark
    if _spark is not None:
        return _spark

    master = os.environ.get("SPARK_MASTER", "local[*]")
    driver_memory = os.environ.get("SPARK_DRIVER_MEMORY", "1g")

    builder = (
        SparkSession.builder.master(master)
        .appName(app_name)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.shuffle.partitions", os.environ.get("SPARK_SHUFFLE_PARTITIONS", "8"))
        .config("spark.ui.enabled", os.environ.get("SPARK_UI_ENABLED", "false"))
    )
    _spark = builder.getOrCreate()
    _spark.sparkContext.setLogLevel(os.environ.get("SPARK_LOG_LEVEL", "WARN"))
    return _spark


def stop_spark() -> None:
    global _spark
    if _spark is not None:
        _spark.stop()
        _spark = None
