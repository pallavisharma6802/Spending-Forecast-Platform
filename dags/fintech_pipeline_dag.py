from datetime import datetime, timedelta
import requests as _requests

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

default_args = {
    "owner": "fintech",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

SPARK_CONN_ID = "spark_default"
SCRIPTS = "/tmp/scripts"


def _check_hdfs_data():
    """Verify raw transaction data is present in HDFS via WebHDFS."""
    url = "http://namenode:9870/webhdfs/v1/user/fintech/transactions/?op=LISTSTATUS"
    r = _requests.get(url, timeout=10)
    r.raise_for_status()
    files = r.json().get("FileStatuses", {}).get("FileStatus", [])
    assert any(f["pathSuffix"].endswith(".csv") for f in files), \
        "No CSV found in /user/fintech/transactions/ — run 01_load_to_hdfs.sh first"


def _reload_api_cache():
    """Signal the serving API to reload HDFS data into memory."""
    try:
        _requests.post("http://api:8000/reload", timeout=60)
    except Exception:
        pass  # non-fatal; API may not be running in all environments


with DAG(
    dag_id="fintech_spending_pipeline",
    default_args=default_args,
    description="Daily: verify data → features → Prophet forecast → CF budget caps → reload API",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["fintech", "etl", "ml"],
) as dag:

    check_data = PythonOperator(
        task_id="check_hdfs_data",
        python_callable=_check_hdfs_data,
    )

    mapreduce_agg = BashOperator(
        task_id="mapreduce_category_aggregation",
        bash_command=f"bash {SCRIPTS}/06_run_mapreduce.sh",
    )

    feature_engineering = SparkSubmitOperator(
        task_id="feature_engineering",
        application=f"{SCRIPTS}/03_feature_engineering.py",
        conn_id=SPARK_CONN_ID,
        name="FintechFeatureEngineering",
        verbose=False,
    )

    time_series_forecast = SparkSubmitOperator(
        task_id="time_series_forecast",
        application=f"{SCRIPTS}/04_time_series_forecast.py",
        conn_id=SPARK_CONN_ID,
        name="FintechTimeSeriesForecast",
        verbose=False,
    )

    collaborative_filter = SparkSubmitOperator(
        task_id="collaborative_filter",
        application=f"{SCRIPTS}/05_collaborative_filter.py",
        conn_id=SPARK_CONN_ID,
        name="FintechCollaborativeFilter",
        verbose=False,
    )

    reload_api = PythonOperator(
        task_id="reload_api_cache",
        python_callable=_reload_api_cache,
    )

    # pipeline order: verify → (MapReduce + features in parallel) → forecast → CF → serve
    check_data >> [mapreduce_agg, feature_engineering]
    feature_engineering >> time_series_forecast >> collaborative_filter >> reload_api
