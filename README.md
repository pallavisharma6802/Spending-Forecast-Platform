# Spending Forecast and Recommendation Platform

A distributed data platform built on a local Hadoop cluster that ingests transaction data, analyzes spending patterns, forecasts future spend using Facebook Prophet, and recommends personalized budget caps per user via collaborative filtering.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Storage | HDFS, Apache Hive |
| Batch processing | Apache Spark, Hadoop MapReduce |
| Forecasting | Facebook Prophet |
| Recommendation | User-based Collaborative Filtering |
| Orchestration | Apache Airflow |
| Serving | FastAPI, Streamlit |
| Infrastructure | Docker Compose |

---

## Architecture

```
Raw CSV (10K transactions, 200 users, 13 categories)
    │
    ▼
HDFS /user/fintech/transactions/
    │
    ├──▶ Hive external tables (SQL access)
    │
    ├──▶ MapReduce (06)
    │       Hadoop Streaming — spend aggregation by category
    │       Output: total, count, avg, spend tier per category
    │       → /user/fintech/mr_category_agg/
    │
    ├──▶ Spark Feature Engineering (03)
    │       Rolling 7/10/30-day spend windows per user-category
    │       Q4 YoY spend velocity (Q4 2024 / Q4 2023)
    │       → /user/fintech/baseline/
    │       → /user/fintech/velocity/
    │
    ├──▶ Prophet Forecasting (04)
    │       Trains on daily category spend (470–500 points per category)
    │       Distributes forecast to each user by their historical spend share
    │       Evaluates accuracy: train 2023, test 2024, MAPE per category
    │       → /user/fintech/forecasts/          (7 / 15 / 30-day horizons)
    │       → /user/fintech/forecast_eval/      (MAPE per category)
    │
    └──▶ Collaborative Filtering (05)
            Feature matrix: forecast + avg_txn + n_txn + max_30d + velocity
            User-user cosine similarity (top-10 neighbors)
            Budget cap = (60% own + 40% CF) × velocity × 1.15 buffer
            → /user/fintech/recommendations/

FastAPI  ──▶  reads all HDFS outputs via WebHDFS, serves in-memory
Streamlit ──▶  calls FastAPI, renders spending overview / forecasts / budget caps
Airflow  ──▶  daily DAG orchestrating the full pipeline end-to-end
```

---

## Dataset

- **10,000 transactions**, 200 customers, 13 spending categories
- Date range: January 2023 – December 2024
- Categories: Fitness, Food, Friend Activities, Gifts, Groceries, Hobbies, Housing & Utilities, Medical/Dental, Personal Hygiene, Shopping, Subscriptions, Transportation, Travel

---

## Forecasting Design

Prophet runs at the **category level** (all users combined), not per user. Each category has 470–500 daily data points over two years — enough for Prophet to learn real seasonal patterns. Each user's forecast is then their historical spend share of that category's total forecast.

This is more statistically sound than per-user Prophet, which would have only 3–11 data points per user-category — too sparse for reliable trend detection.

**Forecast accuracy (train 2023 → test 2024):**

| Category | MAPE |
|---|---|
| Fitness | 41.5% |
| Travel | 55.0% |
| Food | 76.0% |
| Transportation | 89.2% |
| Groceries | 94.9% |
| Shopping | 100.0% |

Higher MAPE categories (Gifts, Friend Activities, Personal Hygiene) are dominated by irregular large transactions that no model can predict from prior year data.

---

## Personalized Forecast Distribution

**Blocker**: Two users with the same historical spend share in a category received identical forecasts — a daily gym-goer and someone with one large equipment purchase looked the same to the model.

**Fix**: A behavior multiplier is applied when distributing the category-level forecast to each user: `user_forecast = category_forecast × share × multiplier`. The multiplier is the geometric mean of three signals — recency (exponential decay since last transaction), frequency ratio (user's transaction count vs. category average, capped 0.5–2×), and Q4 YoY velocity (capped 0.5–2×) — clamped to [0.3, 3.0]. Active, accelerating users receive a higher forecast; dormant or decelerating users receive a lower one.

---

## Collaborative Filtering

User-based CF with a **65-feature matrix** per user (5 features × 13 categories):
- `forecast_30d` — Prophet 30-day forecast
- `avg_per_transaction` — historical average transaction size
- `num_transactions` — transaction frequency
- `max_30d_spend` — peak 30-day spend
- `spend_velocity` — Q4 YoY spend ratio (accelerating or decelerating)

Features are min-max normalized before cosine similarity so no single feature dominates. Top-10 nearest neighbors are used. Budget cap = blended forecast adjusted by velocity × 15% safety buffer.

---

## Services

| Service | URL |
|---|---|
| Streamlit dashboard | http://localhost:8501 |
| FastAPI + docs | http://localhost:8000/docs |
| Airflow UI | http://localhost:8085 (admin / admin) |
| HDFS NameNode | http://localhost:9870 |
| Spark Master | http://localhost:8080 |
| YARN ResourceManager | http://localhost:8088 |

---

## Running the Pipeline

### 1. Start the cluster

```bash
docker compose up -d
```

First run builds the Airflow image (~3 min). All services: Hadoop, Hive, Spark, Airflow, FastAPI, Streamlit, Postgres.

### 2. Load data into HDFS (one-time)

```bash
docker cp data/spending_patterns_detailed.csv fintech-spending-analyzer-namenode-1:/tmp/

docker exec fintech-spending-analyzer-namenode-1 bash -c "
  hdfs dfs -mkdir -p /user/fintech/transactions
  hdfs dfs -chmod -R 777 /user/fintech
  hdfs dfs -put -f /tmp/spending_patterns_detailed.csv /user/fintech/transactions/
"
```

### 3. Create Hive tables (one-time)

```bash
docker exec -it fintech-spending-analyzer-hive-1 \
  beeline -u jdbc:hive2://localhost:10000 -f /tmp/02_create_hive_tables.sql
```

### 4. Install Python dependencies in Spark container (one-time)

```bash
docker exec -u root fintech-spending-analyzer-spark-1 \
  pip install prophet pandas numpy pyarrow
```

### 5. Run the pipeline manually

```bash
# Feature engineering + velocity
docker exec fintech-spending-analyzer-spark-1 bash -c "
  export SPARK_HOME=/opt/spark
  export PYTHONPATH=\$SPARK_HOME/python:\$SPARK_HOME/python/lib/py4j-0.10.9.7-src.zip
  python3 /tmp/03_feature_engineering.py
"

# Prophet forecasting + MAPE evaluation
# (runs in the API container — native ARM64, no Stan crashes)
docker exec fintech-spending-analyzer-api-1 bash -c "
  export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-arm64
  python3 /tmp/04_time_series_forecast.py
"

# Collaborative filtering
docker exec fintech-spending-analyzer-api-1 bash -c "
  export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-arm64
  python3 /tmp/05_collaborative_filter.py
"

# MapReduce category aggregation
bash scripts/06_run_mapreduce.sh
```

### 6. Reload API cache

```bash
curl -X POST http://localhost:8000/reload
```

### 7. Automated daily pipeline via Airflow

Open http://localhost:8085, log in as admin/admin, enable the `fintech_spending_pipeline` DAG. It runs daily and executes the full pipeline in dependency order.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/users` | List all customer IDs |
| GET | `/users/{id}/baseline` | Historical spend by category |
| GET | `/users/{id}/forecasts` | Prophet forecasts (7/15/30-day) |
| GET | `/users/{id}/budget-caps` | CF budget recommendations |
| GET | `/categories` | All spending categories |
| POST | `/reload` | Refresh in-memory cache from HDFS |

---

## Repository Structure

```
├── data/
│   └── spending_patterns_detailed.csv
├── scripts/
│   ├── 01_load_to_hdfs.sh
│   ├── 02_create_hive_tables.sql
│   ├── 03_feature_engineering.py     # Spark — baseline + velocity features
│   ├── 04_time_series_forecast.py    # Prophet — category forecasts + MAPE eval
│   ├── 05_collaborative_filter.py    # User-based CF — budget caps
│   ├── 06_mapreduce_mapper.py        # Hadoop Streaming mapper
│   ├── 06_mapreduce_reducer.py       # Hadoop Streaming reducer
│   └── 06_run_mapreduce.sh           # Runs the MapReduce job
├── api/
│   ├── main.py                       # FastAPI serving layer
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py                        # Streamlit dashboard
│   ├── requirements.txt
│   └── Dockerfile
├── dags/
│   └── fintech_pipeline_dag.py       # Airflow DAG
├── Dockerfile.airflow
├── docker-compose.yml
└── config                            # Hadoop cluster config
```
