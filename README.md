# Spending Forecast and Recommendation Platform

A distributed data platform built on a local Hadoop cluster that ingests transaction data, engineers behavioral features with Spark, forecasts future spend using a recency-weighted baseline model (with SARIMAX comparison), and recommends personalized budget caps via collaborative filtering.

---

## Live Demo

**[Launch on Streamlit Cloud →](https://spending-forecast-and-recommendation-platform.streamlit.app)**

The demo loads pre-computed outputs (forecasts + budget caps for all 200 users) directly from static JSON files in the repo - no Hadoop cluster required to view results. The full distributed pipeline runs locally via Docker Compose.

---

## Tech Stack

| Layer            | Technology                                                                      |
| ---------------- | ------------------------------------------------------------------------------- |
| Storage          | HDFS, Apache Hive                                                               |
| Batch processing | Apache Spark, Hadoop MapReduce                                                  |
| Forecasting      | Recency-weighted baseline (active), SARIMAX (comparison), Prophet path retained |
| Recommendation   | User-based Collaborative Filtering                                              |
| Serving          | FastAPI, Streamlit                                                              |
| Infrastructure   | Docker Compose                                                                  |

---

## Architecture

```
Raw CSV (23K transactions, 200 users, 13 categories, 2020–2024)
    │
    ▼
HDFS /user/fintech/transactions/
    │
    ├──▶ Hive external tables (SQL access)
    │
    ├──▶ MapReduce (06)
    │       Hadoop Streaming - spend aggregation by category
    │       Output: total, count, avg, spend tier per category
    │       → /user/fintech/mr_category_agg/
    │
    ├──▶ Spark Feature Engineering (03)
    │       Rolling 7/10/30-day spend windows per user-category
    │       Q4 YoY spend velocity (Q4 2024 / Q4 2023)
    │       → /user/fintech/baseline/
    │       → /user/fintech/velocity/
    │
    ├──▶ Forecasting (04)
    │       Baseline (active): per-user recency-weighted monthly average
    │         × velocity adjustment
    │       SARIMAX (comparison): category-level daily model for side-by-side
    │         validation vs baseline
    │       Prophet: code path retained but disabled by default
    │       → /user/fintech/forecasts/          (7 / 15 / 30-day horizons)
    │       → /user/fintech/forecast_eval/      (MAPE per category + model)
    │
    └──▶ Collaborative Filtering (05)
            Feature matrix: forecast + avg_txn + n_txn + max_30d + velocity
            User-user cosine similarity (top-10 neighbors)
            Budget cap = (60% own + 40% CF) × velocity × 1.15 buffer
            → /user/fintech/recommendations/

FastAPI   ──▶  reads HDFS outputs via WebHDFS, serves in-memory
Streamlit ──▶  calls FastAPI, renders spending overview / forecasts / budget caps
```

---

## Dataset

- **23,000 transactions**, 200 customers, 13 spending categories
- Date range: January 2020 – December 2024 (2 years real + 3 years synthetic)
- Synthetic history generated with realistic seasonality (holiday spikes, summer travel bumps, January gym surges) and COVID-era suppression in 2020
- Categories: Fitness, Food, Friend Activities, Gifts, Groceries, Hobbies, Housing & Utilities, Medical/Dental, Personal Hygiene, Shopping, Subscriptions, Transportation, Travel

---

## Forecasting Design

Current runtime behavior in `scripts/04_time_series_forecast.py`:

- **Active serving model**: recency-weighted baseline for all 13 categories
- **Comparison model**: SARIMAX evaluation on baseline-routed categories
- **Prophet status**: retained in code, but disabled by default (`PROPHET_CATS = set()`)

Why baseline is the serving path: on this dataset, it remains more stable for per-user
monthly spend forecasting and avoids synthetic seasonality overfit.

**Recency-weighted baseline** - per-user monthly spend history, exponentially weighted
toward recent months (λ=0.15), with a Q4 YoY velocity adjustment on top:

- `forecast = weighted_monthly_avg × (horizon_days / 30) × velocity`
- Recency decay: last month weight=1.0, 6mo ago=0.41, 12mo ago=0.17
- Velocity = Q4-2024 / Q4-2023 spend ratio, clamped to [0.5, 2.0]

**Model comparison:**

The pipeline now prints side-by-side metrics for Prophet, baseline, and SARIMAX
(MAPE, WAPE, RMSE), with per-category winners on the clean evaluation set. Baseline
remains the default forecast source used for downstream recommendations.

Latest comparison from `scripts/04_time_series_forecast.py`:

| Model    | Clean-set mean APE | Clean-set WAPE | Clean-set RMSE |
| -------- | ------------------ | -------------- | -------------- |
| Baseline | 41.3%              | 41.1%          | $4,512         |
| Prophet  | 86.1%              | 62.2%          | $4,970         |
| SARIMAX  | 43.2%              | 37.2%          | $3,472         |

Interpretation: Prophet is still available for comparison, but it is currently the
least competitive on this dataset. Baseline remains the serving default, and SARIMAX
is the strongest aggregate performer on WAPE/RMSE.

**Per-category winners** (clean evaluation set, lowest APE):

| Model    | Categories Won | Example Winners             |
| -------- | -------------- | --------------------------- |
| SARIMAX  | 5              | Gifts (3.2%), Groceries (5.1%), Food (10.0%) |
| Baseline | 4              | Subscriptions (28.5%), Transportation (35.1%) |
| Prophet  | 3              | Gifts (3.2%), Groceries (5.1%), Personal Hygiene (16.2%) |

**Detailed category-wise APE** (Jan 1–13 validation window):

Top performers (APE < 30%):
- Gifts: Prophet 3.2%, Baseline 15.7%, SARIMAX 28.3%
- Groceries: Prophet 5.1%, Baseline 26.0%, SARIMAX 19.7%
- Food: Baseline 10.0%, SARIMAX 26.0%, Prophet 23.5%
- Personal Hygiene: Baseline 16.2%, SARIMAX 25.0%, Prophet 32.5%
- Subscriptions: Baseline 28.5%, SARIMAX 32.6%, Prophet 22.7%

Moderate performers (30% ≤ APE < 70%):
- Transportation: Baseline 35.1%, SARIMAX 29.8%, Prophet 117.8%
- Shopping: Baseline 42.4%, SARIMAX 31.3%, Prophet 35.2%
- Medical/Dental: Baseline 52.6%, SARIMAX 48.2%, Prophet 230.1%
- Fitness: Baseline 40.5%, SARIMAX 57.9%, Prophet 164.4%

Weaker performers (APE ≥ 70%):
- Travel: Baseline 65.3%, SARIMAX 71.4%, Prophet 120.7%
- Hobbies: Baseline 82.4%, SARIMAX 77.2%, Prophet 179.0%
- Friend Activities: Baseline 80.5%, SARIMAX 70.8%, Prophet 98.7%

*Note: Housing & Utilities excluded from analysis (⚠ 3× inflation in Jan 2025 vs training data)*
---

## Personalized Forecast Distribution


- Baseline forecast uses each user's own monthly series with recency decay
- Velocity adjustment scales results using Q4 YoY trend (`spend_velocity`, clamped)

Note: Prophet and SARIMAX are retained in code for model comparison and experimentation,
but only the baseline forecast is active in downstream recommendations.

---

## Collaborative Filtering

User-based CF with a **65-feature matrix** per user (5 features × 13 categories):

- `forecast_30d` - 30-day forecast (baseline active; SARIMAX comparison available)
- `avg_per_transaction` - historical average transaction size
- `num_transactions` - transaction frequency
- `max_30d_spend` - peak 30-day spend
- `spend_velocity` - Q4 YoY spend ratio (accelerating or decelerating)

Features are min-max normalized before cosine similarity so no single feature dominates. Top-10 nearest neighbors. Budget cap = (60% own + 40% CF) × velocity × 1.15 safety buffer.

---

## Services (local cluster)

| Service              | URL                        |
| -------------------- | -------------------------- |
| Streamlit dashboard  | http://localhost:8501      |
| FastAPI + docs       | http://localhost:8000/docs |
| HDFS NameNode        | http://localhost:9870      |
| Spark Master         | http://localhost:8080      |
| YARN ResourceManager | http://localhost:8088      |

---

## Running the Pipeline

### 1. Start the cluster

```bash
docker compose up -d
```

Starts: Hadoop (namenode, datanode, resourcemanager, nodemanager), Hive, Spark, FastAPI, Streamlit.

### 2. Generate and load data (one-time)

```bash
# Generate 5-year dataset with synthetic history
python3 scripts/00_generate_synthetic_history.py

# Upload to HDFS
docker cp data/spending_patterns_5yr.csv fintech-spending-analyzer-namenode-1:/tmp/
docker exec fintech-spending-analyzer-namenode-1 bash -c "
  hdfs dfs -mkdir -p /user/fintech/transactions
  hdfs dfs -chmod -R 777 /user/fintech
  hdfs dfs -put -f /tmp/spending_patterns_5yr.csv /user/fintech/transactions/
"
```

### 3. Create Hive tables (one-time)

```bash
docker exec -it fintech-spending-analyzer-hive-1 \
  beeline -u jdbc:hive2://localhost:10000 -f /tmp/02_create_hive_tables.sql
```

### 4. Run the pipeline

```bash
# Feature engineering (Spark)
docker exec fintech-spending-analyzer-spark-1 bash -c "
  export SPARK_HOME=/opt/spark
  export PYTHONPATH=\$SPARK_HOME/python:\$SPARK_HOME/python/lib/py4j-0.10.9.7-src.zip
  python3 /tmp/03_feature_engineering.py
"

# Forecasting (baseline + SARIMAX comparison; Prophet disabled by default)
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

### 5. Reload API cache

```bash
curl -X POST http://localhost:8000/reload
```

---

## API Endpoints

| Method | Endpoint                     | Description                                              |
| ------ | ---------------------------- | -------------------------------------------------------- |
| GET    | `/users`                     | List all customer IDs                                    |
| GET    | `/users/{id}/baseline`       | Historical spend by category                             |
| GET    | `/users/{id}/forecasts`      | Forecasts (7/15/30-day)                                  |
| GET    | `/users/{id}/budget-caps`    | CF budget recommendations                                |
| GET    | `/categories`                | All spending categories                                  |
| GET    | `/anomalies`                 | Platform-wide anomaly alerts (optional `?severity=high`) |
| GET    | `/users/{id}/anomalies`      | Per-user anomaly alerts                                  |
| GET    | `/users/{id}/peer-benchmark` | Per-user vs peer-average spend delta + percentiles       |
| POST   | `/reload`                    | Refresh in-memory cache from HDFS                        |

---

## Repository Structure

```
├── data/
│   └── spending_patterns_detailed.csv    # original 2yr real data (gitignored)
├── scripts/
│   ├── 00_generate_synthetic_history.py  # extends dataset to 5yr with seasonality
│   ├── 01_load_to_hdfs.sh
│   ├── 02_create_hive_tables.sql
│   ├── 03_feature_engineering.py         # Spark - rolling windows + velocity
│   ├── 04_time_series_forecast.py        # baseline forecasting + SARIMAX comparison (Prophet path retained)
│   ├── 05_collaborative_filter.py        # user-based CF - budget caps
│   ├── 06_mapreduce_mapper.py
│   ├── 06_mapreduce_reducer.py
│   ├── 06_run_mapreduce.sh
│   ├── 07_financial_health_score.py   # composite 0-100 score across 4 dimensions
│   ├── 08_anomaly_detection.py        # Z-score + Isolation Forest mid-month pace check
│   └── 09_peer_benchmarking.py        # CF neighbor delta + category percentile insights
├── api/
│   ├── main.py                           # FastAPI serving layer
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py                            # Streamlit dashboard
│   ├── data/                             # pre-computed JSON for live demo
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml
└── config                                # Hadoop cluster config
```
