# Spending Forecast and Recommendation Platform

A distributed data platform built on a local Hadoop cluster that ingests transaction data, engineers behavioral features with Spark, forecasts future spend using a hybrid Prophet + recency-weighted baseline model, and recommends personalized budget caps via collaborative filtering.

---

## Live Demo

**[Launch on Streamlit Cloud →](https://spending-forecast-and-recommendation-platform.streamlit.app)**

The demo loads pre-computed outputs (forecasts + budget caps for all 200 users) directly from static JSON files in the repo — no Hadoop cluster required to view results. The full distributed pipeline runs locally via Docker Compose.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Storage | HDFS, Apache Hive |
| Batch processing | Apache Spark, Hadoop MapReduce |
| Forecasting | Facebook Prophet + recency-weighted baseline (hybrid) |
| Recommendation | User-based Collaborative Filtering |
| Serving | FastAPI, Streamlit |
| Infrastructure | Docker Compose |

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
    ├──▶ Hybrid Forecasting (04)
    │       Prophet (seasonal categories): category-level daily spend,
    │         5yr history, winsorized at 99th percentile
    │         → distributed to users via spend share × behavior multiplier
    │       Baseline (irregular categories): per-user recency-weighted
    │         monthly average × velocity adjustment
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

## Hybrid Forecasting Design

Categories are routed to different models based on their statistical behavior:

**Prophet** — categories with real repeating seasonal patterns:
Fitness, Food, Friend Activities, Hobbies, Medical/Dental, Personal Hygiene, Shopping, Subscriptions, Transportation, Travel

- Runs at the **category level** (all users combined) on 5 years of daily data
- Winsorized at 99th percentile per category to reduce outlier influence
- Each user's forecast = `category_forecast × spend_share × behavior_multiplier`

**Recency-weighted baseline** — irregular categories where Prophet overfits noise:
Gifts, Groceries, Housing & Utilities

- Per-user monthly spend history, exponentially weighted toward recent months (λ=0.15)
- Velocity adjustment applied on top for trend signal
- These categories have no reliable seasonal pattern — the right tool is a stable estimate of "what does this user typically spend per month?"

**Forecast accuracy (train ≤2023, test=2024):**

| Category | Model | MAPE |
|---|---|---|
| Groceries | Baseline | 20.4% |
| Housing & Utilities | Baseline | 21.6% |
| Subscriptions | Prophet | 30.6% |
| Gifts | Baseline | 37.1% |
| Travel | Prophet | 56.1% |
| Shopping | Prophet | 68.9% |
| Medical/Dental | Prophet | 72.1% |
| Hobbies | Prophet | 76.5% |
| Friend Activities | Prophet | 85.0% |
| Fitness | Prophet | 94.1% |
| Food | Prophet | 95.3% |
| Transportation | Prophet | 121.0% |
| **Overall mean** | | **95.5%** |

---

## Personalized Forecast Distribution

**Blocker**: Two users with the same historical spend share got identical forecasts — a daily gym-goer and someone with one large equipment purchase looked the same.

**Fix**: A behavior multiplier is applied to Prophet-category forecasts: `user_forecast = category_forecast × share × multiplier`. The multiplier is the geometric mean of recency (exponential decay since last transaction), frequency ratio (user's transaction count vs. category average), and Q4 YoY velocity — clamped to [0.3, 3.0].

**Result**: Fitness 30-day forecasts across 195 users range $0.65–$469 (std dev = 88% of mean). A decelerating user (velocity=0.5) receives a cap ~38% lower than their raw forecast.

---

## Collaborative Filtering

User-based CF with a **65-feature matrix** per user (5 features × 13 categories):
- `forecast_30d` — 30-day forecast (Prophet or baseline)
- `avg_per_transaction` — historical average transaction size
- `num_transactions` — transaction frequency
- `max_30d_spend` — peak 30-day spend
- `spend_velocity` — Q4 YoY spend ratio (accelerating or decelerating)

Features are min-max normalized before cosine similarity so no single feature dominates. Top-10 nearest neighbors. Budget cap = (60% own + 40% CF) × velocity × 1.15 safety buffer.

---

## Services (local cluster)

| Service | URL |
|---|---|
| Streamlit dashboard | http://localhost:8501 |
| FastAPI + docs | http://localhost:8000/docs |
| HDFS NameNode | http://localhost:9870 |
| Spark Master | http://localhost:8080 |
| YARN ResourceManager | http://localhost:8088 |

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

# Hybrid forecasting — runs in API container (native ARM64 for Prophet)
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

| Method | Endpoint | Description |
|---|---|---|
| GET | `/users` | List all customer IDs |
| GET | `/users/{id}/baseline` | Historical spend by category |
| GET | `/users/{id}/forecasts` | Forecasts (7/15/30-day) |
| GET | `/users/{id}/budget-caps` | CF budget recommendations |
| GET | `/categories` | All spending categories |
| GET | `/anomalies` | Platform-wide anomaly alerts (optional `?severity=high`) |
| GET | `/users/{id}/anomalies` | Per-user anomaly alerts |
| GET | `/users/{id}/peer-benchmark` | Per-user vs peer-average spend delta + percentiles |
| POST | `/reload` | Refresh in-memory cache from HDFS |

---

## Repository Structure

```
├── data/
│   └── spending_patterns_detailed.csv    # original 2yr real data (gitignored)
├── scripts/
│   ├── 00_generate_synthetic_history.py  # extends dataset to 5yr with seasonality
│   ├── 01_load_to_hdfs.sh
│   ├── 02_create_hive_tables.sql
│   ├── 03_feature_engineering.py         # Spark — rolling windows + velocity
│   ├── 04_time_series_forecast.py        # hybrid Prophet + baseline forecasting
│   ├── 05_collaborative_filter.py        # user-based CF — budget caps
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
