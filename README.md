# Spending Forecast and Recommendation Platform

Personal finance analytics: per-category spend forecasting with backtested model selection,
collaborative-filtering budget recommendations, anomaly detection, a composite financial health
score, and peer benchmarking. Ships with a 200-user reference dataset, or upload your own
transactions and get the same analytics computed live over any day/month/year range. Includes a
tool-calling chat agent and a Kafka-based streaming ingestion path.

## Features

- **Forecasting** - per-category models (recency-weighted baseline, hierarchical shrinkage,
  SARIMAX, and an inverse-error ensemble), selected per category by walk-forward backtest and
  served with confidence intervals
- **Budget recommendations** - user-based collaborative filtering with cold-start support for
  new users
- **Anomaly detection** - z-score and isolation-forest checks against historical spend pace
- **Financial health score** - a composite 0-100 score across stability, essentials ratio,
  volatility, and savings potential
- **Peer benchmarking** - percentile ranking and forecast comparison against similar users
- **File upload** - bring your own transactions CSV, any date range, common header formats
  matched automatically
- **Chat agent** - ask questions about your spending; every answer is grounded in a tool call,
  not model guesswork
- **Streaming** - transactions publish to Kafka and are consumed into rolling 7/30/365-day
  window state by an always-on consumer

## Architecture

A single analytics core (`core/`, PySpark in local mode) is shared by two deployment shapes:

```
                         core/                    (PySpark, master("local[*]") - no HDFS/YARN)
                         ├── spark_session.py   shared local-mode SparkSession
                         ├── data_loader.py     any transactions CSV -> canonical schema
                         ├── forecast_engine.py backtested per-category model choice + serving
                         ├── recommender.py     collaborative filtering + cold-start blending
                         ├── anomaly.py         z-score + isolation forest, arbitrary window
                         ├── health_score.py    composite 0-100 score, arbitrary window
                         ├── peer_bench.py      percentile + CF delta vs. reference population
                         └── streaming/
                             ├── event_bus.py       KafkaEventBus (real) / InMemoryEventBus (dev fallback)
                             └── window_consumer.py incremental rolling-window state

        ┌─────────────────────────────────┐        ┌────────────────────────────────────┐
        │  Deployed app (Dash + gunicorn) │        │  Local pipeline (docker compose)    │
        ├─────────────────────────────────┤        ├────────────────────────────────────┤
        │ dash_app/ imports core/         │        │ Same core/, same Dockerfile pattern │
        │ directly, in one container:     │        │ + Kafka/Redpanda container, its own │
        │  - upload -> live core/ compute │        │   perpetual stream-consumer service │
        │  - background thread runs a     │        │ FastAPI (api/) exposes /upload and  │
        │    real always-on Kafka         │        │   /agent/chat for local parity      │
        │    consumer against a managed   │        │                                     │
        │    broker (Upstash/Confluent)   │        │                                     │
        │  - Claude agent via Anthropic   │        │                                     │
        │    SDK                          │        │                                     │
        └─────────────────────────────────┘        └────────────────────────────────────┘
```

PySpark runs in local mode (`master("local[*]")`), so its parallel execution
(`groupBy(...).applyInPandas(...)` across local cores) works inside a single container with no
HDFS or YARN dependency.

`KafkaEventBus` talks to any Kafka wire-compatible broker - a local Redpanda container for the
docker-compose pipeline, a managed Upstash/Confluent Cloud cluster for the deployed app. Uploaded
transactions are published to the `transactions.raw` topic and consumed into rolling
7/30/365-day window state in SQLite by a background thread (deployed app) or a dedicated
`stream-consumer` container (local pipeline). `InMemoryEventBus` is a dev fallback used when no
broker is configured.

## Forecasting

`core/forecast_engine.py` selects a model per category via a 10-fold walk-forward backtest (each
fold a 30-day holdout, matching the horizon actually served) from four candidates:

| Model | Description |
| --- | --- |
| `baseline` | Recency-weighted exponential moving average of monthly spend |
| `hierarchical` | The user's own trend, empirical-Bayes shrunk toward the category trend scaled by their historical spend share and weighted by transaction count |
| `sarimax` | Category-level SARIMAX (5 candidate orders), allocated to users by spend share |
| `ensemble` | Inverse-WAPE-weighted blend of the other three |

Whichever candidate wins a category's backtest is what that category actually serves. Confidence
intervals are derived from backtest-fold residuals via `scipy.stats`.

**Accuracy** (mean WAPE across 10 walk-forward folds):

| Category | Model | WAPE |
| --- | --- | --- |
| Shopping | ensemble | 12.2% |
| Subscriptions | ensemble | 13.2% |
| Medical/Dental | hierarchical | 16.5% |
| Groceries | baseline | 19.3% |
| Friend Activities | baseline | 19.8% |
| Personal Hygiene | baseline | 21.3% |
| Fitness | baseline | 22.2% |
| Travel | baseline | 22.4% |
| Hobbies | hierarchical | 22.5% |
| Gifts | ensemble | 22.9% |
| Housing and Utilities | ensemble | 27.5% |
| Transportation | hierarchical | 28.9% |
| Food | baseline | 35.9% |
| **Mean (all 13 categories)** | | **21.9%** |

Portfolio-level monthly spend has an inherent coefficient of variation of about 32%
(`std / mean` of monthly totals across the 5-year history), which sets a practical floor on
achievable accuracy: a forecast can't systematically beat the volatility of the process it's
predicting without additional signal (calendar events, recurring-bill schedules, income changes)
that transaction history alone doesn't provide. Categories with lower inherent volatility
(Shopping, Subscriptions) forecast well; lumpy discretionary categories (Travel, Gifts, Food) do
not, regardless of model choice.

Regenerate the backtest and demo dataset:

```bash
python scripts/01_backtest_and_forecast.py   # ~25 min: 5 SARIMAX candidates x 10 folds x 13 categories
python scripts/02_recommend_budgets.py
python scripts/03_detect_anomalies.py
python scripts/04_health_scores.py
python scripts/05_peer_benchmarks.py
```

## Collaborative filtering

User-based CF over a 65-feature matrix (5 features x 13 categories: forecast, average
transaction, transaction count, peak 30-day spend, spend velocity), min-max normalized, cosine
similarity, top-10 neighbors. Uploaded users with little or no history are matched against the
200-user reference population (`frontend/data/reference_feature_matrix.csv`) rather than against
themselves - normalizing a single row against its own min/max would otherwise collapse every
feature to zero and silently disable the recommendation. The same reference-population pattern
(`reference_health_raw.csv`, category totals from `baseline.json`) applies to health-score
normalization and peer-percentile ranking.

## Getting started

### Run locally without Docker

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r dash_app/requirements.txt
brew install openjdk@17   # any JDK 17+ - PySpark requires one, even in local mode
export JAVA_HOME=$(brew --prefix openjdk@17)
python dash_app/app.py    # http://localhost:8501
```

Set `ANTHROPIC_API_KEY` and `KAFKA_BOOTSTRAP_SERVERS` per `.env.example` to enable the chat agent
and a real Kafka broker; both are optional and the app runs without them.

### Run the full pipeline with Docker Compose

```bash
docker compose up -d --build
```

Starts Redpanda (`kafka`), FastAPI (`api`, [localhost:8000/docs](http://localhost:8000/docs)),
an always-on rolling-window consumer (`stream-consumer`), and the Dash app (`dash`,
[localhost:8501](http://localhost:8501)).

### Deploy

A `render.yaml` Blueprint is included for Render; `dash_app/Dockerfile` works on any host that
runs a Dockerfile (Railway, Fly.io, etc.). Set the environment variables listed in `.env.example`
in the host's dashboard.

## Dataset

- 23,000 transactions, 200 customers, 13 spending categories
- Date range: January 2020 - January 2025 (2 years real data, 3 years synthetic, generated by
  `scripts/00_generate_synthetic_history.py` with seasonality - holiday spikes, summer travel,
  January gym surges - and COVID-era suppression in 2020)
- Categories: Fitness, Food, Friend Activities, Gifts, Groceries, Hobbies, Housing & Utilities,
  Medical/Dental, Personal Hygiene, Shopping, Subscriptions, Transportation, Travel

## API reference

FastAPI service (`api/main.py`), local pipeline only:

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/health` | Liveness and demo dataset size |
| GET | `/users` | List demo customer IDs |
| GET | `/categories` | List demo categories |
| GET | `/users/{id}/view` | Baseline, forecasts, budget caps, health, anomalies, and peer benchmark for one demo customer |
| GET | `/anomalies` | Platform-wide anomaly alerts (optional `?severity=`) |
| POST | `/upload` | Upload a CSV and run the live pipeline; returns a `cache_key` |
| GET | `/uploads/{cache_key}/users/{id}/view` | Same shape as `/users/{id}/view`, for an uploaded customer |
| GET | `/uploads/{cache_key}/users/{id}/rolling-windows?category=` | Kafka-consumer-built rolling 7/30/365-day sums |
| POST | `/agent/chat` | `{customer_id, message, history?, cache_key?}` -> tool-calling chat response |

## Project structure

```
├── core/                        # analytics engine - PySpark local mode
│   ├── data_loader.py
│   ├── forecast_engine.py
│   ├── recommender.py
│   ├── anomaly.py
│   ├── health_score.py
│   ├── peer_bench.py
│   └── streaming/
│       ├── event_bus.py
│       └── window_consumer.py
├── agent/                       # chat agent + narrative summaries
│   ├── tools.py
│   └── claude_agent.py
├── dash_app/                    # frontend (Dash)
│   ├── app.py
│   ├── render.py                # view dict -> Dash component tree, one function per tab
│   ├── live.py                  # runs core/ against an uploaded file
│   └── data_store.py            # loads the pre-computed 200-user demo dataset
├── api/                         # FastAPI serving layer, local pipeline only
│   └── main.py
├── scripts/
│   ├── 00_generate_synthetic_history.py
│   ├── 01_backtest_and_forecast.py
│   ├── 02_recommend_budgets.py
│   ├── 03_detect_anomalies.py
│   ├── 04_health_scores.py
│   ├── 05_peer_benchmarks.py
│   └── 06_stream_consumer.py    # perpetual local Kafka consumer
├── frontend/data/                 # pre-computed demo dataset
├── docker-compose.yml
├── render.yaml
└── .env.example
```
