# Spending Forecast and Recommendation Platform

A personal-finance analytics platform: forecasts future spend per category with backtested
model selection, recommends budget caps via collaborative filtering, flags anomalous spend
pace, scores financial health, and benchmarks you against similar users. Runs against a
200-user reference dataset out of the box, or upload your own transactions (any day/month/year
range) and get the same analytics live. Includes a Claude tool-calling chat agent and a real
Kafka streaming path.

---

## Deploying your own instance

There's no shared public URL for this app - forecasts and budget caps are computed live per
deployment, and the chat agent needs your own Anthropic API key. To run your own:

1. Create a free [Upstash Kafka](https://upstash.com/) or [Confluent Cloud](https://www.confluent.io/confluent-cloud/) cluster (for the streaming path) and get an
   [Anthropic API key](https://console.anthropic.com/) (for the chat agent) - both optional, the
   app degrades gracefully without them.
2. Push this repo to your own GitHub, then create a Render Blueprint from `render.yaml` (or use
   `dash_app/Dockerfile` directly on Railway/Fly.io/anywhere that runs a Dockerfile). Fill in the
   env vars from `.env.example` in the host's dashboard.
3. Or run it entirely locally - see [Running locally](#running-locally) below.

---

## Architecture

Two run modes, one shared analytics core (`core/`), so "the deployed app" and "the local
pipeline" aren't two different codebases:

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

**Why no Hadoop cluster.** PySpark's own parallel execution (`groupBy(...).applyInPandas(...)`
across local cores) is genuinely distributed compute - it just doesn't need HDFS or YARN to get
it. `SparkSession.builder.master("local[*]")` is a JVM library call inside the same process, so
it runs inside a single deployed container. Dropping the Hadoop/HDFS/YARN/Hive layer wasn't a
compromise for deployability - nothing was using its distributed-filesystem properties at 23K
rows; the part that mattered (parallel execution) now runs everywhere, including production.

**Why Kafka is real, not decorative.** `KafkaEventBus` talks to any Kafka wire-compatible broker
- a local Redpanda container for the docker-compose pipeline, a managed Upstash/Confluent Cloud
cluster for the deployed app. On upload, transactions are actually published to the
`transactions.raw` topic; a background thread (deployed app) or a dedicated `stream-consumer`
container (local pipeline) consumes them and builds rolling 7/30/365-day window state
incrementally in SQLite - not recomputed from the uploaded file directly. `InMemoryEventBus` is
only a dev-convenience fallback for before you've set up a broker.

---

## Fixing the forecast (this used to be the whole reason for this rewrite)

The old baseline model ran at ~41% mean WAPE, validated on a single noisy 13-day window, and
its own per-category backtest winner (SARIMAX won 5/13 categories) was computed but never
actually served - the app always forced the baseline model regardless of what the evaluation
said. `core/forecast_engine.py` fixes both problems:

- **Walk-forward backtest**, 10 folds, each a fixed 30-day holdout - not one lucky/unlucky window.
  More folds means a more statistically stable estimate of true accuracy, and the tuned decay and
  shrinkage hyperparameters generalize better instead of overfitting to a handful of folds.
- **The winner is actually served.** Whichever of three candidates wins a category's backtest is
  what real forecasts use for that category, per category, not a single model forced everywhere.
- **A new candidate**, `hierarchical`: the user's own recency-weighted trend, empirical-Bayes
  shrunk toward the category trend scaled by the user's historical spend share, weighted by
  their transaction count - sparse users lean on the category signal, active users lean on their
  own trend. (The old approach forecast at the category level only and split by a static spend
  share, discarding user-level trend entirely for SARIMAX/Prophet-routed categories.)
- **A fourth candidate**, `ensemble`: an inverse-WAPE-weighted blend of the other three. Forecast
  combination usually beats picking one "best" model outright, especially where baseline and
  hierarchical are frequently near-ties - it wins outright for Gifts, Shopping, and (in a close
  race) Groceries.
- **Prophet dropped.** It was the worst performer (86% mean APE in the original evaluation) and a
  heavy native dependency (cmdstan) that's painful to deploy.
- **Wider SARIMAX search** (5 candidate orders instead of 3, including a plain non-seasonal
  ARIMA and a seasonally-differenced variant) and the backtest holdout moved from 13 days to 30 -
  matching the horizon actually served (budget caps, health score, and the default forecast view
  all key off `horizon_days == 30`), and a longer window averages out more of the day-to-day noise
  inherent to lumpy discretionary spend.
- **Confidence intervals** from backtest-fold residuals, via `scipy.stats`.

**Backtest results** (mean WAPE across 10 walk-forward folds, actual winner served per category):

| Category | Original (baseline forced, 1x 13-day window) | Current (winner served, 10x 30-day folds) | Winner |
| --- | --- | --- | --- |
| Shopping | 31.3% (SARIMAX won, unused) | 12.2% | ensemble |
| Subscriptions | 28.5% | 13.2% | ensemble |
| Medical/Dental | 52.6% | 16.5% | hierarchical |
| Groceries | 19.7% (SARIMAX won, unused) | 19.3% | baseline |
| Friend Activities | 80.5% | 19.8% | baseline |
| Personal Hygiene | 16.2% | 21.3% | baseline |
| Fitness | 40.5% | 22.2% | baseline |
| Travel | 65.3% | 22.4% | baseline |
| Hobbies | 82.4% | 22.5% | hierarchical |
| Gifts | 15.7% | 22.9% | ensemble |
| Housing and Utilities | excluded* | 27.5% | ensemble |
| Transportation | 29.8% (SARIMAX won, unused) | 28.9% | hierarchical |
| Food | 10.0% | 35.9% | baseline |
| **Mean (all 13 categories)** | **41.1% WAPE*** | **21.9% WAPE** | |

\* The original number excluded Housing and Utilities: its Jan-2025 evaluation window looked
3x cheaper per transaction than its own training history. That turned out to be a real, findable
cause, not an unfixable data defect - the old 13-day evaluation window ended before that
category's monthly rent-sized charge posted, so the "actual" side of the comparison was
structurally incomplete. Moving to a 30-day window fixed it outright (104% to 27.5% WAPE) and let
it rejoin the clean set.

Going from 6 to 10 folds moved individual categories in both directions - Shopping and
Subscriptions look worse (they were partly benefiting from a small, favorable fold sample at 6),
Groceries and Housing and Utilities look better - and the mean ticked up slightly (20.8% to
21.9%). That's the correct, expected result of a more robust sample size, not a regression: some
of the earlier per-category numbers were partly noise from too few folds, and this is a truer
estimate of what the models actually do.

**Why not push further, and what a lower number would actually require.** Portfolio-level monthly
spend has an inherent coefficient of variation of about 32% - that's the volatility of the
underlying process itself, before any model touches it, computed straight from the 5-year history
(`std / mean` of monthly totals). A forecast can't systematically beat the volatility of what it's
predicting without new information; sub-10% aggregate WAPE on categories like Travel or Gifts
(individually lumpy, CV 0.89-0.96) would need signals this dataset doesn't have - calendar events,
recurring-bill schedules, income changes - not more tuning of the same transaction history.
Shopping and Subscriptions are already close on their own merits (12-13%), which is exactly what
the CV table predicts: the categories with low inherent volatility forecast well, the lumpy
discretionary ones don't, and no amount of remodeling changes which is which.

Regenerate this yourself: `python scripts/01_backtest_and_forecast.py` (takes roughly 25 minutes -
5 SARIMAX candidates x 10 folds x 13 categories, plus the final per-category fits).

---

## Collaborative filtering + cold start

`core/recommender.py` keeps user-based CF (cosine similarity over a 65-feature matrix: 5
features x 13 categories - forecast, avg transaction, transaction count, peak 30-day spend,
spend velocity - min-max normalized). New: **cold-start handling**. An uploaded user - possibly
just one person with a few weeks of history - gets matched against the 200-user reference
population (`frontend/data/reference_feature_matrix.csv`) instead of against themselves. This
matters more than it sounds: normalizing a single row against its own min/max makes every
feature collapse to 0 (min == max == the one value present), which silently zeroes out
similarity and made every cold-start recommendation fall back to "just your own forecast" - a
real bug caught while building this, fixed by always normalizing against the larger reference
population's column statistics. The same pattern (`reference_health_raw.csv`,
`baseline.json`-derived category totals) fixes the equivalent issue in health-score normalization
and peer-percentile ranking.

---

## Upload your own data

The "Upload Your Data" tab (and `POST /upload` on the local FastAPI) accepts a CSV with a date,
category, and amount column - common header variants (`Amount`, `Transaction Date`, `Date`,
etc.) are matched automatically, not just this project's exact column names. Any day, month, or
year range works: pick "All uploaded data," "Last 30/90/365 days," and every `core/` function
takes that window directly - there's no separate code path for different granularities. Model
selection (which model family per category) comes from the pre-learned backtest config; serving
computes fresh numbers from whatever data you gave it, whether that's the reference population
or your own file.

---

## Ask Your Data (Claude agent)

A tool-calling chat agent (`agent/`) answers free-form questions about the current customer's
data - "why is my Travel forecast so high," "how do I compare to similar users on groceries."
Every number in its answer comes from a tool call into the same computed data shown in the other
tabs (`agent/tools.py` wraps `core/` outputs; the LLM has no other way to produce a number). A
cheap narrative summary (Haiku) runs automatically on the Overview tab. Both features check
`ANTHROPIC_API_KEY` at runtime and simply don't appear if it's unset - the rest of the app is
fully functional without one.

---

## Running locally

### Option A: just the app, no Docker

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r dash_app/requirements.txt
brew install openjdk@17   # or any JDK 17+ - PySpark needs one, even in local mode
export JAVA_HOME=$(brew --prefix openjdk@17)
python dash_app/app.py    # http://localhost:8501
```

Set `ANTHROPIC_API_KEY` / `KAFKA_BOOTSTRAP_SERVERS` per `.env.example` to enable the chat agent
and real Kafka; both are optional.

### Option B: full pipeline via Docker Compose

```bash
docker compose up -d --build
```

Starts: Redpanda (`kafka`), FastAPI (`api`, [localhost:8000/docs](http://localhost:8000/docs)),
the always-on rolling-window consumer (`stream-consumer`), and the Dash app (`dash`,
[localhost:8501](http://localhost:8501)).

### Regenerating the demo dataset

```bash
python scripts/01_backtest_and_forecast.py   # walk-forward backtest -> model_config.json, forecasts.json
python scripts/02_recommend_budgets.py       # collaborative filtering -> budget_caps.json
python scripts/03_detect_anomalies.py        # z-score + isolation forest -> anomalies.json
python scripts/04_health_scores.py           # composite score -> health_scores.json
python scripts/05_peer_benchmarks.py         # peer comparison -> peer_benchmarks.json
```

Each script is a thin wrapper around `core/` - the actual logic lives there and is unit-testable
without Docker, HDFS, or a browser.

---

## Dataset

- 23,000 transactions, 200 customers, 13 spending categories
- Date range: January 2020 - January 2025 (2 years real + 3 years synthetic, generated by
  `scripts/00_generate_synthetic_history.py` with realistic seasonality - holiday spikes, summer
  travel bumps, January gym surges - and COVID-era suppression in 2020)
- Categories: Fitness, Food, Friend Activities, Gifts, Groceries, Hobbies, Housing & Utilities,
  Medical/Dental, Personal Hygiene, Shopping, Subscriptions, Transportation, Travel

---

## API endpoints (local FastAPI, `api/main.py`)

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/health` | Liveness + demo dataset size |
| GET | `/users` | List demo customer IDs |
| GET | `/categories` | List demo categories |
| GET | `/users/{id}/view` | Baseline, forecasts, budget caps, health, anomalies, peer benchmark for one demo customer |
| GET | `/anomalies` | Platform-wide anomaly alerts (optional `?severity=`) |
| POST | `/upload` | Upload a CSV, run the live pipeline, get back a `cache_key` |
| GET | `/uploads/{cache_key}/users/{id}/view` | Same shape as `/users/{id}/view`, for an uploaded customer |
| GET | `/uploads/{cache_key}/users/{id}/rolling-windows?category=` | Kafka-consumer-built rolling 7/30/365-day sums |
| POST | `/agent/chat` | `{customer_id, message, history?, cache_key?}` -> tool-calling chat response |

---

## Repository structure

```
├── core/                      # shared analytics engine - PySpark local mode, no Docker needed to run it
│   ├── data_loader.py
│   ├── forecast_engine.py
│   ├── recommender.py
│   ├── anomaly.py
│   ├── health_score.py
│   ├── peer_bench.py
│   └── streaming/
│       ├── event_bus.py
│       └── window_consumer.py
├── agent/                     # Claude tool-calling chat agent + narrative summaries
│   ├── tools.py
│   └── claude_agent.py
├── dash_app/                  # deployed frontend (Dash)
│   ├── app.py
│   ├── render.py              # view dict -> Dash component tree, one function per tab
│   ├── live.py                # runs core/ against an uploaded file
│   └── data_store.py          # loads the pre-computed 200-user demo dataset
├── api/                        # FastAPI serving layer, local pipeline only
│   └── main.py
├── scripts/
│   ├── 00_generate_synthetic_history.py
│   ├── 01_backtest_and_forecast.py
│   ├── 02_recommend_budgets.py
│   ├── 03_detect_anomalies.py
│   ├── 04_health_scores.py
│   ├── 05_peer_benchmarks.py
│   └── 06_stream_consumer.py   # perpetual local Kafka consumer
├── frontend/data/               # pre-computed demo dataset (checked in)
├── docker-compose.yml           # Redpanda + api + stream-consumer + dash
├── render.yaml                  # Render Blueprint for the deployed app
└── .env.example
```
