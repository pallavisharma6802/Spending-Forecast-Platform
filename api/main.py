from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from typing import Optional
import json
import os
import requests
import pandas as pd
import io

NAMENODE_HTTP = "http://namenode:9870"

_cache: dict = {}


def _list_hdfs_csvs(hdfs_dir: str) -> list:
    url = f"{NAMENODE_HTTP}/webhdfs/v1{hdfs_dir}?op=LISTSTATUS"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    statuses = r.json().get("FileStatuses", {}).get("FileStatus", [])
    return [s["pathSuffix"] for s in statuses if s["pathSuffix"].endswith(".csv")]


def _read_hdfs_csv(hdfs_path: str) -> pd.DataFrame:
    url = f"{NAMENODE_HTTP}/webhdfs/v1{hdfs_path}?op=OPEN"
    r = requests.get(url, allow_redirects=True, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def _load_directory(hdfs_dir: str) -> pd.DataFrame:
    try:
        files = _list_hdfs_csvs(hdfs_dir)
    except Exception:
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            frames.append(_read_hdfs_csv(f"{hdfs_dir}{f}"))
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


_ANOMALY_FILE = os.path.join(os.path.dirname(__file__), "anomalies.json")
_PEER_FILE    = os.path.join(os.path.dirname(__file__), "peer_benchmarks.json")


def _load_anomalies():
    if os.path.exists(_ANOMALY_FILE):
        with open(_ANOMALY_FILE) as f:
            return json.load(f)
    return {"alerts": [], "generated_at": None}


def _load_peer_benchmarks():
    if os.path.exists(_PEER_FILE):
        with open(_PEER_FILE) as f:
            data = json.load(f)
        # index by customer_id for O(1) lookup
        return {u["customer_id"]: u["categories"] for u in data.get("users", [])}
    return {}


def _reload_cache():
    _cache["forecasts"] = _load_directory("/user/fintech/forecasts/")
    _cache["recommendations"] = _load_directory("/user/fintech/recommendations/")
    _cache["baseline"] = _load_directory("/user/fintech/baseline/")
    _cache["anomalies"] = _load_anomalies()
    _cache["peer_benchmarks"] = _load_peer_benchmarks()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _reload_cache()
    yield
    _cache.clear()


app = FastAPI(title="Fintech Spending Analyzer API", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "rows_cached": {k: len(v) for k, v in _cache.items()},
    }


@app.post("/reload")
def reload():
    _reload_cache()
    return {"status": "reloaded", "rows_cached": {k: len(v) for k, v in _cache.items()}}


@app.get("/users")
def list_users():
    df = _cache.get("baseline", pd.DataFrame())
    if df.empty:
        return []
    return sorted(df["customer_id"].unique().tolist())


@app.get("/users/{customer_id}/baseline")
def get_baseline(customer_id: str):
    df = _cache.get("baseline", pd.DataFrame())
    if df.empty:
        raise HTTPException(404, "No baseline data loaded")
    rows = df[df["customer_id"] == customer_id]
    if rows.empty:
        raise HTTPException(404, f"No baseline for customer {customer_id}")
    return rows.to_dict(orient="records")


@app.get("/users/{customer_id}/forecasts")
def get_forecasts(customer_id: str, horizon: Optional[int] = None):
    df = _cache.get("forecasts", pd.DataFrame())
    if df.empty:
        raise HTTPException(404, "No forecast data loaded")
    rows = df[df["customer_id"] == customer_id]
    if rows.empty:
        raise HTTPException(404, f"No forecasts for customer {customer_id}")
    if horizon is not None:
        rows = rows[rows["horizon_days"] == horizon]
    return rows.to_dict(orient="records")


@app.get("/users/{customer_id}/budget-caps")
def get_budget_caps(customer_id: str):
    df = _cache.get("recommendations", pd.DataFrame())
    if df.empty:
        raise HTTPException(404, "No recommendation data loaded")
    rows = df[df["customer_id"] == customer_id]
    if rows.empty:
        raise HTTPException(404, f"No recommendations for customer {customer_id}")
    return rows.to_dict(orient="records")


@app.get("/categories")
def get_categories():
    df = _cache.get("baseline", pd.DataFrame())
    if df.empty:
        return []
    return sorted(df["category"].unique().tolist())


@app.get("/anomalies")
def get_anomalies(severity: Optional[str] = None):
    data = _cache.get("anomalies", {"alerts": []})
    alerts = data.get("alerts", [])
    if severity:
        alerts = [a for a in alerts if a["severity"] == severity]
    return {
        "generated_at":  data.get("generated_at"),
        "current_month": data.get("current_month"),
        "days_elapsed":  data.get("days_elapsed"),
        "days_in_month": data.get("days_in_month"),
        "total_alerts":  len(alerts),
        "alerts":        alerts,
    }


@app.get("/users/{customer_id}/anomalies")
def get_user_anomalies(customer_id: str):
    data  = _cache.get("anomalies", {"alerts": []})
    alerts = [a for a in data.get("alerts", []) if a["customer_id"] == customer_id]
    return {
        "customer_id":   customer_id,
        "generated_at":  data.get("generated_at"),
        "current_month": data.get("current_month"),
        "days_elapsed":  data.get("days_elapsed"),
        "days_in_month": data.get("days_in_month"),
        "alert_count":   len(alerts),
        "alerts":        alerts,
    }


@app.get("/users/{customer_id}/peer-benchmark")
def get_peer_benchmark(customer_id: str, category: Optional[str] = None):
    index = _cache.get("peer_benchmarks", {})
    cats = index.get(customer_id)
    if cats is None:
        raise HTTPException(404, f"No peer benchmark for customer {customer_id}")
    if category:
        cats = [c for c in cats if c["category"] == category]
        if not cats:
            raise HTTPException(404, f"No peer benchmark for {customer_id} / {category}")
    above = [c for c in cats if c["direction"] == "above"]
    below = [c for c in cats if c["direction"] == "below"]
    return {
        "customer_id":   customer_id,
        "categories":    cats,
        "summary": {
            "above_peer_count": len(above),
            "below_peer_count": len(below),
            "top_overspend":    sorted(above, key=lambda x: x["vs_peers_pct"] or 0, reverse=True)[:3],
            "top_underspend":   sorted(below, key=lambda x: x["vs_peers_pct"] or 0)[:3],
        },
    }
