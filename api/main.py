from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from typing import Optional
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


def _reload_cache():
    _cache["forecasts"] = _load_directory("/user/fintech/forecasts/")
    _cache["recommendations"] = _load_directory("/user/fintech/recommendations/")
    _cache["baseline"] = _load_directory("/user/fintech/baseline/")


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
