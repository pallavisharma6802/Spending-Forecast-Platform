"""FastAPI serving layer for the local docker-compose pipeline - local
parity with the deployed Dash app's Upload Your Data and Ask Your Data
tabs. Reads the same frontend/data/*.json demo dataset and runs the same
core/ pipeline for uploads; no HDFS, no webhdfs, no Hadoop.
"""

import os
import sys
import uuid
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from agent import claude_agent
from core.streaming.event_bus import get_event_bus
from core.streaming.window_consumer import get_connection, get_rolling_windows, start_background_consumer
from dash_app import live
from dash_app.data_store import get_demo

app = FastAPI(title="Fintech Spending Analyzer API")

_event_bus = get_event_bus()
start_background_consumer(_event_bus)
_window_conn = get_connection()
_upload_cache: dict[str, dict] = {}


def _view_json(view: dict) -> dict:
    return {
        "customer_id": view["customer_id"],
        "baseline": view["baseline"].to_dict(orient="records"),
        "forecasts": view["forecasts"].to_dict(orient="records"),
        "budget_caps": view["caps"].to_dict(orient="records"),
        "health": view["health_row"].to_dict(orient="records"),
        "anomalies": view["alerts_user"].to_dict(orient="records") if not view["alerts_user"].empty else [],
        "peer_benchmark": view["peer_categories"],
    }


@app.get("/health")
def health():
    demo = get_demo()
    return {"status": "ok", "demo_users": len(demo.users), "demo_categories": len(demo.categories)}


@app.get("/users")
def list_users():
    return get_demo().users


@app.get("/categories")
def list_categories():
    return get_demo().categories


@app.get("/users/{customer_id}/view")
def get_user_view(customer_id: str):
    demo = get_demo()
    if customer_id not in demo.users:
        raise HTTPException(404, f"Unknown customer {customer_id}")
    return _view_json(demo.user_view(customer_id))


@app.get("/anomalies")
def get_anomalies(severity: Optional[str] = None):
    demo = get_demo()
    alerts = demo.alerts.to_dict(orient="records") if not demo.alerts.empty else []
    if severity:
        alerts = [a for a in alerts if a.get("severity") == severity]
    return {"generated_at": demo.anomaly_meta.get("generated_at"), "total_alerts": len(alerts), "alerts": alerts}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), window_days: Optional[int] = None):
    content = await file.read()
    try:
        result = live.run_pipeline(content, window_days=window_days, event_bus=_event_bus)
    except live.UploadError as exc:
        raise HTTPException(400, str(exc)) from exc

    key = str(uuid.uuid4())
    _upload_cache[key] = result
    return {
        "cache_key": key,
        "customers": result["customers"],
        "categories": result["categories"],
        "rows_loaded": result["load_report"].n_rows_out,
    }


@app.get("/uploads/{cache_key}/users/{customer_id}/view")
def get_upload_view(cache_key: str, customer_id: str):
    if cache_key not in _upload_cache:
        raise HTTPException(404, "Unknown or expired cache_key - re-upload the file")
    result = _upload_cache[cache_key]
    if customer_id not in result["customers"]:
        raise HTTPException(404, f"Unknown customer {customer_id} in this upload")
    return _view_json(live.user_view(result, customer_id))


@app.get("/uploads/{cache_key}/users/{customer_id}/rolling-windows")
def get_upload_rolling_windows(cache_key: str, customer_id: str, category: str):
    if cache_key not in _upload_cache:
        raise HTTPException(404, "Unknown or expired cache_key - re-upload the file")
    return get_rolling_windows(_window_conn, customer_id, category)


class ChatRequest(BaseModel):
    customer_id: str
    message: str
    history: Optional[list] = None
    cache_key: Optional[str] = None


@app.post("/agent/chat")
def agent_chat(req: ChatRequest):
    if not claude_agent.is_configured():
        raise HTTPException(503, "ANTHROPIC_API_KEY is not configured on this server")

    if req.cache_key:
        if req.cache_key not in _upload_cache:
            raise HTTPException(404, "Unknown or expired cache_key - re-upload the file")
        view = live.user_view(_upload_cache[req.cache_key], req.customer_id)
    else:
        demo = get_demo()
        if req.customer_id not in demo.users:
            raise HTTPException(404, f"Unknown customer {req.customer_id}")
        view = demo.user_view(req.customer_id)

    answer, history = claude_agent.chat(req.message, view, history=req.history)
    return {"answer": answer, "history": history}
