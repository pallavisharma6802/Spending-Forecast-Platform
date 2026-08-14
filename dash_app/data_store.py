"""Loads the pre-computed demo dataset (frontend/data/*.json, produced by
scripts/01-05) once at process start. This is the "browse the 200 synthetic
demo users" path - the upload path (live.py) computes the same shapes on
demand from core/ instead of reading these files."""

import json
import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "data")


def _load_json(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


class DemoData:
    def __init__(self):
        self.users = _load_json("users.json") or []
        self.categories = _load_json("categories.json") or []
        self.baseline = pd.DataFrame(_load_json("baseline.json") or [])
        self.health = pd.DataFrame(_load_json("health_scores.json") or [])
        self.forecasts = pd.DataFrame(_load_json("forecasts.json") or [])
        self.caps = pd.DataFrame(_load_json("budget_caps.json") or [])
        anomaly_raw = _load_json("anomalies.json") or {"alerts": []}
        self.anomaly_meta = anomaly_raw
        self.alerts = pd.DataFrame(anomaly_raw.get("alerts", []))
        peer_raw = _load_json("peer_benchmarks.json") or {"users": []}
        self.peer_index = {u["customer_id"]: u["categories"] for u in peer_raw.get("users", [])}

        if not self.baseline.empty:
            self.baseline["total_spend"] = self.baseline["total_spend"].astype(float)
            self.baseline["avg_per_transaction"] = self.baseline["avg_per_transaction"].astype(float)
            self.baseline["num_transactions"] = self.baseline["num_transactions"].astype(int)
            self.baseline["max_30d_spend"] = self.baseline["max_30d_spend"].astype(float)
        if not self.forecasts.empty:
            self.forecasts["horizon_days"] = self.forecasts["horizon_days"].astype(int)
            self.forecasts["forecasted_spend"] = self.forecasts["forecasted_spend"].astype(float)
        if not self.caps.empty:
            for col in ["cf_predicted_spend", "own_forecast_30d", "recommended_budget_cap"]:
                self.caps[col] = self.caps[col].astype(float)

    def user_view(self, customer_id: str) -> dict:
        return {
            "customer_id": customer_id,
            "baseline": self.baseline[self.baseline["customer_id"] == customer_id],
            "forecasts": self.forecasts[self.forecasts["customer_id"] == customer_id],
            "caps": self.caps[self.caps["customer_id"] == customer_id],
            "health_row": self.health[self.health["customer_id"] == customer_id],
            "health_all": self.health,
            "alerts_user": self.alerts[self.alerts["customer_id"] == customer_id] if not self.alerts.empty else pd.DataFrame(),
            "alerts_all": self.alerts,
            "anomaly_meta": self.anomaly_meta,
            "peer_categories": self.peer_index.get(customer_id, []),
            "categories": self.categories,
        }


_demo = None


def get_demo() -> DemoData:
    global _demo
    if _demo is None:
        _demo = DemoData()
    return _demo
