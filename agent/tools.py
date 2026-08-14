"""Tool schema + dispatcher for the chat agent. Every tool reads from an
already-computed `view` dict (the same shape data_store.DemoData.user_view /
dash_app.live.user_view produce) - the LLM never invents a number, it can
only retrieve ones core/ already calculated.
"""

import pandas as pd

TOOLS = [
    {
        "name": "list_categories",
        "description": "List the spending categories available for this customer.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_spend_summary",
        "description": "Historical total spend, average transaction size, and transaction count, "
                        "optionally filtered to one category. Omit category for all categories combined.",
        "input_schema": {
            "type": "object",
            "properties": {"category": {"type": "string", "description": "Optional category filter"}},
        },
    },
    {
        "name": "get_forecast",
        "description": "This customer's forecasted spend for a category at a given horizon.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "horizon_days": {"type": "integer", "enum": [7, 30, 365]},
            },
            "required": ["category", "horizon_days"],
        },
    },
    {
        "name": "get_budget_cap",
        "description": "Recommended budget cap for a category, plus the underlying own-forecast and "
                        "collaborative-filtering peer-predicted spend it was blended from.",
        "input_schema": {
            "type": "object",
            "properties": {"category": {"type": "string"}},
            "required": ["category"],
        },
    },
    {
        "name": "get_health_score",
        "description": "This customer's financial health score (0-100), letter grade, and the four "
                        "underlying dimension sub-scores (stability, essentials ratio, volatility, savings).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_anomalies",
        "description": "This customer's currently active anomaly/overspend alerts, if any.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_peer_comparison",
        "description": "How this customer's spend in a category compares to similar users - percentile "
                        "rank and percent delta vs the peer average.",
        "input_schema": {
            "type": "object",
            "properties": {"category": {"type": "string"}},
            "required": ["category"],
        },
    },
]


def _round(x) -> float:
    return round(float(x), 2)


def execute_tool(name: str, tool_input: dict, view: dict) -> dict:
    tool_input = tool_input or {}

    if name == "list_categories":
        return {"categories": view.get("categories", [])}

    if name == "get_spend_summary":
        baseline: pd.DataFrame = view["baseline"]
        category = tool_input.get("category")
        if category:
            baseline = baseline[baseline["category"] == category]
        if baseline.empty:
            return {"error": f"No historical spend data for category '{category}'" if category else "No historical spend data"}
        return {
            "category": category or "all",
            "total_spend": _round(baseline["total_spend"].sum()),
            "avg_per_transaction": _round(baseline["avg_per_transaction"].mean()),
            "num_transactions": int(baseline["num_transactions"].sum()),
        }

    if name == "get_forecast":
        forecasts: pd.DataFrame = view["forecasts"]
        category, horizon = tool_input.get("category"), tool_input.get("horizon_days")
        row = forecasts[(forecasts["category"] == category) & (forecasts["horizon_days"] == horizon)]
        if row.empty:
            return {"error": f"No {horizon}-day forecast for category '{category}'"}
        r = row.iloc[0]
        out = {"category": category, "horizon_days": horizon, "forecasted_spend": _round(r["forecasted_spend"])}
        if "model_used" in row.columns:
            out["model_used"] = r["model_used"]
        if "ci_low" in row.columns and pd.notna(r.get("ci_low")):
            out["confidence_interval_80pct"] = [_round(r["ci_low"]), _round(r["ci_high"])]
        return out

    if name == "get_budget_cap":
        caps: pd.DataFrame = view["caps"]
        category = tool_input.get("category")
        row = caps[caps["category"] == category]
        if row.empty:
            return {"error": f"No budget cap for category '{category}'"}
        r = row.iloc[0]
        return {
            "category": category,
            "own_forecast_30d": _round(r["own_forecast_30d"]),
            "cf_peer_predicted_spend": _round(r["cf_predicted_spend"]),
            "recommended_budget_cap": _round(r["recommended_budget_cap"]),
        }

    if name == "get_health_score":
        health: pd.DataFrame = view["health_row"]
        if health.empty:
            return {"error": "No health score available for this customer"}
        r = health.iloc[0]
        return {
            "score": float(r["score"]), "grade": r["grade"], "label": r["label"],
            "stability_score": _round(r["stability_score"] * 100),
            "essentials_score": _round(r["essentials_score"] * 100),
            "volatility_score": _round(r["volatility_score"] * 100),
            "savings_score": _round(r["savings_score"] * 100),
        }

    if name == "get_anomalies":
        alerts: pd.DataFrame = view["alerts_user"]
        if alerts.empty:
            return {"alerts": [], "message": "No anomalies currently detected for this customer."}
        cols = [c for c in ["category", "severity", "message", "projected_spend", "budget_cap", "overage_pct"] if c in alerts.columns]
        return {"alerts": alerts[cols].to_dict(orient="records")}

    if name == "get_peer_comparison":
        category = tool_input.get("category")
        for entry in view.get("peer_categories", []):
            if entry["category"] == category:
                return entry
        return {"error": f"No peer comparison available for category '{category}'"}

    return {"error": f"Unknown tool '{name}'"}
