"""Dash entry point. Two halves:
  - demo tabs (Overview..What-If): browse the 200-user reference dataset,
    driven by a customer dropdown, data loaded once from frontend/data/*.json
  - Upload Your Data tab: run the same core/ pipeline live against an
    uploaded file, with its own customer dropdown scoped to that file
"""

import functools
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import dash
from dash import Input, Output, State, dcc, html

from dash_app import live, render
from dash_app.data_store import get_demo
from core.health_score import ESSENTIALS
from core.streaming.event_bus import KafkaEventBus, get_event_bus
from core.streaming.window_consumer import get_connection, get_rolling_windows, start_background_consumer
from agent import claude_agent

app = dash.Dash(__name__, title="Spending Forecast & Recommendation Platform", suppress_callback_exceptions=True)
server = app.server  # WSGI entrypoint for gunicorn in production

# session-scoped cache for uploaded-file pipeline results - DataFrames don't
# fit in a JSON dcc.Store, so only the cache key travels through the store
_UPLOAD_CACHE: dict[str, dict] = {}

demo = get_demo()

# Real always-on Kafka consumer thread (or the in-memory dev fallback if no
# KAFKA_BOOTSTRAP_SERVERS is configured) - started once at process boot,
# same window_consumer code the local docker-compose stream-consumer runs.
_event_bus = get_event_bus()
_consumer_thread = start_background_consumer(_event_bus)
_window_state_conn = get_connection()

DEMO_TABS = [
    ("overview", "Spending Overview"),
    ("forecasts", "Forecasts"),
    ("budget", "Budget Recommendations"),
    ("health", "Financial Health"),
    ("anomalies", "Anomaly Alerts"),
    ("peer", "Peer Benchmarking"),
    ("whatif", "What-If Scenarios"),
]
if claude_agent.is_configured():
    DEMO_TABS = DEMO_TABS + [("chat", "Ask Your Data")]

app.layout = html.Div([
    html.Div([html.H1("Spending Forecast & Recommendation Platform"),
              html.P("PySpark (local mode) forecasting + collaborative filtering, backtested per category. "
                     "Browse the 200-user demo set or upload your own transactions.", className="caption")],
             className="app-header"),

    html.Div([
        html.Div([
            html.H4("Demo customer"),
            dcc.Dropdown(id="customer-dropdown", options=demo.users, value=(demo.users[0] if demo.users else None)),
            html.Hr(),
            html.P("Stack", style={"fontWeight": 700}),
            html.P("PySpark (local mode) - SARIMAX / recency-weighted / hierarchical shrinkage forecasting - "
                   "collaborative filtering - Isolation Forest anomaly detection", className="caption"),
            html.Hr(),
            html.P("Dataset", style={"fontWeight": 700}),
            html.P(f"{len(demo.users)} users - {len(demo.categories)} categories", className="caption"),
        ], className="sidebar"),

        html.Div([
            dcc.Tabs(id="main-tabs", value="overview", children=[
                dcc.Tab(label=label, value=key) for key, label in DEMO_TABS
            ] + [dcc.Tab(label="Upload Your Data", value="upload")]),
            html.Div(id="tab-content", className="tabs-content"),
        ], className="main-content"),
    ], className="layout"),

    dcc.Store(id="upload-cache-key"),
    dcc.Store(id="upload-customers"),
    dcc.Store(id="chat-history", data=[]),
])


def _whatif_controls(view):
    categories = sorted(view["categories"]) if view["categories"] else []
    return html.Div([
        html.Div([
            html.Label("Category to adjust"),
            dcc.Dropdown(id="whatif-category", options=categories, value=(categories[0] if categories else None)),
        ], className="col-4"),
        html.Div([
            html.Label("Spending multiplier"),
            dcc.Slider(id="whatif-multiplier", min=0.1, max=3.0, step=0.05, value=1.0,
                       marks={0.1: "0.1x", 1.0: "1.0x", 2.0: "2.0x", 3.0: "3.0x"}),
        ], className="col-6"),
    ], className="row")


@functools.lru_cache(maxsize=256)
def _cached_demo_narrative(customer_id: str) -> str | None:
    return claude_agent.generate_narrative(demo.user_view(customer_id))


def _narrative_banner(text: str | None):
    if not text:
        return None
    return html.Div([
        html.Span("Claude's take: ", style={"fontWeight": 700}), text,
    ], style={"padding": "12px 16px", "borderRadius": "8px", "background": "#4F8BF922",
              "border": "1px solid #4F8BF9", "marginBottom": "16px"})


@app.callback(Output("tab-content", "children"), Input("main-tabs", "value"), Input("customer-dropdown", "value"))
def render_tab(tab, customer_id):
    if tab == "upload":
        return _upload_layout()
    if not customer_id:
        return html.P("No customer selected.")

    view = demo.user_view(customer_id)
    if tab == "overview":
        banner = _narrative_banner(_cached_demo_narrative(customer_id)) if claude_agent.is_configured() else None
        return html.Div([banner, render.render_overview(view)])
    if tab == "chat":
        return _chat_layout()
    if tab == "forecasts":
        return render.render_forecasts(view)
    if tab == "budget":
        return render.render_budget(view)
    if tab == "health":
        return render.render_health(view)
    if tab == "anomalies":
        return render.render_anomalies(view)
    if tab == "peer":
        return render.render_peer(view)
    if tab == "whatif":
        return html.Div([_whatif_controls(view), html.Div(id="whatif-output")])
    return html.P("Unknown tab.")


@app.callback(
    Output("whatif-output", "children"),
    Input("whatif-category", "value"), Input("whatif-multiplier", "value"),
    State("customer-dropdown", "value"),
)
def update_whatif(category, multiplier, customer_id):
    if not category or not customer_id:
        return None
    view = demo.user_view(customer_id)
    return render.render_whatif(view, category, multiplier, ESSENTIALS)


def _upload_layout():
    return html.Div([
        html.P(
            "Upload a CSV with a date, category, and amount column (any common header names work) - "
            "any day/month/year range. Runs the same backtested models as the demo, served fresh on "
            "your data, with new users matched against the 200-user reference set for budget caps, "
            "health-score normalization, and peer comparison.",
            className="caption",
        ),
        html.Div([
            html.Div([
                dcc.Upload(id="upload-data", children=html.Div(["Drag and drop or ", html.A("select a CSV")]),
                           className="upload-box", multiple=False),
            ], className="col-6"),
            html.Div([
                html.Label("Analysis window"),
                dcc.RadioItems(
                    id="upload-window",
                    options=[
                        {"label": "All uploaded data", "value": 0},
                        {"label": "Last 30 days", "value": 30},
                        {"label": "Last 90 days", "value": 90},
                        {"label": "Last 365 days", "value": 365},
                    ],
                    value=0, inline=True,
                ),
            ], className="col-6"),
        ], className="row"),
        html.Div(id="upload-status", className="caption"),
        html.Div([
            html.Label("Customer"),
            dcc.Dropdown(id="upload-customer-dropdown"),
        ], id="upload-customer-picker", style={"display": "none", "margin": "12px 0"}),
        html.Div(id="upload-result"),
    ])


@app.callback(
    Output("upload-status", "children"),
    Output("upload-cache-key", "data"),
    Output("upload-customers", "data"),
    Input("upload-data", "contents"),
    Input("upload-window", "value"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def handle_upload(contents, window_value, filename):
    if contents is None:
        return "", None, None
    try:
        content_bytes = live.parse_upload_contents(contents)
        window_days = window_value if window_value else None
        result = live.run_pipeline(content_bytes, window_days=window_days, event_bus=_event_bus)
    except live.UploadError as exc:
        return f"Could not process {filename}: {exc}", None, None
    except Exception as exc:  # unexpected failure - surface it, don't silently 500
        return f"Unexpected error processing {filename}: {exc}", None, None

    key = str(uuid.uuid4())
    _UPLOAD_CACHE[key] = result
    report = result["load_report"]
    status = (
        f"Loaded {filename}: {report.n_rows_out:,} usable rows "
        f"(dropped {report.n_dropped_bad_amount + report.n_dropped_bad_date:,}), "
        f"{len(result['customers'])} customer(s), {len(result['categories'])} categories."
    )
    return status, key, result["customers"]


@app.callback(
    Output("upload-customer-picker", "style"),
    Output("upload-customer-dropdown", "options"),
    Output("upload-customer-dropdown", "value"),
    Input("upload-customers", "data"),
)
def populate_upload_customers(customers):
    if not customers:
        return {"display": "none"}, [], None
    return {"display": "block", "margin": "12px 0"}, customers, customers[0]


@app.callback(
    Output("upload-result", "children"),
    Input("upload-customer-dropdown", "value"),
    State("upload-cache-key", "data"),
)
def render_upload_result(customer_id, cache_key):
    if not customer_id or not cache_key or cache_key not in _UPLOAD_CACHE:
        return None
    result = _UPLOAD_CACHE[cache_key]
    view = live.user_view(result, customer_id)

    top_category = (
        view["baseline"].sort_values("total_spend", ascending=False).iloc[0]["category"]
        if not view["baseline"].empty else None
    )
    kafka_panel = None
    if top_category:
        time.sleep(0.3)  # let the background consumer thread catch up
        windows = get_rolling_windows(_window_state_conn, customer_id, top_category)
        broker_kind = "real broker" if isinstance(_event_bus, KafkaEventBus) else "in-memory dev fallback (no KAFKA_BOOTSTRAP_SERVERS set)"
        kafka_panel = html.Div([
            html.P(
                f"Rows were published to the '{broker_kind}' Kafka topic 'transactions.raw' and folded into "
                "this rolling-window state by the always-on background consumer thread - not recomputed "
                "from the uploaded file directly.",
                className="caption",
            ),
            render.kpi_row([
                (f"{top_category} - last 7d", f"${windows[7]:,.2f}"),
                (f"{top_category} - last 30d", f"${windows[30]:,.2f}"),
                (f"{top_category} - last 365d", f"${windows[365]:,.2f}"),
            ]),
        ])

    narrative = _narrative_banner(claude_agent.generate_narrative(view)) if claude_agent.is_configured() else None

    return html.Div([
        html.H4("Live rolling windows (via Kafka)"), kafka_panel, html.Hr(),
        narrative,
        html.H4("Overview"), render.render_overview(view), html.Hr(),
        html.H4("Forecasts"), render.render_forecasts(view), html.Hr(),
        html.H4("Budget Recommendations"), render.render_budget(view), html.Hr(),
        html.H4("Financial Health"), render.render_health(view), html.Hr(),
        html.H4("Anomaly Alerts"), render.render_anomalies(view), html.Hr(),
        html.H4("Peer Benchmarking"), render.render_peer(view),
    ])


def _chat_layout():
    return html.Div([
        html.P(
            "Ask a question about this customer's spending. Every number the assistant gives you comes from "
            "a tool call into the same forecast/health/anomaly/peer data shown in the other tabs - it can't "
            "invent figures.",
            className="caption",
        ),
        html.Div(id="chat-messages", style={"marginBottom": "12px"}),
        dcc.Textarea(id="chat-input", placeholder="e.g. Why is my Travel forecast so high?",
                     style={"width": "100%", "height": "60px"}),
        html.Button("Send", id="chat-send", n_clicks=0, style={"marginTop": "8px"}),
    ])


@app.callback(
    Output("chat-messages", "children"),
    Output("chat-history", "data"),
    Output("chat-input", "value"),
    Input("chat-send", "n_clicks"),
    State("chat-input", "value"),
    State("chat-history", "data"),
    State("customer-dropdown", "value"),
    prevent_initial_call=True,
)
def handle_chat(n_clicks, message, history, customer_id):
    if not message or not message.strip() or not customer_id:
        return dash.no_update, dash.no_update, ""

    view = demo.user_view(customer_id)
    answer, new_history = claude_agent.chat(message.strip(), view, history=history)

    display = []
    for turn in new_history:
        role = turn.get("role")
        content = turn.get("content")
        if isinstance(content, str):
            display.append((role, content))
        elif isinstance(content, list):
            texts = [b.get("text") for b in content if isinstance(b, dict) and b.get("type") == "text"]
            if texts:
                display.append((role, " ".join(texts)))

    bubbles = [
        html.Div(text, style={
            "padding": "8px 12px", "borderRadius": "8px", "marginBottom": "6px", "maxWidth": "80%",
            "marginLeft": "auto" if role == "user" else "0",
            "background": "#4F8BF9" if role == "user" else "#2a2d36",
        })
        for role, text in display
    ]
    return bubbles, new_history, ""


if __name__ == "__main__":
    debug = os.environ.get("DASH_DEBUG", "false").lower() == "true"
    port = int(os.environ.get("PORT", "8501"))
    app.run(host="0.0.0.0", port=port, debug=debug)
