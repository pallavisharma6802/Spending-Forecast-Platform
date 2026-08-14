"""Pure functions: view dict -> Dash component tree. One function per tab,
each a direct port of the corresponding section in the retired
frontend/app.py (Streamlit), same Plotly figures, same layout intent."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dash_table, dcc, html

GRADE_COLOR = {"A": "#2ecc71", "B": "#27ae60", "C": "#f39c12", "D": "#e67e22", "F": "#e74c3c"}
SEV_COLOR = {"high": "#e74c3c", "medium": "#e67e22", "low": "#f1c40f"}
DIR_COLOR = {"above": "#e74c3c", "below": "#2ecc71", "neutral": "#95a5a6"}


def kpi_row(items):
    return html.Div(
        [
            html.Div(
                [html.Div(label, className="kpi-label"), html.Div(value, className="kpi-value")],
                className="kpi-card",
            )
            for label, value in items
        ],
        className="kpi-row",
    )


def _table(df: pd.DataFrame, columns=None):
    cols = columns or [{"name": c, "id": c} for c in df.columns]
    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=cols,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "6px", "fontSize": "13px"},
        page_size=15,
        sort_action="native",
    )


def render_overview(view: dict):
    baseline = view["baseline"]
    forecasts = view["forecasts"]
    health_row = view["health_row"]

    total_hist = baseline["total_spend"].sum() if not baseline.empty else 0.0
    total_fc30 = forecasts[forecasts["horizon_days"] == 30]["forecasted_spend"].sum() if not forecasts.empty else 0.0
    top_category = baseline.sort_values("total_spend", ascending=False).iloc[0]["category"] if not baseline.empty else "-"
    health_score = (
        f"{health_row.iloc[0]['score']:.0f} / 100 ({health_row.iloc[0]['grade']})"
        if not health_row.empty else "-"
    )

    kpis = kpi_row([
        ("Total Historical Spend", f"${total_hist:,.0f}"),
        ("30-Day Forecast (all categories)", f"${total_fc30:,.0f}"),
        ("Top Spending Category", top_category),
        ("Financial Health Score", health_score),
    ])

    if baseline.empty:
        return html.Div([kpis, html.P("No spending data for this customer.")])

    sorted_bl = baseline.sort_values("total_spend", ascending=False)
    fig = px.bar(
        sorted_bl, x="category", y="total_spend", color="category",
        title="Total Spend by Category",
        labels={"total_spend": "Total Spend ($)", "category": "Category"},
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-30)

    table_df = sorted_bl[["category", "total_spend", "avg_per_transaction", "num_transactions", "max_30d_spend"]].round(2)
    table_df.columns = ["Category", "Total Spend ($)", "Avg per Txn ($)", "# Transactions", "Peak 30d Spend ($)"]

    return html.Div([kpis, dcc.Graph(figure=fig), _table(table_df)])


def render_forecasts(view: dict):
    forecasts = view["forecasts"]
    if forecasts.empty:
        return html.P("No forecast data for this customer.")

    fig = px.bar(
        forecasts, x="category", y="forecasted_spend",
        color=forecasts["horizon_days"].astype(str), barmode="group",
        title="Forecasted Spend by Category & Horizon",
        labels={"forecasted_spend": "Forecasted Spend ($)", "color": "Horizon (days)"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(xaxis_tickangle=-30, legend_title="Horizon (days)")

    model_note = ""
    if "model_used" in forecasts.columns:
        models = forecasts["model_used"].value_counts()
        model_note = " · ".join(f"{m}: {n} rows" for m, n in models.items())

    pivot = forecasts.pivot_table(index="category", columns="horizon_days", values="forecasted_spend").reset_index()
    pivot.columns = ["Category"] + [f"{c}d ($)" for c in pivot.columns[1:]]

    return html.Div([
        html.P(f"Model(s) used: {model_note}", className="caption") if model_note else None,
        dcc.Graph(figure=fig),
        _table(pivot.round(2)),
    ])


def render_budget(view: dict):
    caps = view["caps"]
    if caps.empty:
        return html.P("No budget recommendations for this customer.")

    sorted_caps = caps.sort_values("recommended_budget_cap", ascending=False)
    fig2 = px.scatter(
        sorted_caps, x="own_forecast_30d", y="recommended_budget_cap",
        text="category", size="cf_predicted_spend",
        title="Own Forecast vs. Budget Cap",
        labels={"own_forecast_30d": "Own Forecast ($)", "recommended_budget_cap": "Budget Cap ($)"},
    )
    fig2.update_traces(textposition="top center")

    table_df = sorted_caps[["category", "own_forecast_30d", "cf_predicted_spend", "spend_velocity", "recommended_budget_cap"]].round(2)
    table_df.columns = ["Category", "Own Forecast ($)", "CF Predicted ($)", "Velocity", "Budget Cap ($)"]

    return html.Div([
        html.Div([
            html.Div(dcc.Graph(figure=fig2), className="col-6"),
            html.Div(_table(table_df), className="col-6"),
        ], className="row"),
    ])


def render_health(view: dict):
    health_row = view["health_row"]
    health_all = view["health_all"]
    if health_row.empty:
        return html.P("No health score for this customer.")

    h = health_row.iloc[0]
    score, grade, label = float(h["score"]), h["grade"], h["label"]
    color = GRADE_COLOR.get(grade, "#95a5a6")

    score_card = html.Div([
        html.Div(f"{score:.0f}", style={"fontSize": "64px", "fontWeight": 800, "color": color}),
        html.Div(f"Grade {grade}", style={"fontSize": "24px", "fontWeight": 700, "color": color}),
        html.Div(label, style={"fontSize": "15px"}),
        html.Div("out of 100", style={"fontSize": "12px", "color": "#888"}),
    ], style={"textAlign": "center", "padding": "20px", "borderRadius": "12px",
              "border": f"2px solid {color}", "background": color + "18"})

    fig_hist = px.histogram(health_all, x="score", nbins=20, color_discrete_sequence=["#4F8BF9"],
                             labels={"score": "Health Score"})
    fig_hist.add_vline(x=score, line_dash="dash", line_color=color, annotation_text="You")
    fig_hist.update_layout(showlegend=False, height=220, bargap=0.05, margin=dict(t=20, b=20, l=10, r=10))

    dims = {
        "Stability": float(h["stability_score"]), "Essentials Ratio": float(h["essentials_score"]),
        "Low Volatility": float(h["volatility_score"]), "Savings Potential": float(h["savings_score"]),
    }
    dim_df = pd.DataFrame({"Dimension": list(dims.keys()), "Score": [round(v * 100, 1) for v in dims.values()]})
    fig_dims = px.bar(
        dim_df, x="Score", y="Dimension", orientation="h", color="Score",
        color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"], range_color=[0, 100],
        title="Dimension Breakdown (0-100)",
    )
    fig_dims.update_layout(coloraxis_showscale=False, xaxis_range=[0, 100], height=280, margin=dict(t=40, b=20))

    signals = pd.DataFrame([
        {"Signal": "Spend CV (stability)", "Value": f"{float(h['spend_cv']):.2f}", "Interpretation": "lower = more stable"},
        {"Signal": "Essentials ratio", "Value": f"{float(h['essentials_ratio']):.0%}", "Interpretation": "ideal ~= 50% on necessities"},
        {"Signal": "Avg MoM volatility", "Value": f"{float(h['mom_volatility']):.1f}x", "Interpretation": "lower = smoother month-to-month"},
        {"Signal": "Savings gap", "Value": f"{float(h['savings_gap']):+.0%}", "Interpretation": "positive = spending below forecast"},
    ])

    rank = int((health_all["score"] < score).sum()) + 1
    pct = round((1 - (rank - 1) / max(len(health_all), 1)) * 100, 1)
    grade_dist = health_all["grade"].value_counts().sort_index()
    fig_grades = px.bar(x=grade_dist.index, y=grade_dist.values, color=grade_dist.index,
                         color_discrete_map=GRADE_COLOR, labels={"x": "Grade", "y": "Users"},
                         title=f"Grade distribution across all {len(health_all)} users")
    fig_grades.update_layout(showlegend=False, height=280, margin=dict(t=40, b=20))

    return html.Div([
        html.Div([
            html.Div([score_card, dcc.Graph(figure=fig_hist)], className="col-4"),
            html.Div([dcc.Graph(figure=fig_dims), _table(signals)], className="col-8"),
        ], className="row"),
        html.Hr(),
        kpi_row([
            ("Rank", f"#{rank} of {len(health_all)}"),
            ("Better than", f"{100 - pct:.0f}% of users"),
            ("Population median score", f"{health_all['score'].median():.1f}"),
        ]),
        dcc.Graph(figure=fig_grades),
    ])


def render_anomalies(view: dict):
    meta = view["anomaly_meta"]
    user_alerts = view["alerts_user"]
    all_alerts = view["alerts_all"]

    caption = (
        f"As of {meta.get('as_of', '-')}, {meta.get('window_days', '?')}-day window "
        f"projected to a {meta.get('target_period_days', '?')}-day equivalent via a "
        f"{meta.get('pace_factor', '?')}x pace factor."
    )

    cards = []
    if user_alerts.empty:
        cards.append(html.P("No anomalies detected for this customer in the current period.", className="success-text"))
    else:
        cards.append(html.P(f"{len(user_alerts)} alert(s) detected.", className="warning-text"))
        for _, alert in user_alerts.sort_values("overage_pct", ascending=False).iterrows():
            color = SEV_COLOR.get(alert["severity"], "#888")
            cards.append(html.Div([
                html.Div([
                    html.B(f"{alert['category']} - "),
                    html.Span(alert["severity"].upper(), style={"color": color, "fontWeight": 600}),
                ]),
                html.Div(alert["message"], style={"fontSize": "0.9em"}),
                kpi_row([
                    ("Spent so far", f"${alert['window_spend']:,.0f}"),
                    ("Projected", f"${alert['projected_spend']:,.0f}"),
                    ("Budget cap", f"${alert['budget_cap']:,.0f}" if alert["budget_cap"] else "-"),
                    ("Overage", f"+{alert['overage_pct']:.0f}%" if alert["overage_pct"] else "-"),
                ]),
            ], style={"borderLeft": f"4px solid {color}", "padding": "10px 16px", "marginBottom": "10px",
                      "borderRadius": "4px", "background": color + "18"}))

    platform = []
    if not all_alerts.empty:
        platform.append(kpi_row([
            ("Total alerts", str(len(all_alerts))),
            ("Users affected", str(all_alerts["customer_id"].nunique())),
            ("High-severity alerts", str(len(all_alerts[all_alerts["severity"] == "high"]))),
        ]))
        cat_counts = all_alerts["category"].value_counts().reset_index()
        cat_counts.columns = ["category", "alerts"]
        fig_cat = px.bar(cat_counts.head(10), x="alerts", y="category", orientation="h",
                          title="Alerts by Category", color="alerts", color_continuous_scale="Reds")
        fig_cat.update_layout(coloraxis_showscale=False, height=320, yaxis_title="")

        sev_counts = all_alerts["severity"].value_counts().reset_index()
        sev_counts.columns = ["severity", "count"]
        fig_sev = px.pie(sev_counts, names="severity", values="count", title="Alerts by Severity",
                          color="severity", color_discrete_map=SEV_COLOR)

        top_table = all_alerts.nlargest(10, "overage_pct")[
            ["customer_id", "category", "severity", "projected_spend", "budget_cap", "overage_pct", "z_score"]
        ].round(2)
        top_table.columns = ["Customer", "Category", "Severity", "Projected ($)", "Cap ($)", "Overage %", "Z-score"]

        platform.append(html.Div([
            html.Div(dcc.Graph(figure=fig_cat), className="col-6"),
            html.Div(dcc.Graph(figure=fig_sev), className="col-6"),
        ], className="row"))
        platform.append(_table(top_table))
    else:
        platform.append(html.P("No alerts in current period."))

    return html.Div([html.P(caption, className="caption"), html.Div(cards), html.Hr(),
                      html.H4("Platform-wide alert summary"), html.Div(platform)])


def render_peer(view: dict):
    peer_cats = view["peer_categories"]
    if not peer_cats:
        return html.P("No peer benchmark data for this customer.")

    peer_df = pd.DataFrame(peer_cats)
    peer_df["vs_peers_pct"] = peer_df["vs_peers_pct"].astype(float)
    above = peer_df[peer_df["direction"] == "above"]
    below = peer_df[peer_df["direction"] == "below"]
    neutral = peer_df[peer_df["direction"] == "neutral"]

    kpis = kpi_row([
        ("Categories above peers", str(len(above))),
        ("Categories below peers", str(len(below))),
        ("Categories in line", str(len(neutral))),
    ])

    chart_df = peer_df.sort_values("vs_peers_pct").copy()
    chart_df["color"] = chart_df["direction"].map(DIR_COLOR)
    fig_wf = go.Figure(go.Bar(
        x=chart_df["vs_peers_pct"], y=chart_df["category"], orientation="h",
        marker_color=chart_df["color"],
        text=chart_df["vs_peers_pct"].apply(lambda v: f"{v:+.0f}%"), textposition="outside",
    ))
    fig_wf.update_layout(title="Your Forecast vs Peer Average (30-day)", xaxis_title="delta vs peers (%)",
                          height=420, margin=dict(l=160, t=50, b=30, r=60))

    melt_df = peer_df.melt(id_vars="category", value_vars=["own_forecast_30d", "peer_forecast_30d"],
                            var_name="source", value_name="amount")
    melt_df["source"] = melt_df["source"].map({"own_forecast_30d": "Your forecast", "peer_forecast_30d": "Peer average"})
    fig_comp = px.bar(melt_df, x="category", y="amount", color="source", barmode="group",
                       color_discrete_map={"Your forecast": "#4F8BF9", "Peer average": "#f39c12"},
                       title="Your 30-Day Forecast vs Peer Average", labels={"amount": "Forecasted Spend ($)"})
    fig_comp.update_layout(xaxis_tickangle=-30, legend_title="")

    insight_cards = []
    for row in peer_df.sort_values("vs_peers_pct", ascending=False, key=abs).itertuples():
        color = DIR_COLOR.get(row.direction, "#888")
        insight_cards.append(html.Div([
            html.B(f"{row.category}"),
            html.Span(f"{row.vs_peers_pct:+.0f}%", style={"float": "right", "color": color, "fontWeight": 700}),
            html.Div(row.insight, style={"fontSize": "0.85em"}),
        ], style={"borderLeft": f"3px solid {color}", "padding": "8px 12px", "marginBottom": "8px",
                  "borderRadius": "4px", "background": color + "15"}))

    return html.Div([kpis, dcc.Graph(figure=fig_wf), dcc.Graph(figure=fig_comp),
                      html.H4("Insights"), html.Div(insight_cards)])


def render_whatif(view: dict, category: str, multiplier: float, essentials: set):
    forecasts = view["forecasts"]
    health_row = view["health_row"]
    health_all = view["health_all"]
    baseline = view["baseline"]

    if health_row.empty or forecasts.empty:
        return html.P("Not enough data for a what-if scenario for this customer.")

    h = health_row.iloc[0]
    user_fc30 = forecasts[forecasts["horizon_days"] == 30].set_index("category")["forecasted_spend"]
    original_cat_fc = float(user_fc30.get(category, 0.0))
    adjusted_cat_fc = original_cat_fc * multiplier
    delta_fc = adjusted_cat_fc - original_cat_fc

    actual_avg_monthly = float(h["actual_avg_monthly"])
    original_forecast30 = float(h["forecast_30d"])
    adj_actual_monthly = actual_avg_monthly + delta_fc

    def savings_score(forecast, actual):
        gap = float(np.clip((forecast - actual) / max(forecast, 1), -1, 1))
        raw = (gap + 1) / 2
        all_gaps = health_all["savings_gap"].to_numpy()
        raw_all = (np.clip(all_gaps, -1, 1) + 1) / 2
        lo, hi = raw_all.min(), raw_all.max()
        return (float((raw - lo) / (hi - lo)) if hi > lo else 0.5), gap

    new_norm_savings, new_savings_gap = savings_score(original_forecast30, adj_actual_monthly)

    def essentials_score(ratio):
        raw = max(0.0, 1.0 - 2.0 * abs(ratio - 0.5))
        all_ratios = health_all["essentials_ratio"].to_numpy()
        raw_all = np.maximum(0.0, 1.0 - 2.0 * np.abs(all_ratios - 0.5))
        lo, hi = raw_all.min(), raw_all.max()
        return float((raw - lo) / (hi - lo)) if hi > lo else 0.5

    if category in essentials and not baseline.empty:
        orig_cat_hist = float(baseline[baseline["category"] == category]["total_spend"].sum())
        orig_total_hist = float(baseline["total_spend"].sum())
        orig_ess_hist = float(baseline[baseline["category"].isin(essentials)]["total_spend"].sum())
        adj_cat_hist = orig_cat_hist * multiplier
        adj_ess_hist = orig_ess_hist - orig_cat_hist + adj_cat_hist
        adj_total_hist = orig_total_hist - orig_cat_hist + adj_cat_hist
        new_ess_ratio = adj_ess_hist / adj_total_hist if adj_total_hist > 0 else 0.0
        new_norm_ess = essentials_score(new_ess_ratio)
    else:
        new_norm_ess = float(h["essentials_score"])
        new_ess_ratio = float(h["essentials_ratio"])

    original_score = float(h["score"])
    new_score = (float(h["stability_score"]) + new_norm_ess + float(h["volatility_score"]) + new_norm_savings) * 25.0

    def grade(s):
        if s >= 80: return "A"
        if s >= 65: return "B"
        if s >= 50: return "C"
        if s >= 35: return "D"
        return "F"

    new_grade = grade(new_score)
    score_delta = new_score - original_score

    kpis = html.Div([
        html.Div([
            html.Div("Baseline", style={"fontWeight": 700}),
            html.Div(f"Monthly actual avg: ${actual_avg_monthly:,.0f}"),
            html.Div(f"Health score: {original_score:.1f} ({h['grade']})"),
        ], className="col-5"),
        html.Div("->", className="col-2", style={"textAlign": "center", "fontSize": "24px", "paddingTop": "20px"}),
        html.Div([
            html.Div("Scenario", style={"fontWeight": 700}),
            html.Div(f"Monthly actual avg: ${adj_actual_monthly:,.0f} ({delta_fc:+,.0f})"),
            html.Div(f"Health score: {new_score:.1f} ({new_grade}) ({score_delta:+.1f} pts)"),
        ], className="col-5"),
    ], className="row")

    dim_data = pd.DataFrame([
        {"Dimension": "Stability", "Before": float(h["stability_score"]) * 100, "After": float(h["stability_score"]) * 100},
        {"Dimension": "Essentials Ratio", "Before": float(h["essentials_score"]) * 100, "After": new_norm_ess * 100},
        {"Dimension": "Low Volatility", "Before": float(h["volatility_score"]) * 100, "After": float(h["volatility_score"]) * 100},
        {"Dimension": "Savings Potential", "Before": float(h["savings_score"]) * 100, "After": new_norm_savings * 100},
    ])
    dim_melt = dim_data.melt(id_vars="Dimension", var_name="Scenario", value_name="Score")
    fig_dims = px.bar(dim_melt, x="Score", y="Dimension", color="Scenario", barmode="group", orientation="h",
                       color_discrete_map={"Before": "#555", "After": "#4F8BF9"},
                       title="Health Score Dimensions - Before vs After", range_x=[0, 100])
    fig_dims.update_layout(height=280, margin=dict(t=40, b=20, l=160))

    fc_df = user_fc30.reset_index()
    fc_df.columns = ["category", "forecast"]
    fc_df["type"] = fc_df["category"].apply(lambda c: "Adjusted" if c == category else "Unchanged")
    fc_df.loc[fc_df["category"] == category, "forecast"] = adjusted_cat_fc
    fc_df = fc_df.sort_values("forecast")
    fig_fc = px.bar(fc_df, x="forecast", y="category", color="type", orientation="h",
                     color_discrete_map={"Unchanged": "#4F8BF9", "Adjusted": "#e67e22"},
                     title="30-Day Forecast by Category (after adjustment)")
    fig_fc.update_layout(height=400, margin=dict(l=160, t=50, b=20))

    explanation = []
    if category in essentials:
        explanation.append(f"Essentials ratio: {float(h['essentials_ratio']):.1%} -> {new_ess_ratio:.1%}")
    explanation.append(
        f"Savings potential: gap {float(h['savings_gap']):+.1%} -> {new_savings_gap:+.1%}. "
        f"Forecast stays fixed at ${original_forecast30:,.0f}/mo; actual spend moves from "
        f"${actual_avg_monthly:,.0f} to ${adj_actual_monthly:,.0f}/mo."
    )

    return html.Div([
        html.H4(f"Scenario: {category} at {multiplier:.2f}x baseline"),
        kpis, html.Hr(),
        dcc.Graph(figure=fig_dims),
        dcc.Graph(figure=fig_fc),
        html.Details([
            html.Summary("How the score was recalculated"),
            html.Ul([html.Li(e) for e in explanation]),
        ]),
    ])
