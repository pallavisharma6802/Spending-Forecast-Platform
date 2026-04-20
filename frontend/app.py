import json
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Spending Forecast & Recommendation Platform", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@st.cache_data
def load_json(filename):
    with open(os.path.join(DATA_DIR, filename)) as f:
        return json.load(f)


@st.cache_data
def load_all():
    users      = load_json("users.json")
    categories = load_json("categories.json")
    baseline   = pd.DataFrame(load_json("baseline.json"))
    health     = pd.DataFrame(load_json("health_scores.json"))
    forecasts  = pd.DataFrame(load_json("forecasts.json"))
    caps       = pd.DataFrame(load_json("budget_caps.json"))
    anomalies  = load_json("anomalies.json")
    peer_raw   = load_json("peer_benchmarks.json")
    peer_index = {u["customer_id"]: u["categories"] for u in peer_raw.get("users", [])}
    return users, categories, baseline, health, forecasts, caps, anomalies, peer_index


users, categories, df_baseline, df_health, df_forecasts, df_caps, anomaly_data, peer_index = load_all()
all_alerts     = pd.DataFrame(anomaly_data.get("alerts", []))

# cast types once
df_baseline["total_spend"]         = df_baseline["total_spend"].astype(float)
df_baseline["avg_per_transaction"]  = df_baseline["avg_per_transaction"].astype(float)
df_baseline["num_transactions"]     = df_baseline["num_transactions"].astype(int)
df_baseline["max_30d_spend"]        = df_baseline["max_30d_spend"].astype(float)
df_forecasts["horizon_days"]        = df_forecasts["horizon_days"].astype(int)
df_forecasts["forecasted_spend"]    = df_forecasts["forecasted_spend"].astype(float)
for col in ["cf_predicted_spend", "own_forecast_30d", "recommended_budget_cap"]:
    df_caps[col] = df_caps[col].astype(float)

st.title("Spending Forecast & Recommendation Platform")
st.caption(
    "End-to-end pipeline: HDFS · Hive · Spark · Prophet · Collaborative Filtering. "
    "Forecasts and budget caps are pre-computed from 23K transactions across 200 users and 13 categories."
)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Customer")
    customer_id = st.selectbox("Select customer", users)
    st.markdown("---")
    st.markdown("**Stack**")
    st.markdown(
        "HDFS · Apache Hive · PySpark  \n"
        "Facebook Prophet · Collaborative Filtering  \n"
        "Apache Airflow · FastAPI · Streamlit"
    )
    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown("23,000 transactions · 200 users · 13 categories · 2020–2024")

# ── filter to selected user ───────────────────────────────────────────────────
user_baseline  = df_baseline[df_baseline["customer_id"] == customer_id]
user_forecasts = df_forecasts[df_forecasts["customer_id"] == customer_id]
user_caps      = df_caps[df_caps["customer_id"] == customer_id]

# ── top KPI strip ─────────────────────────────────────────────────────────────
total_hist  = user_baseline["total_spend"].sum()
total_fc30  = user_forecasts[user_forecasts["horizon_days"] == 30]["forecasted_spend"].sum()
top_category = (
    user_baseline.sort_values("total_spend", ascending=False)
    .iloc[0]["category"] if not user_baseline.empty else "—"
)
max_cap = user_caps["recommended_budget_cap"].max() if not user_caps.empty else 0

health_row = df_health[df_health["customer_id"] == customer_id]
health_score = f"{health_row.iloc[0]['score']:.0f} / 100 ({health_row.iloc[0]['grade']})" \
    if not health_row.empty else "—"

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Historical Spend", f"${total_hist:,.0f}")
k2.metric("30-Day Forecast (all categories)", f"${total_fc30:,.0f}")
k3.metric("Top Spending Category", top_category)
k4.metric("Financial Health Score", health_score)

st.markdown("---")

# ── tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Spending Overview", "Prophet Forecasts", "Budget Recommendations", "Financial Health", "Anomaly Alerts", "Peer Benchmarking", "What-If Scenarios"])

# ── Tab 1: historical baseline ────────────────────────────────────────────────
with tab1:
    st.subheader(f"Historical Spending — {customer_id}")

    if user_baseline.empty:
        st.info("No baseline data for this customer.")
    else:
        sorted_bl = user_baseline.sort_values("total_spend", ascending=False)

        fig = px.bar(
            sorted_bl,
            x="category", y="total_spend",
            color="category",
            title="Total Spend by Category (2023–2024)",
            labels={"total_spend": "Total Spend ($)", "category": "Category"},
        )
        fig.update_layout(showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            sorted_bl[["category", "total_spend", "avg_per_transaction",
                        "num_transactions", "max_30d_spend"]]
            .rename(columns={
                "total_spend": "Total Spend ($)",
                "avg_per_transaction": "Avg per Txn ($)",
                "num_transactions": "# Transactions",
                "max_30d_spend": "Peak 30d Spend ($)",
            })
            .reset_index(drop=True),
            use_container_width=True,
        )

# ── Tab 2: Prophet forecasts ──────────────────────────────────────────────────
with tab2:
    st.subheader(f"Prophet Forecasts — {customer_id}")
    st.caption(
        "Category-level Prophet (470–500 daily points per category). "
        "Each user's forecast = category forecast × spend share × behavior multiplier "
        "(recency · frequency · Q4 velocity)."
    )

    if user_forecasts.empty:
        st.info("No forecast data for this customer.")
    else:
        fig = px.bar(
            user_forecasts,
            x="category", y="forecasted_spend",
            color=user_forecasts["horizon_days"].astype(str),
            barmode="group",
            title="Forecasted Spend by Category & Horizon",
            labels={"forecasted_spend": "Forecasted Spend ($)", "color": "Horizon (days)"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(xaxis_tickangle=-30, legend_title="Horizon (days)")
        st.plotly_chart(fig, use_container_width=True)

        pivot = (
            user_forecasts
            .pivot_table(index="category", columns="horizon_days", values="forecasted_spend")
            .round(2)
        )
        pivot.columns = [f"{c}-day ($)" for c in pivot.columns]
        st.dataframe(pivot, use_container_width=True)

        with st.expander("Forecast accuracy (MAPE — train 2023, test 2024)"):
            mape_data = {
                "Category": ["Fitness", "Travel", "Food", "Transportation",
                              "Groceries", "Housing and Utilities"],
                "MAPE": ["41.5%", "55.0%", "76.0%", "89.2%", "94.9%", "96.9%"],
                "Note": ["Good", "Good", "Moderate", "Moderate", "Moderate", "Moderate"],
            }
            st.dataframe(pd.DataFrame(mape_data), use_container_width=True)
            st.caption(
                "Higher MAPE categories (Gifts, Personal Hygiene) are dominated by "
                "irregular large transactions — no model can reliably predict these from prior-year data."
            )

# ── Tab 3: budget recommendations ────────────────────────────────────────────
with tab3:
    st.subheader(f"Recommended Budget Caps — {customer_id}")
    st.caption(
        "User-based CF · 65-feature matrix (5 features × 13 categories) · "
        "top-10 neighbors by cosine similarity · "
        "cap = (60% own + 40% CF) × velocity × 1.15 buffer."
    )

    if user_caps.empty:
        st.info("No recommendation data for this customer.")
    else:
        sorted_caps = user_caps.sort_values("recommended_budget_cap", ascending=False)

        fig = px.bar(
            sorted_caps,
            x="category", y="recommended_budget_cap",
            color="category",
            title="Recommended Monthly Budget Cap by Category",
            labels={"recommended_budget_cap": "Budget Cap ($)"},
        )
        fig.update_layout(showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            fig2 = px.scatter(
                sorted_caps,
                x="own_forecast_30d", y="recommended_budget_cap",
                text="category", size="cf_predicted_spend",
                title="Own Forecast vs. Budget Cap",
                labels={
                    "own_forecast_30d": "Own Forecast ($)",
                    "recommended_budget_cap": "Budget Cap ($)",
                },
            )
            fig2.update_traces(textposition="top center")
            st.plotly_chart(fig2, use_container_width=True)

        with col_b:
            st.dataframe(
                sorted_caps[["category", "own_forecast_30d", "cf_predicted_spend",
                              "spend_velocity", "recommended_budget_cap"]]
                .rename(columns={
                    "own_forecast_30d": "Own Forecast ($)",
                    "cf_predicted_spend": "CF Predicted ($)",
                    "spend_velocity": "Velocity",
                    "recommended_budget_cap": "Budget Cap ($)",
                })
                .reset_index(drop=True),
                use_container_width=True,
            )

# ── Tab 4: financial health score ─────────────────────────────────────────────
with tab4:
    user_health = df_health[df_health["customer_id"] == customer_id]

    if user_health.empty:
        st.info("No health score for this customer.")
    else:
        h = user_health.iloc[0]
        score = float(h["score"])
        grade = h["grade"]
        label = h["label"]

        GRADE_COLOR = {"A": "#2ecc71", "B": "#27ae60", "C": "#f39c12",
                       "D": "#e67e22", "F": "#e74c3c"}
        color = GRADE_COLOR.get(grade, "#95a5a6")

        st.subheader(f"Financial Health — {customer_id}")
        st.caption(
            "Composite score across four behavioural dimensions: "
            "spending stability, essentials coverage, month-over-month volatility, "
            "and savings potential vs forecast."
        )

        # ── score hero ───────────────────────────────────────────────────────
        col_score, col_dims = st.columns([1, 2])

        with col_score:
            st.markdown(
                f"""
                <div style="text-align:center; padding:24px; border-radius:12px;
                            background:{color}22; border: 2px solid {color};">
                    <div style="font-size:72px; font-weight:800; color:{color};">{score:.0f}</div>
                    <div style="font-size:28px; font-weight:700; color:{color};">Grade {grade}</div>
                    <div style="font-size:16px; color:#FAFAFA; margin-top:6px;">{label}</div>
                    <div style="font-size:12px; color:#aaa; margin-top:4px;">out of 100</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("")
            st.markdown("**Score distribution (all 200 users)**")
            fig_hist = px.histogram(
                df_health, x="score", nbins=20,
                color_discrete_sequence=["#4F8BF9"],
                labels={"score": "Health Score"},
            )
            fig_hist.add_vline(x=score, line_dash="dash", line_color=color,
                               annotation_text="You", annotation_position="top right")
            fig_hist.update_layout(showlegend=False, margin=dict(t=20, b=20, l=10, r=10),
                                   height=220, bargap=0.05)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_dims:
            dims = {
                "Stability":         float(h["stability_score"]),
                "Essentials Ratio":  float(h["essentials_score"]),
                "Low Volatility":    float(h["volatility_score"]),
                "Savings Potential": float(h["savings_score"]),
            }
            dim_df = pd.DataFrame({
                "Dimension": list(dims.keys()),
                "Score":     [round(v * 100, 1) for v in dims.values()],
            })

            fig_dims = px.bar(
                dim_df, x="Score", y="Dimension", orientation="h",
                color="Score", color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
                range_color=[0, 100],
                title="Dimension Breakdown (0–100)",
                labels={"Score": "Score (0–100)"},
            )
            fig_dims.update_layout(coloraxis_showscale=False,
                                   yaxis={"categoryorder": "array",
                                          "categoryarray": list(dims.keys())[::-1]},
                                   xaxis_range=[0, 100], height=280,
                                   margin=dict(t=40, b=20))
            st.plotly_chart(fig_dims, use_container_width=True)

            # raw signal table
            st.markdown("**Underlying signals**")
            signals = pd.DataFrame([{
                "Signal":                "Spend CV (stability)",
                "Value":                 f"{float(h['spend_cv']):.2f}",
                "Interpretation":        "lower = more stable",
            }, {
                "Signal":                "Essentials ratio",
                "Value":                 f"{float(h['essentials_ratio']):.0%}",
                "Interpretation":        "ideal ≈ 50% on necessities",
            }, {
                "Signal":                "Avg MoM volatility",
                "Value":                 f"{float(h['mom_volatility']):.1f}×",
                "Interpretation":        "lower = smoother month-to-month",
            }, {
                "Signal":                "Savings gap",
                "Value":                 f"{float(h['savings_gap']):+.0%}",
                "Interpretation":        "positive = spending below forecast",
            }])
            st.dataframe(signals, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── population leaderboard ────────────────────────────────────────────
        st.markdown("**Where does this user rank?**")
        rank      = int((df_health["score"] < score).sum()) + 1
        pct       = round((1 - (rank - 1) / len(df_health)) * 100, 1)
        grade_dist = df_health["grade"].value_counts().sort_index()

        r1, r2, r3 = st.columns(3)
        r1.metric("Rank", f"#{rank} of {len(df_health)}")
        r2.metric("Better than", f"{100 - pct:.0f}% of users")
        r3.metric("Population median score", f"{df_health['score'].median():.1f}")

        fig_grades = px.bar(
            x=grade_dist.index, y=grade_dist.values,
            color=grade_dist.index,
            color_discrete_map=GRADE_COLOR,
            labels={"x": "Grade", "y": "Users"},
            title="Grade distribution across all 200 users",
        )
        fig_grades.update_layout(showlegend=False, height=280,
                                 margin=dict(t=40, b=20))
        st.plotly_chart(fig_grades, use_container_width=True)

# ── Tab 5: anomaly alerts ─────────────────────────────────────────────────────
with tab5:
    meta = anomaly_data
    st.subheader(f"Anomaly Alerts — {customer_id}")
    st.caption(
        f"Mid-month spend pace check for {meta.get('current_month','—')} · "
        f"{meta.get('days_elapsed','?')}/{meta.get('days_in_month','?')} days elapsed · "
        f"projected to month-end via {meta.get('pace_factor','?')}× pace factor. "
        "Alerts require both a statistical signal (Z-score > 2.5 or Isolation Forest) "
        "and a projected budget overage > 25%."
    )

    SEV_COLOR  = {"high": "#e74c3c", "medium": "#e67e22", "low": "#f1c40f"}
    SEV_EMOJI  = {"high": "🔴", "medium": "🟠", "low": "🟡"}

    user_alerts = all_alerts[all_alerts["customer_id"] == customer_id] \
        if not all_alerts.empty else pd.DataFrame()

    # ── user-level alert cards ────────────────────────────────────────────────
    if user_alerts.empty:
        st.success("No anomalies detected for this customer in the current period.")
    else:
        st.warning(f"{len(user_alerts)} alert{'s' if len(user_alerts)>1 else ''} detected for {customer_id}")
        for _, alert in user_alerts.sort_values("overage_pct", ascending=False).iterrows():
            sev   = alert["severity"]
            color = SEV_COLOR.get(sev, "#888")
            emoji = SEV_EMOJI.get(sev, "⚪")
            with st.container():
                st.markdown(
                    f"""<div style="border-left:4px solid {color}; padding:10px 16px;
                        margin-bottom:10px; border-radius:4px; background:{color}18;">
                    <b>{emoji} {alert['category']}</b> &nbsp;·&nbsp;
                    <span style="color:{color}; font-weight:600;">{sev.upper()}</span><br>
                    <span style="font-size:0.9em;">{alert['message']}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Spent so far",    f"${alert['current_spend']:,.0f}")
                c2.metric("Projected (30d)", f"${alert['projected_spend']:,.0f}")
                c3.metric("Budget cap",      f"${alert['budget_cap']:,.0f}")
                c4.metric("Overage",         f"+{alert['overage_pct']:.0f}%",
                          delta=f"Z = {alert['z_score']:+.1f}σ",
                          delta_color="inverse")

    st.markdown("---")

    # ── platform-wide summary ─────────────────────────────────────────────────
    st.markdown("**Platform-wide alert summary**")

    if all_alerts.empty:
        st.info("No alerts in current period.")
    else:
        a1, a2, a3 = st.columns(3)
        a1.metric("Total alerts",          len(all_alerts))
        a2.metric("Users affected",        all_alerts["customer_id"].nunique())
        a3.metric("High-severity alerts",  len(all_alerts[all_alerts["severity"] == "high"]))

        col_cat, col_sev = st.columns(2)

        with col_cat:
            cat_counts = all_alerts["category"].value_counts().reset_index()
            cat_counts.columns = ["category", "alerts"]
            fig_cat = px.bar(
                cat_counts.head(10), x="alerts", y="category",
                orientation="h", title="Alerts by Category",
                color="alerts", color_continuous_scale="Reds",
            )
            fig_cat.update_layout(coloraxis_showscale=False, height=320,
                                  margin=dict(t=40, b=10), yaxis_title="")
            st.plotly_chart(fig_cat, use_container_width=True)

        with col_sev:
            sev_counts = all_alerts["severity"].value_counts().reset_index()
            sev_counts.columns = ["severity", "count"]
            fig_sev = px.pie(
                sev_counts, names="severity", values="count",
                title="Alerts by Severity",
                color="severity",
                color_discrete_map=SEV_COLOR,
            )
            fig_sev.update_layout(height=320, margin=dict(t=40, b=10))
            st.plotly_chart(fig_sev, use_container_width=True)

        # top overage table
        st.markdown("**Most at-risk users (projected overage)**")
        top_table = (
            all_alerts.nlargest(10, "overage_pct")
            [["customer_id", "category", "severity", "projected_spend",
              "budget_cap", "overage_pct", "z_score"]]
            .rename(columns={
                "customer_id":     "Customer",
                "category":        "Category",
                "severity":        "Severity",
                "projected_spend": "Projected ($)",
                "budget_cap":      "Cap ($)",
                "overage_pct":     "Overage %",
                "z_score":         "Z-score",
            })
            .reset_index(drop=True)
        )
        st.dataframe(top_table, use_container_width=True, hide_index=True)

# ── Tab 6: peer benchmarking ──────────────────────────────────────────────────
with tab6:
    st.subheader(f"Peer Benchmarking — {customer_id}")
    st.caption(
        "Compares your 30-day forecast against the weighted-average forecast of your "
        "top-10 most similar users (cosine similarity on 65-feature matrix). "
        "Category percentile is your rank among all 200 users by historical total spend."
    )

    peer_cats = peer_index.get(customer_id, [])

    if not peer_cats:
        st.info("No peer benchmark data for this customer.")
    else:
        peer_df = pd.DataFrame(peer_cats)
        peer_df["vs_peers_pct"] = peer_df["vs_peers_pct"].astype(float)

        above = peer_df[peer_df["direction"] == "above"].sort_values("vs_peers_pct", ascending=False)
        below = peer_df[peer_df["direction"] == "below"].sort_values("vs_peers_pct")
        neutral = peer_df[peer_df["direction"] == "neutral"]

        # ── KPI strip ─────────────────────────────────────────────────────────
        p1, p2, p3 = st.columns(3)
        p1.metric("Categories above peers", len(above))
        p2.metric("Categories below peers", len(below))
        p3.metric("Categories in line",     len(neutral))

        # ── waterfall bar chart ───────────────────────────────────────────────
        chart_df = peer_df.sort_values("vs_peers_pct", ascending=True).copy()
        chart_df["color"] = chart_df["direction"].map(
            {"above": "#e74c3c", "below": "#2ecc71", "neutral": "#95a5a6"}
        )
        fig_wf = go.Figure(go.Bar(
            x=chart_df["vs_peers_pct"],
            y=chart_df["category"],
            orientation="h",
            marker_color=chart_df["color"],
            text=chart_df["vs_peers_pct"].apply(lambda v: f"{v:+.0f}%"),
            textposition="outside",
        ))
        fig_wf.update_layout(
            title="Your Forecast vs Peer Average (30-day)",
            xaxis_title="Δ vs peers (%)",
            xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor="#555"),
            height=420,
            margin=dict(l=160, t=50, b=30, r=60),
        )
        st.plotly_chart(fig_wf, use_container_width=True)

        # ── own vs peer side-by-side bars ─────────────────────────────────────
        melt_df = peer_df.melt(
            id_vars="category",
            value_vars=["own_forecast_30d", "peer_forecast_30d"],
            var_name="source", value_name="amount",
        )
        melt_df["source"] = melt_df["source"].map({
            "own_forecast_30d":  "Your forecast",
            "peer_forecast_30d": "Peer average",
        })
        fig_comp = px.bar(
            melt_df,
            x="category", y="amount", color="source",
            barmode="group",
            color_discrete_map={"Your forecast": "#4F8BF9", "Peer average": "#f39c12"},
            title="Your 30-Day Forecast vs Peer Average",
            labels={"amount": "Forecasted Spend ($)", "category": "Category"},
        )
        fig_comp.update_layout(xaxis_tickangle=-30, legend_title="")
        st.plotly_chart(fig_comp, use_container_width=True)

        # ── insight cards ─────────────────────────────────────────────────────
        st.markdown("**Insights**")
        DIR_COLOR = {"above": "#e74c3c", "below": "#2ecc71", "neutral": "#95a5a6"}
        DIR_ICON  = {"above": "↑", "below": "↓", "neutral": "≈"}

        cols = st.columns(2)
        for i, row in enumerate(peer_df.sort_values("vs_peers_pct", ascending=False, key=abs).itertuples()):
            color = DIR_COLOR.get(row.direction, "#888")
            icon  = DIR_ICON.get(row.direction, "")
            pct_label = f"top {100 - int(row.category_percentile)}th pct" \
                if row.category_percentile >= 75 \
                else (f"bottom {int(row.category_percentile)}th pct"
                      if row.category_percentile <= 25
                      else f"{int(row.category_percentile)}th pct")
            with cols[i % 2]:
                st.markdown(
                    f"""<div style="border-left:3px solid {color}; padding:8px 12px;
                        margin-bottom:8px; border-radius:4px; background:{color}15;">
                        <b>{icon} {row.category}</b>
                        <span style="float:right; color:{color}; font-weight:700;">
                            {row.vs_peers_pct:+.0f}%</span><br>
                        <span style="font-size:0.85em; color:#ccc;">{row.insight}</span>
                        </div>""",
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ── full table ────────────────────────────────────────────────────────
        with st.expander("Full category table"):
            display_df = peer_df[[
                "category", "own_forecast_30d", "peer_forecast_30d",
                "vs_peers_pct", "category_percentile", "direction",
            ]].rename(columns={
                "own_forecast_30d":    "Your Forecast ($)",
                "peer_forecast_30d":   "Peer Avg ($)",
                "vs_peers_pct":        "Δ vs Peers (%)",
                "category_percentile": "Spend Percentile",
                "direction":           "Direction",
            }).sort_values("Δ vs Peers (%)", ascending=False).reset_index(drop=True)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

# ── Tab 7: what-if scenario modeling ─────────────────────────────────────────
with tab7:
    st.subheader(f"What-If Scenarios — {customer_id}")
    st.caption(
        "Adjust your expected spending in one category and see how it changes "
        "your 30-day projection and financial health score — entirely in-memory, no retraining."
    )

    ESSENTIALS = {"Groceries", "Housing and Utilities", "Medical/Dental"}

    # ── controls ──────────────────────────────────────────────────────────────
    ctrl_col, _ = st.columns([2, 3])
    with ctrl_col:
        wi_category = st.selectbox(
            "Category to adjust",
            options=sorted(categories),
            key="wi_cat",
        )
        wi_multiplier = st.slider(
            "Spending multiplier",
            min_value=0.1, max_value=3.0, value=1.0, step=0.05,
            format="%.2f×",
            help="1.0 = no change · 0.5 = spend 50% less · 2.0 = spend twice as much",
            key="wi_mult",
        )

    # ── baseline numbers for this user ───────────────────────────────────────
    user_fc30 = (
        df_forecasts[
            (df_forecasts["customer_id"] == customer_id) &
            (df_forecasts["horizon_days"] == 30)
        ]
        .set_index("category")["forecasted_spend"]
    )

    original_cat_fc = float(user_fc30.get(wi_category, 0.0))
    adjusted_cat_fc = original_cat_fc * wi_multiplier
    delta_fc        = adjusted_cat_fc - original_cat_fc

    original_total_fc = float(user_fc30.sum())
    adjusted_total_fc = original_total_fc + delta_fc

    user_h = df_health[df_health["customer_id"] == customer_id]
    if user_h.empty:
        st.warning("Health score data not available for this user.")
        st.stop()

    h = user_h.iloc[0]
    actual_avg_monthly  = float(h["actual_avg_monthly"])
    original_forecast30 = float(h["forecast_30d"])

    # The multiplier represents a change in *actual* spending behaviour in that
    # category.  The forecast (baseline expectation) stays fixed; what changes is
    # how much the user will actually spend, so the savings gap
    # = (forecast - new_actual) / forecast narrows or widens accordingly.
    adj_actual_monthly = actual_avg_monthly + delta_fc

    # ── recompute savings dimension ───────────────────────────────────────────
    def _savings_score(forecast, actual):
        gap     = np.clip((forecast - actual) / max(forecast, 1), -1, 1)
        raw     = (gap + 1) / 2
        all_gaps = df_health["savings_gap"].values
        raw_all  = (np.clip(all_gaps, -1, 1) + 1) / 2
        lo, hi   = raw_all.min(), raw_all.max()
        return float((raw - lo) / (hi - lo)) if hi > lo else 0.5, float(gap)

    new_norm_savings, new_savings_gap = _savings_score(original_forecast30, adj_actual_monthly)

    # ── recompute essentials dimension (only if adjusted cat is essential) ────
    def _essentials_score(ratio):
        raw     = max(0.0, 1.0 - 2.0 * abs(ratio - 0.5))
        all_ratios = df_health["essentials_ratio"].values
        raw_all    = np.maximum(0.0, 1.0 - 2.0 * np.abs(all_ratios - 0.5))
        lo, hi     = raw_all.min(), raw_all.max()
        return float((raw - lo) / (hi - lo)) if hi > lo else 0.5

    if wi_category in ESSENTIALS and not user_baseline.empty:
        # Scale the category's historical contribution to match the new spend rate.
        # Essentials ratio = essentials_spend / total_spend (from 5yr baseline).
        orig_cat_hist   = float(user_baseline[user_baseline["category"] == wi_category]["total_spend"].sum())
        orig_total_hist = float(user_baseline["total_spend"].sum())
        orig_ess_hist   = float(user_baseline[user_baseline["category"].isin(ESSENTIALS)]["total_spend"].sum())

        adj_cat_hist   = orig_cat_hist * wi_multiplier
        adj_ess_hist   = orig_ess_hist - orig_cat_hist + adj_cat_hist
        adj_total_hist = orig_total_hist - orig_cat_hist + adj_cat_hist
        new_ess_ratio  = adj_ess_hist / adj_total_hist if adj_total_hist > 0 else 0.0
        new_norm_ess   = _essentials_score(new_ess_ratio)
    else:
        new_norm_ess  = float(h["essentials_score"])
        new_ess_ratio = float(h["essentials_ratio"])

    # ── new composite score ───────────────────────────────────────────────────
    original_score = float(h["score"])
    new_score = (
        float(h["stability_score"]) +
        new_norm_ess +
        float(h["volatility_score"]) +
        new_norm_savings
    ) * 25.0

    def _grade(s):
        if s >= 80: return "A"
        if s >= 65: return "B"
        if s >= 50: return "C"
        if s >= 35: return "D"
        return "F"

    original_grade = h["grade"]
    new_grade      = _grade(new_score)
    score_delta    = new_score - original_score

    # ── KPI strip: before vs after ────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"#### Scenario: **{wi_category}** at **{wi_multiplier:.2f}×** baseline")

    col_orig, col_arrow, col_new = st.columns([5, 1, 5])

    GRADE_COLOR = {"A": "#2ecc71", "B": "#27ae60", "C": "#f39c12",
                   "D": "#e67e22", "F": "#e74c3c"}

    with col_orig:
        st.markdown("**Baseline**")
        m1, m2 = st.columns(2)
        m1.metric("Monthly actual avg", f"${actual_avg_monthly:,.0f}")
        m2.metric("Health score", f"{original_score:.1f} ({original_grade})")

    with col_arrow:
        st.markdown("<div style='text-align:center; font-size:2rem; padding-top:28px;'>→</div>",
                    unsafe_allow_html=True)

    with col_new:
        st.markdown("**Scenario**")
        m3, m4 = st.columns(2)
        m3.metric(
            "Monthly actual avg", f"${adj_actual_monthly:,.0f}",
            delta=f"${delta_fc:+,.0f}",
            delta_color="inverse",
        )
        score_sign = "+" if score_delta >= 0 else ""
        m4.metric(
            "Health score", f"{new_score:.1f} ({new_grade})",
            delta=f"{score_sign}{score_delta:.1f} pts",
            delta_color="normal" if score_delta >= 0 else "inverse",
        )

    st.markdown("---")

    # ── dimension breakdown: before vs after ──────────────────────────────────
    dim_data = pd.DataFrame([
        {"Dimension": "Stability",        "Before": float(h["stability_score"]) * 100,  "After": float(h["stability_score"]) * 100},
        {"Dimension": "Essentials Ratio", "Before": float(h["essentials_score"]) * 100, "After": new_norm_ess * 100},
        {"Dimension": "Low Volatility",   "Before": float(h["volatility_score"]) * 100, "After": float(h["volatility_score"]) * 100},
        {"Dimension": "Savings Potential","Before": float(h["savings_score"]) * 100,    "After": new_norm_savings * 100},
    ])

    dim_melt = dim_data.melt(id_vars="Dimension", var_name="Scenario", value_name="Score")
    fig_dims = px.bar(
        dim_melt, x="Score", y="Dimension", color="Scenario",
        barmode="group", orientation="h",
        color_discrete_map={"Before": "#555", "After": "#4F8BF9"},
        title="Health Score Dimensions — Before vs After",
        labels={"Score": "Dimension Score (0–100)"},
        range_x=[0, 100],
    )
    fig_dims.update_layout(height=280, margin=dict(t=40, b=20, l=160),
                           legend_title="", yaxis_title="")
    st.plotly_chart(fig_dims, use_container_width=True)

    # ── category forecast waterfall ───────────────────────────────────────────
    fc_df = user_fc30.reset_index()
    fc_df.columns = ["category", "forecast"]
    fc_df["type"] = fc_df["category"].apply(
        lambda c: "Adjusted" if c == wi_category else "Unchanged"
    )
    fc_df.loc[fc_df["category"] == wi_category, "forecast"] = adjusted_cat_fc
    fc_df = fc_df.sort_values("forecast", ascending=True)

    fig_fc = px.bar(
        fc_df, x="forecast", y="category", color="type", orientation="h",
        color_discrete_map={"Unchanged": "#4F8BF9", "Adjusted": "#e67e22"},
        title=f"30-Day Forecast by Category (after adjustment)",
        labels={"forecast": "Forecasted Spend ($)", "category": ""},
    )
    fig_fc.update_layout(height=400, legend_title="", margin=dict(l=160, t=50, b=20))
    st.plotly_chart(fig_fc, use_container_width=True)

    # ── what changed and why ──────────────────────────────────────────────────
    with st.expander("How the score was recalculated"):
        changed_dims = []
        if wi_category in ESSENTIALS:
            changed_dims.append(
                f"**Essentials ratio**: {float(h['essentials_ratio']):.1%} → {new_ess_ratio:.1%}  \n"
                f"_Ideal ≈ 50% on essentials (Housing, Groceries, Medical). "
                f"Score peaks there and falls away on both sides._"
            )
        changed_dims.append(
            f"**Savings potential**: gap {float(h['savings_gap']):+.1%} → {new_savings_gap:+.1%}  \n"
            f"_Forecast stays fixed at ${original_forecast30:,.0f}/mo (the model's expectation). "
            f"Actual spend moves from ${actual_avg_monthly:,.0f} → ${adj_actual_monthly:,.0f}/mo. "
            f"A wider gap (spending below forecast) signals more headroom to save._"
        )
        st.markdown("\n\n".join(changed_dims))
        st.info(
            "Stability and volatility are derived from historical monthly patterns "
            "and stay fixed in this scenario — they require months of real data to move."
        )
