import json
import os
import pandas as pd
import plotly.express as px
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
    return users, categories, baseline, health, forecasts, caps, anomalies


users, categories, df_baseline, df_health, df_forecasts, df_caps, anomaly_data = load_all()
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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Spending Overview", "Prophet Forecasts", "Budget Recommendations", "Financial Health", "Anomaly Alerts"])

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
