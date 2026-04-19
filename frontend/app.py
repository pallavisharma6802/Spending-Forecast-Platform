import json
import os
import pandas as pd
import plotly.express as px
import streamlit as st

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
    forecasts  = pd.DataFrame(load_json("forecasts.json"))
    caps       = pd.DataFrame(load_json("budget_caps.json"))
    return users, categories, baseline, forecasts, caps


users, categories, df_baseline, df_forecasts, df_caps = load_all()

# cast types once
df_baseline["total_spend"]         = df_baseline["total_spend"].astype(float)
df_baseline["avg_per_transaction"]  = df_baseline["avg_per_transaction"].astype(float)
df_baseline["num_transactions"]     = df_baseline["num_transactions"].astype(int)
df_baseline["max_30d_spend"]        = df_baseline["max_30d_spend"].astype(float)
df_forecasts["horizon_days"]        = df_forecasts["horizon_days"].astype(int)
df_forecasts["forecasted_spend"]    = df_forecasts["forecasted_spend"].astype(float)
for col in ["cf_predicted_spend", "own_forecast_30d", "recommended_budget_cap"]:
    df_caps[col] = df_caps[col].astype(float)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Spending Forecast & Recommendation Platform", layout="wide")

st.title("Spending Forecast & Recommendation Platform")
st.caption(
    "End-to-end pipeline: HDFS · Hive · Spark · Prophet · Collaborative Filtering · Airflow. "
    "Forecasts and budget caps are pre-computed from 10K transactions across 200 users and 13 categories."
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
    st.markdown("10,000 transactions · 200 users · 13 categories · 2023–2024")

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

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Historical Spend", f"${total_hist:,.0f}")
k2.metric("30-Day Forecast (all categories)", f"${total_fc30:,.0f}")
k3.metric("Top Spending Category", top_category)
k4.metric("Highest Budget Cap", f"${max_cap:,.0f}")

st.markdown("---")

# ── tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Spending Overview", "Prophet Forecasts", "Budget Recommendations"])

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
