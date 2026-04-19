import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = "http://api:8000"

st.set_page_config(page_title="Fintech Spending Analyzer", layout="wide")
st.title("Fintech Spending Analyzer")

# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Customer")
    try:
        users = requests.get(f"{API_URL}/users", timeout=5).json()
    except Exception:
        users = []
        st.error("Cannot connect to API at " + API_URL)

    if users:
        customer_id = st.selectbox("Select customer", users)
    else:
        customer_id = st.text_input("Customer ID", value="C_001")

    if st.button("Reload pipeline data"):
        try:
            requests.post(f"{API_URL}/reload", timeout=30)
            st.success("Cache refreshed")
        except Exception as e:
            st.error(str(e))

# ── tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Spending Overview", "Prophet Forecasts", "Budget Recommendations"])

# ── Tab 1: historical baseline ────────────────────────────────────────────────
with tab1:
    st.subheader(f"Historical Spending — {customer_id}")
    try:
        data = requests.get(f"{API_URL}/users/{customer_id}/baseline", timeout=10).json()
        df = pd.DataFrame(data)
        if not df.empty:
            df["total_spend"] = df["total_spend"].astype(float)
            df["avg_per_transaction"] = df["avg_per_transaction"].astype(float)
            df["num_transactions"] = df["num_transactions"].astype(int)

            fig = px.bar(
                df.sort_values("total_spend", ascending=False),
                x="category", y="total_spend",
                color="category",
                title="Total Spend by Category",
                labels={"total_spend": "Total Spend ($)", "category": "Category"},
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                df[["category", "total_spend", "avg_per_transaction", "num_transactions",
                    "max_30d_spend"]]
                .sort_values("total_spend", ascending=False)
                .reset_index(drop=True),
                use_container_width=True,
            )
        else:
            st.info("No baseline data for this customer.")
    except Exception as e:
        st.error(f"Error loading baseline: {e}")

# ── Tab 2: Prophet forecasts ──────────────────────────────────────────────────
with tab2:
    st.subheader(f"Prophet Forecasts — {customer_id}")
    st.caption("Forecasted spend over 7, 15, and 30-day horizons using Facebook Prophet.")
    try:
        data = requests.get(f"{API_URL}/users/{customer_id}/forecasts", timeout=10).json()
        df = pd.DataFrame(data)
        if not df.empty:
            df["horizon_days"] = df["horizon_days"].astype(int)
            df["forecasted_spend"] = df["forecasted_spend"].astype(float)

            fig = px.bar(
                df,
                x="category", y="forecasted_spend",
                color=df["horizon_days"].astype(str),
                barmode="group",
                title="Forecasted Spend by Category & Horizon",
                labels={
                    "forecasted_spend": "Forecasted Spend ($)",
                    "color": "Horizon (days)",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

            pivot = (
                df.pivot_table(index="category", columns="horizon_days",
                               values="forecasted_spend")
                  .round(2)
            )
            pivot.columns = [f"{c}-day" for c in pivot.columns]
            st.dataframe(pivot, use_container_width=True)
        else:
            st.info("No forecast data for this customer.")
    except Exception as e:
        st.error(f"Error loading forecasts: {e}")

# ── Tab 3: collaborative-filter budget caps ───────────────────────────────────
with tab3:
    st.subheader(f"Recommended Budget Caps — {customer_id}")
    st.caption(
        "User-based collaborative filtering on Prophet 30-day forecasts: "
        "top-10 similar users (cosine similarity), blended 60% own / 40% CF, +15% buffer."
    )
    try:
        data = requests.get(f"{API_URL}/users/{customer_id}/budget-caps", timeout=10).json()
        df = pd.DataFrame(data)
        if not df.empty:
            for col in ["cf_predicted_spend", "own_forecast_30d", "recommended_budget_cap"]:
                df[col] = df[col].astype(float)

            fig = px.bar(
                df.sort_values("recommended_budget_cap", ascending=False),
                x="category", y="recommended_budget_cap",
                color="category",
                title="Recommended Monthly Budget Cap by Category",
                labels={"recommended_budget_cap": "Budget Cap ($)"},
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                df[["category", "own_forecast_30d", "cf_predicted_spend",
                    "recommended_budget_cap"]]
                .sort_values("recommended_budget_cap", ascending=False)
                .reset_index(drop=True),
                use_container_width=True,
            )
        else:
            st.info("No recommendation data for this customer.")
    except Exception as e:
        st.error(f"Error loading recommendations: {e}")
