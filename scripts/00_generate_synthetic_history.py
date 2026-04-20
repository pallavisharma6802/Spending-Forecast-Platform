"""
Extends the real 2023-2024 transaction dataset backwards to 2020,
adding realistic category-level seasonality, COVID-era suppression,
and year-over-year growth so Prophet has 5 years of signal to learn from.
"""

import pandas as pd
import numpy as np

SEED = 42
rng  = np.random.default_rng(SEED)

# ── Load real data ────────────────────────────────────────────────────────────
REAL_CSV = "data/spending_patterns_detailed.csv"
OUT_CSV  = "data/spending_patterns_5yr.csv"

real = pd.read_csv(REAL_CSV, parse_dates=["Transaction Date"])
real = real.rename(columns={
    "Customer ID": "customer_id", "Category": "category",
    "Item": "item", "Quantity": "quantity",
    "Price Per Unit": "price_per_unit", "Total Spent": "total_spent",
    "Payment Method": "payment_method", "Location": "location",
    "Transaction Date": "transaction_date",
})

# ── Category metadata ─────────────────────────────────────────────────────────
ITEMS = {
    "Fitness":               ["Yoga Class", "Personal Trainer", "Workout Equipment", "Gym Supplement"],
    "Food":                  ["Fast Food", "Restaurant Meal", "Coffee", "Takeout"],
    "Friend Activities":     ["Dinner with Friends", "Movie Tickets", "Concert Tickets", "Bar Night"],
    "Gifts":                 ["Flowers", "Kids Games", "Jewelry", "Gift Cards", "Birthday Gift"],
    "Groceries":             ["Milk", "Bread", "Chicken", "Snacks", "Vegetables", "Eggs"],
    "Hobbies":               ["Crochet Supplies", "Art Supplies", "Books", "Video Games", "Board Games"],
    "Housing and Utilities": ["Water Bill", "Gas Bill", "Electricity Bill", "Rent", "Internet Bill"],
    "Medical/Dental":        ["Dentist Visit", "Doctor Visit", "Medicine", "Eye Exam"],
    "Personal Hygiene":      ["Toothpaste", "Shampoo", "Skin Care Products", "Soap", "Razor"],
    "Shopping":              ["Car", "Shoes", "Clothes", "Electronics", "Furniture"],
    "Subscriptions":         ["Streaming Service", "Magazine", "Gym Membership", "Software Sub"],
    "Transportation":        ["Car Repair", "Public Transit", "Gas", "Parking"],
    "Travel":                ["Plane Ticket", "Taxi/Uber", "Hotel Stay", "Vacation Package"],
}

PAYMENT_METHODS = ["Debit Card", "Digital Wallet", "Cash", "Credit Card"]
LOCATIONS       = ["Mobile App", "In-store", "Online"]

# Monthly seasonality multiplier [Jan..Dec] per category
# Values > 1 = busier than average, < 1 = quieter
SEASONALITY = {
    "Fitness": [1.9, 1.4, 1.1, 1.0, 1.0, 0.9, 0.8, 0.8, 1.0, 1.0, 0.9, 0.8],
    "Food":    [0.9, 0.9, 1.0, 1.0, 1.1, 1.1, 1.1, 1.0, 1.0, 1.0, 1.1, 1.2],
    "Friend Activities": [0.7, 1.2, 1.0, 1.1, 1.2, 1.3, 1.3, 1.2, 1.0, 1.0, 0.9, 1.1],
    "Gifts":   [0.4, 1.6, 0.7, 0.8, 1.3, 0.9, 0.8, 0.8, 0.9, 1.0, 1.2, 2.6],
    "Groceries": [0.9, 0.9, 1.0, 1.0, 1.0, 1.1, 1.1, 1.1, 1.0, 1.0, 1.3, 1.3],
    "Hobbies": [0.8, 0.8, 0.9, 1.1, 1.2, 1.3, 1.3, 1.2, 1.0, 1.0, 0.9, 0.8],
    "Housing and Utilities": [1.2, 1.2, 1.0, 0.9, 0.9, 1.0, 1.2, 1.2, 1.0, 0.9, 1.0, 1.1],
    "Medical/Dental": [1.4, 1.1, 1.0, 1.0, 1.0, 0.9, 0.9, 0.9, 1.0, 1.0, 1.0, 1.2],
    "Personal Hygiene": [1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0],
    "Shopping": [0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.1, 1.0, 1.1, 1.2, 2.0, 2.2],
    "Subscriptions": [1.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.0, 1.0, 1.0],
    "Transportation": [0.9, 0.9, 1.0, 1.0, 1.0, 1.1, 1.2, 1.1, 1.0, 1.0, 1.0, 1.1],
    "Travel":  [0.6, 0.7, 0.9, 1.1, 1.2, 1.6, 1.9, 1.8, 1.1, 0.9, 0.7, 1.2],
}

# Year-over-year base multiplier (relative to 2023 baseline)
# Captures inflation + spending growth + COVID suppression in 2020
YEAR_MULTIPLIER = {
    2020: 0.78,   # COVID year — overall suppression
    2021: 0.87,   # partial recovery
    2022: 0.94,   # near normal
    # 2023-2024 already in real data
}

# COVID-era per-category adjustments (multiplicative on top of year multiplier)
# Applied monthly within 2020 and early 2021
COVID_OVERRIDE = {
    # (year, month): {category: multiplier}
    (2020, 3):  {"Travel": 0.3, "Friend Activities": 0.3, "Fitness": 0.4, "Food": 0.5},
    (2020, 4):  {"Travel": 0.05,"Friend Activities": 0.1, "Fitness": 0.2, "Food": 0.3, "Groceries": 1.6, "Subscriptions": 1.7},
    (2020, 5):  {"Travel": 0.08,"Friend Activities": 0.15,"Fitness": 0.25,"Food": 0.4, "Groceries": 1.5, "Subscriptions": 1.6},
    (2020, 6):  {"Travel": 0.15,"Friend Activities": 0.3, "Fitness": 0.4, "Food": 0.6, "Groceries": 1.3, "Subscriptions": 1.4},
    (2020, 7):  {"Travel": 0.25,"Friend Activities": 0.5, "Fitness": 0.6},
    (2020, 8):  {"Travel": 0.3, "Friend Activities": 0.6, "Fitness": 0.7},
    (2020, 9):  {"Travel": 0.35,"Friend Activities": 0.65},
    (2020, 10): {"Travel": 0.4, "Friend Activities": 0.7, "Fitness": 0.8},
    (2020, 11): {"Travel": 0.3, "Friend Activities": 0.5},
    (2020, 12): {"Travel": 0.3, "Friend Activities": 0.5},
    (2021, 1):  {"Travel": 0.3, "Friend Activities": 0.5, "Fitness": 0.7},
    (2021, 2):  {"Travel": 0.35,"Friend Activities": 0.6},
    (2021, 3):  {"Travel": 0.5, "Friend Activities": 0.7},
    (2021, 4):  {"Travel": 0.6, "Friend Activities": 0.8},
}

# ── Compute per user-category statistics from real data ───────────────────────
# We'll use the 2023-2024 data as the calibration baseline
stats = (
    real.groupby(["customer_id", "category"])
    .agg(
        n_txn       = ("total_spent", "count"),
        mean_spend  = ("total_spent", "mean"),
        std_spend   = ("total_spent", "std"),
    )
    .reset_index()
)
# annualise: real data covers ~2 years
stats["annual_txn"] = stats["n_txn"] / 2.0
stats["std_spend"]  = stats["std_spend"].fillna(stats["mean_spend"] * 0.3)

users      = real["customer_id"].unique().tolist()
categories = list(ITEMS.keys())

# ── Generate synthetic rows year by year ──────────────────────────────────────
synthetic_rows = []

for year in [2020, 2021, 2022]:
    year_mult = YEAR_MULTIPLIER[year]
    print(f"Generating {year}...")

    for _, row in stats.iterrows():
        uid      = row["customer_id"]
        cat      = row["category"]
        base_n   = row["annual_txn"]
        mu       = row["mean_spend"]
        sigma    = row["std_spend"]

        items_pool = ITEMS.get(cat, ["Item"])
        season     = SEASONALITY.get(cat, [1.0] * 12)

        for month in range(1, 13):
            month_mult = season[month - 1]

            # apply COVID override if applicable
            covid_adj = COVID_OVERRIDE.get((year, month), {}).get(cat, 1.0)
            effective_mult = year_mult * month_mult * covid_adj

            # expected transactions this month (Poisson)
            expected_n = base_n / 12.0 * effective_mult
            n_txn = int(rng.poisson(max(expected_n, 0)))

            if n_txn == 0:
                continue

            # generate transaction dates spread across the month
            days_in_month = pd.Timestamp(year, month, 1).days_in_month
            days = sorted(rng.choice(days_in_month, size=n_txn, replace=False) + 1)

            for day in days:
                # log-normal spend — mu/sigma from real data, scaled by seasonality
                scaled_mu    = mu * month_mult * covid_adj
                scaled_sigma = max(sigma * month_mult * covid_adj, scaled_mu * 0.1)
                # convert to log-normal params
                cv2       = (scaled_sigma / scaled_mu) ** 2
                log_mu    = np.log(scaled_mu / np.sqrt(1 + cv2))
                log_sigma = np.sqrt(np.log(1 + cv2))
                amount    = round(float(rng.lognormal(log_mu, log_sigma)), 2)
                amount    = max(0.50, amount)

                qty   = int(rng.choice([1, 1, 1, 2, 2, 3], p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05]))
                price = round(amount / qty, 2)

                synthetic_rows.append({
                    "customer_id":      uid,
                    "category":         cat,
                    "item":             rng.choice(items_pool),
                    "quantity":         qty,
                    "price_per_unit":   price,
                    "total_spent":      amount,
                    "payment_method":   rng.choice(PAYMENT_METHODS),
                    "location":         rng.choice(LOCATIONS),
                    "transaction_date": pd.Timestamp(year, month, int(day)),
                })

synth_df = pd.DataFrame(synthetic_rows)
print(f"Synthetic rows generated: {len(synth_df):,}")

# ── Combine with real data ────────────────────────────────────────────────────
real_renamed = real.rename(columns={
    "customer_id": "customer_id", "category": "category",
    "item": "item", "quantity": "quantity",
    "price_per_unit": "price_per_unit", "total_spent": "total_spent",
    "payment_method": "payment_method", "location": "location",
    "transaction_date": "transaction_date",
})

combined = pd.concat([synth_df, real_renamed], ignore_index=True)
combined  = combined.sort_values("transaction_date").reset_index(drop=True)

# Rename back to original column names for pipeline compatibility
combined = combined.rename(columns={
    "customer_id": "Customer ID", "category": "Category",
    "item": "Item", "quantity": "Quantity",
    "price_per_unit": "Price Per Unit", "total_spent": "Total Spent",
    "payment_method": "Payment Method", "location": "Location",
    "transaction_date": "Transaction Date",
})

combined.to_csv(OUT_CSV, index=False)
print(f"\nSaved {len(combined):,} rows → {OUT_CSV}")
print(f"Date range: {combined['Transaction Date'].min()} → {combined['Transaction Date'].max()}")
print(f"\nRows per year:")
combined["year"] = pd.to_datetime(combined["Transaction Date"]).dt.year
print(combined.groupby("year").size().to_string())
print(f"\nCategory row counts:")
print(combined.groupby("Category").size().sort_values(ascending=False).to_string())
