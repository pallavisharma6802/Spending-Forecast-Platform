"""Normalize any transactions CSV (the project's own 5yr file, or an arbitrary
upload) into the canonical schema used by every core/ module:

    customer_id       string
    category          string
    total_spent       double
    transaction_date  date

Header matching is case/whitespace/punctuation-insensitive so common export
formats (e.g. "Amount", "Transaction Amount", "Date", "Posted Date") map
without the uploader needing to match this project's exact column names.
"""

import io
import re
from dataclasses import dataclass, field

import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, StringType, StructField, StructType, TimestampType

CANONICAL_COLUMNS = ["customer_id", "category", "total_spent", "transaction_date"]

_ALIASES = {
    "customer_id": {
        "customerid", "customer_id", "customer", "userid", "user_id", "user",
        "accountid", "account_id", "account", "clientid", "client_id", "id",
    },
    "category": {
        "category", "spendingcategory", "merchantcategory", "type",
        "transactiontype", "spendcategory",
    },
    "total_spent": {
        "totalspent", "total_spent", "amount", "amt", "transactionamount",
        "value", "price", "cost", "spend", "debit",
    },
    "transaction_date": {
        "transactiondate", "transaction_date", "date", "posteddate",
        "txndate", "txn_date", "postingdate",
    },
}

SPARK_SCHEMA = StructType([
    StructField("customer_id", StringType(), False),
    StructField("category", StringType(), False),
    StructField("total_spent", DoubleType(), False),
    StructField("transaction_date", TimestampType(), False),
])


class SchemaMappingError(ValueError):
    """Raised when a CSV's columns can't be mapped to the canonical schema."""


@dataclass
class LoadReport:
    n_rows_in: int
    n_rows_out: int
    n_dropped_bad_date: int = 0
    n_dropped_bad_amount: int = 0
    column_mapping: dict = field(default_factory=dict)


def _normalize_header(h: str) -> str:
    return re.sub(r"[^a-z0-9]", "", h.strip().lower())


def _infer_column_mapping(columns: list[str]) -> dict[str, str]:
    """Map raw column names -> canonical names. Raises if any canonical
    column has no match."""
    normalized = {_normalize_header(c): c for c in columns}
    mapping: dict[str, str] = {}
    for canonical, aliases in _ALIASES.items():
        hit = next((normalized[a] for a in aliases if a in normalized), None)
        if hit is None:
            raise SchemaMappingError(
                f"Could not find a column for '{canonical}' among: {columns}. "
                f"Expected one of: {sorted(aliases)}"
            )
        mapping[canonical] = hit
    return mapping


def normalize_dataframe(raw: pd.DataFrame) -> tuple[pd.DataFrame, LoadReport]:
    """Map an arbitrary raw pandas DataFrame onto the canonical schema."""
    n_in = len(raw)
    mapping = _infer_column_mapping(list(raw.columns))

    out = pd.DataFrame({
        "customer_id": raw[mapping["customer_id"]].astype(str).str.strip(),
        "category": raw[mapping["category"]].astype(str).str.strip(),
        "total_spent": pd.to_numeric(raw[mapping["total_spent"]], errors="coerce"),
        "transaction_date": pd.to_datetime(raw[mapping["transaction_date"]], errors="coerce"),
    })

    n_dropped_amount = int(out["total_spent"].isna().sum())
    n_dropped_date = int(out["transaction_date"].isna().sum())
    out = out.dropna(subset=["total_spent", "transaction_date"])
    out = out[out["total_spent"] > 0]

    report = LoadReport(
        n_rows_in=n_in,
        n_rows_out=len(out),
        n_dropped_bad_date=n_dropped_date,
        n_dropped_bad_amount=n_dropped_amount,
        column_mapping=mapping,
    )
    return out.reset_index(drop=True), report


def load_csv_path(spark: SparkSession, path: str) -> tuple[SparkDataFrame, LoadReport]:
    """Load the project's static CSV (or any local file at `path`) into a
    canonical Spark DataFrame."""
    raw = pd.read_csv(path)
    normalized, report = normalize_dataframe(raw)
    sdf = spark.createDataFrame(normalized, schema=SPARK_SCHEMA)
    return sdf, report


def load_uploaded_bytes(
    spark: SparkSession, content_bytes: bytes
) -> tuple[SparkDataFrame, LoadReport]:
    """Load an uploaded CSV's raw bytes (from a browser file picker) into a
    canonical Spark DataFrame."""
    raw = pd.read_csv(io.BytesIO(content_bytes))
    normalized, report = normalize_dataframe(raw)
    sdf = spark.createDataFrame(normalized, schema=SPARK_SCHEMA)
    return sdf, report
