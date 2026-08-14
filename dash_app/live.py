"""Runs the same core/ pipeline used to build the demo dataset, but against a
freshly uploaded file. Model selection (which model family per category) and
its tuned hyperparameters come from `model_config.json`, learned once from
the 200-user reference population by scripts/01 - an uploaded user doesn't
need enough history to backtest their own model, they just get served with
whatever already won each category's backtest. Cold start (CF neighbors,
health-score normalization, peer percentile) borrows the reference
population's precomputed artifacts the same way.
"""

import base64
import os

import pandas as pd

from core.anomaly import detect_anomalies
from core.data_loader import SchemaMappingError, load_uploaded_bytes
from core.forecast_engine import DEFAULT_HORIZONS, forecast, load_model_config
from core.health_score import compute_health_scores
from core.peer_bench import category_totals, compute_peer_benchmarks
from core.recommender import build_feature_matrix, budget_caps
from core.spark_session import get_spark

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "data")


class UploadError(Exception):
    pass


def _load_reference_artifacts():
    model_config = load_model_config(os.path.join(DATA_DIR, "model_config.json"))

    ref_fm_path = os.path.join(DATA_DIR, "reference_feature_matrix.csv")
    reference_matrix = pd.read_csv(ref_fm_path, index_col=0) if os.path.exists(ref_fm_path) else None

    ref_health_path = os.path.join(DATA_DIR, "reference_health_raw.csv")
    reference_health_raw = pd.read_csv(ref_health_path) if os.path.exists(ref_health_path) else None

    baseline_path = os.path.join(DATA_DIR, "baseline.json")
    reference_totals = None
    if os.path.exists(baseline_path):
        base = pd.read_json(baseline_path)
        if not base.empty:
            reference_totals = base[["customer_id", "category", "total_spend"]]

    return model_config, reference_matrix, reference_health_raw, reference_totals


def parse_upload_contents(contents: str) -> bytes:
    """`contents` is the base64 data-URI string dcc.Upload hands back."""
    try:
        _, content_string = contents.split(",", 1)
        return base64.b64decode(content_string)
    except Exception as exc:
        raise UploadError(f"Could not decode the uploaded file: {exc}") from exc


def publish_transactions(event_bus, sdf, topic: str = "transactions.raw") -> int:
    """Publishes every row to the event bus (real Kafka round trip when a
    broker is configured) so the streaming window_consumer can build its
    incremental rolling-window state from this upload. Buffered - one flush
    at the end, not one network round trip per row."""
    pdf = sdf.select("customer_id", "category", "total_spent", "transaction_date").toPandas()
    for row in pdf.itertuples(index=False):
        event_bus.publish(topic, {
            "customer_id": row.customer_id, "category": row.category,
            "total_spent": float(row.total_spent), "transaction_date": str(row.transaction_date.date()),
        })
    event_bus.flush()
    return len(pdf)


def run_pipeline(
    content_bytes: bytes,
    window_days: int | None = None,
    horizons: tuple = DEFAULT_HORIZONS,
    event_bus=None,
) -> dict:
    """Returns a dict with the same keys as DemoData.user_view expects, plus
    `customers` (all customer_ids found in the upload) and `load_report`."""
    spark = get_spark("dash-live")
    try:
        sdf, report = load_uploaded_bytes(spark, content_bytes)
    except SchemaMappingError as exc:
        raise UploadError(str(exc)) from exc

    if report.n_rows_out == 0:
        raise UploadError(
            "No valid rows after parsing - check that dates and amounts are in a "
            "recognizable format."
        )

    if window_days is not None:
        pdf_dates = sdf.select("transaction_date").toPandas()["transaction_date"]
        as_of = pd.to_datetime(pdf_dates).max()
        window_start = as_of - pd.Timedelta(days=window_days - 1)
        sdf = sdf.filter(sdf.transaction_date >= window_start)

    if event_bus is not None:
        publish_transactions(event_bus, sdf)

    model_config, reference_matrix, reference_health_raw, reference_totals = _load_reference_artifacts()

    fc = forecast(sdf, model_config, horizons=horizons)
    fm = build_feature_matrix(sdf, fc)
    caps = budget_caps(fm, fc, reference_matrix=reference_matrix)

    pdf_dates = sdf.select("transaction_date").toPandas()["transaction_date"]
    as_of = pd.to_datetime(pdf_dates).max()
    anomaly_window = min(30, max((as_of - pd.to_datetime(pdf_dates).min()).days + 1, 1))
    anomalies = detect_anomalies(
        sdf, as_of=as_of, window_days=anomaly_window, target_period_days=30, budget_caps=caps
    )

    health = compute_health_scores(sdf, fc, as_of=as_of, reference_raw=reference_health_raw)

    totals = category_totals(sdf)
    peer = compute_peer_benchmarks(caps, totals, reference_totals=reference_totals)
    peer_index = {u["customer_id"]: u["categories"] for u in peer.get("users", [])}

    customers = sorted(totals["customer_id"].unique().tolist())
    baseline_pdf = sdf.select("customer_id", "category", "total_spent", "transaction_date").toPandas()
    baseline = (
        baseline_pdf.groupby(["customer_id", "category"])
        .agg(total_spend=("total_spent", "sum"), avg_per_transaction=("total_spent", "mean"),
             num_transactions=("total_spent", "count"))
        .reset_index()
    )
    baseline["max_30d_spend"] = 0.0  # not needed for the upload summary view

    categories = sorted(totals["category"].unique().tolist())
    alerts = pd.DataFrame(anomalies.get("alerts", []))

    return {
        "customers": customers,
        "categories": categories,
        "load_report": report,
        "baseline_all": baseline,
        "forecasts_all": fc,
        "caps_all": caps,
        "health_all": health,
        "alerts_all": alerts,
        "anomaly_meta": anomalies,
        "peer_index": peer_index,
    }


def user_view(pipeline_result: dict, customer_id: str) -> dict:
    baseline, forecasts, caps, health_all, alerts_all = (
        pipeline_result["baseline_all"], pipeline_result["forecasts_all"], pipeline_result["caps_all"],
        pipeline_result["health_all"], pipeline_result["alerts_all"],
    )
    return {
        "customer_id": customer_id,
        "baseline": baseline[baseline["customer_id"] == customer_id],
        "forecasts": forecasts[forecasts["customer_id"] == customer_id],
        "caps": caps[caps["customer_id"] == customer_id],
        "health_row": health_all[health_all["customer_id"] == customer_id],
        "health_all": health_all,
        "alerts_user": alerts_all[alerts_all["customer_id"] == customer_id] if not alerts_all.empty else pd.DataFrame(),
        "alerts_all": alerts_all,
        "anomaly_meta": pipeline_result["anomaly_meta"],
        "peer_categories": pipeline_result["peer_index"].get(customer_id, []),
        "categories": pipeline_result["categories"],
    }
