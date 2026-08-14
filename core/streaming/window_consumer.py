"""Incremental rolling-window state, built from a stream of transaction
events rather than recomputed from scratch. Same code runs as a perpetual
background thread inside the deployed app (against a managed broker) and as
its own always-on container in the local docker-compose pipeline (against
Redpanda) - only which EventBus it's handed differs.

State is a small SQLite table (not Postgres/DuckDB - this is genuinely tiny
data, one row per transaction event) so it survives a process restart.
"""

import os
import sqlite3
import threading
from datetime import datetime, timedelta

from core.streaming.event_bus import EventBus

DB_PATH = os.environ.get(
    "WINDOW_STATE_DB", os.path.join(os.path.dirname(__file__), "..", "..", "window_state.db")
)
WINDOW_DAYS = (7, 30, 365)

_write_lock = threading.Lock()


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS transaction_events (
            customer_id TEXT NOT NULL,
            category TEXT NOT NULL,
            amount REAL NOT NULL,
            txn_date TEXT NOT NULL
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_customer_category ON transaction_events(customer_id, category)"
    )
    conn.commit()
    return conn


def ingest_event(conn: sqlite3.Connection, event: dict) -> None:
    with _write_lock:
        conn.execute(
            "INSERT INTO transaction_events (customer_id, category, amount, txn_date) VALUES (?, ?, ?, ?)",
            (event["customer_id"], event["category"], float(event["total_spent"]), event["transaction_date"]),
        )
        conn.commit()


def get_rolling_windows(
    conn: sqlite3.Connection, customer_id: str, category: str, as_of: datetime | None = None
) -> dict[int, float]:
    as_of = as_of or datetime.utcnow()
    out = {}
    for days in WINDOW_DAYS:
        start = (as_of - timedelta(days=days - 1)).date().isoformat()
        row = conn.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM transaction_events "
            "WHERE customer_id = ? AND category = ? AND txn_date >= ?",
            (customer_id, category, start),
        ).fetchone()
        out[days] = float(row[0])
    return out


def run_consumer(event_bus: EventBus, topic: str = "transactions.raw", group_id: str = "window-consumer") -> None:
    """Blocking. Run in a background thread (deployed app) or its own
    process (local `stream-consumer` service)."""
    conn = get_connection()
    for event in event_bus.subscribe(topic, group_id=group_id):
        try:
            ingest_event(conn, event)
        except (KeyError, ValueError, TypeError):
            continue


def start_background_consumer(event_bus: EventBus, topic: str = "transactions.raw") -> threading.Thread:
    thread = threading.Thread(target=run_consumer, args=(event_bus, topic), daemon=True)
    thread.start()
    return thread
