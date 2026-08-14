"""EventBus: one real Kafka implementation, used identically by the deployed
app (against a managed broker - Upstash Kafka / Confluent Cloud) and the
local docker-compose pipeline (against a local Redpanda container) - only
the bootstrap servers/credentials differ. InMemoryEventBus is a
dev-convenience fallback for when no broker is configured at all (e.g.
before you've created a free Upstash cluster), so core/ and its tests don't
hard-fail without one.
"""

import json
import os
import queue
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator


class EventBus(ABC):
    @abstractmethod
    def publish(self, topic: str, message: dict) -> None:
        """Buffered - call flush() after a batch, not once per message."""

    @abstractmethod
    def flush(self) -> None: ...

    @abstractmethod
    def subscribe(self, topic: str, group_id: str) -> Iterator[dict]:
        """Blocking generator yielding messages published to `topic`."""


class InMemoryEventBus(EventBus):
    """Per-topic queue.Queue, in-process, zero external dependencies."""

    def __init__(self):
        self._topics: dict[str, queue.Queue] = {}
        self._lock = threading.Lock()

    def _get_queue(self, topic: str) -> queue.Queue:
        with self._lock:
            if topic not in self._topics:
                self._topics[topic] = queue.Queue()
            return self._topics[topic]

    def publish(self, topic: str, message: dict) -> None:
        self._get_queue(topic).put(message)

    def flush(self) -> None:
        pass

    def subscribe(self, topic: str, group_id: str = "default") -> Iterator[dict]:
        q = self._get_queue(topic)
        while True:
            yield q.get()


class KafkaEventBus(EventBus):
    """Real Kafka producer/consumer via kafka-python-ng. Works against any
    Kafka wire-compatible broker (Apache Kafka, Redpanda, Upstash, Confluent
    Cloud) - only the connection kwargs change."""

    def __init__(
        self,
        bootstrap_servers: list[str],
        security_protocol: str = "PLAINTEXT",
        sasl_mechanism: str | None = None,
        sasl_username: str | None = None,
        sasl_password: str | None = None,
    ):
        self._connection_kwargs = {"bootstrap_servers": bootstrap_servers, "security_protocol": security_protocol}
        if sasl_mechanism:
            self._connection_kwargs.update(
                sasl_mechanism=sasl_mechanism,
                sasl_plain_username=sasl_username,
                sasl_plain_password=sasl_password,
            )
        from kafka import KafkaProducer

        self._producer = KafkaProducer(
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            **self._connection_kwargs,
        )

    def publish(self, topic: str, message: dict) -> None:
        self._producer.send(topic, message)

    def flush(self) -> None:
        self._producer.flush()

    def subscribe(self, topic: str, group_id: str = "fintech-consumer") -> Iterator[dict]:
        from kafka import KafkaConsumer

        consumer = KafkaConsumer(
            topic,
            group_id=group_id,
            auto_offset_reset="earliest",
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            **self._connection_kwargs,
        )
        for record in consumer:
            yield record.value


def get_event_bus(max_retries: int = 5, retry_delay_seconds: float = 3.0) -> EventBus:
    """`KAFKA_BOOTSTRAP_SERVERS` unset -> InMemoryEventBus (dev fallback).
    Set -> real KafkaEventBus, works locally (Redpanda) or deployed (managed
    broker), same code either way.

    Retries with a short delay before giving up: docker-compose's
    `depends_on` only waits for the broker *container* to start, not for
    Redpanda to actually be ready to accept client connections, so the very
    first connection attempt racing that startup window is expected, not a
    real failure - only fall back after genuinely exhausting retries, and
    always print why, so a real misconfiguration doesn't look identical to
    "no broker configured"."""
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS")
    if not bootstrap:
        return InMemoryEventBus()

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return KafkaEventBus(
                bootstrap_servers=bootstrap.split(","),
                security_protocol=os.environ.get("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT"),
                sasl_mechanism=os.environ.get("KAFKA_SASL_MECHANISM"),
                sasl_username=os.environ.get("KAFKA_SASL_USERNAME"),
                sasl_password=os.environ.get("KAFKA_SASL_PASSWORD"),
            )
        except Exception as exc:
            last_error = exc
            print(
                f"[event_bus] KafkaEventBus connection attempt {attempt}/{max_retries} to "
                f"{bootstrap} failed: {exc}", file=sys.stderr, flush=True,
            )
            if attempt < max_retries:
                time.sleep(retry_delay_seconds)

    print(
        f"[event_bus] Giving up on {bootstrap} after {max_retries} attempts "
        f"({last_error}) - falling back to InMemoryEventBus.", file=sys.stderr, flush=True,
    )
    return InMemoryEventBus()
