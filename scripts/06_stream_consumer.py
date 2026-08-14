"""Perpetual rolling-window consumer for the local pipeline - the
always-on version of the same background thread the deployed Dash app runs
at startup. Run as its own container (see docker-compose.yml's
`stream-consumer` service); blocks forever.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.streaming.event_bus import get_event_bus
from core.streaming.window_consumer import run_consumer

if __name__ == "__main__":
    bus = get_event_bus()
    print(f"stream-consumer starting, bus={type(bus).__name__}, topic=transactions.raw", flush=True)
    run_consumer(bus)
