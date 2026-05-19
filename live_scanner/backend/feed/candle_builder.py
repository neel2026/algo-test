"""Tick to OHLCV candle aggregation."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd


class CandleBuilder:
    """Aggregate ticks into fixed interval candles."""

    def __init__(self, interval_minutes: int = 5) -> None:
        """Create a candle builder with the requested interval."""

        self.interval = int(interval_minutes)
        self.candles: dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.current: dict[str, dict] = {}

    def on_tick(self, instrument_key: str, ltp: float, volume: int, ts_ms: int) -> dict | None:
        """Consume one tick and return a closed candle when the bucket changes."""

        ts_sec = int(ts_ms) // 1000
        bucket = (ts_sec // (self.interval * 60)) * (self.interval * 60)
        ist_dt = datetime.fromtimestamp(bucket, tz=ZoneInfo("Asia/Kolkata"))

        current = self.current.get(instrument_key)
        if current is None:
            self.current[instrument_key] = {
                "datetime": ist_dt,
                "bucket": bucket,
                "open": float(ltp),
                "high": float(ltp),
                "low": float(ltp),
                "close": float(ltp),
                "volume": int(volume),
            }
            return None

        if bucket != current["bucket"]:
            closed = dict(current)
            closed["is_closed"] = True
            self.candles[instrument_key].append(closed)
            self.current[instrument_key] = {
                "datetime": ist_dt,
                "bucket": bucket,
                "open": float(ltp),
                "high": float(ltp),
                "low": float(ltp),
                "close": float(ltp),
                "volume": int(volume),
            }
            return closed

        current["high"] = max(float(current["high"]), float(ltp))
        current["low"] = min(float(current["low"]), float(ltp))
        current["close"] = float(ltp)
        current["volume"] = int(current["volume"]) + int(volume)
        return None

    def get_candles_df(self, instrument_key: str) -> pd.DataFrame:
        """Return the closed candle history as a dataframe."""

        return pd.DataFrame(list(self.candles[instrument_key]))

    def get_current_candle(self, instrument_key: str) -> dict | None:
        """Return the current forming candle for an instrument."""

        return self.current.get(instrument_key)
