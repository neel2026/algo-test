"""Indicator calculations for the live scanner."""

from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nifty_backtest.indicators.bollinger import calculate_bollinger_bands
from nifty_backtest.indicators.rsi import calculate_rsi_features
from nifty_backtest.indicators.support_resistance import add_support_resistance_features

logger = logging.getLogger(__name__)


class IndicatorEngine:
    """Compute the live indicator bundle from rolling candles."""

    MIN_CANDLES = 25

    def __init__(self) -> None:
        """Create a lightweight previous-day cache."""

        self._prev_day_cache: dict[str, tuple[date, dict]] = {}

    def _build_feature_frame(self, candles_df: pd.DataFrame) -> pd.DataFrame:
        """Build a normalized feature frame from the candle history."""

        df = candles_df.copy().sort_values("datetime").reset_index(drop=True)
        close_series = df["close"].astype(float).reset_index(drop=True)

        bb = calculate_bollinger_bands(close_series, period=20, std_dev=2).reset_index(drop=True)
        rsi = calculate_rsi_features(close_series, period=9, ma_short=3, ma_long=21).reset_index(drop=True)

        feature_df = pd.concat([df, bb, rsi], axis=1)
        feature_df = add_support_resistance_features(feature_df, window=20, confluence_threshold=15)
        return feature_df

    def _extract_sr_levels(self, last_row: pd.Series) -> tuple[list[float], list[float]]:
        """Extract support and resistance lists from the latest feature row."""

        support_columns = ("support_levels", "support", "sr_support", "sr_levels")
        resistance_columns = ("resistance_levels", "resistance", "sr_resistance", "sr_levels")

        support_levels: list[float] = []
        resistance_levels: list[float] = []

        for column in support_columns:
            value = last_row.get(column)
            if isinstance(value, list):
                support_levels = [float(item) for item in value if pd.notna(item)]
                break

        for column in resistance_columns:
            value = last_row.get(column)
            if isinstance(value, list):
                resistance_levels = [float(item) for item in value if pd.notna(item)]
                break

        if not support_levels and not resistance_levels:
            raw_levels = last_row.get("sr_levels")
            if isinstance(raw_levels, list):
                close = float(last_row.get("close")) if pd.notna(last_row.get("close")) else 0.0
                support_levels = [float(level) for level in raw_levels if pd.notna(level) and float(level) <= close]
                resistance_levels = [float(level) for level in raw_levels if pd.notna(level) and float(level) >= close]

        return support_levels, resistance_levels

    def _calculate_pivots(self, prev_day: dict) -> dict:
        """Calculate classic pivot levels from the prior day."""

        if not prev_day:
            return {"pivot": None, "r1": None, "r2": None, "s1": None, "s2": None}
        high = float(prev_day.get("high", 0.0))
        low = float(prev_day.get("low", 0.0))
        close = float(prev_day.get("close", 0.0))
        pivot = (high + low + close) / 3.0
        r1 = (2 * pivot) - low
        r2 = pivot + (high - low)
        s1 = (2 * pivot) - high
        s2 = pivot - (high - low)
        return {"pivot": pivot, "r1": r1, "r2": r2, "s1": s1, "s2": s2}

    def compute(self, candles_df: pd.DataFrame, prev_day: dict | None, instrument_key: str | None = None) -> dict | None:
        """Compute the latest indicator snapshot for the current candle set."""

        if candles_df is None or len(candles_df) < self.MIN_CANDLES:
            return None

        df = self._build_feature_frame(candles_df)

        pivots = self._calculate_pivots(prev_day or {})
        for key, value in pivots.items():
            df[key] = value

        last = df.iloc[-1]
        prev = df.iloc[-2]
        percent_b = last.get("percent_b")
        if pd.isna(percent_b) and pd.notna(last.get("upper_band")) and pd.notna(last.get("lower_band")):
            band_span = float(last["upper_band"]) - float(last["lower_band"])
            percent_b = ((float(last["close"]) - float(last["lower_band"])) / band_span) if band_span else 0.5

        rsi_cross_above_ma21 = bool(
            pd.notna(prev.get("rsi"))
            and pd.notna(prev.get("rsi_ma21"))
            and pd.notna(last.get("rsi"))
            and pd.notna(last.get("rsi_ma21"))
            and float(prev["rsi"]) < float(prev["rsi_ma21"])
            and float(last["rsi"]) >= float(last["rsi_ma21"])
        )
        rsi_cross_below_ma21 = bool(
            pd.notna(prev.get("rsi"))
            and pd.notna(prev.get("rsi_ma21"))
            and pd.notna(last.get("rsi"))
            and pd.notna(last.get("rsi_ma21"))
            and float(prev["rsi"]) > float(prev["rsi_ma21"])
            and float(last["rsi"]) <= float(last["rsi_ma21"])
        )
        support_levels, resistance_levels = self._extract_sr_levels(last)

        result = {
            "bb_upper": float(last.get("upper_band")) if pd.notna(last.get("upper_band")) else None,
            "bb_lower": float(last.get("lower_band")) if pd.notna(last.get("lower_band")) else None,
            "bb_middle": float(last.get("middle_band")) if pd.notna(last.get("middle_band")) else None,
            "percent_b": float(percent_b) if percent_b is not None and pd.notna(percent_b) else None,
            "rsi": float(last.get("rsi")) if pd.notna(last.get("rsi")) else None,
            "rsi_ma3": float(last.get("rsi_ma3")) if pd.notna(last.get("rsi_ma3")) else None,
            "rsi_ma21": float(last.get("rsi_ma21")) if pd.notna(last.get("rsi_ma21")) else None,
            "rsi_cross_above_ma3": bool(last.get("rsi_cross_above_ma3", False)),
            "rsi_cross_below_ma3": bool(last.get("rsi_cross_below_ma3", False)),
            "rsi_cross_above_ma21": rsi_cross_above_ma21,
            "rsi_cross_below_ma21": rsi_cross_below_ma21,
            "pivot": float(last.get("pivot")) if pd.notna(last.get("pivot")) else None,
            "r1": float(last.get("r1")) if pd.notna(last.get("r1")) else None,
            "r2": float(last.get("r2")) if pd.notna(last.get("r2")) else None,
            "s1": float(last.get("s1")) if pd.notna(last.get("s1")) else None,
            "s2": float(last.get("s2")) if pd.notna(last.get("s2")) else None,
            "support_levels": support_levels,
            "resistance_levels": resistance_levels,
            "current_close": float(last.get("close")) if pd.notna(last.get("close")) else None,
        }
        print(f"[INDICATORS] {instrument_key or 'unknown'}")
        print(f"  RSI: {result['rsi']:.2f}" if result["rsi"] is not None else "  RSI: N/A")
        print(f"  BB Upper: {result['bb_upper']:.2f}" if result["bb_upper"] is not None else "  BB Upper: N/A")
        print(f"  BB Lower: {result['bb_lower']:.2f}" if result["bb_lower"] is not None else "  BB Lower: N/A")
        print(f"  Candles used: {len(candles_df)}")
        return result

    def fetch_prev_day(self, instrument_key: str, access_token: str, reference_date: date | None = None) -> dict:
        """Fetch the previous daily OHLC candle for the selected instrument."""

        today = reference_date or date.today()
        yesterday = today - timedelta(days=1)
        while yesterday.weekday() >= 5:
            yesterday -= timedelta(days=1)

        cache_key = f"{instrument_key}|{today.isoformat()}|{yesterday.isoformat()}"
        cached = self._prev_day_cache.get(cache_key)
        if cached and cached[0] == yesterday:
            return cached[1]

        url = (
            "https://api.upstox.com/v2/historical-candle/"
            f"{requests.utils.quote(instrument_key, safe='')}/day/{yesterday.isoformat()}/{yesterday.isoformat()}"
        )
        resp = requests.get(
            url,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
            },
            timeout=20,
        )
        resp.raise_for_status()
        payload = resp.json()
        candles = payload.get("data", {}).get("candles", [])
        if not candles:
            self._prev_day_cache[cache_key] = (yesterday, {})
            return {}
        candle = candles[0]
        prev_day = {"open": candle[1], "high": candle[2], "low": candle[3], "close": candle[4]}
        self._prev_day_cache[cache_key] = (yesterday, prev_day)
        return prev_day
