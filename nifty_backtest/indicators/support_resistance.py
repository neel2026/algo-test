"""Swing-based support and resistance detection."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def detect_swing_levels(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Detect trailing-window swing highs and lows."""

    if df.empty:
        return df.copy()
    if not {"high", "low"}.issubset(df.columns):
        raise ValueError("Dataframe must contain high and low columns.")

    working = df.copy()
    rolling_high = working["high"].rolling(window=window, min_periods=window)
    rolling_low = working["low"].rolling(window=window, min_periods=window)
    working["swing_high"] = working["high"] == rolling_high.max()
    working["swing_low"] = working["low"] == rolling_low.min()
    return working


def build_session_levels(df: pd.DataFrame, window: int = 20) -> dict[pd.Timestamp, list[float]]:
    """Build a list of support/resistance levels for each trading session."""

    if df.empty:
        return {}

    working = df.copy()
    working["trade_date"] = pd.to_datetime(working["datetime"]).dt.normalize()
    working = detect_swing_levels(working, window=window)
    session_levels: dict[pd.Timestamp, list[float]] = {}

    for trade_date, group in working.groupby("trade_date"):
        levels: list[float] = []
        if "swing_high" in group.columns:
            levels.extend(group.loc[group["swing_high"], "high"].dropna().astype(float).tolist())
        if "swing_low" in group.columns:
            levels.extend(group.loc[group["swing_low"], "low"].dropna().astype(float).tolist())
        cleaned = sorted({round(level, 2) for level in levels})
        session_levels[pd.Timestamp(trade_date)] = cleaned

    return session_levels


def add_support_resistance_features(
    df: pd.DataFrame,
    window: int = 20,
    confluence_threshold: int | float = 15,
) -> pd.DataFrame:
    """Attach session-level support/resistance lists and confluence flags."""

    if df.empty:
        return df.copy()
    if "datetime" not in df.columns:
        raise ValueError("Dataframe must contain datetime.")

    working = df.copy()
    working["trade_date"] = pd.to_datetime(working["datetime"]).dt.normalize()
    session_levels = build_session_levels(working, window=window)
    working["sr_levels"] = working["trade_date"].map(session_levels)

    def has_confluence(row: pd.Series) -> bool:
        """Return whether pivot levels are close to any detected session level."""

        levels = row.get("sr_levels") or []
        pivot_levels = [row.get(column) for column in ["pivot", "r1", "r2", "s1", "s2"] if pd.notna(row.get(column))]
        for pivot_level in pivot_levels:
            for level in levels:
                if abs(float(pivot_level) - float(level)) <= confluence_threshold:
                    return True
        return False

    working["sr_confluence"] = working.apply(has_confluence, axis=1)
    working["sr_level_count"] = working["sr_levels"].apply(lambda value: len(value) if isinstance(value, list) else 0)
    return working

