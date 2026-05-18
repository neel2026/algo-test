"""Classic pivot level calculations."""

from __future__ import annotations

import pandas as pd


def calculate_daily_pivots(daily_ohlc: pd.DataFrame) -> pd.DataFrame:
    """Calculate classic daily pivot levels from prior-session data."""

    if daily_ohlc.empty:
        return daily_ohlc.copy()

    df = daily_ohlc.copy()
    df = df.sort_values("trade_date").reset_index(drop=True)
    df["pivot"] = (df["high"].shift(1) + df["low"].shift(1) + df["close"].shift(1)) / 3.0
    df["r1"] = (2 * df["pivot"]) - df["low"].shift(1)
    df["r2"] = df["pivot"] + (df["high"].shift(1) - df["low"].shift(1))
    df["s1"] = (2 * df["pivot"]) - df["high"].shift(1)
    df["s2"] = df["pivot"] - (df["high"].shift(1) - df["low"].shift(1))
    return df


def add_classic_pivots(df: pd.DataFrame) -> pd.DataFrame:
    """Attach previous-day classic pivot levels to intraday candles."""

    if df.empty:
        return df.copy()
    if "datetime" not in df.columns:
        raise ValueError("Dataframe must contain datetime.")
    if not {"high", "low", "close"}.issubset(df.columns):
        raise ValueError("Dataframe must contain high, low, and close.")

    working = df.copy()
    working["trade_date"] = pd.to_datetime(working["datetime"]).dt.normalize()
    daily = (
        working.groupby("trade_date", as_index=False)
        .agg({"high": "max", "low": "min", "close": "last"})
        .sort_values("trade_date")
        .reset_index(drop=True)
    )
    pivots = calculate_daily_pivots(daily)
    working = working.merge(
        pivots[["trade_date", "pivot", "r1", "r2", "s1", "s2"]],
        on="trade_date",
        how="left",
    )
    return working

