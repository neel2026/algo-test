"""RSI indicator helpers."""

from __future__ import annotations

import pandas as pd


def calculate_rsi_features(
    close: pd.Series,
    period: int = 9,
    ma_short: int = 3,
    ma_long: int = 21,
) -> pd.DataFrame:
    """Calculate RSI and moving-average overlays."""

    try:
        from ta.momentum import RSIIndicator

        rsi = RSIIndicator(close=close.astype(float), window=period).rsi()
    except Exception:
        delta = close.astype(float).diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

    rsi_ma_short = rsi.rolling(ma_short, min_periods=ma_short).mean()
    rsi_ma_long = rsi.rolling(ma_long, min_periods=ma_long).mean()
    cross_above = (rsi > rsi_ma_short) & (rsi.shift(1) <= rsi_ma_short.shift(1))
    cross_below = (rsi < rsi_ma_short) & (rsi.shift(1) >= rsi_ma_short.shift(1))

    return pd.DataFrame(
        {
            "rsi": rsi,
            "rsi_ma3": rsi_ma_short,
            "rsi_ma21": rsi_ma_long,
            "rsi_cross_above_ma3": cross_above.fillna(False),
            "rsi_cross_below_ma3": cross_below.fillna(False),
        }
    )


def add_rsi_features(
    df: pd.DataFrame,
    close_column: str = "close",
    period: int = 9,
    ma_short: int = 3,
    ma_long: int = 21,
) -> pd.DataFrame:
    """Attach RSI columns to a dataframe."""

    if df.empty:
        return df.copy()
    if close_column not in df.columns:
        raise ValueError(f"Missing close column: {close_column}")
    features = calculate_rsi_features(df[close_column], period=period, ma_short=ma_short, ma_long=ma_long)
    return pd.concat([df.reset_index(drop=True), features.reset_index(drop=True)], axis=1)

