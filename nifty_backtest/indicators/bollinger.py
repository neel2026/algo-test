"""Bollinger Band indicator helpers."""

from __future__ import annotations

import pandas as pd


def calculate_bollinger_bands(close: pd.Series, period: int = 20, std_dev: int | float = 2) -> pd.DataFrame:
    """Calculate Bollinger Bands and %B for a close series."""

    try:
        from ta.volatility import BollingerBands

        indicator = BollingerBands(close=close.astype(float), window=period, window_dev=std_dev)
        result = pd.DataFrame(
            {
                "upper_band": indicator.bollinger_hband(),
                "middle_band": indicator.bollinger_mavg(),
                "lower_band": indicator.bollinger_lband(),
                "percent_b": indicator.bollinger_pband(),
            }
        )
    except Exception:
        middle = close.astype(float).rolling(period, min_periods=period).mean()
        deviation = close.astype(float).rolling(period, min_periods=period).std(ddof=0)
        upper = middle + (deviation * std_dev)
        lower = middle - (deviation * std_dev)
        percent_b = (close.astype(float) - lower) / (upper - lower)
        result = pd.DataFrame(
            {
                "upper_band": upper,
                "middle_band": middle,
                "lower_band": lower,
                "percent_b": percent_b,
            }
        )

    return result


def add_bollinger_bands(
    df: pd.DataFrame,
    close_column: str = "close",
    period: int = 20,
    std_dev: int | float = 2,
) -> pd.DataFrame:
    """Attach Bollinger Band columns to a dataframe."""

    if df.empty:
        return df.copy()
    if close_column not in df.columns:
        raise ValueError(f"Missing close column: {close_column}")
    bands = calculate_bollinger_bands(df[close_column], period=period, std_dev=std_dev)
    return pd.concat([df.reset_index(drop=True), bands.reset_index(drop=True)], axis=1)

