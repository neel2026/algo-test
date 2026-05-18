"""Jugaad-data backed historical fetch helpers."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)


def _normalize_daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize historical frames into the project schema."""

    if df.empty:
        return df.copy()
    result = df.copy()
    result.columns = [column.strip().lower().replace(" ", "_").replace(".", "").replace("__", "_") for column in result.columns]
    if "historicaldate" in result.columns and "date" not in result.columns:
        result = result.rename(columns={"historicaldate": "date"})
    if "date" in result.columns:
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
        result["datetime"] = result["date"]
    if "index_name" in result.columns and "symbol" not in result.columns:
        result["symbol"] = result["index_name"]
    if "option_type" in result.columns:
        result["option_type"] = result["option_type"].astype(str).str.upper()
    numeric_columns = ["open", "high", "low", "close", "settle_pr", "strike_price", "open_int", "change_in_oi", "volume", "contracts"]
    for column in numeric_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    if "open_int" in result.columns and "open_interest" not in result.columns:
        result = result.rename(columns={"open_int": "open_interest"})
    if "contracts" in result.columns and "volume" not in result.columns:
        result = result.rename(columns={"contracts": "volume"})
    if "open_interest" not in result.columns:
        result["open_interest"] = pd.NA
    if "volume" not in result.columns:
        result["volume"] = pd.NA
    return result.sort_values("date").reset_index(drop=True)


def fetch_spot_daily(from_dt: date, to_dt: date) -> pd.DataFrame:
    """Fetch NIFTY 50 daily spot data from jugaad-data."""

    from jugaad_data.nse import index_df

    df = index_df(symbol="NIFTY 50", from_date=from_dt, to_date=to_dt)
    return _normalize_daily_frame(df)


def fetch_options_daily(strike: float | int, expiry_dt: date, option_type: str, from_dt: date, to_dt: date) -> pd.DataFrame:
    """Fetch NIFTY index option daily OHLC data from jugaad-data."""

    try:
        from jugaad_data.nse import index_pe_df

        option_type = option_type.upper()
        if option_type == "CE":
            logger.warning("jugaad-data API changed — skipping to bhav fallback")
            return pd.DataFrame()
        df = index_pe_df(symbol="NIFTY", from_date=from_dt, to_date=to_dt)
        df = _normalize_daily_frame(df)
        if df.empty:
            return df
        if "strike_price" in df.columns and "strike" not in df.columns:
            df = df.rename(columns={"strike_price": "strike"})
        if "strike" not in df.columns:
            df["strike"] = float(strike)
        if "expiry_dt" in df.columns:
            df["expiry_dt"] = pd.to_datetime(df["expiry_dt"], errors="coerce").dt.date
            df = df[df["expiry_dt"] == expiry_dt]
        if "strike" in df.columns:
            df = df[pd.to_numeric(df["strike"], errors="coerce").round(0) == round(float(strike))]
        df["option_type"] = option_type
        return df
    except Exception:
        logger.warning("jugaad-data API changed — skipping to bhav fallback")
        return pd.DataFrame()
