"""Data cleaning, resampling, and feature assembly for backtests."""

from __future__ import annotations

import hashlib
import logging
from typing import Iterable

import numpy as np
import pandas as pd

from nifty_backtest import config
from nifty_backtest.indicators.bollinger import add_bollinger_bands
from nifty_backtest.indicators.pivots import add_classic_pivots
from nifty_backtest.indicators.rsi import add_rsi_features
from nifty_backtest.indicators.support_resistance import add_support_resistance_features

logger = logging.getLogger(__name__)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lower-case snake style."""

    rename_map = {column: column.strip().lower() for column in df.columns}
    df = df.rename(columns=rename_map).copy()
    if "oi" in df.columns and "open_interest" not in df.columns:
        df = df.rename(columns={"oi": "open_interest"})
    return df


def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the dataframe has a parsed datetime column."""

    df = standardize_columns(df)
    candidates = ["datetime", "date", "timestamp", "time"]
    source = next((column for column in candidates if column in df.columns), None)
    if source is None:
        raise ValueError("A datetime-like column is required.")
    df["datetime"] = pd.to_datetime(df[source], errors="coerce")
    df = df.dropna(subset=["datetime"]).copy()
    return df.sort_values("datetime").reset_index(drop=True)


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Clean OHLCV-like data and coerce numeric columns."""

    if df.empty:
        return df.copy()
    df = ensure_datetime(df)
    numeric_columns = ["open", "high", "low", "close", "volume", "open_interest", "strike"]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "option_type" in df.columns:
        df["option_type"] = df["option_type"].astype(str).str.upper()
    df = df.drop_duplicates(
        subset=[column for column in ["datetime", "strike", "option_type"] if column in df.columns],
        keep="last",
    )
    return df.reset_index(drop=True)


def resample_ohlcv(df: pd.DataFrame, interval: str = "5minute") -> pd.DataFrame:
    """Resample candles to the requested interval."""

    if df.empty:
        return df.copy()

    df = clean_ohlcv(df)
    rule_map = {
        "1minute": "1min",
        "3minute": "3min",
        "5minute": "5min",
        "15minute": "15min",
        "30minute": "30min",
        "1hour": "1H",
        "1day": "1D",
        "day": "1D",
    }
    rule = rule_map.get(interval.lower(), interval)

    agg_map = {column: "first" for column in ["open", "high", "low", "close", "volume", "open_interest"] if column in df.columns}
    if not agg_map:
        return df

    df = df.set_index("datetime")
    resampled = df.resample(rule).agg(agg_map)
    if "high" in resampled.columns:
        resampled["high"] = df["high"].resample(rule).max()
    if "low" in resampled.columns:
        resampled["low"] = df["low"].resample(rule).min()
    if "close" in resampled.columns:
        resampled["close"] = df["close"].resample(rule).last()
    if "open" in resampled.columns:
        resampled["open"] = df["open"].resample(rule).first()
    if "volume" in resampled.columns:
        resampled["volume"] = df["volume"].resample(rule).sum()
    if "open_interest" in resampled.columns:
        resampled["open_interest"] = df["open_interest"].resample(rule).last()

    resampled = resampled.dropna(how="all").reset_index()
    return resampled


def compute_atm_strike(spot_close: pd.Series, rounding: int | float = 50) -> pd.Series:
    """Compute the ATM strike rounded to the configured increment."""

    return (spot_close / rounding).round() * rounding


def pivot_option_contracts(option_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot CE and PE candles into a single wide dataframe keyed by datetime and strike."""

    if option_df.empty:
        return option_df.copy()

    option_df = clean_ohlcv(option_df)
    required = {"datetime", "strike", "option_type"}
    if not required.issubset(option_df.columns):
        raise ValueError("Option data must include datetime, strike, and option_type columns.")

    value_columns = [column for column in ["open", "high", "low", "close", "volume", "open_interest"] if column in option_df.columns]
    if not value_columns:
        return option_df[["datetime", "strike", "option_type"]].copy()

    frames = []
    for option_type, group in option_df.groupby(option_df["option_type"].str.upper()):
        renamed = group[["datetime", "strike"] + value_columns].copy()
        renamed = renamed.rename(columns={column: f"{option_type.lower()}_{column}" for column in value_columns})
        frames.append(renamed)

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["datetime", "strike"], how="outer")
    return merged.sort_values(["datetime", "strike"]).reset_index(drop=True)


def merge_spot_and_options(spot_df: pd.DataFrame, option_df: pd.DataFrame, rounding: int | float | None = None) -> pd.DataFrame:
    """Merge spot candles with matching ATM option candles."""

    if spot_df.empty:
        return spot_df.copy()

    rounding = rounding or config.ATM_ROUNDING
    spot_df = clean_ohlcv(spot_df)
    if "close" not in spot_df.columns:
        raise ValueError("Spot data must include a close column.")

    spot_df["atm_strike"] = compute_atm_strike(spot_df["close"], rounding=rounding)
    spot_df["trade_date"] = spot_df["datetime"].dt.normalize()

    if option_df.empty:
        return spot_df

    option_df = clean_ohlcv(option_df)
    if "strike" not in option_df.columns:
        raise ValueError("Option data must include a strike column.")

    wide_options = pivot_option_contracts(option_df)
    merged = spot_df.merge(
        wide_options,
        left_on=["datetime", "atm_strike"],
        right_on=["datetime", "strike"],
        how="left",
        suffixes=("", "_option"),
    )
    if "strike_option" in merged.columns:
        merged = merged.drop(columns=["strike_option"])
    return merged


def attach_vix(frame: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """Merge daily VIX close onto the backtest frame by date."""

    if frame.empty or vix_df.empty:
        return frame.copy()

    CLOSE_ALIASES = [
        "close",
        "Close",
        "Adj Close",
        "vix_close",
        "VIX Close",
        "CLOSE",
        "closing",
        "Closing",
        "vixclose",
        "vix",
        "VIX",
    ]

    vix = vix_df.copy()
    column_lookup = {str(column).strip().lower(): column for column in vix.columns}
    found_col = next((column_lookup.get(alias.strip().lower()) for alias in CLOSE_ALIASES if alias.strip().lower() in column_lookup), None)

    if found_col is None:
        numeric_cols = vix.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            found_col = numeric_cols[0]
            logger.warning(
                "VIX DataFrame has no recognized close column. Available columns: %s. Using '%s' as fallback.",
                list(vix.columns),
                found_col,
            )
        else:
            raise ValueError(f"VIX data has no usable numeric column. Columns found: {list(vix.columns)}")

    vix = vix.rename(columns={found_col: "vix_close"})

    date_aliases = ["date", "Date", "INDEX", "index", "index_date", "historicaldate", "tradetd", "trad_dt"]
    found_date = next((column_lookup.get(alias.strip().lower()) for alias in date_aliases if alias.strip().lower() in column_lookup), None)
    if found_date is None:
        if isinstance(vix.index, pd.DatetimeIndex):
            vix = vix.reset_index().rename(columns={"index": "date", "Date": "date"})
        else:
            vix = vix.reset_index()
    else:
        vix = vix.rename(columns={found_date: "date"})

    vix["date"] = pd.to_datetime(vix["date"], errors="coerce").dt.date
    vix = vix[["date", "vix_close"]].drop_duplicates("date")

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["datetime"], errors="coerce").dt.date
    frame = frame.merge(vix, on="date", how="left")
    frame["vix_close"] = frame["vix_close"].ffill().bfill()
    frame["vix_close"] = frame["vix_close"].fillna(14.0)
    logger.info("VIX attached. NaN remaining: %s", frame["vix_close"].isna().sum())
    return frame


def build_backtest_frame(
    spot_df: pd.DataFrame,
    option_df: pd.DataFrame,
    vix_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create the fully enriched dataframe consumed by the backtester."""

    frame = merge_spot_and_options(spot_df, option_df, rounding=config.ATM_ROUNDING)
    if vix_df is not None and not vix_df.empty:
        frame = attach_vix(frame, vix_df)

    frame = add_bollinger_bands(frame, close_column="close", period=config.BB_PERIOD, std_dev=config.BB_STD)
    frame = add_rsi_features(
        frame,
        close_column="close",
        period=config.RSI_PERIOD,
        ma_short=config.RSI_MA_SHORT,
        ma_long=config.RSI_MA_LONG,
    )
    frame = add_classic_pivots(frame)
    frame = add_support_resistance_features(frame, window=20, confluence_threshold=config.SR_CONFLUENCE_THRESHOLD)
    return frame.sort_values("datetime").reset_index(drop=True)


def synthesize_intraday(daily_ohlc_df: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
    """Generate regime-aware synthetic 5-minute candles from daily OHLC data."""

    if daily_ohlc_df.empty:
        return daily_ohlc_df.copy()

    working = standardize_columns(daily_ohlc_df.copy())
    if "date" not in working.columns and "datetime" in working.columns:
        working["date"] = pd.to_datetime(working["datetime"], errors="coerce").dt.date
    elif "date" in working.columns:
        working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.date
    else:
        raise ValueError("Daily OHLC data must contain a date or datetime column.")

    group_cols = [column for column in ["date", "strike", "option_type", "expiry_dt", "symbol"] if column in working.columns]
    if not group_cols:
        group_cols = ["date"]

    seed = config.SYNTHETIC_SEED if seed is None else seed
    generated: list[pd.DataFrame] = []

    for group_key, group in working.groupby(group_cols, dropna=False):
        group = group.sort_values("date").copy()
        row = group.iloc[0]
        trade_date = pd.Timestamp(row["date"]).date()
        key_text = f"{group_key!r}|{seed}"
        key_digest = hashlib.sha256(key_text.encode("utf-8")).hexdigest()
        rng = np.random.default_rng(int(key_digest[:8], 16))

        o = float(row.get("open", row.get("close", 0.0)) or 0.0)
        h = float(row.get("high", max(o, row.get("close", o))) or max(o, row.get("close", o)))
        l = float(row.get("low", min(o, row.get("close", o))) or min(o, row.get("close", o)))
        c = float(row.get("close", o) or o)
        if h < max(o, c):
            h = max(o, c)
        if l > min(o, c):
            l = min(o, c)
        if h <= l:
            h = l + max(abs(c - o), 1.0)

        day_range = max(h - l, 1.0)
        body = abs(c - o)
        is_trending = body > (day_range * 0.5)
        is_bullish = c > o

        times = pd.date_range(start=f"{trade_date} 09:15", periods=75, freq="5min")

        if is_trending and is_bullish:
            path = np.linspace(o, c, 75)
            noise = rng.normal(0, day_range * 0.04, 75)
            path[:18] += np.linspace(0, -(day_range * 0.35), 18)
            path[18:] += np.linspace(-(day_range * 0.35), 0, 57)
        elif is_trending and not is_bullish:
            path = np.linspace(o, c, 75)
            noise = rng.normal(0, day_range * 0.04, 75)
            path[:18] += np.linspace(0, day_range * 0.35, 18)
            path[18:] += np.linspace(day_range * 0.35, 0, 57)
        else:
            mid = (h + l) / 2.0
            cycles = int(rng.integers(3, 7))
            t = np.linspace(0, cycles * 2 * np.pi, 75)
            path = mid + (day_range * 0.45 * np.sin(t))
            noise = rng.normal(0, day_range * 0.03, 75)

        prices = np.clip(path + noise, l, h)
        prices[0] = o
        prices[-1] = c

        candles = []
        for i, timestamp in enumerate(times):
            p = float(prices[i])
            candle_open = float(prices[i - 1]) if i > 0 else o
            candle_close = p
            candle_range = day_range * float(rng.uniform(0.015, 0.06))
            candle_high = min(max(candle_open, candle_close) + candle_range * 0.5, h)
            candle_low = max(min(candle_open, candle_close) - candle_range * 0.5, l)
            candles.append(
                {
                    "datetime": timestamp,
                    "date": trade_date,
                    "open": round(candle_open, 2),
                    "high": round(candle_high, 2),
                    "low": round(candle_low, 2),
                    "close": round(candle_close, 2),
                    "is_synthetic": True,
                }
            )

        out = pd.DataFrame(candles)
        vol_curve = np.concatenate(
            [
                np.linspace(1.8, 0.6, 15),
                np.linspace(0.6, 0.4, 45),
                np.linspace(0.4, 1.5, 15),
            ]
        )
        daily_vol = row.get("volume", row.get("contracts", 500000))
        if pd.isna(daily_vol):
            daily_vol = 500000
        daily_vol = float(daily_vol)
        volumes = (vol_curve / vol_curve.sum()) * daily_vol
        out["volume"] = volumes.astype(int)

        if "strike" in row.index:
            out["strike"] = row.get("strike")
        if "option_type" in row.index:
            out["option_type"] = str(row.get("option_type")).upper()
        if "expiry_dt" in row.index:
            out["expiry_dt"] = row.get("expiry_dt")
        if "open_interest" in row.index or "open_int" in row.index:
            open_interest_base = row.get("open_interest", row.get("open_int", pd.NA))
            if pd.isna(open_interest_base):
                out["open_interest"] = pd.NA
            else:
                oi = float(open_interest_base)
                out["open_interest"] = np.linspace(oi * 1.02, oi * 0.98, 75)
        if "symbol" in row.index:
            out["symbol"] = row.get("symbol")

        generated.append(out)

    combined = pd.concat(generated, ignore_index=True)
    return combined.sort_values("datetime").reset_index(drop=True)
