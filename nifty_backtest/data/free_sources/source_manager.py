"""Source priority manager and synthetic intraday generator for free data sources."""

from __future__ import annotations

import hashlib
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from nifty_backtest import config
from nifty_backtest.data.free_sources import jugaad_fetcher, nse_bhav, nse_indices

logger = logging.getLogger(__name__)


def _holiday_set() -> set[date]:
    """Return the configured NSE holiday set."""

    values = pd.to_datetime(config.NSE_HOLIDAYS_2023 + config.NSE_HOLIDAYS_2024, errors="coerce")
    return {value.date() for value in values if not pd.isna(value)}


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a source dataframe into the project schema."""

    if df.empty:
        return df.copy()
    result = df.copy()
    result.columns = [str(column).strip().lower().replace(" ", "_").replace(".", "").replace("-", "_") for column in result.columns]
    if "historicaldate" in result.columns and "date" not in result.columns:
        result = result.rename(columns={"historicaldate": "date"})
    if "index_date" in result.columns and "date" not in result.columns:
        result = result.rename(columns={"index_date": "date"})
    if "timestamp" in result.columns and "date" not in result.columns:
        result = result.rename(columns={"timestamp": "date"})
    if "date" in result.columns:
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
        result["datetime"] = result["date"]
    if "closing" in result.columns and "close" not in result.columns:
        result = result.rename(columns={"closing": "close"})
    if "vix_close" in result.columns and "close" not in result.columns:
        result["close"] = result["vix_close"]
    if "oi" in result.columns and "open_interest" not in result.columns:
        result = result.rename(columns={"oi": "open_interest"})
    if "open_int" in result.columns and "open_interest" not in result.columns:
        result = result.rename(columns={"open_int": "open_interest"})
    if "volume_traded" in result.columns and "volume" not in result.columns:
        result = result.rename(columns={"volume_traded": "volume"})
    if "contracts" in result.columns and "volume" not in result.columns:
        result = result.rename(columns={"contracts": "volume"})
    for column in ["open", "high", "low", "close", "volume", "open_interest", "strike", "strike_price"]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    if "strike_price" in result.columns and "strike" not in result.columns:
        result = result.rename(columns={"strike_price": "strike"})
    if "option type" in result.columns and "option_type" not in result.columns:
        result = result.rename(columns={"option type": "option_type"})
    if "option_type" in result.columns:
        result["option_type"] = result["option_type"].astype(str).str.upper()
    if "volume" not in result.columns:
        result["volume"] = pd.NA
    if "open_interest" not in result.columns:
        result["open_interest"] = pd.NA
    return result.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _business_days(from_dt: date, to_dt: date) -> list[date]:
    """Return all business days within the requested range."""

    return [value.date() for value in pd.bdate_range(from_dt, to_dt)]


def _daterange(from_dt: date, to_dt: date) -> Iterable[date]:
    """Yield calendar dates between two bounds, inclusive."""

    current = from_dt
    while current <= to_dt:
        yield current
        current += timedelta(days=1)


def next_expiry_for_date(trade_date: date) -> date:
    """Return the next weekly expiry date on or after a given trade date."""

    trade_day = pd.Timestamp(trade_date).date()
    offset = (3 - trade_day.weekday()) % 7
    expiry = trade_day + timedelta(days=offset)
    if expiry in _holiday_set():
        expiry = expiry - timedelta(days=1)
    while expiry.weekday() >= 5:
        expiry -= timedelta(days=1)
    return expiry


def get_all_expiries_in_range(from_dt: date, to_dt: date) -> list[date]:
    """Return all weekly expiry dates in the requested date range."""

    expiries = {next_expiry_for_date(day) for day in _daterange(from_dt, to_dt)}
    return sorted(expiry for expiry in expiries if from_dt <= expiry <= to_dt)


def _download_bhav_zip(trade_date: date) -> pd.DataFrame:
    """Download and parse a single NSE bhav copy date."""

    return nse_bhav.download_bhav_zip(trade_date)


def _try_sources(sources: list[str], fetchers: list[tuple[str, Callable[[], pd.DataFrame]]]) -> pd.DataFrame:
    """Try source fetchers in priority order and merge gap-filling results."""

    frames: list[pd.DataFrame] = []
    for name, fetcher in fetchers:
        if name not in sources:
            continue
        try:
            df = fetcher()
            if df is not None and not df.empty:
                frames.append(_normalize_frame(df))
        except Exception as exc:
            logger.warning("Source %s failed: %s", name, exc)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if "date" in combined.columns:
        combined = combined.drop_duplicates(subset=["date"] + [column for column in ["strike", "option_type"] if column in combined.columns], keep="first")
        return combined.sort_values("date").reset_index(drop=True)
    return combined.drop_duplicates().reset_index(drop=True)


def get_spot_daily(from_dt: date, to_dt: date) -> pd.DataFrame:
    """Fetch daily NIFTY spot data using configured free sources."""

    priorities = config.DATA_SOURCE_PRIORITY
    fetchers = [
        ("jugaad", lambda: jugaad_fetcher.fetch_spot_daily(from_dt, to_dt)),
        ("niftyindices", lambda: nse_indices.fetch_index_history("NIFTY 50", from_dt, to_dt)),
        ("nse_direct", lambda: nse_indices.fetch_nse_spot_history(from_dt, to_dt)),
        ("yfinance", lambda: nse_indices.fetch_index_history("NIFTY 50", from_dt, to_dt)),
    ]
    result = _try_sources(priorities + ["nse_direct", "yfinance"], fetchers)
    if result.empty:
        return result
    result = result.rename(columns={"close": "close"})
    if "volume" not in result.columns:
        result["volume"] = pd.NA
    return result[["date", "datetime", "open", "high", "low", "close", "volume"]].copy() if {"open", "high", "low", "close"}.issubset(result.columns) else result


def get_vix_daily(from_dt: date, to_dt: date) -> pd.DataFrame:
    """Fetch daily India VIX data."""

    result = nse_indices.fetch_vix_history(from_dt, to_dt)
    result = _normalize_frame(result)
    if "close" in result.columns and "vix_close" not in result.columns:
        result["vix_close"] = result["close"]
    if "vix_close" not in result.columns:
        result["vix_close"] = pd.NA
    return result[["date", "datetime", "vix_close"]].copy()


def _synthetic_option_daily_from_spot(
    strike: float | int,
    expiry_dt: date,
    option_type: str,
    from_dt: date,
    to_dt: date,
) -> pd.DataFrame:
    """Synthesize daily option candles from spot and VIX history."""

    logger.warning("Using synthetic daily options derived from spot/VIX because source data was unavailable.")
    spot = get_spot_daily(from_dt, to_dt)
    if spot.empty:
        return pd.DataFrame()

    spot = spot.copy()
    spot["trade_date"] = pd.to_datetime(spot["date"]).dt.date
    vix = get_vix_daily(from_dt, to_dt)
    if not vix.empty:
        vix = vix.copy()
        vix["trade_date"] = pd.to_datetime(vix["date"]).dt.date
        spot = spot.merge(vix[["trade_date", "vix_close"]], on="trade_date", how="left")
    else:
        spot["vix_close"] = pd.NA

    option_type = option_type.upper()
    rows: list[dict[str, object]] = []
    for _, row in spot.iterrows():
        trade_date = pd.Timestamp(row["trade_date"]).date()
        open_price = float(row.get("open", row.get("close", 0.0)))
        high_price = float(row.get("high", max(open_price, row.get("close", open_price))))
        low_price = float(row.get("low", min(open_price, row.get("close", open_price))))
        close_price = float(row.get("close", open_price))
        vix_value = float(row.get("vix_close", 15.0) or 15.0)
        days_to_expiry = max((expiry_dt - trade_date).days, 0)
        theta_decay = max(0.25, 1.0 / max(days_to_expiry + 1, 1))
        time_value = max(5.0, float(strike) * 0.0025 * (1.0 + max(vix_value - 13.0, 0.0) / 25.0) * theta_decay)

        def _premium(spot_price: float) -> float:
            intrinsic = max(spot_price - float(strike), 0.0) if option_type == "CE" else max(float(strike) - spot_price, 0.0)
            return intrinsic + time_value

        premiums = [
            _premium(open_price),
            _premium(high_price),
            _premium(low_price),
            _premium(close_price),
        ]
        open_opt = premiums[0]
        high_opt = max(premiums)
        low_opt = min(premiums)
        close_opt = premiums[3]
        base_volume = row.get("volume", 0)
        if pd.isna(base_volume):
            base_volume = 0
        volume = max(1000, int(float(base_volume) * 0.02))
        open_interest = max(1000, int(volume * 1.5))

        rows.append(
            {
                "date": pd.Timestamp(trade_date),
                "datetime": pd.Timestamp(trade_date),
                "strike": float(strike),
                "option_type": option_type,
                "open": open_opt,
                "high": high_opt,
                "low": low_opt,
                "close": close_opt,
                "open_interest": open_interest,
                "volume": volume,
                "expiry_dt": expiry_dt,
            }
        )

    return pd.DataFrame(rows)


def get_options_daily(
    strike: float | int,
    expiry_dt: date,
    option_type: str,
    from_dt: date,
    to_dt: date,
) -> pd.DataFrame:
    """Fetch daily NIFTY index option data using the configured free sources."""

    option_type = option_type.strip().upper()
    if option_type in {"CALL", "C"}:
        option_type = "CE"
    elif option_type in {"PUT", "P"}:
        option_type = "PE"
    priorities = config.DATA_SOURCE_PRIORITY

    def _jugaad() -> pd.DataFrame:
        return jugaad_fetcher.fetch_options_daily(strike, expiry_dt, option_type, from_dt, to_dt)

    def _bhav() -> pd.DataFrame:
        rows: list[pd.DataFrame] = []
        for trade_date in _business_days(from_dt, to_dt):
            bhav = _download_bhav_zip(trade_date)
            if bhav.empty:
                continue
            bhav = _normalize_frame(bhav)
            if bhav.empty:
                continue
            symbol_col = bhav["symbol"].astype(str).str.upper() if "symbol" in bhav.columns else pd.Series("", index=bhav.index)
            instrument_col = bhav["instrument"].astype(str).str.upper() if "instrument" in bhav.columns else pd.Series("", index=bhav.index)
            option_col = (
                bhav["option_typ"].astype(str).str.upper()
                if "option_typ" in bhav.columns
                else bhav["option_type"].astype(str).str.upper()
                if "option_type" in bhav.columns
                else pd.Series("", index=bhav.index)
            )
            filtered = bhav[
                (symbol_col == "NIFTY")
                & (instrument_col == "OPTIDX")
                & (option_col == option_type)
            ].copy()
            if "strike_pr" in filtered.columns:
                filtered["strike"] = pd.to_numeric(filtered["strike_pr"], errors="coerce")
                filtered = filtered[filtered["strike"].round(0) == round(float(strike))]
            if "expiry_dt" in filtered.columns:
                filtered["expiry_dt"] = pd.to_datetime(filtered["expiry_dt"], errors="coerce").dt.date
                filtered = filtered[filtered["expiry_dt"] == expiry_dt]
            if "timestamp" in filtered.columns:
                filtered["date"] = pd.to_datetime(filtered["timestamp"], errors="coerce")
                filtered["datetime"] = filtered["date"]
            if not filtered.empty:
                rows.append(filtered)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    fetchers = [("jugaad", _jugaad), ("bhav", _bhav)]
    result = _try_sources(priorities, fetchers)
    if result.empty:
        return _synthetic_option_daily_from_spot(strike, expiry_dt, option_type, from_dt, to_dt)
    if "strike" not in result.columns:
        result["strike"] = float(strike)
    result["option_type"] = option_type
    if "open_interest" not in result.columns and "open_int" in result.columns:
        result["open_interest"] = result["open_int"]
    if "volume" not in result.columns and "contracts" in result.columns:
        result["volume"] = result["contracts"]
    if "expiry_dt" in result.columns:
        result["expiry_dt"] = pd.to_datetime(result["expiry_dt"], errors="coerce").dt.date
    if "date" in result.columns:
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
        result["datetime"] = result["date"]
    keep = [column for column in ["date", "datetime", "strike", "option_type", "open", "high", "low", "close", "open_interest", "volume", "expiry_dt"] if column in result.columns]
    return result[keep].drop_duplicates().sort_values("date").reset_index(drop=True)


def _seed_for_frame(df: pd.DataFrame, seed: int) -> int:
    """Derive a stable seed from the frame contents."""

    key_parts = [str(seed)]
    for column in ["date", "strike", "option_type"]:
        if column in df.columns and not df.empty:
            key_parts.append(str(df.iloc[0][column]))
    digest = hashlib.sha256("|".join(key_parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _volume_profile(count: int) -> np.ndarray:
    """Return a volume weighting profile with higher activity near open and close."""

    profile = np.ones(count, dtype=float)
    first_block = min(6, count)
    last_block = min(6, count)
    if first_block:
        profile[:first_block] *= np.linspace(1.8, 1.2, first_block)
    if last_block:
        profile[-last_block:] *= np.linspace(1.2, 1.8, last_block)
    mid_start = first_block
    mid_end = count - last_block
    if mid_end > mid_start:
        profile[mid_start:mid_end] *= 0.9
    return profile / profile.sum()


def _generate_day_path(open_price: float, high_price: float, low_price: float, close_price: float, count: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a bounded synthetic close path for one trading day."""

    day_range = max(high_price - low_price, max(abs(close_price - open_price), 1.0))
    pivot = (high_price + low_price + close_price) / 3.0
    direction = 1.0 if close_price >= open_price else -1.0
    morning_target = np.clip(open_price + direction * 0.30 * day_range, low_price, high_price)
    midday_target = np.clip(pivot, low_price, high_price)
    afternoon_target = np.clip(close_price, low_price, high_price)

    anchors_x = np.array([0.0, 0.30, 0.70, 1.0])
    anchors_y = np.array([open_price, morning_target, midday_target, afternoon_target], dtype=float)
    x = np.linspace(0.0, 1.0, count)
    path = np.interp(x, anchors_x, anchors_y)
    noise = rng.normal(0.0, day_range * 0.012, size=count)
    path = np.clip(path + noise, low_price, high_price)
    path[0] = open_price
    path[-1] = close_price
    return path


def synthesize_intraday(daily_ohlc_df: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
    """Synthesize 5-minute intraday candles from daily OHLC data."""

    if daily_ohlc_df.empty:
        return daily_ohlc_df.copy()

    daily = _normalize_frame(daily_ohlc_df)
    if "date" not in daily.columns:
        raise ValueError("Daily OHLC data must contain a date or datetime column.")
    seed = config.SYNTHETIC_SEED if seed is None else seed
    logger.warning(
        "Using SYNTHETIC intraday data. Results are directional only. "
        "For real validation, use a broker API."
    )

    generated: list[pd.DataFrame] = []
    candle_times = pd.date_range("09:15", "15:25", freq="5min").time
    count = len(candle_times)
    volume_weights = _volume_profile(count)

    for _, row in daily.iterrows():
        trade_date = pd.Timestamp(row["date"]).date()
        row_seed = _seed_for_frame(pd.DataFrame([row]), seed)
        rng = np.random.default_rng(row_seed)

        open_price = float(row.get("open", row.get("close", 0.0)))
        high_price = float(row.get("high", max(open_price, row.get("close", open_price))))
        low_price = float(row.get("low", min(open_price, row.get("close", open_price))))
        close_price = float(row.get("close", open_price))
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        if high_price == low_price:
            high_price = low_price + 0.01

        close_path = _generate_day_path(open_price, high_price, low_price, close_price, count, rng)
        open_path = np.empty(count, dtype=float)
        open_path[0] = open_price
        open_path[1:] = close_path[:-1]

        highs = np.maximum(open_path, close_path) + np.abs(rng.normal(0.0, max((high_price - low_price) * 0.03, 0.01), size=count))
        lows = np.minimum(open_path, close_path) - np.abs(rng.normal(0.0, max((high_price - low_price) * 0.03, 0.01), size=count))
        highs = np.clip(highs, low_price, high_price)
        lows = np.clip(lows, low_price, high_price)
        highs = np.maximum(highs, np.maximum(open_path, close_path))
        lows = np.minimum(lows, np.minimum(open_path, close_path))
        highs[-1] = max(highs[-1], close_price)
        lows[-1] = min(lows[-1], close_price)

        daily_volume = row.get("volume", row.get("contracts", 0.0))
        if pd.isna(daily_volume):
            daily_volume = 0.0
        daily_volume = float(daily_volume)
        if daily_volume <= 0:
            daily_volume = max(1000.0, (high_price - low_price) * 150.0)
        volumes = np.maximum(1.0, daily_volume * volume_weights).round().astype(int)

        open_interest_base = row.get("open_interest", row.get("open_int", pd.NA))
        if pd.isna(open_interest_base):
            open_interest_series = np.full(count, pd.NA)
        else:
            oi = float(open_interest_base)
            open_interest_series = np.linspace(oi * 1.02, oi * 0.98, count)

        timestamps = [pd.Timestamp.combine(trade_date, time_obj) for time_obj in candle_times]
        out = pd.DataFrame(
            {
                "datetime": timestamps,
                "date": pd.to_datetime([trade_date] * count),
                "open": open_path,
                "high": highs,
                "low": lows,
                "close": close_path,
                "volume": volumes,
                "is_synthetic": True,
            }
        )

        if "strike" in row.index:
            out["strike"] = row.get("strike")
        if "option_type" in row.index:
            out["option_type"] = row.get("option_type")
        if "expiry_dt" in row.index:
            out["expiry_dt"] = row.get("expiry_dt")
        if "open_interest" in row.index or "open_int" in row.index:
            out["open_interest"] = open_interest_series
        if "symbol" in row.index:
            out["symbol"] = row.get("symbol")
        generated.append(out)

    combined = pd.concat(generated, ignore_index=True)
    return combined.sort_values("datetime").reset_index(drop=True)
