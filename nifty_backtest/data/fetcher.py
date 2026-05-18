"""Free-source historical data fetching with CSV caching and synthetic intraday support."""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from nifty_backtest import config
from nifty_backtest.data.free_sources import source_manager

logger = logging.getLogger(__name__)


def _ensure_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Create and return the configured cache directory."""

    cache_path = Path(cache_dir or config.CACHE_DIR).resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def _safe_key(value: Any) -> str:
    """Convert a value into a filesystem-safe cache segment."""

    text = str(value)
    for char in [":", " ", "/", "\\", ".", ",", "(", ")", "[", "]", "{", "}"]:
        text = text.replace(char, "_")
    return text


def _cache_path(kind: str, cache_dir: str | Path | None = None, **parts: Any) -> Path:
    """Build a deterministic cache file path."""

    cache_root = _ensure_cache_dir(cache_dir)
    key = "__".join([kind] + [f"{_safe_key(name)}-{_safe_key(value)}" for name, value in sorted(parts.items())])
    return cache_root / f"{key}.csv"


def _load_cache(path: Path) -> pd.DataFrame | None:
    """Load a cached CSV if it exists."""

    if not path.exists():
        return None
    logger.info("Loading cached file: %s", path)
    return pd.read_csv(path)


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    """Persist a dataframe to cache."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _to_date(value: str | datetime | date) -> date:
    """Convert a date-like value to a date object."""

    return pd.Timestamp(value).date()


def _resolve_interval(interval: str | None) -> str:
    """Normalize interval strings to a comparable form."""

    return (interval or config.INTERVAL).lower()


def _normalize_option_side(right: str) -> str:
    """Map option side names to CE/PE labels."""

    value = right.strip().upper()
    if value in {"CALL", "CE", "C"}:
        return "CE"
    if value in {"PUT", "PE", "P"}:
        return "PE"
    return value


def _maybe_synthesize_intraday(df: pd.DataFrame, seed_offset: int = 0) -> pd.DataFrame:
    """Return synthetic intraday candles when configured for 5-minute data."""

    if df.empty or not config.USE_SYNTHETIC_INTRADAY:
        return df.copy()
    synthetic = source_manager.synthesize_intraday(df, seed=config.SYNTHETIC_SEED + seed_offset)
    return synthetic


class FreeDataFetcher:
    """Fetch NIFTY spot, options, and VIX data from free public sources."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        """Initialize the fetcher with an optional cache directory."""

        self.cache_dir = _ensure_cache_dir(cache_dir)

    def get_spot_daily(self, from_dt: str | date, to_dt: str | date) -> pd.DataFrame:
        """Fetch daily NIFTY spot data using the configured source priority."""

        start = _to_date(from_dt)
        end = _to_date(to_dt)
        cache_path = _cache_path("spot", self.cache_dir, from_date=start, to_date=end)
        cached = _load_cache(cache_path)
        if cached is not None:
            return cached

        df = source_manager.get_spot_daily(start, end)
        if not df.empty:
            _save_cache(df, cache_path)
        return df

    def get_options_daily(
        self,
        strike: float | int,
        expiry_dt: str | date,
        option_type: str,
        from_dt: str | date,
        to_dt: str | date,
    ) -> pd.DataFrame:
        """Fetch daily ATM option data using the configured source priority."""

        start = _to_date(from_dt)
        end = _to_date(to_dt)
        expiry = _to_date(expiry_dt)
        option_type = option_type.upper()
        cache_path = _cache_path(
            "opt",
            self.cache_dir,
            strike=strike,
            option_type=option_type,
            expiry=expiry,
            from_date=start,
            to_date=end,
        )
        cached = _load_cache(cache_path)
        if cached is not None:
            return cached

        df = source_manager.get_options_daily(strike=float(strike), expiry_dt=expiry, option_type=option_type, from_dt=start, to_dt=end)
        if not df.empty:
            _save_cache(df, cache_path)
        return df

    def get_vix_daily(self, from_dt: str | date, to_dt: str | date) -> pd.DataFrame:
        """Fetch daily India VIX data."""

        start = _to_date(from_dt)
        end = _to_date(to_dt)
        cache_path = _cache_path("vix", self.cache_dir, from_date=start, to_date=end)
        cached = _load_cache(cache_path)
        if cached is not None:
            return cached

        df = source_manager.get_vix_daily(start, end)
        if not df.empty and "vix_close" not in df.columns:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if numeric_cols:
                df = df.rename(columns={numeric_cols[0]: "vix_close"})
            else:
                logger.warning("VIX fetch returned no numeric data. Using default VIX=14.")
                df = pd.DataFrame({"date": pd.date_range(start, end, freq="B"), "vix_close": 14.0})
        if not df.empty:
            _save_cache(df, cache_path)
        return df

    def get_all_expiries_in_range(self, from_dt: str | date, to_dt: str | date) -> list[date]:
        """Return all weekly expiry dates within the requested range."""

        return source_manager.get_all_expiries_in_range(_to_date(from_dt), _to_date(to_dt))


def fetch_spot_data(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str | None = None,
    exchange_code: str | None = None,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch spot data and synthesize intraday candles when requested."""

    _ = symbol, exchange_code
    fetcher = FreeDataFetcher(cache_dir=cache_dir)
    daily = fetcher.get_spot_daily(start_date, end_date)
    if _resolve_interval(interval) in {"5minute", "5min", "5m"}:
        return _maybe_synthesize_intraday(daily)
    return daily


def fetch_india_vix(
    start_date: str,
    end_date: str,
    interval: str | None = "1day",
    symbol: str | None = None,
    exchange_code: str | None = None,
) -> pd.DataFrame:
    """Fetch daily India VIX data."""

    _ = symbol, exchange_code, interval
    fetcher = FreeDataFetcher()
    vix_df = fetcher.get_vix_daily(start_date, end_date)
    if not vix_df.empty and "vix_close" not in vix_df.columns:
        numeric_cols = vix_df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            vix_df = vix_df.rename(columns={numeric_cols[0]: "vix_close"})
        else:
            logger.warning("VIX fetch returned no numeric data. Using default VIX=14.")
            vix_df = pd.DataFrame({"date": pd.date_range(start_date, end_date, freq="B"), "vix_close": 14.0})
    return vix_df


def fetch_option_data(
    symbol: str,
    start_date: str,
    end_date: str,
    expiry_date: str,
    strike_price: float | int,
    right: str,
    interval: str | None = None,
    exchange_code: str | None = None,
) -> pd.DataFrame:
    """Fetch option data in daily or synthetic 5-minute form."""

    _ = symbol, exchange_code
    right = _normalize_option_side(right)
    fetcher = FreeDataFetcher()
    daily = fetcher.get_options_daily(strike=strike_price, expiry_dt=expiry_date, option_type=right, from_dt=start_date, to_dt=end_date)
    if _resolve_interval(interval) in {"5minute", "5min", "5m"}:
        return _maybe_synthesize_intraday(daily, seed_offset=int(float(strike_price)))
    return daily


def fetch_atm_option_series(
    spot_df: pd.DataFrame,
    right: str,
    symbol: str | None = None,
    interval: str | None = None,
    exchange_code: str | None = None,
) -> pd.DataFrame:
    """Fetch the ATM option series implied by a spot dataframe."""

    _ = symbol, exchange_code
    right = _normalize_option_side(right)
    if spot_df.empty:
        return pd.DataFrame()

    df = spot_df.copy()
    if "datetime" not in df.columns:
        raise ValueError("Spot dataframe must contain a datetime column.")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    df["trade_date"] = df["datetime"].dt.normalize()

    if "close" not in df.columns:
        raise ValueError("Spot dataframe must contain a close column.")

    fetcher = FreeDataFetcher()
    option_frames: list[pd.DataFrame] = []
    for trade_date, group in df.groupby("trade_date"):
        group = group.sort_values("datetime")
        spot_close = float(group["close"].iloc[-1])
        atm_strike = round(spot_close / config.ATM_ROUNDING) * config.ATM_ROUNDING
        expiry_dt = source_manager.next_expiry_for_date(pd.Timestamp(trade_date).date())
        option_daily = fetcher.get_options_daily(
            strike=atm_strike,
            expiry_dt=expiry_dt,
            option_type=right,
            from_dt=pd.Timestamp(trade_date).date(),
            to_dt=pd.Timestamp(trade_date).date(),
        )
        if option_daily.empty:
            continue
        if _resolve_interval(interval) in {"5minute", "5min", "5m"}:
            option_intraday = source_manager.synthesize_intraday(option_daily, seed=config.SYNTHETIC_SEED + int(atm_strike))
            option_intraday["trade_date"] = pd.Timestamp(trade_date)
            option_frames.append(option_intraday)
        else:
            option_daily = option_daily.copy()
            option_daily["trade_date"] = pd.Timestamp(trade_date)
            option_frames.append(option_daily)

    if not option_frames:
        return pd.DataFrame()

    combined = pd.concat(option_frames, ignore_index=True)
    return combined.sort_values("datetime").reset_index(drop=True)
