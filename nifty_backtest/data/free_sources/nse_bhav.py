"""NSE bhav copy downloader and parser for options fallback."""

from __future__ import annotations

import io
import logging
import time
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from nifty_backtest import config

logger = logging.getLogger(__name__)

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
]


def _cache_dir() -> Path:
    """Return the configured cache directory."""

    cache_dir = config.resolve_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _cache_path(trade_date: date) -> Path:
    """Return the cache path for a bhav copy date."""

    return _cache_dir() / f"bhav_{trade_date.isoformat()}.csv"


def _request_with_retry(url: str, max_retries: int = 3) -> requests.Response:
    """Download a URL with retry, backoff, and user-agent rotation."""

    last_error: Exception | None = None
    headers = {"User-Agent": _USER_AGENTS[0], "Referer": "https://www.nseindia.com"}
    session = requests.Session()
    for attempt in range(max_retries):
        headers["User-Agent"] = _USER_AGENTS[attempt % len(_USER_AGENTS)]
        try:
            response = session.get(url, headers=headers, timeout=30)
            if response.status_code == 403:
                logger.warning("403 received for %s; rotating User-Agent and retrying.", url)
                time.sleep(2 ** (attempt + 1))
                continue
            if response.status_code == 429:
                logger.warning("429 received for %s; cooling down for 60 seconds.", url)
                time.sleep(60)
                continue
            response.raise_for_status()
            return response
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
    assert last_error is not None
    raise last_error


def _bhav_url(trade_date: date) -> str:
    """Build the NSE bhav copy URL for a given trading date."""

    month = trade_date.strftime("%b").upper()
    return (
        "https://nsearchives.nseindia.com/content/historical/DERIVATIVES/"
        f"{trade_date:%Y}/{month}/fo{trade_date:%d}{month}{trade_date:%Y}bhav.csv.zip"
    )


def _read_zip_csv(content: bytes) -> pd.DataFrame:
    """Read the first CSV file from a zip payload."""

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        csv_name = next((name for name in zf.namelist() if name.lower().endswith(".csv")), None)
        if csv_name is None:
            return pd.DataFrame()
        with zf.open(csv_name) as handle:
            return pd.read_csv(handle)


def download_bhav_zip(trade_date: date) -> pd.DataFrame:
    """Download and parse the NSE bhav copy for a single date."""

    cache_path = _cache_path(trade_date)
    if cache_path.exists():
        return pd.read_csv(cache_path)

    url = _bhav_url(trade_date)
    response = _request_with_retry(url)
    df = _read_zip_csv(response.content)
    if not df.empty:
        df.to_csv(cache_path, index=False)
    return df
