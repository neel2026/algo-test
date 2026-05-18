"""Niftyindices.com historical index data fetchers."""

from __future__ import annotations

import json
import logging
import time
from datetime import date
from typing import Any

import pandas as pd
import requests

from nifty_backtest import config

logger = logging.getLogger(__name__)

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
]


def _request_with_retry(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    session: requests.Session | None = None,
) -> requests.Response:
    """Perform an HTTP request with retry, backoff, and user-agent rotation."""

    session = session or requests.Session()
    last_error: Exception | None = None
    headers = dict(headers or {})
    for attempt in range(3):
        headers["User-Agent"] = _USER_AGENTS[attempt % len(_USER_AGENTS)]
        try:
            if method.upper() == "POST":
                response = session.post(url, headers=headers, data=data, json=json_body, timeout=30)
            else:
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
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
    assert last_error is not None
    raise last_error


def _warm_session(session: requests.Session, headers: dict[str, str]) -> None:
    """Warm up an NSE session cookie for direct API access."""

    try:
        session.get("https://www.nseindia.com", headers=headers, timeout=30)
    except Exception:
        pass


def _parse_payload(payload: Any) -> pd.DataFrame:
    """Convert a JSON payload into a normalized dataframe."""

    if isinstance(payload, pd.DataFrame):
        return payload.copy()
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        for key in ["data", "Data", "result", "rows", "records", "aaData"]:
            if key in payload:
                value = payload[key]
                if isinstance(value, list):
                    return pd.DataFrame(value)
                if isinstance(value, dict):
                    return pd.DataFrame(value)
        return pd.DataFrame(payload)
    return pd.DataFrame()


def _normalize_yfinance_history(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a yfinance history dataframe into the project schema."""

    if df.empty:
        return df.copy()

    result = df.copy().reset_index()
    result.columns = [str(column).strip().lower().replace(" ", "_").replace(".", "").replace("-", "_") for column in result.columns]
    if "datetime" in result.columns and "date" not in result.columns:
        result = result.rename(columns={"datetime": "date"})
    if "date" not in result.columns and "index" in result.columns:
        result = result.rename(columns={"index": "date"})
    if "date" in result.columns:
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
        result["datetime"] = result["date"]
    for column in ["open", "high", "low", "close", "adj_close", "volume"]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    if "close" not in result.columns and "adj_close" in result.columns:
        result["close"] = result["adj_close"]
    if "volume" not in result.columns:
        result["volume"] = pd.NA
    return result.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _fetch_yfinance_history(ticker: str, from_dt: date, to_dt: date) -> pd.DataFrame:
    """Fetch daily history from Yahoo Finance as a free fallback."""

    try:
        import yfinance as yf
    except Exception as exc:
        logger.warning("yfinance unavailable: %s", exc)
        return pd.DataFrame()

    try:
        history = yf.Ticker(ticker).history(start=pd.Timestamp(from_dt), end=pd.Timestamp(to_dt) + pd.Timedelta(days=1), auto_adjust=False)
        return _normalize_yfinance_history(history)
    except Exception as exc:
        logger.warning("yfinance fetch failed for %s: %s", ticker, exc)
        return pd.DataFrame()


def _normalize_index_df(df: pd.DataFrame, close_name: str = "close") -> pd.DataFrame:
    """Normalize index data into the project schema."""

    if df.empty:
        return df.copy()
    result = df.copy()
    result.columns = [column.strip().lower().replace(" ", "_").replace(".", "").replace("-", "_") for column in result.columns]
    if "index_date" in result.columns and "date" not in result.columns:
        result = result.rename(columns={"index_date": "date"})
    if "historicaldate" in result.columns and "date" not in result.columns:
        result = result.rename(columns={"historicaldate": "date"})
    if "date" in result.columns:
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
        result["datetime"] = result["date"]
    if "closing" in result.columns and close_name not in result.columns:
        result = result.rename(columns={"closing": close_name})
    if "close" not in result.columns and close_name in result.columns:
        result = result.rename(columns={close_name: "close"})
    for column in ["open", "high", "low", "close", "volume"]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    if "volume" not in result.columns:
        result["volume"] = pd.NA
    return result.sort_values("date").reset_index(drop=True)


def _normalize_vix_output(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize VIX output to a date/vix_close dataframe."""

    if df.empty:
        return pd.DataFrame(columns=["date", "vix_close"])

    result = df.copy()
    result.columns = [str(column).strip().lower().replace(" ", "_").replace(".", "").replace("-", "_") for column in result.columns]
    date_col = next((col for col in ["date", "index_date", "historicaldate", "tradetd", "trad_dt"] if col in result.columns), None)
    if date_col is None and isinstance(result.index, pd.DatetimeIndex):
        result = result.reset_index().rename(columns={"index": "date"})
        date_col = "date"
    elif date_col is None and "index" in result.columns:
        date_col = "index"
        result = result.rename(columns={"index": "date"})

    if date_col is not None and date_col != "date":
        result = result.rename(columns={date_col: "date"})

    close_col = next((col for col in ["vix_close", "close", "closing", "adj_close", "vix", "c"] if col in result.columns), None)
    if close_col is None:
        numeric_cols = result.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            close_col = numeric_cols[0]
        else:
            return pd.DataFrame(columns=["date", "vix_close"])

    result = result.rename(columns={close_col: "vix_close"})
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.date
    result["vix_close"] = pd.to_numeric(result["vix_close"], errors="coerce")
    result = result[["date", "vix_close"]].dropna(subset=["date"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return result


def fetch_index_history(name: str, from_dt: date, to_dt: date) -> pd.DataFrame:
    """Fetch historical index data from niftyindices.com."""

    url = "https://www.niftyindices.com/Backpage.aspx/getHistoricaldatatabletoString"
    headers = {"Referer": "https://www.niftyindices.com", "Content-Type": "application/json; charset=utf-8"}
    payload = {
        "name": name,
        "indexName": name,
        "startDate": pd.Timestamp(from_dt).strftime("%d-%b-%Y"),
        "endDate": pd.Timestamp(to_dt).strftime("%d-%b-%Y"),
    }
    try:
        response = _request_with_retry(url, method="POST", headers=headers, data=json.dumps({"cinfo": json.dumps(payload)}))
        body = response.json()
        data = body.get("d", body)
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                pass
        df = _parse_payload(data)
        if not df.empty:
            if name.strip().upper() == "INDIA VIX":
                return _normalize_vix_output(df)
            return _normalize_index_df(df)
    except Exception as exc:
        logger.warning("niftyindices.com fetch failed for %s: %s", name, exc)

    ticker = "^NSEI" if name.strip().upper() in {"NIFTY 50", "NIFTY50"} else "^INDIAVIX"
    fallback = _fetch_yfinance_history(ticker, from_dt, to_dt)
    if name.strip().upper() == "INDIA VIX":
        return _normalize_vix_output(fallback)
    return _normalize_index_df(fallback)


def fetch_nse_spot_history(from_dt: date, to_dt: date) -> pd.DataFrame:
    """Fetch NIFTY 50 historical data from NSE direct API."""

    session = requests.Session()
    headers = {
        "User-Agent": _USER_AGENTS[0],
        "Referer": "https://www.nseindia.com",
        "Accept": "application/json",
    }
    _warm_session(session, headers)
    url = (
        "https://www.nseindia.com/api/historical/indicesHistory"
        f"?indexType=NIFTY%2050&from={pd.Timestamp(from_dt).strftime('%d-%m-%Y')}&to={pd.Timestamp(to_dt).strftime('%d-%m-%Y')}"
    )
    response = _request_with_retry(url, method="GET", headers=headers, session=session)
    data = response.json()
    df = _parse_payload(data.get("data", data))
    if not df.empty:
        return _normalize_index_df(df)
    return _fetch_yfinance_history("^NSEI", from_dt, to_dt)


def fetch_vix_history(from_dt: date, to_dt: date) -> pd.DataFrame:
    """Fetch India VIX daily data from niftyindices.com."""

    df = fetch_index_history("India VIX", from_dt, to_dt)
    if df.empty:
        fallback = _fetch_yfinance_history("^INDIAVIX", from_dt, to_dt)
        return _normalize_vix_output(fallback)
    return _normalize_vix_output(df)
