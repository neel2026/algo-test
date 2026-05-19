"""FastAPI entrypoint for the live trading signal scanner."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import json
import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.routing import APIRouter

from auth.token_store import TokenStore
from auth.upstox_auth import UpstoxAuth
from feed.candle_builder import CandleBuilder
from feed.upstox_feed import UpstoxFeed
from instruments.instrument_list import INDICES, TOP_FO_STOCKS, search_instruments
from signals.indicator_engine import IndicatorEngine
from signals.signal_checker import SignalChecker

ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(path: Path | str | None = None) -> bool:
        """Fallback .env loader when python-dotenv is unavailable."""

        env_path = Path(path) if path else ROOT / ".env"
        if not env_path.exists():
            return False
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()
        return True

load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
HISTORY_FETCH_EXECUTOR = ThreadPoolExecutor(max_workers=2)

app = FastAPI(title="NIFTY Live Scanner")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

state: dict[str, Any] = {
    "access_token": None,
    "feed": None,
    "loop": None,
    "auth": None,
    "candle_builder": CandleBuilder(interval_minutes=int(os.getenv("DEFAULT_INTERVAL", 5))),
    "indicator_engine": IndicatorEngine(),
    "signal_checker": SignalChecker(),
    "sse_queues": [],
    "candle_history": {},
    "signal_history": [],
    "current_ltp": {},
    "current_vix": None,
    "latest_indicators": {},
    "vix_task": None,
    "current_instrument": os.getenv("DEFAULT_INSTRUMENT", "NSE_INDEX|Nifty 50"),
    "interval": int(os.getenv("DEFAULT_INTERVAL", 5)),
}

api = APIRouter(prefix="/api")


def _enqueue(queue: asyncio.Queue, event: dict) -> None:
    """Thread-safe queue insertion helper."""

    try:
        queue.put_nowait(event)
    except asyncio.QueueFull:
        pass


def broadcast(event_type: str, data: dict) -> None:
    """Broadcast a server-sent event to all connected clients."""

    loop = state.get("loop")
    event = {"type": event_type, "data": data, "ts": pd.Timestamp.now(tz="Asia/Kolkata").isoformat()}
    for queue in list(state["sse_queues"]):
        if loop is not None:
            loop.call_soon_threadsafe(_enqueue, queue, event)
        else:
            _enqueue(queue, event)


def _load_access_token() -> str | None:
    """Load the current Upstox token from disk if it is valid today."""

    token = TokenStore.load()
    if token:
        state["access_token"] = token
    return token


def _start_feed() -> None:
    """Start or refresh the websocket feed."""

    if not state.get("access_token"):
        return
    current_instrument = state["current_instrument"]
    if state["feed"] is None:
        state["feed"] = UpstoxFeed(state["access_token"], state["candle_builder"], on_candle_close, on_tick)
        state["feed"].start([current_instrument])
    else:
        state["feed"].token = state["access_token"]
        state["feed"].builder = state["candle_builder"]
        state["feed"].change_instruments([current_instrument])


async def poll_vix(token: str) -> None:
    """Poll India VIX periodically and cache the latest value."""

    while True:
        try:
            response = await asyncio.to_thread(
                requests.get,
                "https://api.upstox.com/v2/market-quote/quotes",
                params={"instrument_key": "NSE_INDEX|India VIX"},
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                },
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json().get("data", {})
            vix_data = (
                payload.get("NSE_INDEX:India VIX")
                or payload.get("NSE_INDEX|India VIX")
                or next(iter(payload.values()), {})
            )
            vix_val = vix_data.get("last_price")
            if vix_val is not None:
                state["current_vix"] = float(vix_val)
                state["signal_checker"].current_vix = float(vix_val)
                logger.info("VIX updated: %s", vix_val)
        except Exception as exc:
            logger.warning("VIX poll failed: %s", exc)
        await asyncio.sleep(60)


def _ensure_vix_poller() -> None:
    """Start the VIX polling task once per process."""

    loop = state.get("loop")
    token = state.get("access_token")
    task = state.get("vix_task")
    if loop is None or not token:
        return
    if task and not task.done():
        return
    state["vix_task"] = loop.create_task(poll_vix(token))


def _on_token(token: str) -> None:
    """Handle a fresh token from the OAuth flow."""

    state["access_token"] = token
    _start_feed()
    _ensure_vix_poller()


state["auth"] = UpstoxAuth(on_token=_on_token)


def _interval_to_str(minutes: int) -> str:
    """Map a numeric interval to the Upstox candle string."""

    return {
        1: "1minute",
        3: "3minute",
        5: "5minute",
        15: "15minute",
        30: "30minute",
        60: "60minute",
    }.get(int(minutes), "5minute")


def _previous_trading_day(reference: date | None = None) -> date:
    """Return the most recent completed trading day."""

    current = reference or date.today()
    previous = current - timedelta(days=1)
    while previous.weekday() >= 5:
        previous -= timedelta(days=1)
    return previous


def _resample_candles(frame: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    """Resample 1-minute candles to the requested interval."""

    if frame.empty or interval_minutes <= 1:
        return frame.copy()

    working = frame.copy()
    working["datetime"] = pd.to_datetime(working["datetime"])
    working = working.set_index("datetime").sort_index()
    resampled = (
        working.resample(f"{interval_minutes}min", label="left", closed="left")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    return resampled


def _format_history_candle(candle: list) -> dict:
    """Format one Upstox candle into lightweight-charts format."""

    return {
        "time": int(pd.Timestamp(candle[0]).timestamp()),
        "open": candle[1],
        "high": candle[2],
        "low": candle[3],
        "close": candle[4],
        "volume": candle[5] if len(candle) > 5 else 0,
    }


def _fetch_historical_frame(instrument_key: str, interval_minutes: int, days: int = 2) -> pd.DataFrame:
    """Fetch a historical candle frame for the requested instrument and interval."""

    if not state.get("access_token"):
        token = _load_access_token()
        if not token:
            raise HTTPException(status_code=401, detail="Not authenticated")

    interval_str = _interval_to_str(interval_minutes)
    to_dt = _previous_trading_day()
    from_dt = to_dt - timedelta(days=max(10, days * 5))
    safe_key = requests.utils.quote(instrument_key, safe="")

    attempts = [
        {
            "label": interval_str,
            "url": f"https://api.upstox.com/v2/historical-candle/{safe_key}/{interval_str}/{to_dt.isoformat()}/{from_dt.isoformat()}",
            "source": "upstox_history",
            "resample_to": None,
        }
    ]
    if interval_minutes > 1:
        attempts.append(
            {
                "label": "1minute",
                "url": f"https://api.upstox.com/v2/historical-candle/{safe_key}/1minute/{to_dt.isoformat()}/{from_dt.isoformat()}",
                "source": "upstox_history_resampled",
                "resample_to": interval_minutes,
            }
        )

    last_error = "No history attempt executed."
    for attempt in attempts:
        try:
            future = HISTORY_FETCH_EXECUTOR.submit(
                requests.get,
                attempt["url"],
                headers={
                    "Authorization": f"Bearer {state['access_token']}",
                    "Accept": "application/json",
                },
                timeout=3,
            )
            resp = future.result(timeout=4)
            resp.raise_for_status()
            candles = resp.json().get("data", {}).get("candles", [])
            if not candles:
                last_error = f"Upstox returned zero candles for {attempt['label']}."
                continue

            rows = []
            for candle in reversed(candles):
                candle_time = pd.Timestamp(candle[0])
                if candle_time.tzinfo is None:
                    candle_time = candle_time.tz_localize("UTC")
                candle_time = candle_time.tz_convert("Asia/Kolkata")
                rows.append(
                    {
                        "datetime": candle_time.to_pydatetime(),
                        "open": candle[1],
                        "high": candle[2],
                        "low": candle[3],
                        "close": candle[4],
                        "volume": candle[5] if len(candle) > 5 else 0,
                    }
                )

            frame = pd.DataFrame(rows)
            if attempt["resample_to"]:
                frame = _resample_candles(frame, int(attempt["resample_to"]))

            frame["date"] = pd.to_datetime(frame["datetime"]).dt.date
            unique_dates = sorted(frame["date"].unique())
            if len(unique_dates) > days + 1:
                keep_dates = unique_dates[-(days + 1) :]
                frame = frame[frame["date"].isin(keep_dates)].copy()
            frame.attrs["history_meta"] = {
                "status": "ok",
                "source": attempt["source"],
                "reason": "loaded",
                "detail": f"Loaded {len(frame)} candles via {attempt['label']} ending {to_dt.isoformat()}.",
            }
            return frame.reset_index(drop=True)
        except FutureTimeoutError as exc:
            last_error = f"{attempt['label']} timed out: {exc}"
            logger.warning("Historical fetch timed out for %s via %s: %s", instrument_key, attempt["label"], exc)
        except Exception as exc:
            last_error = str(exc)
            logger.warning("Historical fetch failed for %s via %s: %s", instrument_key, attempt["label"], exc)

    fallback = state["candle_history"].get(instrument_key, [])
    if fallback:
        frame = pd.DataFrame(
            [
                {
                    "datetime": item.get("datetime"),
                    "open": item.get("open"),
                    "high": item.get("high"),
                    "low": item.get("low"),
                    "close": item.get("close"),
                    "volume": item.get("volume", 0),
                }
                for item in fallback
            ]
        )
        frame.attrs["history_meta"] = {
            "status": "fallback",
            "source": "live_cache",
            "reason": "request_failed",
            "detail": last_error,
        }
        return frame

    empty = pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    empty.attrs["history_meta"] = {
        "status": "error",
        "source": "upstox_history",
        "reason": "request_failed",
        "detail": last_error,
    }
    return empty


def _build_history_payload(instrument_key: str, interval_minutes: int, days: int = 2) -> dict:
    """Build a historical payload containing candles and computed signals."""

    frame = _fetch_historical_frame(instrument_key, interval_minutes, days=days)
    history_meta = frame.attrs.get("history_meta", {})
    if frame.empty:
        return {
            "candles": [],
            "signals": [],
            "indicators": {"vix": state.get("current_vix")},
            "meta": {
                "instrument": instrument_key,
                "interval": interval_minutes,
                "days": days,
                "candle_count": 0,
                "signal_count": 0,
                **history_meta,
            },
        }

    working = frame.copy().sort_values("datetime").reset_index(drop=True)
    unique_dates = sorted(working["date"].unique())
    target_dates = unique_dates[-days:]

    day_ohlc: dict[date, dict] = {}
    for candle_date, day_frame in working.groupby("date", sort=True):
        day_ohlc[candle_date] = {
            "open": float(day_frame.iloc[0]["open"]),
            "high": float(day_frame["high"].max()),
            "low": float(day_frame["low"].min()),
            "close": float(day_frame.iloc[-1]["close"]),
        }

    signals: list[dict] = []
    latest_indicators: dict[str, Any] = {"vix": state.get("current_vix")}
    for idx in range(len(working)):
        candle = working.iloc[idx]
        candle_date = candle["date"]
        prev_day = {}
        if candle_date in unique_dates:
            prev_index = unique_dates.index(candle_date) - 1
            if prev_index >= 0:
                prev_day = day_ohlc.get(unique_dates[prev_index], {})
        try:
            indicators = state["indicator_engine"].compute(
                working.iloc[: idx + 1].drop(columns=["date"]),
                prev_day,
                instrument_key=instrument_key,
            )
        except Exception as exc:
            logger.warning("Indicator compute failed for %s on %s: %s", instrument_key, candle_date, exc)
            continue
        if not indicators:
            continue
        indicators["vix"] = state.get("current_vix")
        latest_indicators = indicators
        try:
            signal = state["signal_checker"].check(
                instrument_key,
                candle.to_dict(),
                indicators,
                indicators,
            )
        except Exception as exc:
            logger.warning("Signal check failed for %s on %s: %s", instrument_key, candle_date, exc)
            signal = None
        if signal and pd.to_datetime(signal["datetime"]).date() in target_dates:
            signals.append(signal)

    visible = working[working["date"].isin(target_dates)].copy()
    visible = visible.sort_values("datetime").reset_index(drop=True)
    candles = [
        {
            "time": int(pd.Timestamp(row["datetime"]).timestamp()),
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row["volume"],
        }
        for _, row in visible.iterrows()
    ]
    return {
        "candles": candles,
        "signals": signals,
        "indicators": latest_indicators,
        "meta": {
            "instrument": instrument_key,
            "interval": interval_minutes,
            "days": days,
            "candle_count": len(candles),
            "signal_count": len(signals),
            **history_meta,
        },
    }


def on_tick(instrument_key: str, ltp: float, ts_ms: int) -> None:
    """Handle live tick updates from the websocket."""

    state["current_ltp"][instrument_key] = ltp
    current = state["candle_builder"].get_current_candle(instrument_key) or {}
    broadcast(
        "tick",
        {
            "instrument": instrument_key,
            "ltp": ltp,
            "time": int(ts_ms // 1000),
            "candle_time_ts": int(current["datetime"].timestamp()) if current.get("datetime") else None,
        },
    )


def on_candle_close(instrument_key: str, candle: dict) -> None:
    """Handle one closed candle from the websocket."""

    state["candle_history"].setdefault(instrument_key, []).append(candle)
    broadcast(
        "candle",
        {
            "instrument": instrument_key,
            "time": int(pd.Timestamp(candle["datetime"]).timestamp()),
            "open": candle["open"],
            "high": candle["high"],
            "low": candle["low"],
            "close": candle["close"],
            "volume": candle["volume"],
        },
    )

    df = state["candle_builder"].get_candles_df(instrument_key)
    prev_day = {}
    if state.get("access_token"):
        try:
            prev_day = state["indicator_engine"].fetch_prev_day(instrument_key, state["access_token"])
        except Exception as exc:
            logger.warning("Previous day fetch failed for %s: %s", instrument_key, exc)

    indicators = state["indicator_engine"].compute(df, prev_day, instrument_key=instrument_key)
    if indicators:
        indicators["vix"] = state.get("current_vix")
        state["latest_indicators"] = indicators
        signal = state["signal_checker"].check(instrument_key, candle, indicators, indicators)
        if signal:
            state["signal_history"].append(signal)
            state["signal_history"] = state["signal_history"][-50:]
            broadcast("signal", signal)
        numeric_indicators = {k: v for k, v in indicators.items() if isinstance(v, (int, float, bool))}
        numeric_indicators["instrument"] = instrument_key
        broadcast("indicators", numeric_indicators)


@api.get("/auth/login")
def login():
    """Redirect to the Upstox login page."""

    auth: UpstoxAuth = state["auth"]
    auth.start_callback_server()
    return RedirectResponse(auth.get_login_url())


@api.get("/auth/status")
def auth_status():
    """Return authentication status and lazily start the live feed."""

    token = _load_access_token()
    if token and state["feed"] is None:
        _start_feed()
    return {
        "authenticated": bool(token),
        "login_url": "/api/auth/login",
    }


@api.get("/state")
def get_state():
    """Return the current scanner state for frontend initialization."""

    return {
        "current_instrument": state["current_instrument"],
        "interval": state["interval"],
        "authenticated": bool(state.get("access_token")),
        "signal_count": len(state["signal_history"]),
        "latest_signal": state["signal_history"][-1] if state["signal_history"] else None,
        "current_vix": state.get("current_vix"),
        "latest_indicators": state.get("latest_indicators", {}),
    }


@api.get("/instruments")
def get_instruments():
    """Return the supported instrument groups."""

    return {"indices": INDICES, "stocks": TOP_FO_STOCKS}


@api.get("/instruments/search")
def search(q: str = Query(default="")):
    """Search the instrument universe."""

    return search_instruments(q)


@api.post("/settings/instrument")
def set_instrument(instrument_key: str):
    """Change the active instrument subscription."""

    state["current_instrument"] = instrument_key
    if state["feed"]:
        state["feed"].change_instruments([instrument_key])
    return {"ok": True, "instrument": instrument_key}


@api.post("/settings/interval")
def set_interval(minutes: int):
    """Change the candle aggregation interval."""

    state["interval"] = int(minutes)
    state["candle_builder"] = CandleBuilder(interval_minutes=int(minutes))
    if state["feed"]:
        state["feed"].builder = state["candle_builder"]
    return {"ok": True, "interval": int(minutes)}


@api.get("/candles/{instrument_key:path}")
def get_candles(instrument_key: str, interval: int = Query(default=None)):
    """Return historical candles for the chart."""

    interval_minutes = int(interval or state["interval"])
    payload = _build_history_payload(instrument_key, interval_minutes, days=2)
    return {"candles": payload["candles"]}


@api.get("/history/{instrument_key:path}")
def get_history(instrument_key: str, interval: int = Query(default=None), days: int = Query(default=2, ge=1, le=10)):
    """Return candles and computed historical signals for the selected window."""

    interval_minutes = int(interval or state["interval"])
    return _build_history_payload(instrument_key, interval_minutes, days=days)


@api.get("/stream")
async def stream():
    """Server-Sent Events endpoint for live ticks, candles, and signals."""

    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    state["sse_queues"].append(queue)
    initial_event = {
        "type": "status",
        "data": {
            "connected": True,
            "authenticated": bool(state.get("access_token")),
            "current_instrument": state["current_instrument"],
            "interval": state["interval"],
        },
        "ts": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
    }

    async def event_generator():
        try:
            yield f"data: {json.dumps(initial_event)}\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    event = {
                        "type": "heartbeat",
                        "data": {"connected": True},
                        "ts": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
                    }
                yield f"data: {json.dumps(event)}\n\n"
        except asyncio.CancelledError:
            if queue in state["sse_queues"]:
                state["sse_queues"].remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


app.include_router(api)


@app.on_event("startup")
async def startup() -> None:
    """Initialize runtime state and auto-connect when a token exists."""

    state["loop"] = asyncio.get_running_loop()
    token = _load_access_token()
    if token:
        _start_feed()
        _ensure_vix_poller()
