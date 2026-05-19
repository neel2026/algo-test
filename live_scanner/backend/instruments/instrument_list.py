"""Instrument universe for the live scanner."""

from __future__ import annotations

from typing import Iterable

INDICES = [
    {"key": "NSE_INDEX|Nifty 50", "label": "Nifty 50", "group": "Indices"},
    {"key": "NSE_INDEX|Nifty Bank", "label": "Nifty Bank", "group": "Indices"},
    {"key": "NSE_INDEX|Nifty Fin Service", "label": "Nifty Fin Service", "group": "Indices"},
    {"key": "NSE_INDEX|Nifty Next 50", "label": "Nifty Next 50", "group": "Indices"},
    {"key": "NSE_INDEX|India VIX", "label": "India VIX", "group": "Indices"},
]

TOP_FO_STOCKS = [
    {"key": "NSE_EQ|Reliance", "label": "Reliance", "group": "F&O Stocks"},
    {"key": "NSE_EQ|TCS", "label": "TCS", "group": "F&O Stocks"},
    {"key": "NSE_EQ|HDFCBANK", "label": "HDFCBANK", "group": "F&O Stocks"},
    {"key": "NSE_EQ|ICICIBANK", "label": "ICICIBANK", "group": "F&O Stocks"},
    {"key": "NSE_EQ|INFY", "label": "INFY", "group": "F&O Stocks"},
    {"key": "NSE_EQ|SBIN", "label": "SBIN", "group": "F&O Stocks"},
    {"key": "NSE_EQ|LT", "label": "LT", "group": "F&O Stocks"},
    {"key": "NSE_EQ|ITC", "label": "ITC", "group": "F&O Stocks"},
    {"key": "NSE_EQ|BHARTIARTL", "label": "BHARTIARTL", "group": "F&O Stocks"},
    {"key": "NSE_EQ|HINDUNILVR", "label": "HINDUNILVR", "group": "F&O Stocks"},
    {"key": "NSE_EQ|AXISBANK", "label": "AXISBANK", "group": "F&O Stocks"},
    {"key": "NSE_EQ|KOTAKBANK", "label": "KOTAKBANK", "group": "F&O Stocks"},
]


def all_instruments() -> list[dict]:
    """Return the full instrument universe."""

    return INDICES + TOP_FO_STOCKS


def search_instruments(query: str | None) -> dict[str, list[dict]]:
    """Search instruments by label or key and preserve grouping."""

    q = (query or "").strip().lower()
    if not q:
        return {"indices": INDICES, "stocks": TOP_FO_STOCKS}

    def _match(items: Iterable[dict]) -> list[dict]:
        return [item for item in items if q in item["label"].lower() or q in item["key"].lower()]

    return {"indices": _match(INDICES), "stocks": _match(TOP_FO_STOCKS)}
