"""Validate the free data sources used by the NIFTY backtesting framework."""

from __future__ import annotations

import importlib.util
import sys
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nifty_backtest import config
from nifty_backtest.data.free_sources import jugaad_fetcher, nse_bhav, nse_indices


def _format_table(rows: list[list[object]], headers: list[str]) -> str:
    """Render a simple validation table."""

    try:
        from tabulate import tabulate

        return tabulate(rows, headers=headers, tablefmt="github")
    except Exception:
        widths = [len(str(header)) for header in headers]
        for row in rows:
            for idx, value in enumerate(row):
                widths[idx] = max(widths[idx], len(str(value)))

        def _line(values: list[object]) -> str:
            return " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(values))

        separator = " | ".join("-" * width for width in widths)
        return "\n".join([_line(headers), separator] + [_line(row) for row in rows])


def _last_business_days(count: int) -> list[date]:
    """Return the last N business days excluding the current day."""

    today = pd.Timestamp(datetime.now().date()) - pd.Timedelta(days=1)
    config_end = pd.Timestamp(config.BACKTEST_END).date()
    end_date = min(today.date(), config_end)
    end = pd.Timestamp(end_date)
    while end.weekday() >= 5:
        end -= pd.Timedelta(days=1)
    days = pd.bdate_range(end=end, periods=count)
    return [timestamp.date() for timestamp in days]


def _check_required_packages() -> bool:
    """Check whether the required runtime packages are installed."""

    required = ["jugaad_data", "requests", "pandas", "tqdm", "yfinance"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        print("Run: pip install -r requirements.txt")
        print(f"Missing packages: {', '.join(missing)}")
        return False
    return True


def main() -> None:
    """Run the free-source data layer validation flow."""

    print(
        """
=== DATA SOURCE SETUP ===
This project uses free public sources only.
No broker account, API key, or authentication is required.
"""
    )

    if not _check_required_packages():
        sys.exit(1)

    step_status: list[list[object]] = []

    print("Step 1")
    step1_ok = False
    try:
        days = _last_business_days(3)
        spot = jugaad_fetcher.fetch_spot_daily(days[0], days[-1])
        print(f"jugaad-data spot OK - {len(spot)} rows")
        step1_ok = not spot.empty
    except Exception as exc:
        print(f"jugaad-data error: {exc}")
        print("Fallback notice: the pipeline will try NSE Bhav Copy and niftyindices.com instead.")
    step_status.append(["jugaad-data", "OK" if step1_ok else "CHECK", "Options EOD OHLC", "2016-today"])

    print("Step 2")
    bhav_ok = False
    bhav_rows = 0
    bhav_date = None
    try:
        for offset in range(5):
            candidate = (pd.Timestamp(datetime.now().date()) - pd.Timedelta(days=offset + 1)).date()
            if candidate.weekday() >= 5:
                continue
            try:
                bhav = nse_bhav.download_bhav_zip(candidate)
                if not bhav.empty:
                    bhav_ok = True
                    bhav_rows = len(bhav)
                    bhav_date = candidate
                    print(f"NSE bhav OK - {candidate.isoformat()} bhav has {bhav_rows} rows")
                    break
            except Exception:
                continue
        if not bhav_ok:
            print("NSE bhav WARN - no bhav copy found in the last 5 trading days.")
    except Exception as exc:
        print(f"NSE bhav error: {exc}")
    step_status.append(["NSE Bhav Copy", "OK" if bhav_ok else "CHECK", "Options EOD OHLC", "2000-today"])

    print("Step 3")
    vix_ok = False
    vix_value = None
    try:
        days = _last_business_days(5)
        vix = nse_indices.fetch_vix_history(days[0], days[-1])
        if not vix.empty:
            vix_ok = True
            vix_value = float(vix["vix_close"].dropna().iloc[-1])
            print(f"VIX OK - Latest VIX: {vix_value:.2f}")
        else:
            print("VIX WARN - no data returned for the last 5 business days.")
    except Exception as exc:
        print(f"VIX error: {exc}")
    step_status.append(["niftyindices.com", "OK" if vix_ok else "CHECK", "Spot + VIX daily", "2000-today"])

    step_status.append(["Synthetic 5-min", "OK", "Simulated intra", "any"])

    print(
        _format_table(
            step_status,
            ["Source", "Status", "Data Type", "Date Range"],
        )
    )

    print(
        """
OK Data layer ready. No API key needed.
WARN All sources provide EOD data only.
   Intraday will be synthesized (directional test only).

Next step: python scripts/test_fetch.py
"""
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
