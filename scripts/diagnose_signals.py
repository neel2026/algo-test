"""Standalone signal audit for a single month of NIFTY backtest data."""

from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

try:  # pragma: no cover - optional dependency
    from tabulate import tabulate
except Exception:  # pragma: no cover - fallback
    tabulate = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nifty_backtest import config
from nifty_backtest.data.fetcher import FreeDataFetcher, fetch_india_vix, fetch_option_data, fetch_spot_data
from nifty_backtest.data.processor import build_backtest_frame, clean_ohlcv, synthesize_intraday
from nifty_backtest.strategy.atm_strategy import ATMStrategy


def _format_table(rows: list[list[object]], headers: list[str]) -> str:
    """Render a compact text table."""

    if tabulate is not None:
        return tabulate(rows, headers=headers, tablefmt="github")

    widths = [len(str(header)) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(str(value)))

    def _line(values: list[object]) -> str:
        return " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(values))

    separator = " | ".join("-" * width for width in widths)
    return "\n".join([_line(headers), separator] + [_line(row) for row in rows])


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Diagnose ATM strategy signals for one month.")
    parser.add_argument("--month", default="2024-06", help="Month to inspect in YYYY-MM format.")
    return parser.parse_args()


def _clear_synthetic_cache() -> int:
    """Remove synthetic option caches so they can be regenerated."""

    synthetic_files = glob.glob(str(ROOT / "nifty_backtest" / "data" / "cache" / "opt_*.csv"))
    cleared = 0
    for file_name in synthetic_files:
        try:
            df_check = pd.read_csv(file_name, nrows=1)
            if "is_synthetic" in df_check.columns:
                os.remove(file_name)
                cleared += 1
        except Exception:
            pass
    return cleared


def _collapse_for_synthesis(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse intraday candles into daily rows before re-synthesizing."""

    if frame.empty:
        return frame.copy()
    working = clean_ohlcv(frame)
    working["date"] = working["datetime"].dt.normalize()
    group_cols = [column for column in ["date", "strike", "option_type", "expiry_dt", "symbol"] if column in working.columns]
    agg_map: dict[str, str] = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    if "open_interest" in working.columns:
        agg_map["open_interest"] = "last"
    return working.groupby(group_cols, as_index=False).agg(agg_map)


def _rehydrate_synthetic(frame: pd.DataFrame) -> pd.DataFrame:
    """Regenerate better synthetic intraday candles when synthetic data is detected."""

    if frame.empty or "is_synthetic" not in frame.columns or not bool(frame["is_synthetic"].fillna(False).any()):
        return frame
    return synthesize_intraday(_collapse_for_synthesis(frame), seed=config.SYNTHETIC_SEED)


def _month_bounds(month: str) -> tuple[str, str]:
    """Return the start and end date for a YYYY-MM month."""

    period = pd.Period(month, freq="M")
    return period.start_time.strftime("%Y-%m-%d"), period.end_time.strftime("%Y-%m-%d")


def _build_options(daily_summary: pd.DataFrame) -> pd.DataFrame:
    """Fetch CE and PE contracts for the month."""

    frames: list[pd.DataFrame] = []
    fetcher = FreeDataFetcher()
    for _, row in daily_summary.iterrows():
        trade_date = pd.Timestamp(row["trade_date"]).normalize()
        expiry = pd.Timestamp(row["expiry"]).normalize()
        atm_strike = float(row["atm_strike"])
        expiry_text = expiry.strftime("%Y-%m-%dT06:00:00.000Z")
        for right in ("call", "put"):
            frame = fetch_option_data(
                symbol=config.OPTION_SYMBOL,
                start_date=trade_date.strftime("%Y-%m-%dT09:15:00.000Z"),
                end_date=trade_date.strftime("%Y-%m-%dT15:30:00.000Z"),
                expiry_date=expiry_text,
                strike_price=atm_strike,
                right=right,
                interval=config.INTERVAL,
                exchange_code=config.OPTION_EXCHANGE,
            )
            if not frame.empty:
                frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return _rehydrate_synthetic(combined)


def _signal_summary(audit_df: pd.DataFrame, month_label: str) -> str:
    """Build the signal-condition summary text."""

    total = len(audit_df)
    if total == 0:
        return "No rows available for signal audit."

    def hit(col: str) -> int:
        return int(audit_df[col].fillna(False).astype(bool).sum())

    rows = [
        ["BB Upper", hit("at_upper_band"), hit("at_upper_band") / total * 100.0],
        ["BB Lower", hit("at_lower_band"), hit("at_lower_band") / total * 100.0],
        ["RSI Down MA3", hit("rsi_cross_below_ma3"), hit("rsi_cross_below_ma3") / total * 100.0],
        ["RSI Up MA3", hit("rsi_cross_above_ma3"), hit("rsi_cross_above_ma3") / total * 100.0],
        ["Near R1/R2", hit("near_r1r2"), hit("near_r1r2") / total * 100.0],
        ["Near S1/S2", hit("near_s1s2"), hit("near_s1s2") / total * 100.0],
        ["Near SR Res", hit("near_resistance"), hit("near_resistance") / total * 100.0],
        ["Near SR Sup", hit("near_support"), hit("near_support") / total * 100.0],
    ]
    pe_all3 = int((audit_df["pe_cond1"].fillna(False).astype(bool) & audit_df["pe_cond2"].fillna(False).astype(bool) & audit_df["pe_cond3"].fillna(False).astype(bool)).sum())
    ce_all3 = int((audit_df["ce_cond1"].fillna(False).astype(bool) & audit_df["ce_cond2"].fillna(False).astype(bool) & audit_df["ce_cond3"].fillna(False).astype(bool)).sum())
    rows.extend(
        [
            ["PE Cond1", hit("pe_cond1"), hit("pe_cond1") / total * 100.0],
            ["PE Cond2", hit("pe_cond2"), hit("pe_cond2") / total * 100.0],
            ["PE Cond3", hit("pe_cond3"), hit("pe_cond3") / total * 100.0],
            ["PE ALL 3", pe_all3, pe_all3 / total * 100.0],
            ["CE Cond1", hit("ce_cond1"), hit("ce_cond1") / total * 100.0],
            ["CE Cond2", hit("ce_cond2"), hit("ce_cond2") / total * 100.0],
            ["CE Cond3", hit("ce_cond3"), hit("ce_cond3") / total * 100.0],
            ["CE ALL 3", ce_all3, ce_all3 / total * 100.0],
        ]
    )

    out = [
        "",
        f"SIGNAL CONDITION AUDIT - {month_label}",
        f"Total candles scanned    : {total:,}",
        _format_table([[name, count, f"{rate:.1f}%"] for name, count, rate in rows], ["Condition", "Hit Count", "Hit Rate"]),
        f"TOTAL SIGNALS            : {int((audit_df['action'] != 'HOLD').sum()):,}",
        f"Expected trades/month    : {int((audit_df['action'] != 'HOLD').sum())}",
    ]
    if pe_all3 / total * 100.0 < 0.5 and ce_all3 / total * 100.0 < 0.5:
        out.extend(
            [
                "WARN SIGNAL TOO STRICT - conditions almost never align.",
                "Suggestion: relax SR_TOLERANCE from 15 to 25 pts, or allow 'single' signals to trade without BB confirmation.",
            ]
        )
    elif pe_all3 / total * 100.0 > 8.0 or ce_all3 / total * 100.0 > 8.0:
        out.extend(
            [
                "WARN SIGNAL TOO LOOSE - overtrading risk.",
                "Suggestion: tighten SR_TOLERANCE or require full confluence.",
            ]
        )
    return "\n".join(out)


def main() -> None:
    """Run the signal audit for one month."""

    args = _parse_args()
    cleared = _clear_synthetic_cache()
    if cleared:
        print(f"Cleared {cleared} synthetic cache files - will regenerate with improved model")

    start, end = _month_bounds(args.month)
    print(f"Loading month: {args.month} ({start} to {end})")

    spot_df = fetch_spot_data(
        symbol=config.SPOT_SYMBOL,
        start_date=start,
        end_date=end,
        interval=config.INTERVAL,
        exchange_code=config.SPOT_EXCHANGE,
    )
    spot_df = _rehydrate_synthetic(spot_df)

    daily = clean_ohlcv(spot_df)
    daily["trade_date"] = daily["datetime"].dt.normalize()
    daily_summary = (
        daily.groupby("trade_date", as_index=False)
        .agg(close=("close", "last"))
        .sort_values("trade_date")
        .reset_index(drop=True)
    )
    daily_summary["expiry"] = daily_summary["trade_date"].apply(lambda value: value + pd.Timedelta(days=(3 - pd.Timestamp(value).weekday()) % 7))
    daily_summary["atm_strike"] = (daily_summary["close"] / config.ATM_ROUNDING).round() * config.ATM_ROUNDING

    option_df = _build_options(daily_summary)
    vix_df = fetch_india_vix(start, end, interval="1day", symbol=config.VIX_SYMBOL, exchange_code=config.VIX_EXCHANGE)

    enriched = build_backtest_frame(spot_df, option_df, vix_df)
    strategy = ATMStrategy()
    audit_df = strategy.audit_signals(enriched)

    out_path = ROOT / "nifty_backtest" / "data" / "cache" / f"signal_audit_{args.month.replace('-', '')}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(out_path, index=False)

    print(_signal_summary(audit_df, args.month))
    print(f"Audit CSV saved to: {out_path.as_posix()}")


if __name__ == "__main__":
    main()
