"""Run the NIFTY ATM strategy on end-of-day data only."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import time
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nifty_backtest import config
from nifty_backtest.data.fetcher import FreeDataFetcher
from nifty_backtest.data.processor import build_backtest_frame, clean_ohlcv
from nifty_backtest.engine.portfolio import Portfolio
from nifty_backtest.engine.trade import Trade
from nifty_backtest.reports.stats import compute_statistics, monthly_pnl_breakdown, signal_hit_frequency

logger = logging.getLogger(__name__)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def _fmt_inr(value: float) -> str:
    """Format a numeric value as INR text."""

    sign = "-" if value < 0 else ""
    value = abs(float(value))
    integer = int(value)
    decimals = f"{value:.2f}".split(".")[1]
    s = str(integer)
    if len(s) <= 3:
        grouped = s
    else:
        grouped = s[-3:]
        s = s[:-3]
        while s:
            grouped = f"{s[-2:]},{grouped}"
            s = s[:-2]
    return f"{sign}₹{grouped}.{decimals}"


def _format_pct(value: float) -> str:
    """Format a percentage with one decimal place."""

    return f"{float(value):.1f}%"


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    default_output = ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Run an EOD-only NIFTY ATM backtest.")
    parser.add_argument("--start", default=config.BACKTEST_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=config.BACKTEST_END, help="End date YYYY-MM-DD")
    parser.add_argument("--year", default=None, help="Shortcut for a full year run, e.g. 2024")
    parser.add_argument("--month", default=None, help="Shortcut for a single month run, e.g. 2024-06")
    parser.add_argument("--capital", type=float, default=config.CAPITAL, help="Starting capital")
    parser.add_argument("--log-level", default="INFO", help="DEBUG / INFO / WARNING")
    parser.add_argument("--dry-run", action="store_true", help="Print configuration and exit")
    parser.add_argument("--no-cache", action="store_true", help="Clear CSV cache before running")
    parser.add_argument("--output-dir", default=str(default_output), help="Where to save the results")
    return parser.parse_args()


def _resolve_period(args: argparse.Namespace) -> tuple[str, str]:
    """Resolve the final date range from CLI inputs."""

    if args.month:
        period = pd.Period(args.month, freq="M")
        return period.start_time.normalize().strftime("%Y-%m-%d"), period.end_time.normalize().strftime("%Y-%m-%d")
    if args.year:
        return f"{int(args.year):04d}-01-01", f"{int(args.year):04d}-12-31"
    return args.start, args.end


def _clear_cache() -> None:
    """Remove cached CSV files."""

    cache_dir = config.resolve_cache_dir()
    if not cache_dir.exists():
        return
    for path in cache_dir.glob("*.csv"):
        path.unlink(missing_ok=True)


def _load_daily_frame(fetcher: FreeDataFetcher, start: str, end: str) -> pd.DataFrame:
    """Load the real daily spot and VIX frame and calculate indicators."""

    t0 = time.perf_counter()
    logger.info("Fetching spot data...")
    spot = fetcher.get_spot_daily(start, end)
    logger.info("Fetching VIX data...")
    vix = fetcher.get_vix_daily(start, end)
    if spot.empty:
        raise ValueError("No spot data available for the selected period.")
    frame = build_backtest_frame(spot, pd.DataFrame(), vix)
    if frame.empty:
        raise ValueError("Failed to build the daily analysis frame.")
    elapsed = time.perf_counter() - t0
    logger.info("Loaded %s daily rows in %.2fs", len(frame), elapsed)
    return frame


def _select_levels(close: float, sr_levels: list[float], tolerance: float) -> tuple[list[float], list[float]]:
    """Split swing levels into resistance and support buckets."""

    resistance_levels: list[float] = []
    support_levels: list[float] = []
    for raw_level in sr_levels:
        try:
            level = float(raw_level)
        except (TypeError, ValueError):
            continue
        if level >= close - tolerance:
            resistance_levels.append(level)
        if level <= close + tolerance:
            support_levels.append(level)
    return resistance_levels, support_levels


def _evaluate_daily_signal(row: pd.Series) -> dict[str, Any]:
    """Evaluate the daily EOD signal conditions for one row."""

    close = float(row.get("close"))
    tolerance = float(config.SR_TOLERANCE)
    sr_levels = row.get("sr_levels") if isinstance(row.get("sr_levels"), list) else []
    resistance_levels, support_levels = _select_levels(close, sr_levels, tolerance)

    pivot = row.get("pivot")
    r1 = row.get("r1")
    r2 = row.get("r2")
    s1 = row.get("s1")
    s2 = row.get("s2")

    near_resistance = any(close >= float(level) - tolerance for level in resistance_levels)
    near_support = any(close <= float(level) + tolerance for level in support_levels)
    near_r1r2 = any(pd.notna(level) and close >= float(level) - tolerance for level in [r1, r2])
    near_s1s2 = any(pd.notna(level) and close <= float(level) + tolerance for level in [s1, s2])

    at_upper_band = pd.notna(row.get("upper_band")) and close >= float(row.get("upper_band"))
    at_lower_band = pd.notna(row.get("lower_band")) and close <= float(row.get("lower_band"))
    rsi_cross_below_ma3 = bool(row.get("rsi_cross_below_ma3", False))
    rsi_cross_above_ma3 = bool(row.get("rsi_cross_above_ma3", False))

    pe_cond1a = near_resistance
    pe_cond1b = near_r1r2
    pe_cond1 = pe_cond1a or pe_cond1b
    pe_cond2 = at_upper_band
    pe_cond3 = rsi_cross_below_ma3
    pe_confluence = pe_cond1a and pe_cond1b

    ce_cond1a = near_support
    ce_cond1b = near_s1s2
    ce_cond1 = ce_cond1a or ce_cond1b
    ce_cond2 = at_lower_band
    ce_cond3 = rsi_cross_above_ma3
    ce_confluence = ce_cond1a and ce_cond1b

    pe_score = int(pe_cond1a) + int(pe_cond1b) + int(pe_cond2) + int(pe_cond3)
    ce_score = int(ce_cond1a) + int(ce_cond1b) + int(ce_cond2) + int(ce_cond3)

    action = "HOLD"
    strength = "single"
    pe_trade = pe_cond1 and (pe_cond2 or pe_cond3)
    ce_trade = ce_cond1 and (ce_cond2 or ce_cond3)
    if pe_trade and not ce_trade:
        action = "BUY_PE"
        strength = "full" if (pe_cond2 and pe_cond3) else "confluence"
    elif ce_trade and not pe_trade:
        action = "BUY_CE"
        strength = "full" if (ce_cond2 and ce_cond3) else "confluence"
    elif pe_trade and ce_trade:
        if pe_score > ce_score:
            action = "BUY_PE"
            strength = "full" if (pe_cond2 and pe_cond3) else "confluence"
        elif ce_score > pe_score:
            action = "BUY_CE"
            strength = "full" if (ce_cond2 and ce_cond3) else "confluence"

    signal_log = {
        "at_upper_band": bool(at_upper_band),
        "at_lower_band": bool(at_lower_band),
        "rsi_cross_below_ma3": bool(rsi_cross_below_ma3),
        "rsi_cross_above_ma3": bool(rsi_cross_above_ma3),
        "near_resistance": bool(near_resistance),
        "near_support": bool(near_support),
        "near_r1r2": bool(near_r1r2),
        "near_s1s2": bool(near_s1s2),
        "pe_cond1a": bool(pe_cond1a),
        "pe_cond1b": bool(pe_cond1b),
        "pe_cond1": bool(pe_cond1),
        "pe_cond2": bool(pe_cond2),
        "pe_cond3": bool(pe_cond3),
        "pe_trade": bool(pe_trade),
        "pe_confluence": bool(pe_confluence),
        "ce_cond1a": bool(ce_cond1a),
        "ce_cond1b": bool(ce_cond1b),
        "ce_cond1": bool(ce_cond1),
        "ce_cond2": bool(ce_cond2),
        "ce_cond3": bool(ce_cond3),
        "ce_trade": bool(ce_trade),
        "ce_confluence": bool(ce_confluence),
        "action": action,
        "strength": strength,
    }

    return {
        "action": action,
        "strength": strength,
        "signal_log": signal_log,
        "near_resistance": bool(near_resistance),
        "near_support": bool(near_support),
        "near_r1r2": bool(near_r1r2),
        "near_s1s2": bool(near_s1s2),
        "pe_cond1": bool(pe_cond1),
        "pe_cond2": bool(pe_cond2),
        "pe_cond3": bool(pe_cond3),
        "pe_trade": bool(pe_trade),
        "ce_cond1": bool(ce_cond1),
        "ce_cond2": bool(ce_cond2),
        "ce_cond3": bool(ce_cond3),
        "ce_trade": bool(ce_trade),
        "pe_score": pe_score,
        "ce_score": ce_score,
        "resistance_levels": resistance_levels,
        "support_levels": support_levels,
    }


def _entry_timestamp(trade_date: pd.Timestamp) -> pd.Timestamp:
    """Return the timestamp used for an entry or exit on a daily candle."""

    return pd.Timestamp(trade_date).normalize() + pd.Timedelta(hours=9, minutes=15)


def _close_timestamp(trade_date: pd.Timestamp) -> pd.Timestamp:
    """Return the timestamp used for an exit at the daily close."""

    return pd.Timestamp(trade_date).normalize() + pd.Timedelta(hours=15, minutes=30)


def _next_expiry(trade_date: pd.Timestamp) -> pd.Timestamp:
    """Return the next weekly expiry date on or after the given trading date."""

    expiry = pd.Timestamp(trade_date).normalize()
    offset = (3 - expiry.weekday()) % 7
    expiry = expiry + pd.Timedelta(days=offset)
    holidays = {pd.Timestamp(value).date() for value in config.NSE_HOLIDAYS_2023 + config.NSE_HOLIDAYS_2024}
    while expiry.date() in holidays or expiry.weekday() >= 5:
        expiry -= pd.Timedelta(days=1)
    return expiry.normalize()


def _format_expiry(expiry: pd.Timestamp) -> str:
    """Format expiry as a plain date string."""

    return pd.Timestamp(expiry).strftime("%Y-%m-%d")


def _fetch_option_series(
    fetcher: FreeDataFetcher,
    strike: float,
    option_type: str,
    entry_date: pd.Timestamp,
    expiry_dt: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch the option series needed for one EOD trade."""

    hold_end = (pd.Timestamp(entry_date) + pd.offsets.BDay(config.EOD_MAX_HOLD_DAYS - 1)).normalize()
    end_date = min(pd.Timestamp(expiry_dt).normalize(), hold_end)
    frame = fetcher.get_options_daily(
        strike=strike,
        expiry_dt=pd.Timestamp(expiry_dt).date(),
        option_type=option_type,
        from_dt=pd.Timestamp(entry_date).date(),
        to_dt=end_date.date(),
    )
    frame = clean_ohlcv(frame)
    if frame.empty:
        return frame
    frame["trade_date"] = pd.to_datetime(frame["datetime"], errors="coerce").dt.normalize()
    return frame.sort_values("datetime").reset_index(drop=True)


def _option_lookup(frame: pd.DataFrame, option_type: str) -> dict[pd.Timestamp, pd.Series]:
    """Build a lookup from trade date to option row."""

    if frame.empty:
        return {}
    working = frame.copy()
    if "option_type" in working.columns:
        working["option_type"] = working["option_type"].astype(str).str.upper()
        working = working[working["option_type"] == option_type.upper()]
    lookup: dict[pd.Timestamp, pd.Series] = {}
    for _, row in working.iterrows():
        trade_date = pd.Timestamp(row["trade_date"]).normalize()
        lookup[trade_date] = row
    return lookup


def _option_price(row: pd.Series | None, field: str = "open") -> float | None:
    """Extract an option price from a row when available."""

    if row is None or field not in row.index:
        return None
    value = row.get(field)
    if pd.isna(value):
        return None
    return float(value)


def _open_trade(
    portfolio: Portfolio,
    fetcher: FreeDataFetcher,
    pending: dict[str, Any],
    entry_date: pd.Timestamp,
    current_row: pd.Series,
    trade_id: int,
) -> tuple[Trade | None, dict[pd.Timestamp, pd.Series]]:
    """Open a trade at the current day's open using the prior signal."""

    signal_row = pending["signal_row"]
    signal = pending["signal"]
    action = signal["action"]
    direction = "CE" if action == "BUY_CE" else "PE"
    strike = float(round(float(signal_row["close"]) / config.ATM_ROUNDING) * config.ATM_ROUNDING)
    expiry_dt = _next_expiry(entry_date)
    option_frame = _fetch_option_series(fetcher, strike, direction, entry_date, expiry_dt)
    if option_frame.empty:
        logger.warning("Missing option data for trade_id=%s strike=%s direction=%s", trade_id, strike, direction)
        return None, {}

    lookup = _option_lookup(option_frame, direction)
    entry_date = pd.Timestamp(entry_date).normalize()
    entry_option_row = lookup.get(entry_date)
    if entry_option_row is None:
        future_dates = sorted(date for date in lookup if pd.Timestamp(date).normalize() >= entry_date)
        if future_dates:
            entry_date = pd.Timestamp(future_dates[0]).normalize()
            entry_option_row = lookup.get(entry_date)
    option_entry_price = _option_price(entry_option_row, "open")
    if option_entry_price is None:
        logger.warning("No entry open price for trade_id=%s date=%s", trade_id, entry_date.date())
        return None, {}

    entry_spot = float(signal_row["close"])
    target1 = entry_spot + config.TARGET1_SPOT_MOVE if direction == "CE" else entry_spot - config.TARGET1_SPOT_MOVE
    target2 = entry_spot + config.TARGET2_SPOT_MOVE if direction == "CE" else entry_spot - config.TARGET2_SPOT_MOVE
    stoploss = entry_spot - config.STOPLOSS_SPOT_MOVE if direction == "CE" else entry_spot + config.STOPLOSS_SPOT_MOVE
    qty_multiplier = 1.0
    qty_total = max(1, int(round(config.LOT_SIZE * qty_multiplier)))
    premium_sl = option_entry_price * (
        config.PREMIUM_SL_CONSERVATIVE_MULTIPLIER if config.PREMIUM_SL_TYPE == "conservative" else config.PREMIUM_SL_AGGRESSIVE_MULTIPLIER
    )

    trade = Trade(
        trade_id=trade_id,
        direction=direction,
        entry_time=_entry_timestamp(entry_date),
        entry_spot=entry_spot,
        entry_strike=strike,
        option_entry_price=option_entry_price,
        qty_total=qty_total,
        qty_multiplier=qty_multiplier,
        target1=target1,
        target2=target2,
        stoploss_spot=stoploss,
        conservative_sl_premium=option_entry_price * config.PREMIUM_SL_CONSERVATIVE_MULTIPLIER,
        aggressive_sl_premium=option_entry_price * config.PREMIUM_SL_AGGRESSIVE_MULTIPLIER,
        premium_sl_price=premium_sl,
        signal_strength=signal["strength"],
        vix_at_entry=float(signal_row.get("vix_close")) if pd.notna(signal_row.get("vix_close")) else None,
        rsi_at_entry=float(signal_row.get("rsi")) if pd.notna(signal_row.get("rsi")) else None,
        bb_position_at_entry=float(signal_row.get("percent_b")) if pd.notna(signal_row.get("percent_b")) else None,
        is_expiry_day=bool(pd.Timestamp(entry_date).weekday() == 3),
        is_thursday_trade=bool(pd.Timestamp(entry_date).weekday() == 3),
        entry_reason="eod_signal",
        metadata={
            "signal_date": pd.Timestamp(signal_row["datetime"]).date().isoformat(),
            "entry_date": pd.Timestamp(entry_date).date().isoformat(),
            "expiry_dt": expiry_dt.date().isoformat(),
            "signal_log": signal["signal_log"],
            "option_lookup": lookup,
        },
    )
    portfolio.open(trade)
    logger.info(
        "Opened trade_id=%s direction=%s entry_date=%s strike=%s option_open=%.2f",
        trade.trade_id,
        trade.direction,
        entry_date.date(),
        strike,
        option_entry_price,
    )
    return trade, lookup


def _check_exit_conditions(trade: Trade, candle: pd.Series, option_ltp: float | None, days_held: int) -> tuple[bool, bool, str]:
    """Evaluate exit rules in the required order."""

    close = float(candle.get("close"))
    if option_ltp is None or pd.isna(option_ltp):
        option_ltp = trade.option_entry_price

    if trade.direction == "PE":
        if option_ltp < trade.premium_sl_price:
            return True, False, "premium_sl"
        if trade.stoploss_spot is not None and close > float(trade.stoploss_spot):
            return True, False, "stoploss"
        if not trade.partial_exit_done and trade.target1 is not None and close <= float(trade.target1):
            return False, True, "target1"
        if trade.partial_exit_done and trade.target2 is not None and close <= float(trade.target2):
            return True, False, "target2"
    else:
        if option_ltp < trade.premium_sl_price:
            return True, False, "premium_sl"
        if trade.stoploss_spot is not None and close < float(trade.stoploss_spot):
            return True, False, "stoploss"
        if not trade.partial_exit_done and trade.target1 is not None and close >= float(trade.target1):
            return False, True, "target1"
        if trade.partial_exit_done and trade.target2 is not None and close >= float(trade.target2):
            return True, False, "target2"

    if days_held >= config.EOD_MAX_HOLD_DAYS:
        return True, False, "time_exit"
    if pd.Timestamp(candle.get("datetime")).weekday() == 3:
        return True, False, "expiry_exit"
    return False, False, "open"


def _build_daily_pnl_frame(equity_curve: pd.DataFrame, starting_capital: float) -> pd.DataFrame:
    """Convert an equity curve into a daily PnL table."""

    if equity_curve.empty:
        return pd.DataFrame(columns=["date", "daily_pnl", "cumulative_pnl", "drawdown"])
    frame = equity_curve.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame = frame.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    frame["date"] = frame["datetime"].dt.date
    frame["cumulative_pnl"] = frame["equity"] - float(starting_capital)
    frame["daily_pnl"] = frame["equity"].diff().fillna(frame["equity"] - float(starting_capital))
    if "drawdown_pct" not in frame.columns:
        peak = frame["equity"].cummax()
        frame["drawdown"] = peak - frame["equity"]
    else:
        peak = frame["equity"].cummax()
        frame["drawdown"] = peak - frame["equity"]
    return frame[["date", "daily_pnl", "cumulative_pnl", "drawdown", "equity"]].rename(columns={"drawdown": "drawdown"})


def _print_summary(stats: dict[str, Any], trades_df: pd.DataFrame, start: str, end: str, output_dir: Path) -> None:
    """Print a final human-readable summary."""

    winners = trades_df[trades_df["pnl_amount"] > 0] if not trades_df.empty else pd.DataFrame()
    losers = trades_df[trades_df["pnl_amount"] < 0] if not trades_df.empty else pd.DataFrame()
    gross_profit = float(winners["pnl_amount"].sum()) if not winners.empty else 0.0
    gross_loss = abs(float(losers["pnl_amount"].sum())) if not losers.empty else 0.0
    net_pnl = gross_profit - gross_loss
    capital_pct = (net_pnl / float(config.CAPITAL)) * 100.0 if config.CAPITAL else 0.0
    best_trade = trades_df["pnl_amount"].max() if not trades_df.empty else 0.0
    worst_trade = trades_df["pnl_amount"].min() if not trades_df.empty else 0.0

    print("═" * 44)
    print("BACKTEST COMPLETE")
    print("═" * 44)
    print(f"Period       : {start} → {end}")
    print(f"Trading Days : {len(pd.bdate_range(start=start, end=end))}")
    print("─" * 44)
    print(f"Total Trades : {stats['total_trades']}")
    print(f"Winners      : {len(winners)}  ({stats['win_rate_pct']:.1f}%)")
    print(f"Losers       : {len(losers)}  ({stats['loss_rate_pct']:.1f}%)")
    print("─" * 44)
    print(f"Gross Profit : {_fmt_inr(gross_profit)}")
    print(f"Gross Loss   : {_fmt_inr(gross_loss)}")
    print(f"Net PnL      : {_fmt_inr(net_pnl)}  ({capital_pct:+.1f}% on capital)")
    print(f"Profit Factor: {stats['profit_factor']:.2f}")
    print("─" * 44)
    print(f"Max Drawdown : {stats['max_drawdown_pct']:.1f}%")
    print(f"Avg Win      : {_fmt_inr(float(winners['pnl_amount'].mean()) if not winners.empty else 0.0)}")
    print(f"Avg Loss     : {_fmt_inr(float(losers['pnl_amount'].mean()) if not losers.empty else 0.0)}")
    print(f"Largest Win  : {_fmt_inr(float(best_trade))}")
    print(f"Largest Loss : {_fmt_inr(float(worst_trade))}")
    print("─" * 44)
    print(f"Sharpe Ratio : {stats['sharpe_ratio']:.2f}")
    print("─" * 44)
    print(f"Results saved to: {output_dir}")


def main() -> None:
    """Run the EOD backtest end to end."""

    args = _parse_args()
    start, end = _resolve_period(args)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="%(asctime)s | %(levelname)s | %(message)s")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(json.dumps({"start": start, "end": end, "capital": args.capital, "output_dir": str(output_dir)}, indent=2))
        return
    if args.no_cache:
        _clear_cache()

    print(f"Running EOD backtest: {start} → {end}")
    fetcher = FreeDataFetcher()
    load_start = time.perf_counter()
    frame = _load_daily_frame(fetcher, start, end)
    logger.info("Daily frame built in %.2fs", time.perf_counter() - load_start)

    portfolio = Portfolio(starting_capital=float(args.capital))
    signal_records: list[dict[str, Any]] = []
    pending_entry: dict[str, Any] | None = None
    trade_id = 1

    logger.info("Running daily EOD loop across %s rows", len(frame))
    for idx, row in frame.reset_index(drop=True).iterrows():
        current_date = pd.Timestamp(row["datetime"]).normalize()
        trade_opened_today = False
        trade_closed_today = False
        exit_reason = "open"
        option_ltp: float | None = None

        if pending_entry is not None and current_date >= pending_entry["entry_date"] and portfolio.can_open_trade():
            trade, option_lookup = _open_trade(portfolio, fetcher, pending_entry, pd.Timestamp(pending_entry["entry_date"]), row, trade_id)
            if trade is not None:
                trade_id += 1
            trade_opened_today = portfolio.open_trade is not None
            pending_entry = None
        elif pending_entry is not None and current_date >= pending_entry["entry_date"] and not portfolio.can_open_trade():
            pending_entry = None

        if portfolio.open_trade is not None:
            trade = portfolio.open_trade
            option_lookup = trade.metadata.get("option_lookup", {})
            option_row = option_lookup.get(current_date)
            option_ltp = _option_price(option_row, "close")
            entry_business_date = pd.Timestamp(trade.metadata.get("entry_date", current_date)).normalize()
            days_held = len(pd.bdate_range(start=entry_business_date, end=current_date))
            should_exit_full, should_partial_exit, reason = _check_exit_conditions(trade, row, option_ltp, days_held)
            exit_reason = reason

            if should_partial_exit and not trade.partial_exit_done and option_ltp is not None:
                portfolio.on_partial_exit(trade, option_ltp, _close_timestamp(current_date))
                if trade.target2 is not None:
                    if trade.direction == "PE" and float(row["close"]) <= float(trade.target2):
                        should_exit_full = True
                        exit_reason = "target2"
                    elif trade.direction == "CE" and float(row["close"]) >= float(trade.target2):
                        should_exit_full = True
                        exit_reason = "target2"
                    else:
                        should_exit_full = False
                        exit_reason = "target1"
                else:
                    should_exit_full = False
                    exit_reason = "target1"

            if should_exit_full and option_ltp is not None and portfolio.open_trade is not None:
                portfolio.on_full_exit(trade, option_ltp, exit_reason, _close_timestamp(current_date))
                trade_closed_today = True
        else:
            option_ltp = None

        signal = _evaluate_daily_signal(row)
        signal_records.append(
            {
                "datetime": row["datetime"],
                "date": pd.Timestamp(row["datetime"]).date().isoformat(),
                "close": row.get("close"),
                "bb_upper": row.get("upper_band"),
                "bb_lower": row.get("lower_band"),
                "bb_middle": row.get("middle_band"),
                "percent_b": row.get("percent_b"),
                "rsi": row.get("rsi"),
                "rsi_ma3": row.get("rsi_ma3"),
                "rsi_ma21": row.get("rsi_ma21"),
                "rsi_cross_below_ma3": signal["signal_log"]["rsi_cross_below_ma3"],
                "rsi_cross_above_ma3": signal["signal_log"]["rsi_cross_above_ma3"],
                "near_resistance": signal["signal_log"]["near_resistance"],
                "near_support": signal["signal_log"]["near_support"],
                "near_r1r2": signal["signal_log"]["near_r1r2"],
                "near_s1s2": signal["signal_log"]["near_s1s2"],
                "pe_cond1": signal["signal_log"]["pe_cond1"],
                "pe_cond2": signal["signal_log"]["pe_cond2"],
                "pe_cond3": signal["signal_log"]["pe_cond3"],
                "pe_trade": signal["signal_log"]["pe_trade"],
                "ce_cond1": signal["signal_log"]["ce_cond1"],
                "ce_cond2": signal["signal_log"]["ce_cond2"],
                "ce_cond3": signal["signal_log"]["ce_cond3"],
                "ce_trade": signal["signal_log"]["ce_trade"],
                "action": signal["action"],
                "strength": signal["strength"],
                "trade_opened_today": trade_opened_today,
                "trade_closed_today": trade_closed_today,
                "exit_reason": exit_reason,
                "vix_close": row.get("vix_close"),
            }
        )

        if portfolio.open_trade is None and signal["action"] in {"BUY_CE", "BUY_PE"} and idx + 1 < len(frame):
            next_business_day = (pd.Timestamp(current_date) + pd.offsets.BDay(1)).normalize()
            pending_entry = {
                "entry_date": next_business_day,
                "signal_row": row,
                "signal": signal,
            }

        portfolio.update_equity(current_date, mark_to_market=option_ltp if portfolio.open_trade is not None else None)

    trades_df = portfolio.closed_trades_frame()
    equity_df = portfolio.equity_frame()
    if not trades_df.empty and "exit_time" not in trades_df.columns:
        trades_df = trades_df.copy()
        trades_df["exit_time"] = trades_df["full_exit_time"].fillna(trades_df["partial_exit_time"])
    stats = compute_statistics(trades_df, equity_curve=equity_df, starting_capital=float(args.capital))
    daily_pnl_df = _build_daily_pnl_frame(equity_df, float(args.capital))
    signal_df = pd.DataFrame(signal_records)

    trades_path = output_dir / "trades.csv"
    daily_pnl_path = output_dir / "daily_pnl.csv"
    signal_log_path = output_dir / "signal_log.csv"
    run_config_path = output_dir / "run_config.json"
    summary_path = output_dir / "summary.txt"

    trades_df.to_csv(trades_path, index=False)
    daily_pnl_df.to_csv(daily_pnl_path, index=False)
    signal_df.to_csv(signal_log_path, index=False)

    run_config = {
        "start": start,
        "end": end,
        "capital": float(args.capital),
        "output_dir": str(output_dir),
        "eod_max_hold_days": config.EOD_MAX_HOLD_DAYS,
        "eod_entry_on_next_open": config.EOD_ENTRY_ON_NEXT_OPEN,
        "lot_size": config.LOT_SIZE,
        "atm_rounding": config.ATM_ROUNDING,
        "strategy": "EOD daily ATM backtest",
    }
    run_config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    summary_lines = [
        f"Period: {start} -> {end}",
        f"Capital: {_fmt_inr(float(args.capital))}",
        f"Total trades: {stats['total_trades']}",
        f"Win rate: {stats['win_rate_pct']:.1f}%",
        f"Loss rate: {stats['loss_rate_pct']:.1f}%",
        f"Profit factor: {stats['profit_factor']:.2f}",
        f"Max drawdown: {stats['max_drawdown_pct']:.1f}%",
        f"Sharpe ratio: {stats['sharpe_ratio']:.2f}",
        f"Results folder: {output_dir}",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    _print_summary(stats, trades_df, start, end, output_dir)
    print(f"Trades CSV   : {trades_path}")
    print(f"Daily PnL    : {daily_pnl_path}")
    print(f"Signals CSV  : {signal_log_path}")
    print(f"Run config   : {run_config_path}")
    print(f"Summary      : {summary_path}")

    monthly = monthly_pnl_breakdown(trades_df)
    if not monthly.empty:
        print("\nMonthly PnL breakdown:")
        print(monthly.to_string(index=False))

    signal_freq = signal_hit_frequency(signal_df)
    if not signal_freq.empty:
        print("\nSignal frequency:")
        print(signal_freq.to_string(index=False))


if __name__ == "__main__":
    main()
