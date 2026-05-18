"""Production backtest runner with CLI, progress tracking, and result export."""

from __future__ import annotations

import argparse
import json
import importlib
import logging
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    tqdm = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PACKAGE_ROOT = ROOT / "nifty_backtest"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from nifty_backtest import config
from nifty_backtest.data.fetcher import fetch_india_vix, fetch_option_data, fetch_spot_data
from nifty_backtest.data.processor import build_backtest_frame, clean_ohlcv, synthesize_intraday
from nifty_backtest.engine.portfolio import Portfolio
from nifty_backtest.engine.trade import Trade
from nifty_backtest.reports.stats import compute_statistics

logger = logging.getLogger(__name__)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def _fmt_inr(value: float) -> str:
    """Format a numeric value as Indian rupee text."""

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
    """Format a floating-point percentage."""

    return f"{float(value):.1f}%"


def _fallback_tqdm(iterable: Iterable, total: int | None = None, desc: str | None = None):
    """Provide a minimal tqdm-compatible fallback."""

    if desc:
        print(desc)
    for item in iterable:
        yield item


def _progress(iterable: Iterable, total: int | None = None, desc: str | None = None):
    """Return a tqdm iterator when available, otherwise a plain iterable."""

    if tqdm is None:
        return _fallback_tqdm(iterable, total=total, desc=desc)
    return tqdm(iterable, total=total, desc=desc)


def _business_days(start: str, end: str) -> pd.DatetimeIndex:
    """Return business days between the requested bounds."""

    return pd.bdate_range(start=pd.Timestamp(start).normalize(), end=pd.Timestamp(end).normalize())


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the backtest runner."""

    default_output = ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Run the NIFTY ATM options backtest.")
    parser.add_argument("--start", default=config.BACKTEST_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=config.BACKTEST_END, help="End date YYYY-MM-DD")
    parser.add_argument("--year", default=None, help="Shortcut for a full year run, e.g. 2024")
    parser.add_argument("--month", default=None, help="Shortcut for a single month run, e.g. 2024-06")
    parser.add_argument("--strategy", default=config.STRATEGY_CLASS_PATH, help="Strategy class path module:Class")
    parser.add_argument("--capital", type=float, default=config.CAPITAL, help="Starting capital")
    parser.add_argument("--log-level", default="INFO", help="DEBUG / INFO / WARNING")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit without running")
    parser.add_argument("--no-cache", action="store_true", help="Force re-fetch all data")
    parser.add_argument("--output-dir", default=str(default_output), help="Where to save results")
    return parser.parse_args()


def _resolve_period(args: argparse.Namespace) -> tuple[str, str]:
    """Resolve the final start/end dates from CLI arguments."""

    if args.month:
        start = pd.Period(args.month, freq="M").start_time.normalize()
        end = pd.Period(args.month, freq="M").end_time.normalize()
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    if args.year:
        return f"{int(args.year):04d}-01-01", f"{int(args.year):04d}-12-31"
    return args.start, args.end


def _load_strategy(strategy_path: str):
    """Instantiate the configured strategy class."""

    if not strategy_path:
        raise ValueError("Strategy path is required. Set --strategy or config.STRATEGY_CLASS_PATH.")
    module_path, class_name = strategy_path.split(":", 1)
    module = importlib.import_module(module_path)
    strategy_class = getattr(module, class_name)
    return strategy_class()


def _print_run_config(start: str, end: str, strategy, capital: float, output_dir: Path) -> None:
    """Print a compact run configuration summary."""

    strategy_name = strategy.__class__.__name__ if strategy is not None else "Unknown"
    table = [
        ["Period", f"{start} → {end}"],
        ["Strategy", strategy_name],
        ["Capital", _fmt_inr(capital)],
        ["Lot Size", str(config.LOT_SIZE)],
        ["Data Source", "Breeze API (cached)"],
        ["Output Dir", str(output_dir)],
    ]
    width = max(len(row[0]) for row in table) + 2
    print("┌" + "─" * 41 + "┐")
    print("│  NIFTY ATM Strategy — Backtest Run      │")
    print("├" + "─" * 18 + "┬" + "─" * 22 + "┤")
    for key, value in table:
        print(f"│ {key.ljust(16)} │ {value.ljust(20)} │")
    print("└" + "─" * 18 + "┴" + "─" * 22 + "┘")


def _print_step(message: str, elapsed: float) -> None:
    """Print a completed pipeline step with elapsed time."""

    print(f"{message} ({elapsed:.2f}s)")


def _clear_cache() -> None:
    """Remove cached CSV files under the configured cache directory."""

    cache_dir = config.resolve_cache_dir()
    if not cache_dir.exists():
        return
    for path in cache_dir.glob("*.csv"):
        path.unlink(missing_ok=True)


def _business_day_summary(spot_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate spot candles into daily close and ATM contract metadata."""

    working = clean_ohlcv(spot_df)
    working["trade_date"] = working["datetime"].dt.normalize()
    daily = (
        working.groupby("trade_date", as_index=False)
        .agg(close=("close", "last"))
        .sort_values("trade_date")
        .reset_index(drop=True)
    )
    daily["expiry"] = daily["trade_date"].apply(
        lambda value: value + pd.Timedelta(days=(3 - pd.Timestamp(value).weekday()) % 7)
    )
    daily["atm_strike"] = (daily["close"] / config.ATM_ROUNDING).round() * config.ATM_ROUNDING
    return daily


def _expiry_weeks(daily_summary: pd.DataFrame) -> int:
    """Count distinct expiry weeks in the daily summary."""

    if daily_summary.empty:
        return 0
    return int(pd.to_datetime(daily_summary["expiry"]).dt.normalize().nunique())


def _format_expiry(expiry_date: pd.Timestamp) -> str:
    """Format expiry in the Breeze contract format."""

    return pd.Timestamp(expiry_date).strftime("%Y-%m-%dT06:00:00.000Z")


def _fetch_options_for_day(row: pd.Series) -> pd.DataFrame:
    """Fetch CE and PE option candles for one trading day."""

    trade_date = pd.Timestamp(row["trade_date"]).normalize()
    expiry = pd.Timestamp(row["expiry"]).normalize()
    from_date = trade_date.strftime("%Y-%m-%dT09:15:00.000Z")
    to_date = trade_date.strftime("%Y-%m-%dT15:30:00.000Z")
    atm_strike = float(row["atm_strike"])
    expiry_text = _format_expiry(expiry)

    ce = fetch_option_data(
        symbol=config.OPTION_SYMBOL,
        start_date=from_date,
        end_date=to_date,
        expiry_date=expiry_text,
        strike_price=atm_strike,
        right="call",
        interval=config.INTERVAL,
        exchange_code=config.OPTION_EXCHANGE,
    )
    pe = fetch_option_data(
        symbol=config.OPTION_SYMBOL,
        start_date=from_date,
        end_date=to_date,
        expiry_date=expiry_text,
        strike_price=atm_strike,
        right="put",
        interval=config.INTERVAL,
        exchange_code=config.OPTION_EXCHANGE,
    )
    if not ce.empty:
        ce = ce.copy()
        ce["option_type"] = "CE"
    if not pe.empty:
        pe = pe.copy()
        pe["option_type"] = "PE"
    return pd.concat([ce, pe], ignore_index=True) if not ce.empty or not pe.empty else pd.DataFrame()


def _build_contract_frame(daily_summary: pd.DataFrame) -> pd.DataFrame:
    """Fetch and combine option contracts across all trading days."""

    frames: list[pd.DataFrame] = []
    total_weeks = _expiry_weeks(daily_summary)
    iterator = _progress(daily_summary.iterrows(), total=len(daily_summary), desc=f"Fetching options data for {total_weeks} expiry weeks...")
    for _, row in iterator:
        frame = _fetch_options_for_day(row)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


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
    daily = working.groupby(group_cols, as_index=False).agg(agg_map)
    return daily


def _rehydrate_synthetic_intraday(frame: pd.DataFrame) -> pd.DataFrame:
    """Rebuild synthetic intraday paths from daily rows when synthetic data is detected."""

    if frame.empty or "is_synthetic" not in frame.columns:
        return frame
    if not bool(frame["is_synthetic"].fillna(False).any()):
        return frame
    daily = _collapse_for_synthesis(frame)
    return synthesize_intraday(daily, seed=config.SYNTHETIC_SEED)


def _signal_payload(row: pd.Series) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build indicator and level payloads for strategy evaluation."""

    spot_close = row.get("close")
    sr_levels = row.get("sr_levels") if isinstance(row.get("sr_levels"), list) else []
    resistance_levels = []
    support_levels = []
    if pd.notna(spot_close):
        for level in sr_levels:
            try:
                level_value = float(level)
            except (TypeError, ValueError):
                continue
            if level_value >= float(spot_close):
                resistance_levels.append(level_value)
            if level_value <= float(spot_close):
                support_levels.append(level_value)
    else:
        resistance_levels = [float(level) for level in sr_levels if pd.notna(level)]
        support_levels = [float(level) for level in sr_levels if pd.notna(level)]

    indicators = {
        "bb_upper": row.get("upper_band"),
        "bb_lower": row.get("lower_band"),
        "bb_middle": row.get("middle_band"),
        "rsi": row.get("rsi"),
        "rsi_ma3": row.get("rsi_ma3"),
        "rsi_ma21": row.get("rsi_ma21"),
        "rsi_cross_above_ma3": bool(row.get("rsi_cross_above_ma3", False)),
        "rsi_cross_below_ma3": bool(row.get("rsi_cross_below_ma3", False)),
        "rsi_cross_above_ma21": bool(row.get("rsi_cross_above_ma21", False)),
        "rsi_cross_below_ma21": bool(row.get("rsi_cross_below_ma21", False)),
    }
    levels = {
        "pivot": row.get("pivot"),
        "r1": row.get("r1"),
        "r2": row.get("r2"),
        "s1": row.get("s1"),
        "s2": row.get("s2"),
        "resistance_levels": resistance_levels,
        "support_levels": support_levels,
        "vix": row.get("vix_close"),
    }
    return indicators, levels


def _option_ltp(row: pd.Series, direction: str) -> float | None:
    """Resolve the current option last traded price from a merged row."""

    column = "ce_close" if direction == "CE" else "pe_close"
    value = row.get(column)
    if pd.isna(value):
        return None
    return float(value)


def _signal_audit_summary(audit_df: pd.DataFrame, start: str, end: str) -> str:
    """Format a compact signal audit summary."""

    if audit_df.empty:
        return "Signal audit unavailable: no rows."

    total = len(audit_df)
    pairs = [
        ("BB Upper", int(audit_df["at_upper_band"].sum())),
        ("BB Lower", int(audit_df["at_lower_band"].sum())),
        ("RSI Down MA3", int(audit_df["rsi_cross_below_ma3"].sum())),
        ("RSI Up MA3", int(audit_df["rsi_cross_above_ma3"].sum())),
        ("Near R1/R2", int(audit_df["near_r1r2"].sum())),
        ("Near S1/S2", int(audit_df["near_s1s2"].sum())),
        ("Near SR Res", int(audit_df["near_resistance"].sum())),
        ("Near SR Sup", int(audit_df["near_support"].sum())),
        ("PE Cond1", int(audit_df["pe_cond1"].sum())),
        ("PE Cond2", int(audit_df["pe_cond2"].sum())),
        ("PE Cond3", int(audit_df["pe_cond3"].sum())),
        ("PE ALL 3", int((audit_df["pe_cond1"] & audit_df["pe_cond2"] & audit_df["pe_cond3"]).sum())),
        ("CE Cond1", int(audit_df["ce_cond1"].sum())),
        ("CE Cond2", int(audit_df["ce_cond2"].sum())),
        ("CE Cond3", int(audit_df["ce_cond3"].sum())),
        ("CE ALL 3", int((audit_df["ce_cond1"] & audit_df["ce_cond2"] & audit_df["ce_cond3"]).sum())),
    ]

    def _rate(hit_count: int) -> float:
        return (hit_count / total * 100.0) if total else 0.0

    lines = [
        "",
        f"SIGNAL AUDIT - {pd.Period(start, freq='D').start_time.date()} to {pd.Period(end, freq='D').end_time.date()}",
        f"Total candles scanned    : {total:,}",
        "Condition                | Hit Count | Hit Rate",
        "-------------------------|-----------|---------",
    ]
    for label, count in pairs[:8]:
        lines.append(f"{label.ljust(24)} | {str(count).rjust(9)} | {_rate(count):5.1f}%")
    lines.append("-------------------------|-----------|---------")
    for label, count in pairs[8:]:
        lines.append(f"{label.ljust(24)} | {str(count).rjust(9)} | {_rate(count):5.1f}%")

    pe_all3 = int((audit_df["pe_cond1"] & audit_df["pe_cond2"] & audit_df["pe_cond3"]).sum())
    ce_all3 = int((audit_df["ce_cond1"] & audit_df["ce_cond2"] & audit_df["ce_cond3"]).sum())
    pe_rate = _rate(pe_all3)
    ce_rate = _rate(ce_all3)
    total_signals = int((audit_df["action"] != "HOLD").sum())
    expected_trades = total_signals / max(len(pd.bdate_range(start=start, end=end)), 1)
    lines.extend(
        [
            "-------------------------|-----------|---------",
            f"TOTAL SIGNALS            : {total_signals:,}",
            f"Expected trades/month    : {expected_trades:.1f}",
        ]
    )
    if min(pe_rate, ce_rate) < 0.5:
        lines.extend(
            [
                "WARN SIGNAL TOO STRICT - conditions almost never align.",
                "Suggestion: relax SR_TOLERANCE or allow single signals to trade without BB confirmation.",
            ]
        )
    elif max(pe_rate, ce_rate) > 8.0:
        lines.extend(
            [
                "WARN SIGNAL TOO LOOSE - overtrading risk.",
                "Suggestion: tighten SR_TOLERANCE or require full confluence.",
            ]
        )
    return "\n".join(lines)


def _open_trade_from_signal(trade_id: int, row: pd.Series, signal: dict[str, Any]) -> Trade | None:
    """Create a Trade object from a buy signal."""

    action = str(signal.get("action", "HOLD")).upper()
    if action not in {"BUY_CE", "BUY_PE"}:
        return None

    direction = "CE" if action == "BUY_CE" else "PE"
    option_entry_price = _option_ltp(row, direction)
    if option_entry_price is None:
        return None

    qty_multiplier = float(signal.get("qty_multiplier", 1.0) or 1.0)
    qty_total = max(1, int(round(config.LOT_SIZE * qty_multiplier)))
    premium_sl = signal.get("signal_log", {}).get("premium_sl_price")
    premium_sl_price = float(premium_sl) if premium_sl is not None else option_entry_price * config.PREMIUM_SL_CONSERVATIVE_MULTIPLIER
    trade = Trade(
        trade_id=trade_id,
        direction=direction,
        entry_time=pd.Timestamp(row["datetime"]).to_pydatetime(),
        entry_spot=float(row["close"]),
        entry_strike=float(signal.get("atm_strike", row.get("atm_strike", row.get("strike", row["close"])))),
        option_entry_price=float(option_entry_price),
        qty_total=qty_total,
        qty_multiplier=qty_multiplier,
        target1=float(signal.get("target1_spot")) if signal.get("target1_spot") is not None else None,
        target2=float(signal.get("target2_spot")) if signal.get("target2_spot") is not None else None,
        stoploss_spot=float(signal.get("stoploss_spot")) if signal.get("stoploss_spot") is not None else None,
        conservative_sl_premium=option_entry_price * config.PREMIUM_SL_CONSERVATIVE_MULTIPLIER,
        aggressive_sl_premium=option_entry_price * config.PREMIUM_SL_AGGRESSIVE_MULTIPLIER,
        premium_sl_price=premium_sl_price,
        signal_strength=signal.get("strength"),
        vix_at_entry=float(row["vix_close"]) if pd.notna(row.get("vix_close")) else None,
        rsi_at_entry=float(row["rsi"]) if pd.notna(row.get("rsi")) else None,
        bb_position_at_entry=float(row["percent_b"]) if pd.notna(row.get("percent_b")) else None,
        is_expiry_day=pd.Timestamp(row["datetime"]).day_name() == "Thursday",
        is_thursday_trade=pd.Timestamp(row["datetime"]).day_name() == "Thursday",
        entry_reason=action,
        metadata={"signal_log": signal.get("signal_log", {})},
    )
    return trade


def _check_exit_conditions(trade: Trade, candle: pd.Series, option_ltp: float | None) -> tuple[bool, bool, str | None]:
    """Check stoploss, premium stop, target1, target2, and EOD exit conditions."""

    spot_close = float(candle["close"])
    candle_time = pd.Timestamp(candle["datetime"]).time()
    if trade.direction == "PE":
        stoploss_hit = trade.stoploss_spot is not None and spot_close > float(trade.stoploss_spot)
        target1_hit = trade.target1 is not None and spot_close <= float(trade.target1)
        target2_hit = trade.target2 is not None and spot_close <= float(trade.target2)
    else:
        stoploss_hit = trade.stoploss_spot is not None and spot_close < float(trade.stoploss_spot)
        target1_hit = trade.target1 is not None and spot_close >= float(trade.target1)
        target2_hit = trade.target2 is not None and spot_close >= float(trade.target2)

    premium_sl_hit = option_ltp is not None and trade.premium_sl_price not in (None, 0) and float(option_ltp) < float(trade.premium_sl_price)
    eod_hit = candle_time >= pd.Timestamp(config.EOD_EXIT_TIME).time()

    if stoploss_hit:
        return True, False, "stoploss"
    if premium_sl_hit:
        return True, False, "premium_sl"
    if not trade.partial_exit_done and target1_hit:
        return False, True, "target1"
    if target2_hit:
        return True, False, "target2"
    if eod_hit:
        return True, False, "eod_exit"
    return False, False, None


def _rolling_summary(trades_df: pd.DataFrame, starting_capital: float) -> tuple[int, float, float]:
    """Return running trade count, win rate, and PnL."""

    if trades_df.empty:
        return 0, 0.0, 0.0
    wins = trades_df[trades_df["total_pnl"] > 0]
    return len(trades_df), float(len(wins) / len(trades_df) * 100.0), float(trades_df["total_pnl"].sum())


def _run_backtest_loop(frame: pd.DataFrame, strategy, capital: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Portfolio]:
    """Run a persistent intraday backtest over the merged frame."""

    portfolio = Portfolio(starting_capital=capital)
    signal_rows: list[dict[str, Any]] = []
    trade_counter = 0
    cooldown_remaining = 0
    eod_time = pd.Timestamp(config.EOD_EXIT_TIME).time()

    frame = frame.copy().sort_values("datetime").reset_index(drop=True)
    frame["trade_date"] = pd.to_datetime(frame["datetime"]).dt.normalize()
    trading_days = list(frame["trade_date"].drop_duplicates())

    day_iterator = _progress(trading_days, total=len(trading_days), desc=f"Backtesting [{0:>3}/{len(trading_days)} days]")
    for day_index, trade_date in enumerate(day_iterator, start=1):
        day_frame = frame[frame["trade_date"] == trade_date]
        for _, row in day_frame.iterrows():
            indicators, levels = _signal_payload(row)
            signal = strategy.generate_signals(candle=row, indicators=indicators, levels=levels) or {}
            signal_rows.append(
                {
                    "datetime": row["datetime"],
                    "date": pd.Timestamp(row["datetime"]).date().isoformat(),
                    "action": signal.get("action", "HOLD"),
                    "strength": signal.get("strength"),
                    "atm_strike": signal.get("atm_strike"),
                    "qty_multiplier": signal.get("qty_multiplier"),
                    "target1_spot": signal.get("target1_spot"),
                    "target2_spot": signal.get("target2_spot"),
                    "stoploss_spot": signal.get("stoploss_spot"),
                    "signal_log": json.dumps(signal.get("signal_log", {}), default=str),
                }
            )

            if cooldown_remaining > 0:
                cooldown_remaining -= 1

            if portfolio.open_trade is not None:
                trade = portfolio.open_trade
                option_ltp = _option_ltp(row, trade.direction)
                should_exit_full, should_partial_exit, exit_reason = _check_exit_conditions(trade, row, option_ltp)

                if should_partial_exit and exit_reason == "target1" and option_ltp is not None:
                    portfolio.on_partial_exit(trade, option_ltp, pd.Timestamp(row["datetime"]).to_pydatetime())
                    portfolio.update_equity(row["datetime"], mark_to_market=option_ltp)
                elif should_exit_full and exit_reason is not None:
                    exit_price = option_ltp if option_ltp is not None else trade.option_entry_price
                    portfolio.on_full_exit(trade, exit_price, exit_reason, pd.Timestamp(row["datetime"]).to_pydatetime())
                    cooldown_remaining = config.COOLDOWN_CANDLES
                    portfolio.update_equity(row["datetime"], mark_to_market=None)
                else:
                    portfolio.update_equity(row["datetime"], mark_to_market=option_ltp)
                continue

            if row["datetime"].time() > eod_time:
                portfolio.update_equity(row["datetime"], mark_to_market=None)
                continue

            action = str(signal.get("action", "HOLD")).upper()
            if action in {"BUY_CE", "BUY_PE"} and cooldown_remaining == 0:
                trade_counter += 1
                trade = _open_trade_from_signal(trade_counter, row, signal)
                if trade is not None:
                    portfolio.open(trade)
            mark_to_market = None
            if portfolio.open_trade is not None:
                mark_to_market = _option_ltp(row, portfolio.open_trade.direction)
            portfolio.update_equity(row["datetime"], mark_to_market=mark_to_market)

        if portfolio.open_trade is not None:
            last_row = day_frame.iloc[-1]
            mark = _option_ltp(last_row, portfolio.open_trade.direction) or portfolio.open_trade.option_entry_price
            portfolio.on_full_exit(portfolio.open_trade, mark, "eod_exit", pd.Timestamp(last_row["datetime"]).to_pydatetime())
            cooldown_remaining = config.COOLDOWN_CANDLES
            portfolio.update_equity(last_row["datetime"], mark_to_market=None)

        if day_index % 50 == 0:
            trades_so_far, win_rate, running_pnl = _rolling_summary(pd.DataFrame(portfolio.trade_log), capital)
            print(f"  → Trades so far: {trades_so_far} | Win rate: {win_rate:.1f}% | Running PnL: {_fmt_inr(running_pnl)}")

    trades_df = pd.DataFrame(portfolio.trade_log)
    signals_df = pd.DataFrame(signal_rows)
    equity_df = portfolio.equity_frame()
    return trades_df, signals_df, equity_df, portfolio


def _build_daily_pnl(trades_df: pd.DataFrame, equity_df: pd.DataFrame, capital: float) -> pd.DataFrame:
    """Build daily PnL and drawdown series."""

    if equity_df.empty:
        return pd.DataFrame(columns=["date", "daily_pnl", "cumulative_pnl", "drawdown"])

    curve = equity_df.copy()
    curve["datetime"] = pd.to_datetime(curve["datetime"], errors="coerce")
    curve = curve.dropna(subset=["datetime"]).sort_values("datetime")
    curve["date"] = curve["datetime"].dt.date.astype(str)
    daily_equity = curve.groupby("date", as_index=False).agg(equity=("equity", "last"))
    daily_equity["daily_pnl"] = daily_equity["equity"].diff().fillna(daily_equity["equity"] - capital)
    daily_equity["cumulative_pnl"] = daily_equity["equity"] - capital
    peak = daily_equity["equity"].cummax()
    daily_equity["drawdown"] = peak - daily_equity["equity"]
    return daily_equity[["date", "daily_pnl", "cumulative_pnl", "drawdown"]]


def _write_outputs(output_dir: Path, trades_df: pd.DataFrame, daily_pnl_df: pd.DataFrame, signals_df: pd.DataFrame, config_blob: dict[str, Any], summary_text: str) -> None:
    """Persist all run artifacts to disk."""

    output_dir.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(output_dir / "trades.csv", index=False)
    daily_pnl_df.to_csv(output_dir / "daily_pnl.csv", index=False)
    signals_df.to_csv(output_dir / "signal_log.csv", index=False)
    (output_dir / "run_config.json").write_text(json.dumps(config_blob, indent=2, default=str), encoding="utf-8")
    (output_dir / "summary.txt").write_text(summary_text, encoding="utf-8")


def _build_summary_text(start: str, end: str, stats: dict[str, Any], daily_pnl_df: pd.DataFrame) -> str:
    """Create the human-readable run summary."""

    if daily_pnl_df.empty:
        max_dd = 0.0
    else:
        max_dd = float(daily_pnl_df["drawdown"].max())
    total_pnl = float(daily_pnl_df["cumulative_pnl"].iloc[-1]) if not daily_pnl_df.empty else 0.0
    return (
        f"Period       : {start} → {end}\n"
        f"Total Trades : {stats['total_trades']}\n"
        f"Winners      : {stats['winners']}  ({stats['win_rate_pct']:.1f}%)\n"
        f"Losers       : {stats['losers']}  ({stats['loss_rate_pct']:.1f}%)\n"
        f"Gross Profit : {stats['gross_profit_text']}\n"
        f"Gross Loss   : {stats['gross_loss_text']}\n"
        f"Net PnL      : {_fmt_inr(total_pnl)}\n"
        f"Profit Factor: {stats['profit_factor']:.2f}\n"
        f"Max Drawdown : {_fmt_inr(max_dd)}\n"
        f"Avg Win      : {_fmt_inr(stats['avg_win_amount'])}\n"
        f"Avg Loss     : {_fmt_inr(stats['avg_loss_amount'])}\n"
        f"Largest Win  : {_fmt_inr(stats['largest_win'])}\n"
        f"Largest Loss : {_fmt_inr(abs(stats['largest_loss']))}\n"
        f"Sharpe Ratio : {stats['sharpe_ratio']:.2f}\n"
    )


def main() -> None:
    """Run the full production backtest and write output artifacts."""

    args = _parse_args()
    start, end = _resolve_period(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s | %(levelname)s | %(message)s")

    strategy = _load_strategy(args.strategy)

    if args.dry_run:
        _print_run_config(start, end, strategy, args.capital, output_dir)
        return

    _print_run_config(start, end, strategy, args.capital, output_dir)

    if args.no_cache:
        _clear_cache()

    step_start = time.perf_counter()
    print("Fetching spot data...")
    spot_df = fetch_spot_data(
        symbol=config.SPOT_SYMBOL,
        start_date=start,
        end_date=end,
        interval=config.INTERVAL,
        exchange_code=config.SPOT_EXCHANGE,
    )
    spot_df = _rehydrate_synthetic_intraday(spot_df)
    elapsed = time.perf_counter() - step_start
    _print_step(f"Fetched spot data: {len(spot_df)} candles", elapsed)

    if spot_df.empty:
        raise SystemExit("No spot data returned; aborting.")

    daily_summary = _business_day_summary(spot_df)
    print(f"Fetching options data for {len(pd.to_datetime(daily_summary['expiry']).dt.normalize().unique())} expiry weeks...")
    step_start = time.perf_counter()
    option_df = _build_contract_frame(daily_summary)
    option_df = _rehydrate_synthetic_intraday(option_df)
    elapsed = time.perf_counter() - step_start
    _print_step(f"Fetched options data: {len(option_df)} candles", elapsed)

    print("Fetching India VIX...")
    step_start = time.perf_counter()
    vix_df = fetch_india_vix(start, end, interval="1day", symbol=config.VIX_SYMBOL, exchange_code=config.VIX_EXCHANGE)
    elapsed = time.perf_counter() - step_start
    _print_step(f"Fetched VIX data: {len(vix_df)} candles", elapsed)

    print("Running indicators...")
    step_start = time.perf_counter()
    enriched = build_backtest_frame(spot_df, option_df, vix_df)
    elapsed = time.perf_counter() - step_start
    _print_step(f"Indicators ready: {len(enriched)} rows", elapsed)

    print("Auditing signals...")
    audit_df = strategy.audit_signals(enriched)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not audit_df.empty:
        audit_df.to_csv(output_dir / "signal_audit.csv", index=False)
    print(_signal_audit_summary(audit_df, start, end))

    print("Running backtest...")
    step_start = time.perf_counter()
    trades_df, signals_df, equity_df, portfolio = _run_backtest_loop(enriched, strategy, args.capital)
    elapsed = time.perf_counter() - step_start
    _print_step(f"Backtest complete: {len(trades_df)} trades", elapsed)

    daily_pnl_df = _build_daily_pnl(trades_df, equity_df, args.capital)

    stats = compute_statistics(trades_df, equity_curve=equity_df, starting_capital=args.capital)
    wins = trades_df[trades_df["total_pnl"] > 0] if not trades_df.empty else pd.DataFrame()
    losses = trades_df[trades_df["total_pnl"] < 0] if not trades_df.empty else pd.DataFrame()
    gross_profit = float(wins["total_pnl"].sum()) if not wins.empty else 0.0
    gross_loss = float(abs(losses["total_pnl"].sum())) if not losses.empty else 0.0
    net_pnl = float(trades_df["total_pnl"].sum()) if not trades_df.empty else 0.0
    avg_win = float(wins["total_pnl"].mean()) if not wins.empty else 0.0
    avg_loss = float(abs(losses["total_pnl"].mean())) if not losses.empty else 0.0
    largest_win = float(trades_df["total_pnl"].max()) if not trades_df.empty else 0.0
    largest_loss = float(trades_df["total_pnl"].min()) if not trades_df.empty else 0.0
    winners = int(len(wins))
    losers = int(len(losses))
    stats.update(
        {
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "winners": winners,
            "losers": losers,
            "avg_win_amount": avg_win,
            "avg_loss_amount": avg_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "gross_profit_text": _fmt_inr(gross_profit),
            "gross_loss_text": _fmt_inr(gross_loss),
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_outputs(
        output_dir,
        trades_df,
        daily_pnl_df,
        signals_df,
        {
            "start": start,
            "end": end,
            "year": args.year,
            "month": args.month,
            "strategy": args.strategy,
            "capital": args.capital,
            "log_level": args.log_level,
            "dry_run": args.dry_run,
            "no_cache": args.no_cache,
            "output_dir": str(output_dir),
            "lot_size": config.LOT_SIZE,
        },
        _build_summary_text(start, end, stats, daily_pnl_df),
    )

    trading_days = int(len(pd.bdate_range(start=start, end=end)))
    max_drawdown_amount = float(daily_pnl_df["drawdown"].max()) if not daily_pnl_df.empty else 0.0
    max_drawdown_pct = (max_drawdown_amount / args.capital * 100.0) if args.capital else 0.0
    net_pct = (net_pnl / args.capital * 100.0) if args.capital else 0.0

    print(
        f"""
════════════════════════════════════════════
BACKTEST COMPLETE
════════════════════════════════════════════
Period       : {start} → {end}
Trading Days : {trading_days}
────────────────────────────────────────────
Total Trades : {len(trades_df)}
Winners      : {winners}  ({(winners / len(trades_df) * 100.0) if len(trades_df) else 0.0:.1f}%)
Losers       : {losers}  ({(losers / len(trades_df) * 100.0) if len(trades_df) else 0.0:.1f}%)
────────────────────────────────────────────
Gross Profit : {_fmt_inr(gross_profit)}
Gross Loss   : {_fmt_inr(gross_loss)}
Net PnL      : {_fmt_inr(net_pnl)}  ({net_pct:+.1f}% on capital)
Profit Factor: {stats['profit_factor']:.2f}
────────────────────────────────────────────
Max Drawdown : {_fmt_inr(max_drawdown_amount)}  ({max_drawdown_pct:.1f}%)
Avg Win      : {_fmt_inr(avg_win)}
Avg Loss     : {_fmt_inr(avg_loss)}
Largest Win  : {_fmt_inr(largest_win)}
Largest Loss : {_fmt_inr(abs(largest_loss))}
────────────────────────────────────────────
Sharpe Ratio : {stats['sharpe_ratio']:.2f}
────────────────────────────────────────────
Results saved to: {output_dir.as_posix()}
Run analyze:  python scripts/analyze_results.py --dir {output_dir.as_posix()}
════════════════════════════════════════════
"""
    )


if __name__ == "__main__":
    main()
