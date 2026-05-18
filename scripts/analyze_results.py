"""Post-run analysis and report generator for completed backtests."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Any

import pandas as pd

try:  # pragma: no cover - optional dependency
    from tabulate import tabulate
except Exception:  # pragma: no cover - fallback when tabulate is unavailable
    tabulate = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reports.backtest_report import generate_report

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def _load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file or return an empty dataframe if missing."""

    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _require_results_dir(result_dir: Path) -> None:
    """Validate that the results directory contains the expected artifacts."""

    required = ["trades.csv", "daily_pnl.csv", "signal_log.csv"]
    if not result_dir.exists():
        print(f"❌ Results directory not found: {result_dir}")
        sys.exit(1)
    missing = [name for name in required if not (result_dir / name).exists()]
    if missing:
        print(f"❌ Results directory is incomplete: missing {', '.join(missing)}")
        sys.exit(1)


def _fmt_inr(value: float) -> str:
    """Format a number as Indian rupees text."""

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


def _table(rows: list[list[Any]], headers: list[str]) -> str:
    """Render a table for console output."""

    if tabulate is not None:
        return tabulate(rows, headers=headers, tablefmt="github")

    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(str(value)))
    lines = []
    lines.append(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    lines.append("-+-".join("-" * width for width in widths))
    for row in rows:
        lines.append(" | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)))
    return "\n".join(lines)


def _trade_frame(trades: pd.DataFrame) -> pd.DataFrame:
    """Normalize trade dataframe dtypes and ordering."""

    if trades.empty:
        return trades.copy()
    df = trades.copy()
    for column in ["entry_time", "partial_exit_time", "full_exit_time", "exit_time"]:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    df = df.sort_values("entry_time").reset_index(drop=True)
    if "total_pnl" not in df.columns and "pnl_amount" in df.columns:
        df["total_pnl"] = df["pnl_amount"]
    if "direction" in df.columns:
        df["direction"] = df["direction"].astype(str).str.upper()
    return df


def _exit_time_series(df: pd.DataFrame) -> pd.Series:
    """Return the best available exit-time series from a trade dataframe."""

    if "full_exit_time" in df.columns and "exit_time" in df.columns:
        return pd.to_datetime(df["full_exit_time"].fillna(df["exit_time"]), errors="coerce")
    if "full_exit_time" in df.columns:
        return pd.to_datetime(df["full_exit_time"], errors="coerce")
    if "exit_time" in df.columns:
        return pd.to_datetime(df["exit_time"], errors="coerce")
    return pd.Series(pd.NaT, index=df.index)


def _daily_curve(daily_pnl: pd.DataFrame, capital: float) -> pd.DataFrame:
    """Normalize daily PnL and derive equity/drawdown columns."""

    if daily_pnl.empty:
        return daily_pnl.copy()
    df = daily_pnl.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if "daily_pnl" not in df.columns:
        df["daily_pnl"] = 0.0
    if "cumulative_pnl" not in df.columns:
        df["cumulative_pnl"] = df["daily_pnl"].cumsum()
    if "drawdown" not in df.columns:
        equity = capital + df["cumulative_pnl"]
        peak = equity.cummax()
        df["drawdown"] = peak - equity
    df["equity"] = capital + df["cumulative_pnl"]
    df["drawdown_pct"] = (df["drawdown"] / capital) * 100.0 if capital else 0.0
    return df


def _summary_metrics(trades: pd.DataFrame, daily_curve: pd.DataFrame, capital: float) -> dict[str, Any]:
    """Compute core performance metrics for analysis and comparison."""

    if trades.empty:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "net_pnl": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_amount": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
        }

    pnl_col = "total_pnl" if "total_pnl" in trades.columns else "pnl_amount"
    wins = trades[trades[pnl_col] > 0]
    losses = trades[trades[pnl_col] < 0]
    gross_profit = float(wins[pnl_col].sum())
    gross_loss = float(abs(losses[pnl_col].sum()))
    profit_factor = float("inf") if gross_loss == 0 and gross_profit > 0 else (gross_profit / gross_loss if gross_loss else 0.0)
    net_pnl = float(trades[pnl_col].sum())
    win_rate = float(len(wins) / len(trades) * 100.0)
    sharpe = 0.0
    max_dd_amount = 0.0
    max_dd_pct = 0.0
    if not daily_curve.empty:
        returns = daily_curve["equity"].pct_change().fillna(0.0)
        volatility = float(returns.std(ddof=0))
        sharpe = float((returns.mean() / volatility) * math.sqrt(252)) if volatility else 0.0
        max_dd_amount = float(daily_curve["drawdown"].max())
        max_dd_pct = float((max_dd_amount / capital) * 100.0) if capital else 0.0
    return {
        "total_trades": int(len(trades)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate_pct": win_rate,
        "net_pnl": net_pnl,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "max_drawdown_amount": max_dd_amount,
        "max_drawdown_pct": max_dd_pct,
        "sharpe_ratio": sharpe,
    }


def _monthly_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    """Build a monthly performance table."""

    if trades.empty:
        return pd.DataFrame(columns=["Month", "Trades", "Winners", "Win%", "Net PnL", "Best Trade", "Worst Trade"])
    df = trades.copy()
    df["exit_time"] = _exit_time_series(df)
    df = df.dropna(subset=["exit_time"])
    pnl_col = "total_pnl" if "total_pnl" in df.columns else "pnl_amount"
    df["month"] = df["exit_time"].dt.to_period("M").astype(str)
    rows = []
    for month, group in df.groupby("month"):
        winners = group[group[pnl_col] > 0]
        best_trade = float(group[pnl_col].max())
        worst_trade = float(group[pnl_col].min())
        rows.append(
            [
                month,
                int(len(group)),
                int(len(winners)),
                round((len(winners) / len(group) * 100.0) if len(group) else 0.0, 1),
                _fmt_inr(float(group[pnl_col].sum())),
                _fmt_inr(best_trade),
                _fmt_inr(worst_trade),
            ]
        )
    return pd.DataFrame(rows, columns=["Month", "Trades", "Winners", "Win%", "Net PnL", "Best Trade", "Worst Trade"])


def _signal_quality(trades: pd.DataFrame) -> pd.DataFrame:
    """Summarize performance by signal strength."""

    if trades.empty or "strength" not in trades.columns:
        return pd.DataFrame(columns=["Strength", "Count", "Win Rate", "Avg PnL", "Avg Winner", "Avg Loser", "Profit Factor"])
    df = trades.copy()
    pnl_col = "total_pnl" if "total_pnl" in df.columns else "pnl_amount"
    rows = []
    for strength, group in df.groupby(df["strength"].fillna("NA")):
        winners = group[group[pnl_col] > 0]
        losers = group[group[pnl_col] < 0]
        gross_profit = float(winners[pnl_col].sum())
        gross_loss = float(abs(losers[pnl_col].sum()))
        profit_factor = float("inf") if gross_loss == 0 and gross_profit > 0 else (gross_profit / gross_loss if gross_loss else 0.0)
        rows.append(
            [
                strength,
                int(len(group)),
                round((len(winners) / len(group) * 100.0) if len(group) else 0.0, 1),
                _fmt_inr(float(group[pnl_col].mean())),
                _fmt_inr(float(winners[pnl_col].mean()) if not winners.empty else 0.0),
                _fmt_inr(float(losers[pnl_col].mean()) if not losers.empty else 0.0),
                f"{profit_factor:.2f}" if math.isfinite(profit_factor) else "inf",
            ]
        )
    return pd.DataFrame(rows, columns=["Strength", "Count", "Win Rate", "Avg PnL", "Avg Winner", "Avg Loser", "Profit Factor"])


def _exit_reason_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    """Summarize exits by reason."""

    if trades.empty or "exit_reason" not in trades.columns:
        return pd.DataFrame(columns=["Exit Reason", "Count", "% of Trades", "Avg PnL"])
    df = trades.copy()
    pnl_col = "total_pnl" if "total_pnl" in df.columns else "pnl_amount"
    rows = []
    total = len(df)
    for reason, group in df.groupby(df["exit_reason"].fillna("NA")):
        rows.append(
            [
                reason,
                int(len(group)),
                round((len(group) / total * 100.0) if total else 0.0, 1),
                _fmt_inr(float(group[pnl_col].mean())),
            ]
        )
    return pd.DataFrame(rows, columns=["Exit Reason", "Count", "% of Trades", "Avg PnL"])


def _time_bucket(value: pd.Timestamp) -> str:
    """Bucket a timestamp into the requested entry windows."""

    t = value.time()
    buckets = [
        (dtime(9, 20), dtime(10, 0), "09:20-10:00"),
        (dtime(10, 0), dtime(11, 0), "10:00-11:00"),
        (dtime(11, 0), dtime(12, 0), "11:00-12:00"),
        (dtime(12, 0), dtime(13, 0), "12:00-13:00"),
    ]
    for start, end, label in buckets:
        if start <= t < end or (label == "12:00-13:00" and start <= t <= end):
            return label
    return "Other"


def _time_of_day_analysis(trades: pd.DataFrame) -> pd.DataFrame:
    """Analyze trade performance by entry time bucket."""

    if trades.empty:
        return pd.DataFrame(columns=["Entry Window", "Trade Count", "Win Rate", "Avg PnL"])
    df = trades.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df = df.dropna(subset=["entry_time"])
    df["bucket"] = df["entry_time"].apply(_time_bucket)
    pnl_col = "total_pnl" if "total_pnl" in df.columns else "pnl_amount"
    rows = []
    for bucket in ["09:20-10:00", "10:00-11:00", "11:00-12:00", "12:00-13:00"]:
        group = df[df["bucket"] == bucket]
        winners = group[group[pnl_col] > 0]
        rows.append(
            [
                bucket,
                int(len(group)),
                round((len(winners) / len(group) * 100.0) if len(group) else 0.0, 1),
                _fmt_inr(float(group[pnl_col].mean()) if not group.empty else 0.0),
            ]
        )
    return pd.DataFrame(rows, columns=["Entry Window", "Trade Count", "Win Rate", "Avg PnL"])


def _vix_regime_analysis(trades: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance by VIX regime."""

    if trades.empty:
        return pd.DataFrame(columns=["VIX Regime", "Trade Count", "Win Rate", "Avg PnL", "Avg Qty Multiplier"])
    df = trades.copy()
    pnl_col = "total_pnl" if "total_pnl" in df.columns else "pnl_amount"
    rows = []
    regimes = [
        ("<13", lambda value: pd.notna(value) and float(value) < 13),
        ("13-15", lambda value: pd.notna(value) and 13 <= float(value) <= 15),
        (">15", lambda value: pd.notna(value) and float(value) > 15),
    ]
    for label, predicate in regimes:
        group = df[df["vix_at_entry"].apply(predicate)] if "vix_at_entry" in df.columns else pd.DataFrame()
        winners = group[group[pnl_col] > 0] if not group.empty else pd.DataFrame()
        rows.append(
            [
                label,
                int(len(group)),
                round((len(winners) / len(group) * 100.0) if len(group) else 0.0, 1),
                _fmt_inr(float(group[pnl_col].mean()) if not group.empty else 0.0),
                round(float(group["qty_multiplier"].mean()) if not group.empty and "qty_multiplier" in group.columns else 0.0, 2),
            ]
        )
    return pd.DataFrame(rows, columns=["VIX Regime", "Trade Count", "Win Rate", "Avg PnL", "Avg Qty Multiplier"])


def _day_of_week_analysis(trades: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance by day of week."""

    if trades.empty:
        return pd.DataFrame(columns=["Day", "Trade Count", "Win Rate", "Avg PnL"])
    df = trades.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df = df.dropna(subset=["entry_time"])
    df["day"] = df["entry_time"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday"]
    pnl_col = "total_pnl" if "total_pnl" in df.columns else "pnl_amount"
    rows = []
    for day in order:
        group = df[df["day"] == day]
        winners = group[group[pnl_col] > 0]
        rows.append(
            [
                day,
                int(len(group)),
                round((len(winners) / len(group) * 100.0) if len(group) else 0.0, 1),
                _fmt_inr(float(group[pnl_col].mean()) if not group.empty else 0.0),
            ]
        )
    return pd.DataFrame(rows, columns=["Day", "Trade Count", "Win Rate", "Avg PnL"])


def _consecutive_analysis(trades: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Analyze consecutive win/loss streaks."""

    if trades.empty:
        return pd.DataFrame(columns=["Metric", "Value"]), {"three_loss_followup_win_rate": 0.0, "loss_streak_3plus": 0}
    df = trades.copy()
    df["exit_time"] = _exit_time_series(df)
    df = df.dropna(subset=["exit_time"]).sort_values("exit_time").reset_index(drop=True)
    pnl_col = "total_pnl" if "total_pnl" in df.columns else "pnl_amount"
    outcomes = [1 if value > 0 else -1 if value < 0 else 0 for value in df[pnl_col]]
    max_win = 0
    max_loss = 0
    current_sign = 0
    current_length = 0
    streak_lengths: list[tuple[int, int]] = []
    after_three_loss_followup = []
    for idx, outcome in enumerate(outcomes):
        if outcome == current_sign:
            current_length += 1
        else:
            if current_sign != 0 and current_length > 0:
                streak_lengths.append((current_sign, current_length))
            current_sign = outcome
            current_length = 1 if outcome != 0 else 0
        if outcome > 0:
            max_win = max(max_win, current_length)
        elif outcome < 0:
            max_loss = min(max_loss, -current_length)
    if current_sign != 0 and current_length > 0:
        streak_lengths.append((current_sign, current_length))

    for idx, (sign, length) in enumerate(streak_lengths):
        if sign < 0 and length >= 3:
            next_index = sum(prev_length for _, prev_length in streak_lengths[: idx + 1])
            if next_index < len(outcomes):
                after_three_loss_followup.append(outcomes[next_index] > 0)

    three_plus_loss_streaks = sum(1 for sign, length in streak_lengths if sign < 0 and length >= 3)
    followup_rate = float(sum(after_three_loss_followup) / len(after_three_loss_followup) * 100.0) if after_three_loss_followup else 0.0
    table = pd.DataFrame(
        [
            ["Max consecutive wins", int(max_win)],
            ["Max consecutive losses", int(abs(max_loss))],
            ["3+ loss streaks", int(three_plus_loss_streaks)],
            ["Next trade win rate after 3 losses", f"{followup_rate:.1f}%"],
        ],
        columns=["Metric", "Value"],
    )
    details = {"three_loss_followup_win_rate": followup_rate, "loss_streak_3plus": three_plus_loss_streaks}
    return table, details


def _ce_pe_analysis(trades: pd.DataFrame) -> pd.DataFrame:
    """Analyze CE versus PE performance."""

    if trades.empty:
        return pd.DataFrame(columns=["Direction", "Count", "Win Rate", "Avg PnL"])
    df = trades.copy()
    pnl_col = "total_pnl" if "total_pnl" in df.columns else "pnl_amount"
    rows = []
    for direction in ["CE", "PE"]:
        group = df[df["direction"].astype(str).str.upper() == direction]
        winners = group[group[pnl_col] > 0]
        rows.append(
            [
                direction,
                int(len(group)),
                round((len(winners) / len(group) * 100.0) if len(group) else 0.0, 1),
                _fmt_inr(float(group[pnl_col].mean()) if not group.empty else 0.0),
            ]
        )
    return pd.DataFrame(rows, columns=["Direction", "Count", "Win Rate", "Avg PnL"])


def _drawdown_analysis(daily_curve: pd.DataFrame) -> pd.DataFrame:
    """Analyze drawdown depth and duration."""

    if daily_curve.empty:
        return pd.DataFrame(columns=["Metric", "Value"])
    df = daily_curve.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    equity = df["equity"]
    peak = equity.cummax()
    dd = peak - equity
    dd_pct = dd / peak.replace(0, pd.NA) * 100.0
    in_drawdown = dd > 0
    drawdown_periods = []
    start_idx = None
    peak_idx = 0
    max_dd_amount = 0.0
    max_dd_pct = 0.0
    for idx, is_dd in enumerate(in_drawdown):
        if not is_dd:
            if start_idx is not None:
                period = df.iloc[start_idx:idx]
                drawdown_periods.append(
                    {
                        "start": df.iloc[start_idx]["date"],
                        "end": df.iloc[idx]["date"],
                        "max_dd": float((peak.iloc[start_idx:idx] - equity.iloc[start_idx:idx]).max()),
                        "max_dd_pct": float(dd_pct.iloc[start_idx:idx].max()),
                        "duration_days": int((df.iloc[idx]["date"] - df.iloc[start_idx]["date"]).days),
                    }
                )
            start_idx = None
            peak_idx = idx
        else:
            if start_idx is None:
                start_idx = max(0, peak_idx)
    if start_idx is not None:
        period = df.iloc[start_idx:]
        drawdown_periods.append(
            {
                "start": df.iloc[start_idx]["date"],
                "end": df.iloc[-1]["date"],
                "max_dd": float((peak.iloc[start_idx:] - equity.iloc[start_idx:]).max()),
                "max_dd_pct": float(dd_pct.iloc[start_idx:].max()),
                "duration_days": int((df.iloc[-1]["date"] - df.iloc[start_idx]["date"]).days),
            }
        )
    if not dd.empty:
        max_dd_amount = float(dd.max())
        max_dd_pct = float(dd_pct.max()) if not dd_pct.empty else 0.0
    count_2 = sum(1 for period in drawdown_periods if period["max_dd_pct"] > 2.0)
    count_5 = sum(1 for period in drawdown_periods if period["max_dd_pct"] > 5.0)
    count_10 = sum(1 for period in drawdown_periods if period["max_dd_pct"] > 10.0)
    max_duration = max((period["duration_days"] for period in drawdown_periods), default=0)
    table = pd.DataFrame(
        [
            ["Max drawdown amount", _fmt_inr(max_dd_amount)],
            ["Max drawdown % of capital", f"{max_dd_pct:.2f}%"],
            ["Max drawdown duration (days)", int(max_duration)],
            ["> 2% drawdowns", int(count_2)],
            ["> 5% drawdowns", int(count_5)],
            ["> 10% drawdowns", int(count_10)],
        ],
        columns=["Metric", "Value"],
    )
    return table


def _compare_runs(dir_a: Path, dir_b: Path) -> pd.DataFrame:
    """Build a side-by-side comparison table for two result directories."""

    def _summary_for(path: Path) -> dict[str, Any]:
        trades = _trade_frame(_load_csv(path / "trades.csv"))
        capital = 500000.0
        if (path / "run_config.json").exists():
            try:
                capital = float(json.loads((path / "run_config.json").read_text(encoding="utf-8")).get("capital", capital))
            except Exception:
                pass
        daily = _daily_curve(_load_csv(path / "daily_pnl.csv"), capital=capital)
        return _summary_metrics(trades, daily, capital)

    a = _summary_for(dir_a)
    b = _summary_for(dir_b)
    rows = []
    for metric, key in [
        ("Total Trades", "total_trades"),
        ("Win Rate %", "win_rate_pct"),
        ("Net PnL", "net_pnl"),
        ("Profit Factor", "profit_factor"),
        ("Max Drawdown %", "max_drawdown_pct"),
        ("Sharpe Ratio", "sharpe_ratio"),
    ]:
        rows.append([metric, a.get(key), b.get(key), (b.get(key, 0) or 0) - (a.get(key, 0) or 0)])
    return pd.DataFrame(rows, columns=["Metric", "Run A", "Run B", "Delta (B-A)"])


def _print_section(title: str) -> None:
    """Print a numbered section header."""

    print(f"\n--- {title} ---")


def main() -> None:
    """Run the completed-backtest analysis."""

    parser = argparse.ArgumentParser(description="Analyze backtest results.")
    parser.add_argument("--dir", required=True, help="Path to a results directory")
    parser.add_argument("--compare", default=None, help="Path to a second results dir for A/B comparison")
    parser.add_argument("--export-html", action="store_true", default=True, help="Save full HTML report")
    parser.add_argument("--no-export-html", action="store_false", dest="export_html", help="Disable HTML report generation")
    args = parser.parse_args()

    result_dir = Path(args.dir).expanduser().resolve()
    _require_results_dir(result_dir)

    trades = _trade_frame(_load_csv(result_dir / "trades.csv"))
    daily = _daily_curve(_load_csv(result_dir / "daily_pnl.csv"), capital=500000.0)
    signals = _load_csv(result_dir / "signal_log.csv")
    capital = 500000.0
    if (result_dir / "run_config.json").exists():
        try:
            config_blob = json.loads((result_dir / "run_config.json").read_text(encoding="utf-8"))
            capital = float(config_blob.get("capital", capital))
            daily = _daily_curve(_load_csv(result_dir / "daily_pnl.csv"), capital=capital)
        except Exception:
            pass

    if trades.empty or daily.empty:
        print(f"❌ Results directory is empty or incomplete: {result_dir}")
        sys.exit(1)

    metrics = _summary_metrics(trades, daily, capital)

    _print_section("SECTION 1: Monthly Breakdown")
    monthly = _monthly_breakdown(trades)
    print(_table(monthly.values.tolist(), monthly.columns.tolist()))
    if not monthly.empty:
        month_values = pd.to_numeric(monthly["Net PnL"].astype(str).str.replace("₹", "").str.replace(",", ""), errors="coerce")
        best_idx = month_values.idxmax()
        worst_idx = month_values.idxmin()
        print(f"Best month: {monthly.loc[best_idx, 'Month']}")
        print(f"Worst month: {monthly.loc[worst_idx, 'Month']}")

    _print_section("SECTION 2: Signal Quality Analysis")
    signal_quality = _signal_quality(trades)
    print(_table(signal_quality.values.tolist(), signal_quality.columns.tolist()))
    if not signal_quality.empty and {"Strength", "Win Rate", "Avg PnL"}.issubset(signal_quality.columns):
        print("Key insight: compare 'full' against 'single' in the table above.")

    _print_section("SECTION 3: Exit Reason Breakdown")
    exit_reason = _exit_reason_breakdown(trades)
    print(_table(exit_reason.values.tolist(), exit_reason.columns.tolist()))
    if not exit_reason.empty:
        target_rows = exit_reason[exit_reason["Exit Reason"].isin(["target1", "target2"])]
        stop_rows = exit_reason[exit_reason["Exit Reason"] == "stoploss"]
        target_pct = float(target_rows["% of Trades"].astype(float).sum()) if not target_rows.empty else 0.0
        stop_pct = float(stop_rows["% of Trades"].astype(float).sum()) if not stop_rows.empty else 0.0
        print(f"Key insight: target exits = {target_pct:.1f}% of trades; stoploss exits = {stop_pct:.1f}% of trades.")

    _print_section("SECTION 4: Time-of-Day Analysis")
    tod = _time_of_day_analysis(trades)
    print(_table(tod.values.tolist(), tod.columns.tolist()))
    if not tod.empty:
        best_bucket = tod.iloc[pd.to_numeric(tod["Win Rate"], errors="coerce").idxmax()]
        print(f"Key insight: best entry window appears to be {best_bucket['Entry Window']}.")

    _print_section("SECTION 5: VIX Regime Analysis")
    vix_regime = _vix_regime_analysis(trades)
    print(_table(vix_regime.values.tolist(), vix_regime.columns.tolist()))
    if not vix_regime.empty:
        low_row = vix_regime[vix_regime["VIX Regime"] == "<13"]
        print("Key insight: compare low-IV vs high-IV performance in the table above.")

    _print_section("SECTION 6: Day-of-Week Analysis")
    dow = _day_of_week_analysis(trades)
    print(_table(dow.values.tolist(), dow.columns.tolist()))
    if not dow.empty:
        thursday = dow[dow["Day"] == "Thursday"]
        if not thursday.empty:
            print(f"Key insight: Thursday performance is {thursday.iloc[0]['Win Rate']} win rate.")

    _print_section("SECTION 7: Consecutive Trade Analysis")
    consecutive_table, consecutive_meta = _consecutive_analysis(trades)
    print(_table(consecutive_table.values.tolist(), consecutive_table.columns.tolist()))

    _print_section("SECTION 8: CE vs PE Performance")
    cepe = _ce_pe_analysis(trades)
    print(_table(cepe.values.tolist(), cepe.columns.tolist()))

    _print_section("SECTION 9: Drawdown Analysis")
    drawdown = _drawdown_analysis(daily)
    print(_table(drawdown.values.tolist(), drawdown.columns.tolist()))

    if args.compare:
        _print_section("SECTION 10: A/B Comparison")
        compare_dir = Path(args.compare).expanduser().resolve()
        _require_results_dir(compare_dir)
        comparison = _compare_runs(result_dir, compare_dir)
        print(_table(comparison.values.tolist(), comparison.columns.tolist()))

    if args.export_html:
        report_path = generate_report(result_dir)
        print(f"📊 Report saved: {report_path}")
        print(f"   Open in browser: file://{report_path.resolve()}")


if __name__ == "__main__":
    main()
