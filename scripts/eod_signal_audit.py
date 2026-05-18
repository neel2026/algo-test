"""Audit daily EOD signal conditions for a single month."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:  # pragma: no cover - optional dependency
    from tabulate import tabulate
except Exception:  # pragma: no cover - fallback
    tabulate = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nifty_backtest import config
from nifty_backtest.data.fetcher import FreeDataFetcher
from nifty_backtest.data.processor import build_backtest_frame

logger = logging.getLogger(__name__)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run an EOD signal audit for one month.")
    parser.add_argument("--month", default="2024-06", help="Month to inspect in YYYY-MM format.")
    return parser.parse_args()


def _format_table(rows: list[list[object]], headers: list[str]) -> str:
    """Render a readable table."""

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


def _month_bounds(month: str) -> tuple[str, str]:
    """Return the first and last date of a YYYY-MM month."""

    period = pd.Period(month, freq="M")
    return period.start_time.normalize().strftime("%Y-%m-%d"), period.end_time.normalize().strftime("%Y-%m-%d")


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
    """Evaluate the daily EOD signal rules for one row."""

    close = float(row.get("close"))
    tolerance = float(config.SR_TOLERANCE)
    sr_levels = row.get("sr_levels") if isinstance(row.get("sr_levels"), list) else []
    resistance_levels, support_levels = _select_levels(close, sr_levels, tolerance)

    near_resistance = any(close >= float(level) - tolerance for level in resistance_levels)
    near_support = any(close <= float(level) + tolerance for level in support_levels)
    near_r1r2 = any(pd.notna(level) and close >= float(level) - tolerance for level in [row.get("r1"), row.get("r2")])
    near_s1s2 = any(pd.notna(level) and close <= float(level) + tolerance for level in [row.get("s1"), row.get("s2")])

    at_upper_band = pd.notna(row.get("upper_band")) and close >= float(row.get("upper_band"))
    at_lower_band = pd.notna(row.get("lower_band")) and close <= float(row.get("lower_band"))
    rsi_cross_below_ma3 = bool(row.get("rsi_cross_below_ma3", False))
    rsi_cross_above_ma3 = bool(row.get("rsi_cross_above_ma3", False))

    pe_cond1 = bool(near_resistance or near_r1r2)
    ce_cond1 = bool(near_support or near_s1s2)

    pe_confluence = bool(near_resistance and near_r1r2)
    ce_confluence = bool(near_support and near_s1s2)

    pe_cond2 = bool(at_upper_band)
    pe_cond3 = bool(rsi_cross_below_ma3)
    ce_cond2 = bool(at_lower_band)
    ce_cond3 = bool(rsi_cross_above_ma3)

    pe_score = int(pe_cond1) + int(pe_cond2) + int(pe_cond3) + int(pe_confluence)
    ce_score = int(ce_cond1) + int(ce_cond2) + int(ce_cond3) + int(ce_confluence)

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

    return {
        "datetime": row.get("datetime"),
        "date": pd.Timestamp(row.get("datetime")).date().isoformat(),
        "close": row.get("close"),
        "bb_upper": row.get("upper_band"),
        "bb_lower": row.get("lower_band"),
        "at_upper_band": bool(at_upper_band),
        "at_lower_band": bool(at_lower_band),
        "rsi": row.get("rsi"),
        "rsi_ma3": row.get("rsi_ma3"),
        "rsi_cross_below_ma3": bool(rsi_cross_below_ma3),
        "rsi_cross_above_ma3": bool(rsi_cross_above_ma3),
        "near_resistance": bool(near_resistance),
        "near_support": bool(near_support),
        "near_r1r2": bool(near_r1r2),
        "near_s1s2": bool(near_s1s2),
        "pe_cond1": bool(pe_cond1),
        "pe_cond2": bool(pe_cond2),
        "pe_cond3": bool(pe_cond3),
        "pe_trade": bool(pe_trade),
        "pe_confluence": bool(pe_confluence),
        "ce_cond1": bool(ce_cond1),
        "ce_cond2": bool(ce_cond2),
        "ce_cond3": bool(ce_cond3),
        "ce_trade": bool(ce_trade),
        "ce_confluence": bool(ce_confluence),
        "action": action,
        "strength": strength,
    }


def _signal_summary(audit_df: pd.DataFrame, month_label: str) -> str:
    """Build a compact audit summary."""

    total = len(audit_df)
    if total == 0:
        return "No rows available for signal audit."

    def hit(column: str) -> int:
        return int(audit_df[column].fillna(False).astype(bool).sum())

    rows = [
        ["BB Upper", hit("at_upper_band"), f"{hit('at_upper_band') / total * 100.0:.1f}%"],
        ["BB Lower", hit("at_lower_band"), f"{hit('at_lower_band') / total * 100.0:.1f}%"],
        ["RSI↓ MA3", hit("rsi_cross_below_ma3"), f"{hit('rsi_cross_below_ma3') / total * 100.0:.1f}%"],
        ["RSI↑ MA3", hit("rsi_cross_above_ma3"), f"{hit('rsi_cross_above_ma3') / total * 100.0:.1f}%"],
        ["Near R1/R2", hit("near_r1r2"), f"{hit('near_r1r2') / total * 100.0:.1f}%"],
        ["Near S1/S2", hit("near_s1s2"), f"{hit('near_s1s2') / total * 100.0:.1f}%"],
        ["Near SR Res", hit("near_resistance"), f"{hit('near_resistance') / total * 100.0:.1f}%"],
        ["Near SR Sup", hit("near_support"), f"{hit('near_support') / total * 100.0:.1f}%"],
    ]

    pe_all3 = int((audit_df["pe_cond1"] & audit_df["pe_cond2"] & audit_df["pe_cond3"]).sum())
    ce_all3 = int((audit_df["ce_cond1"] & audit_df["ce_cond2"] & audit_df["ce_cond3"]).sum())
    rows.extend(
        [
            ["PE Cond1", hit("pe_cond1"), f"{hit('pe_cond1') / total * 100.0:.1f}%"],
            ["PE Cond2", hit("pe_cond2"), f"{hit('pe_cond2') / total * 100.0:.1f}%"],
            ["PE Cond3", hit("pe_cond3"), f"{hit('pe_cond3') / total * 100.0:.1f}%"],
            ["PE ALL 3", pe_all3, f"{pe_all3 / total * 100.0:.1f}%"],
            ["CE Cond1", hit("ce_cond1"), f"{hit('ce_cond1') / total * 100.0:.1f}%"],
            ["CE Cond2", hit("ce_cond2"), f"{hit('ce_cond2') / total * 100.0:.1f}%"],
            ["CE Cond3", hit("ce_cond3"), f"{hit('ce_cond3') / total * 100.0:.1f}%"],
            ["CE ALL 3", ce_all3, f"{ce_all3 / total * 100.0:.1f}%"],
        ]
    )

    total_signals = int((audit_df["action"] != "HOLD").sum())
    expected_trades = total_signals / max(len(pd.bdate_range(pd.Period(month_label, freq="M").start_time.normalize(), pd.Period(month_label, freq="M").end_time.normalize())), 1)

    lines = [
        f"SIGNAL CONDITION AUDIT — {month_label}",
        f"Total candles scanned    : {total:,}",
        _format_table(rows, ["Condition", "Hit Count", "Hit Rate"]),
        f"TOTAL SIGNALS            : {total_signals:,}",
        f"Expected trades/month    : {expected_trades:.1f}",
    ]

    pe_rate = pe_all3 / total * 100.0
    ce_rate = ce_all3 / total * 100.0
    if max(pe_rate, ce_rate) < 0.5:
        lines.extend(
            [
                "⚠️  SIGNAL TOO STRICT — conditions almost never align.",
                "Suggestion: relax SR_TOLERANCE from 15 to 25 pts, or allow single signals to trade without BB confirmation.",
            ]
        )
    elif max(pe_rate, ce_rate) > 8.0:
        lines.extend(
            [
                "⚠️  SIGNAL TOO LOOSE — overtrading risk.",
                "Suggestion: tighten SR_TOLERANCE or require full confluence.",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    """Run the EOD signal audit for a single month."""

    args = _parse_args()
    start, end = _month_bounds(args.month)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    fetcher = FreeDataFetcher()
    spot = fetcher.get_spot_daily(start, end)
    vix = fetcher.get_vix_daily(start, end)
    if spot.empty:
        raise SystemExit(f"No spot data returned for {args.month}")

    frame = build_backtest_frame(spot, pd.DataFrame(), vix)
    audit_rows = [_evaluate_daily_signal(row) | {"trade_date": pd.Timestamp(row["datetime"]).date().isoformat()} for _, row in frame.iterrows()]
    audit_df = pd.DataFrame(audit_rows)

    out_path = ROOT / "nifty_backtest" / "data" / "cache" / f"eod_signal_audit_{args.month.replace('-', '')}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(out_path, index=False)

    print(_signal_summary(audit_df, args.month))
    signal_days = audit_df[audit_df["action"] != "HOLD"][["trade_date", "action", "strength", "pe_cond1", "ce_cond1"]]
    if not signal_days.empty:
        print("\nDays with signals:")
        print(signal_days.head(20).to_string(index=False))
    print(f"\nAudit CSV saved to: {out_path}")


if __name__ == "__main__":
    main()
