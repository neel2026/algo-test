"""Smoke test the free-source NIFTY backtest pipeline."""

from __future__ import annotations

import sys
import time
import traceback
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
from nifty_backtest.data.fetcher import FreeDataFetcher, fetch_option_data, fetch_spot_data
from nifty_backtest.data.free_sources.source_manager import next_expiry_for_date
from nifty_backtest.data.processor import attach_vix, clean_ohlcv, merge_spot_and_options
from nifty_backtest.indicators.bollinger import add_bollinger_bands
from nifty_backtest.indicators.pivots import add_classic_pivots
from nifty_backtest.indicators.rsi import add_rsi_features
from nifty_backtest.indicators.support_resistance import add_support_resistance_features
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


def _last_business_days(count: int) -> pd.DatetimeIndex:
    """Return the last N business days ending yesterday."""

    today = pd.Timestamp(datetime.now().date()) - pd.Timedelta(days=1)
    config_end = pd.Timestamp(config.BACKTEST_END).date()
    end_date = min(today.date(), config_end)
    end_date = pd.Timestamp(end_date)
    while end_date.weekday() >= 5:
        end_date -= pd.Timedelta(days=1)
    start_date = end_date - pd.tseries.offsets.BDay(count - 1)
    return pd.bdate_range(start=start_date.normalize(), end=end_date.normalize())


def _daily_close_frame(spot_df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-day close summary from intraday spot candles."""

    working = clean_ohlcv(spot_df)
    working["trade_date"] = working["datetime"].dt.normalize()
    daily = (
        working.groupby("trade_date", as_index=False)
        .agg({"close": "last"})
        .rename(columns={"trade_date": "date"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily["atm_strike"] = (daily["close"] / config.ATM_ROUNDING).round() * config.ATM_ROUNDING
    daily["expiry"] = daily["date"].dt.date.apply(next_expiry_for_date)
    return daily


def _build_indicator_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach all requested indicators to the merged dataset."""

    working = frame.copy()
    working = add_bollinger_bands(working, close_column="close", period=config.BB_PERIOD, std_dev=config.BB_STD)
    working = add_rsi_features(
        working,
        close_column="close",
        period=config.RSI_PERIOD,
        ma_short=config.RSI_MA_SHORT,
        ma_long=config.RSI_MA_LONG,
    )
    working["rsi_cross_above_ma21"] = (working["rsi"] > working["rsi_ma21"]) & (working["rsi"].shift(1) <= working["rsi_ma21"].shift(1))
    working["rsi_cross_below_ma21"] = (working["rsi"] < working["rsi_ma21"]) & (working["rsi"].shift(1) >= working["rsi_ma21"].shift(1))
    working = add_classic_pivots(working)
    working = add_support_resistance_features(working, window=20, confluence_threshold=config.SR_CONFLUENCE_THRESHOLD)
    return working


def _signal_payload(row: pd.Series) -> dict:
    """Build the indicator payload for strategy evaluation."""

    return {
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


def _level_payload(row: pd.Series) -> dict:
    """Build the level payload for strategy evaluation."""

    levels = row.get("sr_levels")
    if not isinstance(levels, list):
        levels = []
    return {
        "pivot": row.get("pivot"),
        "r1": row.get("r1"),
        "r2": row.get("r2"),
        "s1": row.get("s1"),
        "s2": row.get("s2"),
        "resistance_levels": levels,
        "support_levels": levels,
        "vix": row.get("vix_close"),
    }


def main() -> None:
    """Execute the full 5-step data and signal pipeline smoke test."""

    start_time = time.time()
    step = 0
    try:
        step = 1
        print("Step 1")
        print("Step 1 OK Free data layer loaded")

        step = 2
        print("Step 2")
        business_days = _last_business_days(5)
        spot_start = business_days[0].date().isoformat()
        spot_end = business_days[-1].date().isoformat()
        spot_df = fetch_spot_data(
            symbol=config.SPOT_SYMBOL,
            start_date=spot_start,
            end_date=spot_end,
            interval=config.INTERVAL,
            exchange_code=config.SPOT_EXCHANGE,
        )
        if spot_df.empty:
            raise ValueError("No spot data returned.")
        spot_df = clean_ohlcv(spot_df)
        print(f"Step 2 OK Spot data: {len(spot_df)} candles, {spot_df['datetime'].min()} to {spot_df['datetime'].max()}")

        step = 3
        print("Step 3")
        daily = _daily_close_frame(spot_df)
        step3_rows: list[list[object]] = []
        for _, row in daily.iterrows():
            step3_rows.append(
                [
                    row["date"].date().isoformat(),
                    int(round(float(row["close"]))),
                    int(round(float(row["atm_strike"]))),
                    pd.Timestamp(row["expiry"]).date().isoformat(),
                ]
            )
        print(_format_table(step3_rows, ["Date", "Close", "ATM Strike", "Expiry"]))
        print("Step 3 OK ATM strikes determined")

        step = 4
        print("Step 4")
        option_frames: list[pd.DataFrame] = []
        day_table: list[list[object]] = []
        gap_dates: list[str] = []
        for _, row in daily.iterrows():
            trade_date = row["date"].date()
            expiry_date = pd.Timestamp(row["expiry"]).date()
            atm_strike = int(round(float(row["atm_strike"])))
            ce = fetch_option_data(
                symbol=config.OPTION_SYMBOL,
                start_date=trade_date.isoformat(),
                end_date=trade_date.isoformat(),
                expiry_date=expiry_date.isoformat(),
                strike_price=atm_strike,
                right="call",
                interval=config.INTERVAL,
                exchange_code=config.OPTION_EXCHANGE,
            )
            pe = fetch_option_data(
                symbol=config.OPTION_SYMBOL,
                start_date=trade_date.isoformat(),
                end_date=trade_date.isoformat(),
                expiry_date=expiry_date.isoformat(),
                strike_price=atm_strike,
                right="put",
                interval=config.INTERVAL,
                exchange_code=config.OPTION_EXCHANGE,
            )
            ce_count = len(ce)
            pe_count = len(pe)
            day_table.append([trade_date.isoformat(), ce_count, pe_count])
            if min(ce_count, pe_count) < 60:
                gap_dates.append(trade_date.isoformat())
            if not ce.empty:
                option_frames.append(ce)
            if not pe.empty:
                option_frames.append(pe)
        print(_format_table(day_table, ["Date", "CE Candles", "PE Candles"]))
        if gap_dates:
            print(f"Step 4 WARN gaps detected on {', '.join(gap_dates)}")
        print("Step 4 OK Options data fetched" if not gap_dates else f"Step 4 OK Options data fetched (or WARN gaps detected on {', '.join(gap_dates)})")

        step = 5
        print("Step 5")
        option_df = pd.concat(option_frames, ignore_index=True) if option_frames else pd.DataFrame()
        merged = merge_spot_and_options(spot_df, option_df, rounding=config.ATM_ROUNDING)
        expected_columns = [
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "ce_open",
            "ce_high",
            "ce_low",
            "ce_close",
            "pe_open",
            "pe_high",
            "pe_low",
            "pe_close",
        ]
        missing = [column for column in expected_columns if column not in merged.columns]
        if missing:
            raise ValueError(f"Missing merged columns: {missing}")
        print(f"Step 5 OK Data merged: {len(merged)} rows, {len(merged.columns)} columns")

        step = 6
        print("Step 6")
        vix_start = business_days[0].date().isoformat()
        vix_end = business_days[-1].date().isoformat()
        vix_fetcher = FreeDataFetcher()
        vix_df = vix_fetcher.get_vix_daily(vix_start, vix_end)
        vix_for_attach = vix_df.rename(columns={"vix_close": "close"}) if not vix_df.empty else vix_df
        featured = attach_vix(merged, vix_for_attach) if not vix_df.empty else merged.copy()
        featured = _build_indicator_frame(featured)
        print(featured.loc[featured["rsi_cross_above_ma3"] == True, ["datetime", "close", "rsi", "rsi_ma3"]].head(3).to_string(index=False))
        print(featured.loc[featured["rsi_cross_below_ma3"] == True, ["datetime", "close", "rsi", "rsi_ma3"]].head(3).to_string(index=False))
        print("Step 6 OK Indicators computed")

        step = 7
        print("Step 7")
        strategy = ATMStrategy()
        signals: list[dict[str, object]] = []
        for _, row in featured.iterrows():
            signal = strategy.generate_signals(
                candle=row,
                indicators=_signal_payload(row),
                levels=_level_payload(row),
            )
            if signal.get("action") != "HOLD":
                signals.append(
                    {
                        "datetime": row["datetime"],
                        "date": pd.Timestamp(row["datetime"]).date().isoformat(),
                        "action": signal.get("action"),
                        "strength": signal.get("strength"),
                        "signal_log": signal.get("signal_log"),
                    }
                )
        signal_df = pd.DataFrame(signals)
        total_by_day = featured.groupby(featured["datetime"].dt.date).size()
        ce_by_day = signal_df[signal_df["action"] == "BUY_CE"].groupby("date").size() if not signal_df.empty else pd.Series(dtype=int)
        pe_by_day = signal_df[signal_df["action"] == "BUY_PE"].groupby("date").size() if not signal_df.empty else pd.Series(dtype=int)
        full_by_day = signal_df[signal_df["strength"] == "full"].groupby("date").size() if not signal_df.empty else pd.Series(dtype=int)
        single_by_day = signal_df[signal_df["strength"] == "single"].groupby("date").size() if not signal_df.empty else pd.Series(dtype=int)
        summary_rows: list[list[object]] = []
        for day, total in total_by_day.items():
            day_key = day.isoformat()
            summary_rows.append(
                [
                    day_key,
                    int(total),
                    int(ce_by_day.get(day_key, 0)),
                    int(pe_by_day.get(day_key, 0)),
                    int(full_by_day.get(day_key, 0)),
                    int(single_by_day.get(day_key, 0)),
                ]
            )
        summary = pd.DataFrame(summary_rows, columns=["Date", "Total Candles", "BUY_CE", "BUY_PE", "Full Strength", "Single Only"])
        print(_format_table(summary.values.tolist(), list(summary.columns)))
        print("Step 7 OK Signal scan complete")

        step = 8
        print("Step 8")
        output = featured.copy()
        if not signal_df.empty:
            signal_flags = signal_df.groupby("datetime")["action"].apply(lambda values: ",".join(sorted(set(values)))).reset_index()
            output = output.merge(signal_flags, on="datetime", how="left")
            output = output.rename(columns={"action": "signal_actions"})
        cache_dir = config.resolve_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_path = cache_dir / f"test_run_{pd.Timestamp.now().strftime('%Y%m%d')}.csv"
        output.to_csv(out_path, index=False)
        print(f"Step 8 OK Saved to {out_path.as_posix()}")

        elapsed = time.time() - start_time
        print(
            """
============================================
OK PIPELINE HEALTHY - READY FOR FULL BACKTEST
============================================
Next step:
  python main.py --start 2024-01-01 --end 2024-12-31
============================================
"""
        )
        print(f"Elapsed time: {elapsed:.2f} seconds")
    except Exception as exc:
        elapsed = time.time() - start_time
        print(
            f"""
============================================
ERROR PIPELINE FAILED AT STEP {step}
Error: {exc}
Fix the issue above and re-run: python scripts/test_fetch.py
============================================
"""
        )
        print(traceback.format_exc())
        print(f"Elapsed time: {elapsed:.2f} seconds")
        sys.exit(1)


if __name__ == "__main__":
    main()
