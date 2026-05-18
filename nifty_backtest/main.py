"""Entry point for the NIFTY backtesting framework."""

from __future__ import annotations

import argparse
import importlib
import logging
from pathlib import Path

import pandas as pd

from nifty_backtest import config
from nifty_backtest.data.fetcher import fetch_atm_option_series, fetch_india_vix, fetch_spot_data
from nifty_backtest.data.processor import build_backtest_frame
from nifty_backtest.engine.backtester import Backtester
from nifty_backtest.reports.charts import drawdown_figure, equity_curve_figure, save_figure_html, trade_distribution_figure
from nifty_backtest.reports.stats import compute_statistics, monthly_pnl_breakdown, signal_hit_frequency


def configure_logging() -> None:
    """Configure the root logger."""

    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_strategy(strategy_path: str):
    """Load a concrete strategy class from a module path."""

    if not strategy_path:
        raise ValueError("A concrete strategy class path is required, e.g. module.submodule:StrategyClass")

    module_path, class_name = strategy_path.split(":", 1)
    module = importlib.import_module(module_path)
    strategy_class = getattr(module, class_name)
    return strategy_class()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run the NIFTY ATM options backtest.")
    parser.add_argument(
        "--strategy",
        default=config.STRATEGY_CLASS_PATH,
        help="Import path in the form module.path:ClassName",
    )
    return parser.parse_args()


def main() -> None:
    """Run the end-to-end backtest workflow."""

    configure_logging()
    logger = logging.getLogger(__name__)
    args = parse_args()

    strategy = load_strategy(args.strategy)
    logger.info("Fetching spot data for %s to %s", config.BACKTEST_START, config.BACKTEST_END)
    spot_df = fetch_spot_data(
        symbol=config.SPOT_SYMBOL,
        start_date=config.BACKTEST_START,
        end_date=config.BACKTEST_END,
        interval=config.INTERVAL,
        exchange_code=config.SPOT_EXCHANGE,
    )

    logger.info("Fetching ATM CE data")
    ce_df = fetch_atm_option_series(spot_df, right="CE", symbol=config.OPTION_SYMBOL, interval=config.INTERVAL, exchange_code=config.OPTION_EXCHANGE)
    logger.info("Fetching ATM PE data")
    pe_df = fetch_atm_option_series(spot_df, right="PE", symbol=config.OPTION_SYMBOL, interval=config.INTERVAL, exchange_code=config.OPTION_EXCHANGE)
    option_df = pd.concat([ce_df, pe_df], ignore_index=True) if not ce_df.empty or not pe_df.empty else pd.DataFrame()

    logger.info("Fetching India VIX data")
    vix_df = fetch_india_vix(config.BACKTEST_START, config.BACKTEST_END, interval="1day", symbol=config.VIX_SYMBOL, exchange_code=config.VIX_EXCHANGE)

    logger.info("Building enriched backtest frame")
    backtest_df = build_backtest_frame(spot_df, option_df, vix_df)

    logger.info("Running backtest")
    backtester = Backtester(strategy=strategy, data=backtest_df, starting_capital=config.CAPITAL)
    result = backtester.run()

    trades_df = result.trades
    signals_df = result.signals
    equity_df = result.equity_curve

    report_dir = config.resolve_report_dir()
    report_dir.mkdir(parents=True, exist_ok=True)

    if not trades_df.empty:
        trades_df.to_csv(report_dir / "trade_log.csv", index=False)
    if not signals_df.empty:
        signals_df.to_csv(report_dir / "signal_log.csv", index=False)
    if not equity_df.empty:
        equity_df.to_csv(report_dir / "equity_curve.csv", index=False)

    stats = compute_statistics(trades_df, equity_curve=equity_df, starting_capital=config.CAPITAL)
    monthly = monthly_pnl_breakdown(trades_df)
    signal_freq = signal_hit_frequency(signals_df)

    if not monthly.empty:
        monthly.to_csv(report_dir / "monthly_pnl.csv", index=False)
    if not signal_freq.empty:
        signal_freq.to_csv(report_dir / "signal_frequency.csv", index=False)

    equity_fig = equity_curve_figure(equity_df)
    drawdown_fig = drawdown_figure(equity_df)
    distribution_fig = trade_distribution_figure(trades_df)
    save_figure_html(equity_fig, report_dir / "equity_curve.html")
    save_figure_html(drawdown_fig, report_dir / "drawdown.html")
    save_figure_html(distribution_fig, report_dir / "trade_distribution.html")

    logger.info("Backtest complete")
    logger.info("Total trades: %s", stats["total_trades"])
    logger.info("Win rate: %.2f%%", stats["win_rate_pct"])
    logger.info("Loss rate: %.2f%%", stats["loss_rate_pct"])
    logger.info("Average win: %.2f%%", stats["avg_win_pct"])
    logger.info("Average loss: %.2f%%", stats["avg_loss_pct"])
    logger.info("Profit factor: %.2f", stats["profit_factor"])
    logger.info("Max drawdown: %.2f%%", stats["max_drawdown_pct"])
    logger.info("Sharpe ratio: %.2f", stats["sharpe_ratio"])


if __name__ == "__main__":
    main()

