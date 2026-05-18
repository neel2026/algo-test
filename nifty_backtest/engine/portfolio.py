"""Portfolio state tracking for backtests."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from nifty_backtest.config import CAPITAL
from nifty_backtest.engine.trade import Trade

logger = logging.getLogger(__name__)


@dataclass
class Portfolio:
    """Track realized and unrealized PnL, equity, and drawdown."""

    starting_capital: float = CAPITAL
    open_trade: Trade | None = None
    closed_trades: list[Trade] = field(default_factory=list)
    realized_pnl: float = 0.0
    peak_equity: float = CAPITAL
    max_drawdown: float = 0.0
    consecutive_losses: int = 0
    current_equity: float = CAPITAL
    equity_curve: list[dict] = field(default_factory=list)
    trade_log: list[dict] = field(default_factory=list)
    partial_exit_log: list[dict] = field(default_factory=list)

    def can_open_trade(self) -> bool:
        """Return whether the portfolio currently allows a new trade."""

        return self.open_trade is None

    def open(self, trade: Trade) -> None:
        """Register a newly opened trade."""

        self.open_trade = trade

    def close(self, trade: Trade) -> None:
        """Finalize a trade and update realized PnL state."""

        if trade.exit_reason == "open":
            self.on_full_exit(
                trade,
                trade.option_exit_price or trade.option_entry_price,
                "eod_exit",
                trade.full_exit_time or trade.exit_time or trade.entry_time,
            )

    def on_partial_exit(self, trade: Trade, exit_price: float, candle_time) -> float:
        """Book a 50 percent exit, keep the trade open, and move SL to entry."""

        partial_pnl = trade.record_partial_exit(candle_time, exit_price)
        self.realized_pnl += partial_pnl
        self.partial_exit_log.append(
            {
                "trade_id": trade.trade_id,
                "time": candle_time,
                "exit_price": float(exit_price),
                "partial_pnl": partial_pnl,
                "qty_remaining": trade.qty_remaining,
            }
        )
        logger.info(
            "Partial exit trade_id=%s reason=target1 pnl=%.2f qty_remaining=%s",
            trade.trade_id,
            partial_pnl,
            trade.qty_remaining,
        )
        return partial_pnl

    def on_full_exit(self, trade: Trade, exit_price: float, exit_reason: str, candle_time) -> float:
        """Close the remaining quantity, archive the trade, and clear the open slot."""

        total_pnl = trade.record_full_exit(candle_time, exit_price, exit_reason)
        final_pnl = trade.final_pnl
        self.realized_pnl += final_pnl
        self.closed_trades.append(trade)
        self.trade_log.append(trade.to_record())
        if total_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        self.open_trade = None
        logger.info(
            "Full exit trade_id=%s reason=%s total_pnl=%.2f final_pnl=%.2f",
            trade.trade_id,
            exit_reason,
            total_pnl,
            final_pnl,
        )
        return total_pnl

    def update_equity(self, timestamp, mark_to_market: float | None = None) -> None:
        """Update the portfolio equity curve."""

        unrealized = 0.0
        if self.open_trade is not None and mark_to_market is not None:
            unrealized = (float(mark_to_market) - float(self.open_trade.option_entry_price)) * float(self.open_trade.qty_remaining)
        equity = self.starting_capital + self.realized_pnl + unrealized
        self.current_equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - equity) / self.peak_equity * 100.0
            self.max_drawdown = max(self.max_drawdown, drawdown)
        self.equity_curve.append({"datetime": timestamp, "equity": equity, "drawdown_pct": self.max_drawdown})

    def closed_trades_frame(self) -> pd.DataFrame:
        """Return closed trades as a dataframe."""

        if not self.trade_log:
            return pd.DataFrame()
        return pd.DataFrame(self.trade_log)

    def equity_frame(self) -> pd.DataFrame:
        """Return the recorded equity curve as a dataframe."""

        if not self.equity_curve:
            return pd.DataFrame()
        return pd.DataFrame(self.equity_curve)
