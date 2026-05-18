"""Backtest execution loop."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import time
from typing import Any

import pandas as pd

from nifty_backtest import config
from nifty_backtest.engine.portfolio import Portfolio
from nifty_backtest.engine.trade import Trade

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for the backtest outputs."""

    trades: pd.DataFrame
    signals: pd.DataFrame
    equity_curve: pd.DataFrame
    portfolio: Portfolio


class Backtester:
    """Iterate candles, invoke a strategy, and manage trade execution."""

    def __init__(self, strategy: Any, data: pd.DataFrame, starting_capital: float | None = None) -> None:
        """Initialize the backtester."""

        self.strategy = strategy
        self.data = data.copy().sort_values("datetime").reset_index(drop=True)
        self.portfolio = Portfolio(starting_capital=starting_capital or config.CAPITAL)
        self.trade_counter = 0
        self.cooldown_remaining = 0
        self.signal_log: list[dict] = []
        self.entry_start_time = time.fromisoformat(config.ENTRY_START_TIME)
        self.entry_cutoff_time = time.fromisoformat(config.ENTRY_CUTOFF_TIME)
        self.eod_exit_time = time.fromisoformat(config.EOD_EXIT_TIME)

    def _is_entry_window(self, candle_time: time) -> bool:
        """Return whether a candle is inside the entry window."""

        return self.entry_start_time <= candle_time <= self.entry_cutoff_time

    def _select_option_price(self, row: pd.Series, direction: str, field: str) -> float | None:
        """Read the requested CE or PE price field from the candle."""

        prefix = "ce_" if direction.upper() == "CE" else "pe_"
        column = f"{prefix}{field}"
        if column not in row or pd.isna(row[column]):
            return None
        return float(row[column])

    def _current_option_mark(self, row: pd.Series, direction: str) -> float | None:
        """Return the current close price for the active option leg."""

        return self._select_option_price(row, direction, "close")

    def _entry_price(self, row: pd.Series, direction: str) -> float | None:
        """Return the entry price for the selected option leg."""

        return self._select_option_price(row, direction, "close")

    def _check_exit_conditions(
        self,
        trade: Trade,
        candle: pd.Series,
        option_ltp: float | None,
    ) -> tuple[bool, bool, str | None]:
        """Evaluate exit conditions in priority order and return the first match."""

        spot_close = float(candle["close"])
        candle_time = candle["datetime"].time()

        if trade.direction.upper() == "PE":
            stoploss_hit = trade.stoploss_spot is not None and spot_close > float(trade.stoploss_spot)
            target1_hit = trade.target1 is not None and spot_close <= float(trade.target1)
            target2_hit = trade.target2 is not None and spot_close <= float(trade.target2)
        else:
            stoploss_hit = trade.stoploss_spot is not None and spot_close < float(trade.stoploss_spot)
            target1_hit = trade.target1 is not None and spot_close >= float(trade.target1)
            target2_hit = trade.target2 is not None and spot_close >= float(trade.target2)

        premium_sl_hit = (
            option_ltp is not None
            and trade.premium_sl_price not in (None, 0)
            and float(option_ltp) < float(trade.premium_sl_price)
        )
        eod_hit = candle_time >= self.eod_exit_time

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

    def _open_trade_from_signal(self, row: pd.Series, signal: dict[str, Any]) -> None:
        """Open a trade using the strategy-provided signal payload."""

        action = str(signal.get("action", "HOLD")).upper()
        if action not in {"BUY_CE", "BUY_PE"}:
            return
        if not self.portfolio.can_open_trade():
            return
        if self.cooldown_remaining > 0:
            return
        if not self._is_entry_window(row["datetime"].time()):
            return

        direction = "CE" if action == "BUY_CE" else "PE"
        entry_price = self._entry_price(row, direction)
        if entry_price is None:
            return

        self.trade_counter += 1
        trade = Trade(
            trade_id=self.trade_counter,
            direction=direction,
            entry_time=row["datetime"].to_pydatetime(),
            entry_spot=float(row["close"]),
            entry_strike=float(row.get("atm_strike", row.get("strike", row["close"]))),
            option_entry_price=float(entry_price),
            qty_total=max(1, int(round(config.LOT_SIZE * float(signal.get("qty_multiplier", 1.0) or 1.0)))),
            qty_multiplier=float(signal.get("qty_multiplier", 1.0) or 1.0),
            target1=signal.get("target1"),
            target2=signal.get("target2"),
            stoploss_spot=signal.get("stoploss_spot"),
            conservative_sl_premium=(
                float(entry_price) * config.PREMIUM_SL_CONSERVATIVE_MULTIPLIER
            ),
            aggressive_sl_premium=(
                float(entry_price) * config.PREMIUM_SL_AGGRESSIVE_MULTIPLIER
            ),
            premium_sl_price=float(signal.get("signal_log", {}).get("premium_sl_price") or (float(entry_price) * config.PREMIUM_SL_CONSERVATIVE_MULTIPLIER)),
            signal_strength=signal.get("strength"),
            vix_at_entry=float(row["vix_close"]) if "vix_close" in row and pd.notna(row["vix_close"]) else None,
            rsi_at_entry=float(row["rsi"]) if "rsi" in row and pd.notna(row["rsi"]) else None,
            bb_position_at_entry=float(row["percent_b"]) if "percent_b" in row and pd.notna(row["percent_b"]) else None,
            is_expiry_day=row["datetime"].day_name() == "Thursday",
            is_thursday_trade=row["datetime"].day_name() == "Thursday",
            entry_reason=action,
            metadata={
                "pivot": row.get("pivot"),
                "r1": row.get("r1"),
                "r2": row.get("r2"),
                "s1": row.get("s1"),
                "s2": row.get("s2"),
                "sr_confluence": bool(row.get("sr_confluence", False)),
                "signal_log": signal.get("signal_log", {}),
            },
        )
        self.portfolio.open(trade)
        logger.info("Opened trade %s: %s at %.2f", trade.trade_id, trade.direction, trade.option_entry_price)

    def _close_trade(self, row: pd.Series, exit_price: float, exit_reason: str) -> None:
        """Close the active trade and start cooldown."""

        if self.portfolio.open_trade is None:
            return
        trade = self.portfolio.open_trade
        trade.close(row["datetime"].to_pydatetime(), exit_price, exit_reason)
        self.portfolio.close(trade)
        self.cooldown_remaining = config.COOLDOWN_CANDLES
        logger.info(
            "Closed trade %s at %.2f with reason=%s pnl=%.2f",
            trade.trade_id,
            exit_price,
            exit_reason,
            trade.pnl_amount,
        )

    def run(self) -> BacktestResult:
        """Run the backtest over the loaded dataframe."""

        if self.data.empty:
            return BacktestResult(
                trades=pd.DataFrame(),
                signals=pd.DataFrame(),
                equity_curve=pd.DataFrame(),
                portfolio=self.portfolio,
            )

        for _, row in self.data.iterrows():
            signal = self.strategy.generate_signals(
                candle=row.to_dict(),
                indicators=row.to_dict(),
                levels={"sr_levels": row.get("sr_levels"), "pivot": row.get("pivot"), "r1": row.get("r1"), "r2": row.get("r2"), "s1": row.get("s1"), "s2": row.get("s2")},
            )
            signal = signal or {}
            action = str(signal.get("action", "HOLD")).upper()
            if action != "HOLD" or signal.get("strength") is not None:
                self.signal_log.append(
                    {
                        "datetime": row["datetime"],
                        "action": action,
                        "strength": signal.get("strength"),
                        "target1": signal.get("target1"),
                        "target2": signal.get("target2"),
                        "stoploss": signal.get("stoploss_spot"),
                        "qty_multiplier": signal.get("qty_multiplier"),
                        "signal_log": signal.get("signal_log"),
                    }
                )

            if self.cooldown_remaining > 0:
                self.cooldown_remaining -= 1

            if self.portfolio.open_trade is not None:
                trade = self.portfolio.open_trade
                option_ltp = self._current_option_mark(row, trade.direction)
                should_exit_full, should_partial_exit, exit_reason = self._check_exit_conditions(trade, row, option_ltp)
                if should_partial_exit and exit_reason == "target1" and option_ltp is not None:
                    self.portfolio.on_partial_exit(trade, option_ltp, row["datetime"].to_pydatetime())
                    self.portfolio.update_equity(row["datetime"], mark_to_market=option_ltp)
                elif should_exit_full and exit_reason is not None:
                    full_exit_price = option_ltp if option_ltp is not None else trade.option_entry_price
                    self.portfolio.on_full_exit(trade, full_exit_price, exit_reason, row["datetime"].to_pydatetime())
                    self.portfolio.update_equity(row["datetime"], mark_to_market=None)
                else:
                    self.portfolio.update_equity(row["datetime"], mark_to_market=option_ltp)
                continue

            if row["datetime"].time() > self.eod_exit_time:
                self.portfolio.update_equity(row["datetime"], mark_to_market=None)
                continue

            if action in {"BUY_CE", "BUY_PE"}:
                self._open_trade_from_signal(row, signal)

            mark_to_market = None
            if self.portfolio.open_trade is not None:
                mark_to_market = self._current_option_mark(row, self.portfolio.open_trade.direction)
            self.portfolio.update_equity(row["datetime"], mark_to_market=mark_to_market)

        if self.portfolio.open_trade is not None:
            last_row = self.data.iloc[-1]
            mark = self._current_option_mark(last_row, self.portfolio.open_trade.direction)
            if mark is None:
                mark = self.portfolio.open_trade.option_entry_price
            self.portfolio.on_full_exit(self.portfolio.open_trade, mark, "eod_exit", last_row["datetime"].to_pydatetime())
            self.portfolio.update_equity(last_row["datetime"], mark_to_market=None)

        trades_df = self.portfolio.closed_trades_frame()
        signals_df = pd.DataFrame(self.signal_log)
        equity_df = self.portfolio.equity_frame()
        return BacktestResult(trades=trades_df, signals=signals_df, equity_curve=equity_df, portfolio=self.portfolio)
