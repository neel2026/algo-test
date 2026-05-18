"""Trade representation used by the backtesting engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from nifty_backtest import config


@dataclass
class Trade:
    """A single options trade lifecycle."""

    trade_id: int
    direction: str
    entry_time: datetime
    entry_spot: float
    entry_strike: float
    option_entry_price: float
    qty_total: int
    qty_multiplier: float
    target1: float | None = None
    target2: float | None = None
    stoploss_spot: float | None = None
    conservative_sl_premium: float = 0.0
    aggressive_sl_premium: float = 0.0
    premium_sl_price: float = 0.0
    signal_strength: str | None = None
    vix_at_entry: float | None = None
    rsi_at_entry: float | None = None
    bb_position_at_entry: float | None = None
    is_expiry_day: bool = False
    is_thursday_trade: bool = False
    entry_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    partial_exit_done: bool = False
    sl_moved_to_entry: bool = False
    qty_remaining: int = 0
    partial_exit_time: datetime | None = None
    partial_exit_price: float | None = None
    partial_pnl: float = 0.0
    full_exit_time: datetime | None = None
    full_exit_price: float | None = None
    final_pnl: float = 0.0
    total_pnl: float = 0.0
    exit_time: datetime | None = None
    option_exit_price: float | None = None
    exit_reason: str = "open"

    def __post_init__(self) -> None:
        """Initialize derived trade fields after dataclass construction."""

        if self.qty_remaining <= 0:
            self.qty_remaining = int(self.qty_total)
        self.conservative_sl_premium = (
            float(self.option_entry_price) * config.PREMIUM_SL_CONSERVATIVE_MULTIPLIER
            if not self.conservative_sl_premium
            else float(self.conservative_sl_premium)
        )
        self.aggressive_sl_premium = (
            float(self.option_entry_price) * config.PREMIUM_SL_AGGRESSIVE_MULTIPLIER
            if not self.aggressive_sl_premium
            else float(self.aggressive_sl_premium)
        )
        if not self.premium_sl_price:
            self.premium_sl_price = self.conservative_sl_premium
        self.is_expiry_day = bool(self.is_expiry_day or self.is_thursday_trade)
        self.is_thursday_trade = self.is_expiry_day
        self.stoploss_spot = float(self.stoploss_spot) if self.stoploss_spot is not None else None

    def record_partial_exit(self, exit_time: datetime, exit_price: float) -> float:
        """Record a 50 percent partial exit and return the booked PnL."""

        if self.partial_exit_done:
            return 0.0

        partial_qty = self.qty_total // 2
        if partial_qty <= 0:
            partial_qty = self.qty_total

        self.partial_exit_time = exit_time
        self.partial_exit_price = float(exit_price)
        self.partial_exit_done = True
        self.sl_moved_to_entry = True
        self.stoploss_spot = float(self.entry_spot)
        self.qty_remaining = max(0, int(self.qty_remaining) - int(partial_qty))
        self.partial_pnl = (float(exit_price) - float(self.option_entry_price)) * partial_qty
        self.total_pnl = self.partial_pnl + self.final_pnl
        return self.partial_pnl

    def record_full_exit(self, exit_time: datetime, option_exit_price: float, exit_reason: str) -> float:
        """Record the final exit and return the total trade PnL."""

        self.full_exit_time = exit_time
        self.full_exit_price = float(option_exit_price)
        self.exit_time = exit_time
        self.option_exit_price = float(option_exit_price)
        self.exit_reason = exit_reason
        final_qty = int(self.qty_remaining)
        self.final_pnl = (float(option_exit_price) - float(self.option_entry_price)) * final_qty
        self.total_pnl = self.partial_pnl + self.final_pnl
        self.qty_remaining = 0
        return self.total_pnl

    def close(self, exit_time: datetime, option_exit_price: float, exit_reason: str) -> None:
        """Backward-compatible alias for recording a full exit."""

        self.record_full_exit(exit_time, option_exit_price, exit_reason)

    @property
    def pnl_points(self) -> float:
        """Return the per-unit premium PnL."""

        if self.total_pnl:
            return float(self.total_pnl) / float(self.qty_total or 1)
        if self.option_exit_price is None:
            return 0.0
        return float(self.option_exit_price) - float(self.option_entry_price)

    @property
    def pnl_pct(self) -> float:
        """Return the premium return percentage."""

        if not self.option_entry_price:
            return 0.0
        return (self.pnl_points / float(self.option_entry_price)) * 100.0

    @property
    def pnl_amount(self) -> float:
        """Return the trade PnL in rupees for one lot."""

        if self.total_pnl:
            return float(self.total_pnl)
        return self.pnl_points * float(self.qty_total)

    def to_record(self) -> dict[str, Any]:
        """Return a serializable record for reporting and CSV export."""

        record = {
            "trade_id": self.trade_id,
            "date": self.entry_time.date().isoformat(),
            "direction": self.direction,
            "atm_strike": self.entry_strike,
            "entry_time": self.entry_time,
            "entry_spot": self.entry_spot,
            "entry_option_price": self.option_entry_price,
            "partial_exit_time": self.partial_exit_time,
            "partial_exit_price": self.partial_exit_price,
            "partial_pnl": self.partial_pnl,
            "full_exit_time": self.full_exit_time,
            "full_exit_price": self.full_exit_price,
            "final_pnl": self.final_pnl,
            "total_pnl": self.total_pnl,
            "exit_reason": self.exit_reason,
            "strength": self.signal_strength,
            "qty_total": self.qty_total,
            "qty_multiplier": self.qty_multiplier,
            "vix_at_entry": self.vix_at_entry,
            "rsi_at_entry": self.rsi_at_entry,
            "bb_position_at_entry": self.bb_position_at_entry,
            "target1_spot": self.target1,
            "target2_spot": self.target2,
            "stoploss_spot": self.stoploss_spot,
            "premium_sl_price": self.premium_sl_price,
            "sl_moved_to_entry": self.sl_moved_to_entry,
            "is_expiry_day": self.is_expiry_day,
            "pnl_points": self.pnl_points,
            "pnl_pct": self.pnl_pct,
            "pnl_amount": self.pnl_amount,
        }
        return record
