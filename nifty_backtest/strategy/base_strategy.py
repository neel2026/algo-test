"""Abstract strategy interface for the backtesting engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    @abstractmethod
    def generate_signals(self, candle, indicators, levels) -> dict:
        """Return the strategy decision for a single candle."""

        raise NotImplementedError

