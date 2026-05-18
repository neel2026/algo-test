"""Concrete ATM options strategy implementation for NIFTY backtesting."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from nifty_backtest import config
from nifty_backtest.strategy.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class ATMStrategy(BaseStrategy):
    """Pure signal-generation strategy for ATM NIFTY CE/PE entries."""

    def __init__(self) -> None:
        """Initialize the strategy with config-driven parameters."""

        self.atm_rounding = config.ATM_ROUNDING
        self.entry_start_time = config.ENTRY_START_TIME
        self.entry_cutoff_time = config.ENTRY_CUTOFF_TIME
        self.sr_tolerance = config.SR_TOLERANCE
        self.target1_spot_move = config.TARGET1_SPOT_MOVE
        self.target2_spot_move = config.TARGET2_SPOT_MOVE
        self.stoploss_spot_move = config.STOPLOSS_SPOT_MOVE
        self.premium_sl_type = config.PREMIUM_SL_TYPE
        self.vix_low_threshold = config.VIX_LOW_THRESHOLD
        self.vix_high_threshold = config.VIX_HIGH_THRESHOLD
        self.vix_normal_qty_multiplier = config.VIX_NORMAL_QTY_MULTIPLIER
        self.vix_reduced_qty_multiplier = config.VIX_REDUCED_QTY_MULTIPLIER
        self.vix_half_qty_multiplier = config.VIX_HALF_QTY_MULTIPLIER
        self.premium_sl_conservative_multiplier = config.PREMIUM_SL_CONSERVATIVE_MULTIPLIER
        self.premium_sl_aggressive_multiplier = config.PREMIUM_SL_AGGRESSIVE_MULTIPLIER

    def __repr__(self) -> str:
        """Return a compact representation of the strategy configuration."""

        return (
            "ATMStrategy("
            f"atm_rounding={self.atm_rounding}, "
            f"entry_window={self.entry_start_time}-{self.entry_cutoff_time}, "
            f"sr_tolerance={self.sr_tolerance}, "
            f"target1_spot_move={self.target1_spot_move}, "
            f"target2_spot_move={self.target2_spot_move}, "
            f"stoploss_spot_move={self.stoploss_spot_move}, "
            f"premium_sl_type='{self.premium_sl_type}'"
            ")"
        )

    @staticmethod
    def _as_float(value: Any, default: float | None = None) -> float | None:
        """Safely coerce a value to float."""

        try:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return default
            result = float(value)
            if pd.isna(result):
                return default
            return result
        except (TypeError, ValueError):
            return default

    def _parse_time(self, value: str) -> pd.Timestamp:
        """Parse a time string into a pandas time-like timestamp."""

        return pd.Timestamp(value)

    def _is_entry_allowed(self, candle: pd.Series) -> bool:
        """Check whether the candle is inside the strategy entry window."""

        candle_time = pd.Timestamp(candle["datetime"]).time()
        start_time = pd.Timestamp(self.entry_start_time).time()
        cutoff_time = pd.Timestamp(self.entry_cutoff_time).time()
        allowed = start_time <= candle_time <= cutoff_time
        logger.debug("Entry window check time=%s allowed=%s", candle_time, allowed)
        return allowed

    def _round_atm_strike(self, spot_close: float) -> int:
        """Round the NIFTY spot close to the nearest ATM strike."""

        return int(round(spot_close / self.atm_rounding) * self.atm_rounding)

    def _resolve_vix(self, candle: pd.Series, levels: dict) -> float | None:
        """Resolve VIX from either levels or the candle payload."""

        candidates = [
            levels.get("vix"),
            candle.get("vix"),
            candle.get("vix_close"),
            candle.get("india_vix"),
        ]
        for candidate in candidates:
            value = self._as_float(candidate)
            if value is not None:
                return value
        return None

    def _resolve_qty_multiplier(self, vix: float | None) -> float:
        """Resolve the quantity multiplier from the VIX filter."""

        if vix is None:
            logger.debug("VIX unavailable; using normal quantity multiplier=%s", self.vix_normal_qty_multiplier)
            return self.vix_normal_qty_multiplier
        if vix > self.vix_high_threshold:
            logger.debug("VIX=%s above high threshold=%s; multiplier=%s", vix, self.vix_high_threshold, self.vix_half_qty_multiplier)
            return self.vix_half_qty_multiplier
        if vix <= self.vix_low_threshold:
            logger.debug("VIX=%s at/below low threshold=%s; multiplier=%s", vix, self.vix_low_threshold, self.vix_normal_qty_multiplier)
            return self.vix_normal_qty_multiplier
        logger.debug(
            "VIX=%s between thresholds %s and %s; multiplier=%s",
            vix,
            self.vix_low_threshold,
            self.vix_high_threshold,
            self.vix_reduced_qty_multiplier,
        )
        return self.vix_reduced_qty_multiplier

    def _premium_sl_price(self, option_entry_price: float | None) -> float | None:
        """Calculate the premium stop-loss price from the configured mode."""

        if option_entry_price is None:
            return None
        multiplier = (
            self.premium_sl_conservative_multiplier
            if self.premium_sl_type == "conservative"
            else self.premium_sl_aggressive_multiplier
        )
        premium_sl = option_entry_price * multiplier
        logger.debug(
            "Premium SL calculated entry=%s type=%s multiplier=%s sl=%s",
            option_entry_price,
            self.premium_sl_type,
            multiplier,
            premium_sl,
        )
        return premium_sl

    def _extract_levels(self, levels: dict) -> dict[str, list[float]]:
        """Extract resistance, support, and pivot levels from the payload."""

        resistance_levels = levels.get("resistance_levels") or levels.get("sr_levels") or []
        support_levels = levels.get("support_levels") or levels.get("sr_levels") or []
        pivot_resistance_levels = [levels.get(key) for key in ("r1", "r2")]
        pivot_support_levels = [levels.get(key) for key in ("s1", "s2")]
        return {
            "resistance_levels": [level for level in (self._as_float(value) for value in resistance_levels) if level is not None],
            "support_levels": [level for level in (self._as_float(value) for value in support_levels) if level is not None],
            "pivot_resistance_levels": [level for level in (self._as_float(value) for value in pivot_resistance_levels) if level is not None],
            "pivot_support_levels": [level for level in (self._as_float(value) for value in pivot_support_levels) if level is not None],
        }

    def _build_payloads_from_row(self, candle: pd.Series) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build indicator and level payloads from a dataframe row."""

        spot_close = self._as_float(candle.get("close"))
        sr_levels = candle.get("sr_levels") if isinstance(candle.get("sr_levels"), list) else []
        resistance_levels = [level for level in (self._as_float(value) for value in sr_levels) if level is not None and (spot_close is None or level >= spot_close)]
        support_levels = [level for level in (self._as_float(value) for value in sr_levels) if level is not None and (spot_close is None or level <= spot_close)]
        indicators = {
            "bb_upper": candle.get("upper_band"),
            "bb_lower": candle.get("lower_band"),
            "bb_middle": candle.get("middle_band"),
            "rsi": candle.get("rsi"),
            "rsi_ma3": candle.get("rsi_ma3"),
            "rsi_ma21": candle.get("rsi_ma21"),
            "rsi_cross_above_ma3": bool(candle.get("rsi_cross_above_ma3", False)),
            "rsi_cross_below_ma3": bool(candle.get("rsi_cross_below_ma3", False)),
            "rsi_cross_above_ma21": bool(candle.get("rsi_cross_above_ma21", False)),
            "rsi_cross_below_ma21": bool(candle.get("rsi_cross_below_ma21", False)),
        }
        levels = {
            "pivot": candle.get("pivot"),
            "r1": candle.get("r1"),
            "r2": candle.get("r2"),
            "s1": candle.get("s1"),
            "s2": candle.get("s2"),
            "resistance_levels": resistance_levels,
            "support_levels": support_levels,
            "vix": candle.get("vix_close"),
        }
        return indicators, levels

    def _match_resistance_conditions(self, spot_close: float, levels_payload: dict[str, list[float]]) -> dict[str, Any]:
        """Evaluate bearish price-level and pivot conditions."""

        resistance_levels = levels_payload["resistance_levels"]
        pivot_resistance_levels = levels_payload["pivot_resistance_levels"]
        price_level_hit = any(spot_close >= (level - self.sr_tolerance) for level in resistance_levels)
        pivot_hit = any(spot_close >= (level - self.sr_tolerance) for level in pivot_resistance_levels)
        confluence = price_level_hit and pivot_hit
        price_hits = [level for level in resistance_levels if spot_close >= (level - self.sr_tolerance)]
        pivot_hits = [level for level in pivot_resistance_levels if spot_close >= (level - self.sr_tolerance)]

        logger.debug(
            "Bearish level check spot=%s price_level_hit=%s pivot_hit=%s price_hits=%s pivot_hits=%s",
            spot_close,
            price_level_hit,
            pivot_hit,
            price_hits,
            pivot_hits,
        )

        return {
            "price_level_hit": price_level_hit,
            "pivot_hit": pivot_hit,
            "confluence": confluence,
            "price_hits": price_hits,
            "pivot_hits": pivot_hits,
        }

    def _match_support_conditions(self, spot_close: float, levels_payload: dict[str, list[float]]) -> dict[str, Any]:
        """Evaluate bullish price-level and pivot conditions."""

        support_levels = levels_payload["support_levels"]
        pivot_support_levels = levels_payload["pivot_support_levels"]
        price_level_hit = any(spot_close <= (level + self.sr_tolerance) for level in support_levels)
        pivot_hit = any(spot_close <= (level + self.sr_tolerance) for level in pivot_support_levels)
        confluence = price_level_hit and pivot_hit
        price_hits = [level for level in support_levels if spot_close <= (level + self.sr_tolerance)]
        pivot_hits = [level for level in pivot_support_levels if spot_close <= (level + self.sr_tolerance)]

        logger.debug(
            "Bullish level check spot=%s price_level_hit=%s pivot_hit=%s price_hits=%s pivot_hits=%s",
            spot_close,
            price_level_hit,
            pivot_hit,
            price_hits,
            pivot_hits,
        )

        return {
            "price_level_hit": price_level_hit,
            "pivot_hit": pivot_hit,
            "confluence": confluence,
            "price_hits": price_hits,
            "pivot_hits": pivot_hits,
        }

    def _evaluate_bearish(self, candle: pd.Series, indicators: dict, levels: dict, spot_close: float) -> dict[str, Any]:
        """Evaluate the BUY PE bearish setup."""

        level_payload = self._extract_levels(levels)
        level_match = self._match_resistance_conditions(spot_close, level_payload)
        bb_upper = self._as_float(indicators.get("bb_upper"))
        rsi_cross_below_ma3 = bool(indicators.get("rsi_cross_below_ma3", False))
        rsi_cross_below_ma21 = bool(indicators.get("rsi_cross_below_ma21", False))

        cond_price = level_match["price_level_hit"]
        cond_pivot = level_match["pivot_hit"]
        cond_confluence = level_match["confluence"]
        cond_bb = bb_upper is not None and spot_close >= bb_upper
        cond_rsi = rsi_cross_below_ma3 or rsi_cross_below_ma21

        logger.debug(
            "Bearish evaluation spot=%s bb_upper=%s price=%s pivot=%s confluence=%s rsi3=%s rsi21=%s",
            spot_close,
            bb_upper,
            cond_price,
            cond_pivot,
            cond_confluence,
            rsi_cross_below_ma3,
            rsi_cross_below_ma21,
        )

        return {
            "candidate": cond_price or cond_pivot,
            "confluence": cond_confluence,
            "bb": cond_bb,
            "rsi": cond_rsi,
            "level_match": level_match,
            "direction": "BUY_PE",
        }

    def _evaluate_bullish(self, candle: pd.Series, indicators: dict, levels: dict, spot_close: float) -> dict[str, Any]:
        """Evaluate the BUY CE bullish setup."""

        level_payload = self._extract_levels(levels)
        level_match = self._match_support_conditions(spot_close, level_payload)
        bb_lower = self._as_float(indicators.get("bb_lower"))
        rsi_cross_above_ma3 = bool(indicators.get("rsi_cross_above_ma3", False))
        rsi_cross_above_ma21 = bool(indicators.get("rsi_cross_above_ma21", False))

        cond_price = level_match["price_level_hit"]
        cond_pivot = level_match["pivot_hit"]
        cond_confluence = level_match["confluence"]
        cond_bb = bb_lower is not None and spot_close <= bb_lower
        cond_rsi = rsi_cross_above_ma3 or rsi_cross_above_ma21

        logger.debug(
            "Bullish evaluation spot=%s bb_lower=%s price=%s pivot=%s confluence=%s rsi3=%s rsi21=%s",
            spot_close,
            bb_lower,
            cond_price,
            cond_pivot,
            cond_confluence,
            rsi_cross_above_ma3,
            rsi_cross_above_ma21,
        )

        return {
            "candidate": cond_price or cond_pivot,
            "confluence": cond_confluence,
            "bb": cond_bb,
            "rsi": cond_rsi,
            "level_match": level_match,
            "direction": "BUY_CE",
        }

    @staticmethod
    def _strength_from_conditions(candidate: bool, confluence: bool, bb: bool, rsi: bool) -> str | None:
        """Convert condition flags into a signal strength label."""

        if not candidate:
            return None
        if confluence and bb and rsi:
            return "full"
        if confluence:
            return "confluence"
        return "single"

    def generate_signals(self, candle: pd.Series, indicators: dict, levels: dict) -> dict:
        """Generate a pure ATM options signal from the current candle context."""

        spot_close = self._as_float(candle.get("close"))
        if spot_close is None:
            logger.debug("Missing spot close; returning HOLD.")
            return {
                "action": "HOLD",
                "strength": None,
                "atm_strike": 0,
                "target1_spot": None,
                "target2_spot": None,
                "stoploss_spot": None,
                "qty_multiplier": self.vix_normal_qty_multiplier,
                "signal_log": {"reason": "missing_close"},
            }

        vix = self._resolve_vix(candle, levels)
        qty_multiplier = self._resolve_qty_multiplier(vix)
        atm_strike = self._round_atm_strike(spot_close)
        is_thursday = pd.Timestamp(candle["datetime"]).day_name() == "Thursday"
        entry_allowed = self._is_entry_allowed(candle)

        bearish = self._evaluate_bearish(candle, indicators, levels, spot_close)
        bullish = self._evaluate_bullish(candle, indicators, levels, spot_close)

        bearish_strength = self._strength_from_conditions(
            bearish["candidate"], bearish["confluence"], bearish["bb"], bearish["rsi"]
        )
        bullish_strength = self._strength_from_conditions(
            bullish["candidate"], bullish["confluence"], bullish["bb"], bullish["rsi"]
        )

        action = "HOLD"
        strength: str | None = None
        side = None

        if bearish_strength and not bullish_strength:
            action = bearish["direction"]
            strength = bearish_strength
            side = "bearish"
        elif bullish_strength and not bearish_strength:
            action = bullish["direction"]
            strength = bullish_strength
            side = "bullish"
        elif bearish_strength and bullish_strength:
            logger.debug("Conflicting bullish and bearish signals detected; returning HOLD.")
        else:
            logger.debug("No actionable setup detected.")

        if is_thursday and strength not in {"full"}:
            logger.debug("Thursday filter blocked strength=%s; returning HOLD.", strength)
            action = "HOLD"

        if not entry_allowed:
            logger.debug("Entry window filter blocked candle; returning HOLD.")
            action = "HOLD"

        chosen_spot_move = {
            "BUY_PE": (-self.target1_spot_move, -self.target2_spot_move, self.stoploss_spot_move),
            "BUY_CE": (self.target1_spot_move, self.target2_spot_move, -self.stoploss_spot_move),
        }.get(action)

        if chosen_spot_move is None:
            target1_spot = None
            target2_spot = None
            stoploss_spot = None
        else:
            t1_move, t2_move, sl_move = chosen_spot_move
            target1_spot = spot_close + t1_move
            target2_spot = spot_close + t2_move
            stoploss_spot = spot_close + sl_move

        option_entry_price = None
        if action == "BUY_PE":
            option_entry_price = self._as_float(candle.get("pe_close"), self._as_float(candle.get("option_close")))
        elif action == "BUY_CE":
            option_entry_price = self._as_float(candle.get("ce_close"), self._as_float(candle.get("option_close")))
        premium_sl_price = self._premium_sl_price(option_entry_price) if option_entry_price is not None else None

        signal_log = {
            "datetime": candle.get("datetime"),
            "spot_close": spot_close,
            "atm_strike": atm_strike,
            "vix": vix,
            "qty_multiplier": qty_multiplier,
            "entry_allowed": entry_allowed,
            "is_thursday": is_thursday,
            "bearish_price_hit": bearish["candidate"],
            "bearish_confluence": bearish["confluence"],
            "bearish_bb_hit": bearish["bb"],
            "bearish_rsi_hit": bearish["rsi"],
            "bullish_price_hit": bullish["candidate"],
            "bullish_confluence": bullish["confluence"],
            "bullish_bb_hit": bullish["bb"],
            "bullish_rsi_hit": bullish["rsi"],
            "bearish_resistance_hits": bearish["level_match"]["price_hits"],
            "bearish_pivot_hits": bearish["level_match"]["pivot_hits"],
            "bullish_support_hits": bullish["level_match"]["price_hits"],
            "bullish_pivot_hits": bullish["level_match"]["pivot_hits"],
            "at_upper_band": bearish["bb"],
            "at_lower_band": bullish["bb"],
            "near_resistance": bearish["level_match"]["price_level_hit"],
            "near_support": bullish["level_match"]["price_level_hit"],
            "near_r1r2": bearish["level_match"]["pivot_hit"],
            "near_s1s2": bullish["level_match"]["pivot_hit"],
            "pe_cond1": bearish["candidate"],
            "pe_cond2": bearish["bb"],
            "pe_cond3": bearish["rsi"],
            "ce_cond1": bullish["candidate"],
            "ce_cond2": bullish["bb"],
            "ce_cond3": bullish["rsi"],
            "premium_sl_type": self.premium_sl_type,
            "premium_sl_price": premium_sl_price,
            "blocked_by_thursday": is_thursday and strength not in {"full"},
            "blocked_by_time": not entry_allowed,
            "signal_side": side,
        }

        logger.debug("Signal log: %s", signal_log)

        return {
            "action": action,
            "strength": strength,
            "atm_strike": atm_strike,
            "target1_spot": target1_spot,
            "target2_spot": target2_spot,
            "stoploss_spot": stoploss_spot,
            "qty_multiplier": qty_multiplier,
            "signal_log": signal_log,
        }

    def audit_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Audit every candle and return a condition-by-condition breakdown."""

        rows: list[dict[str, Any]] = []
        for _, candle in df.iterrows():
            indicators, levels = self._build_payloads_from_row(candle)
            sig = self.generate_signals(candle, indicators, levels)
            log = sig.get("signal_log", {}) or {}
            rows.append(
                {
                    "datetime": candle.get("datetime"),
                    "close": candle.get("close"),
                    "bb_upper": indicators.get("bb_upper"),
                    "bb_lower": indicators.get("bb_lower"),
                    "at_upper_band": log.get("at_upper_band", False),
                    "at_lower_band": log.get("at_lower_band", False),
                    "rsi": indicators.get("rsi"),
                    "rsi_ma3": indicators.get("rsi_ma3"),
                    "rsi_cross_below_ma3": log.get("pe_cond3", False),
                    "rsi_cross_above_ma3": log.get("ce_cond3", False),
                    "near_resistance": log.get("near_resistance", False),
                    "near_support": log.get("near_support", False),
                    "near_r1r2": log.get("near_r1r2", False),
                    "near_s1s2": log.get("near_s1s2", False),
                    "ce_cond1": log.get("ce_cond1", False),
                    "ce_cond2": log.get("ce_cond2", False),
                    "ce_cond3": log.get("ce_cond3", False),
                    "pe_cond1": log.get("pe_cond1", False),
                    "pe_cond2": log.get("pe_cond2", False),
                    "pe_cond3": log.get("pe_cond3", False),
                    "action": sig.get("action"),
                    "strength": sig.get("strength"),
                }
            )
        return pd.DataFrame(rows)
