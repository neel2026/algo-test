"""ATM strategy condition checks for live signals."""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

ATM_ROUNDING = 50
TARGET1_SPOT_MOVE = 65
TARGET2_SPOT_MOVE = 125
STOPLOSS_SPOT_MOVE = 27
SR_TOLERANCE = 15
ENTRY_START = time(9, 20)
ENTRY_CUTOFF = time(13, 0)
VIX_LOW_THRESHOLD = 13.0
VIX_HIGH_THRESHOLD = 15.0
SIGNAL_COOLDOWN_CANDLES = 4
PRICE_ZONE_COOLDOWN_CANDLES = 8
PRICE_ZONE_TOLERANCE = 100.0


class SignalChecker:
    """Evaluate the live scanner's signal rules."""

    def __init__(self) -> None:
        """Initialize runtime signal state."""

        self.current_vix: float | None = None
        self.candle_counter: dict[str, int] = {}
        self.last_signal_candle_index: dict[str, int] = {}
        self.last_signal_price: dict[str, float] = {}

    def _qty_multiplier(self, vix: float | None) -> float:
        """Map the current VIX regime to a quantity multiplier."""

        if vix is None or pd.isna(vix):
            return 1.0
        if float(vix) > VIX_HIGH_THRESHOLD:
            return 0.5
        if float(vix) <= VIX_LOW_THRESHOLD:
            return 1.0
        return 0.75

    def _entry_window_open(self, candle_dt: datetime) -> bool:
        """Return whether a candle time falls within the live entry window."""

        ist_dt = candle_dt.astimezone(ZoneInfo("Asia/Kolkata"))
        return ENTRY_START <= ist_dt.time() <= ENTRY_CUTOFF

    def _split_levels(self, close: float, levels: list[float], tolerance: float) -> tuple[list[float], list[float]]:
        """Split a level list into resistance and support buckets around price."""

        resistance_levels: list[float] = []
        support_levels: list[float] = []
        for raw_level in levels:
            try:
                level = float(raw_level)
            except (TypeError, ValueError):
                continue
            if level >= close - tolerance:
                resistance_levels.append(level)
            if level <= close + tolerance:
                support_levels.append(level)
        return resistance_levels, support_levels

    def _in_cooldown(self, instrument_key: str, close: float, candle_index: int) -> bool:
        """Return whether the instrument is blocked by candle or price-zone cooldown."""

        last_index = self.last_signal_candle_index.get(instrument_key)
        if last_index is not None and (candle_index - last_index) < SIGNAL_COOLDOWN_CANDLES:
            return True

        last_price = self.last_signal_price.get(instrument_key)
        if last_index is not None and last_price is not None:
            if (candle_index - last_index) < PRICE_ZONE_COOLDOWN_CANDLES and abs(close - last_price) < PRICE_ZONE_TOLERANCE:
                return True

        return False

    def check(self, instrument_key: str, candle: dict, indicators: dict, levels: dict) -> dict | None:
        """Return a signal payload when the live ATM conditions are satisfied."""

        if not candle or not indicators:
            return None

        close = float(candle.get("close", 0.0))
        candle_dt = candle.get("datetime")
        if isinstance(candle_dt, pd.Timestamp):
            candle_dt = candle_dt.to_pydatetime()
        if not isinstance(candle_dt, datetime):
            return None
        candle_index = self.candle_counter.get(instrument_key, 0)
        self.candle_counter[instrument_key] = candle_index + 1

        if self._in_cooldown(instrument_key, close, candle_index):
            return None

        sr_levels = indicators.get("support_levels") or indicators.get("resistance_levels") or []
        resistance_levels, support_levels = self._split_levels(close, sr_levels, SR_TOLERANCE)
        near_resistance = any(close >= float(level) - SR_TOLERANCE for level in resistance_levels)
        near_support = any(close <= float(level) + SR_TOLERANCE for level in support_levels)
        near_r1r2 = any(
            level is not None and pd.notna(level) and close >= float(level) - SR_TOLERANCE
            for level in [indicators.get("r1"), indicators.get("r2")]
        )
        near_s1s2 = any(
            level is not None and pd.notna(level) and close <= float(level) + SR_TOLERANCE
            for level in [indicators.get("s1"), indicators.get("s2")]
        )

        bb_upper = indicators.get("bb_upper")
        bb_lower = indicators.get("bb_lower")
        at_upper_band = bb_upper is not None and pd.notna(bb_upper) and close >= float(bb_upper)
        at_lower_band = bb_lower is not None and pd.notna(bb_lower) and close <= float(bb_lower)

        rsi_cross_below_ma3 = bool(indicators.get("rsi_cross_below_ma3", False))
        rsi_cross_above_ma3 = bool(indicators.get("rsi_cross_above_ma3", False))
        rsi_cross_below_ma21 = bool(indicators.get("rsi_cross_below_ma21", False))
        rsi_cross_above_ma21 = bool(indicators.get("rsi_cross_above_ma21", False))

        pe_cond1a = bool(near_resistance)
        pe_cond1b = bool(near_r1r2)
        pe_cond1 = pe_cond1a or pe_cond1b
        pe_cond2 = bool(at_upper_band)
        pe_cond3 = bool(rsi_cross_below_ma3 or rsi_cross_below_ma21)
        pe_confluence = pe_cond1a and pe_cond1b

        ce_cond1a = bool(near_support)
        ce_cond1b = bool(near_s1s2)
        ce_cond1 = ce_cond1a or ce_cond1b
        ce_cond2 = bool(at_lower_band)
        ce_cond3 = bool(rsi_cross_above_ma3 or rsi_cross_above_ma21)
        ce_confluence = ce_cond1a and ce_cond1b

        pe_trade = pe_cond1
        ce_trade = ce_cond1

        if not pe_trade and not ce_trade:
            return None

        pe_score = int(pe_cond1) + int(pe_cond2) + int(pe_cond3) + int(pe_confluence)
        ce_score = int(ce_cond1) + int(ce_cond2) + int(ce_cond3) + int(ce_confluence)

        action = "HOLD"
        strength = "single"
        if pe_trade and not ce_trade:
            action = "BUY_PE"
        elif ce_trade and not pe_trade:
            action = "BUY_CE"
        elif pe_trade and ce_trade:
            if pe_score > ce_score:
                action = "BUY_PE"
            elif ce_score > pe_score:
                action = "BUY_CE"

        if action == "HOLD":
            return None

        if action == "BUY_PE":
            if pe_cond2 and pe_cond3:
                strength = "full"
            elif pe_cond2 or pe_cond3:
                strength = "confluence"
            else:
                strength = "single"
        elif action == "BUY_CE":
            if ce_cond2 and ce_cond3:
                strength = "full"
            elif ce_cond2 or ce_cond3:
                strength = "confluence"
            else:
                strength = "single"

        atm_strike = int(round(close / ATM_ROUNDING) * ATM_ROUNDING)
        direction = "CE" if action == "BUY_CE" else "PE"
        target1 = close + TARGET1_SPOT_MOVE if direction == "CE" else close - TARGET1_SPOT_MOVE
        target2 = close + TARGET2_SPOT_MOVE if direction == "CE" else close - TARGET2_SPOT_MOVE
        stoploss = close - STOPLOSS_SPOT_MOVE if direction == "CE" else close + STOPLOSS_SPOT_MOVE
        vix = levels.get("vix") or indicators.get("vix") or self.current_vix
        qty_multiplier = self._qty_multiplier(float(vix) if vix is not None and pd.notna(vix) else None)
        entry_window_open = self._entry_window_open(candle_dt)
        signal_id = f"{instrument_key}:{int(candle_dt.timestamp())}:{action}"

        signal_log = {
            "at_upper_band": bool(at_upper_band),
            "at_lower_band": bool(at_lower_band),
            "rsi_cross_below_ma3": bool(rsi_cross_below_ma3),
            "rsi_cross_above_ma3": bool(rsi_cross_above_ma3),
            "rsi_cross_below_ma21": bool(rsi_cross_below_ma21),
            "rsi_cross_above_ma21": bool(rsi_cross_above_ma21),
            "near_resistance": bool(near_resistance),
            "near_support": bool(near_support),
            "near_r1r2": bool(near_r1r2),
            "near_s1s2": bool(near_s1s2),
            "pe_cond1": bool(pe_cond1),
            "pe_cond2": bool(pe_cond2),
            "pe_cond3": bool(pe_cond3),
            "pe_confluence": bool(pe_confluence),
            "ce_cond1": bool(ce_cond1),
            "ce_cond2": bool(ce_cond2),
            "ce_cond3": bool(ce_cond3),
            "ce_confluence": bool(ce_confluence),
            "entry_window_open": bool(entry_window_open),
            "tradeable": bool(entry_window_open),
        }

        signal = {
            "signal_id": signal_id,
            "instrument": instrument_key,
            "datetime": candle_dt.isoformat(),
            "candle_time_ts": int(candle_dt.timestamp()),
            "action": action,
            "strength": strength,
            "atm_strike": atm_strike,
            "target1_spot": target1,
            "target2_spot": target2,
            "stoploss_spot": stoploss,
            "qty_multiplier": qty_multiplier,
            "vix_at_entry": float(vix) if vix is not None and pd.notna(vix) else None,
            "rsi_at_entry": float(indicators.get("rsi")) if indicators.get("rsi") is not None and pd.notna(indicators.get("rsi")) else None,
            "bb_position_at_entry": float(indicators.get("percent_b")) if indicators.get("percent_b") is not None and pd.notna(indicators.get("percent_b")) else None,
            "entry_window_open": bool(entry_window_open),
            "tradeable": bool(entry_window_open),
            "signal_log": signal_log,
        }
        rsi_value = indicators.get("rsi")
        if rsi_value is None or pd.isna(rsi_value):
            signal["strength"] = "single"
            signal["action_note"] = "RSI/BB not yet computed"
        self.last_signal_candle_index[instrument_key] = candle_index
        self.last_signal_price[instrument_key] = close
        return signal
