"""Upstox market data WebSocket integration."""

from __future__ import annotations

import logging
import threading
from typing import Callable

from feed.candle_builder import CandleBuilder

logger = logging.getLogger(__name__)


class UpstoxFeed:
    """Maintain a live market data stream and publish ticks/candles."""

    def __init__(
        self,
        access_token: str,
        candle_builder: CandleBuilder,
        on_candle_close: Callable[[str, dict], None],
        on_tick: Callable[[str, float, int], None],
    ) -> None:
        """Create the feed wrapper."""

        self.token = access_token
        self.builder = candle_builder
        self.on_candle_close = on_candle_close
        self.on_tick = on_tick
        self.streamer = None
        self.subscribed: set[str] = set()

    def start(self, instruments: list[str]) -> None:
        """Connect to the Upstox websocket in a background thread."""

        try:
            import upstox_client
        except ImportError:
            logger.error(
                "upstox-python-sdk is not installed. Live feed is disabled until the package is available."
            )
            return

        config = upstox_client.Configuration()
        config.access_token = self.token
        api_client = upstox_client.ApiClient(config)
        self.streamer = upstox_client.MarketDataStreamerV3(api_client, instruments, "full")
        self.streamer.on("open", self._on_open)
        self.streamer.on("message", self._on_message)
        self.streamer.on("error", self._on_error)
        self.streamer.on("close", self._on_close)
        self.streamer.on("autoReconnectStopped", self._on_reconnect_stopped)
        self.subscribed = set(instruments)
        threading.Thread(target=self.streamer.connect, daemon=True).start()

    def _on_open(self, *args, **kwargs) -> None:
        """Log websocket open."""

        logger.info("Upstox market feed connected.")

    def _on_close(self, *args, **kwargs) -> None:
        """Log websocket close."""

        logger.warning("Upstox market feed closed.")

    def _on_error(self, error: Exception | dict | str) -> None:
        """Log websocket errors."""

        logger.error("Upstox feed error: %s", error)

    def _on_reconnect_stopped(self, *args, **kwargs) -> None:
        """Log reconnect stop events."""

        logger.warning("Upstox feed auto reconnect stopped.")

    def _extract_volume(self, feed_data: dict) -> int:
        """Pull volume from the full market feed payload."""

        ff = feed_data.get("ff", {})
        market_ff = ff.get("marketFF", {})
        ohlc_list = market_ff.get("marketOHLC", {}).get("ohlc", [])
        for ohlc in ohlc_list:
            if ohlc.get("interval") == "I1":
                return int(ohlc.get("vol", 0) or 0)
        return 0

    def _on_message(self, message: dict) -> None:
        """Handle one websocket message from Upstox."""

        feeds = message.get("feeds", {})
        for instrument_key, feed_data in feeds.items():
            ltpc = feed_data.get("ltpc", {})
            ltp = ltpc.get("ltp")
            ltt = int(ltpc.get("ltt", 0) or 0)
            if ltp is None:
                continue

            volume = self._extract_volume(feed_data)
            self.on_tick(instrument_key, float(ltp), ltt)
            closed = self.builder.on_tick(instrument_key, float(ltp), volume, ltt)
            if closed:
                self.on_candle_close(instrument_key, closed)

    def subscribe(self, instruments: list[str]) -> None:
        """Subscribe to additional instruments."""

        new_instruments = [instrument for instrument in instruments if instrument not in self.subscribed]
        if new_instruments and self.streamer:
            self.streamer.subscribe(new_instruments, "full")
            self.subscribed.update(new_instruments)

    def unsubscribe(self, instruments: list[str]) -> None:
        """Unsubscribe from instruments."""

        if self.streamer and instruments:
            self.streamer.unsubscribe(instruments)
        self.subscribed -= set(instruments)

    def change_instruments(self, new_instruments: list[str]) -> None:
        """Swap to a new subscription set."""

        to_remove = list(self.subscribed - set(new_instruments))
        to_add = [instrument for instrument in new_instruments if instrument not in self.subscribed]
        if to_remove:
            self.unsubscribe(to_remove)
        if to_add:
            self.subscribe(to_add)
