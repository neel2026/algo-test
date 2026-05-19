"""Upstox OAuth2 helpers and local callback server."""

from __future__ import annotations

import os
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable
from urllib.parse import parse_qs, urlencode, urlparse

import requests

from auth.token_store import TokenStore

logger = logging.getLogger(__name__)


class UpstoxAuth:
    """Handle the Upstox OAuth2 login flow."""

    def __init__(self, on_token: Callable[[str], None] | None = None) -> None:
        """Create an auth helper with an optional token callback."""

        self._on_token = on_token
        self._callback_started = False
        self._callback_lock = threading.Lock()
        self._auth_code_event = threading.Event()
        self._latest_code: str | None = None
        self._server: HTTPServer | None = None

    def get_login_url(self) -> str:
        """Build the Upstox login URL."""

        params = {
            "response_type": "code",
            "client_id": os.getenv("UPSTOX_API_KEY", ""),
            "redirect_uri": os.getenv("UPSTOX_REDIRECT_URI", "http://127.0.0.1:3000/callback"),
        }
        return "https://api.upstox.com/v2/login/authorization/dialog?" + urlencode(params)

    def exchange_code(self, auth_code: str) -> str:
        """Exchange an authorization code for an access token."""

        import os

        resp = requests.post(
            "https://api.upstox.com/v2/login/authorization/token",
            data={
                "code": auth_code,
                "client_id": os.getenv("UPSTOX_API_KEY", ""),
                "client_secret": os.getenv("UPSTOX_API_SECRET", ""),
                "redirect_uri": os.getenv("UPSTOX_REDIRECT_URI", "http://127.0.0.1:3000/callback"),
                "grant_type": "authorization_code",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=20,
        )
        resp.raise_for_status()
        payload = resp.json()
        token = payload.get("access_token") or payload.get("data", {}).get("access_token")
        if not token:
            raise ValueError(f"Unexpected token response: {payload}")
        TokenStore.save(token)
        if self._on_token:
            self._on_token(token)
        return token

    def start_callback_server(self) -> threading.Event:
        """Start a one-shot local callback server on port 3000."""

        with self._callback_lock:
            if self._callback_started:
                return self._auth_code_event

            auth = self

            class CallbackHandler(BaseHTTPRequestHandler):
                """Handle OAuth callback requests."""

                def do_GET(self) -> None:  # noqa: N802
                    """Capture the authorization code from the callback URL."""

                    parsed = urlparse(self.path)
                    if parsed.path != "/callback":
                        self.send_response(404)
                        self.end_headers()
                        self.wfile.write(b"Not found")
                        return
                    params = parse_qs(parsed.query)
                    error = params.get("error", [None])[0]
                    if error:
                        self.send_response(400)
                        self.send_header("Content-Type", "text/html")
                        self.end_headers()
                        self.wfile.write(f"<h1>OAuth error</h1><p>{error}</p>".encode("utf-8"))
                        return
                    code = params.get("code", [None])[0]
                    if not code:
                        self.send_response(400)
                        self.send_header("Content-Type", "text/html")
                        self.end_headers()
                        self.wfile.write(b"<h1>Missing code</h1>")
                        return

                    auth._latest_code = code
                    auth._auth_code_event.set()
                    threading.Thread(target=auth._finalize_code, args=(code,), daemon=True).start()

                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(
                        (
                            "<html><head><meta http-equiv='refresh' content='1;url=http://localhost:5173/?auth=success' /></head>"
                            "<body style='font-family: monospace; background:#0f0f0f; color:#d1d4dc;'>"
                            "<h2>Upstox authentication complete.</h2>"
                            "<p>You can close this tab. Redirecting to the scanner...</p>"
                            "</body></html>"
                        ).encode("utf-8")
                    )

                def log_message(self, format: str, *args) -> None:  # noqa: A003
                    """Silence the default HTTP server logging."""

                    logger.debug(format, *args)

            self._server = HTTPServer(("127.0.0.1", 3000), CallbackHandler)
            threading.Thread(target=self._server.serve_forever, daemon=True).start()
            self._callback_started = True
            logger.info("OAuth callback server listening on http://127.0.0.1:3000/callback")
            return self._auth_code_event

    def wait_for_code(self, timeout: float | None = None) -> str | None:
        """Block until an auth code is received."""

        if self._auth_code_event.wait(timeout=timeout):
            return self._latest_code
        return None

    def _finalize_code(self, code: str) -> None:
        """Exchange the received code and stop the callback server."""

        try:
            self.exchange_code(code)
            logger.info("Upstox token stored successfully.")
        except Exception as exc:
            logger.exception("Failed to exchange Upstox auth code: %s", exc)
        finally:
            if self._server:
                try:
                    self._server.shutdown()
                except Exception:
                    pass
