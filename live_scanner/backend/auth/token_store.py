"""Token persistence for Upstox OAuth access tokens."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
TOKEN_FILE = ROOT / ".token"


class TokenStore:
    """Persist and retrieve the current access token."""

    @classmethod
    def save(cls, token: str) -> None:
        """Save an access token with its timestamp."""

        payload = {
            "access_token": token,
            "saved_at": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
        }
        TOKEN_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls) -> str | None:
        """Load the access token if it is still valid for today."""

        if not TOKEN_FILE.exists():
            return None
        try:
            raw = TOKEN_FILE.read_text(encoding="utf-8").strip()
            if not raw:
                return None
            try:
                payload = json.loads(raw)
                token = payload.get("access_token") or payload.get("token")
                saved_at = payload.get("saved_at")
            except json.JSONDecodeError:
                token = raw
                saved_at = None
            if not token:
                return None
            if saved_at:
                saved_date = datetime.fromisoformat(saved_at).astimezone(ZoneInfo("Asia/Kolkata")).date()
                today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
                if saved_date != today:
                    return None
            return str(token)
        except Exception:
            return None

    @classmethod
    def clear(cls) -> None:
        """Delete the saved token, if any."""

        TOKEN_FILE.unlink(missing_ok=True)
