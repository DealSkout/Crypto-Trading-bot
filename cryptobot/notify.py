"""Optional Telegram notifications. Never raises: a failed message must not stop trading."""

from __future__ import annotations

import logging
import os

import requests

from cryptobot.config import Config

log = logging.getLogger(__name__)


class Notifier:
    def __init__(
        self,
        enabled: bool = True,
        token: str | None = None,
        chat_id: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.token = token or os.environ.get("TELEGRAM_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(enabled and self.token and self.chat_id)
        self.session = session or requests.Session()
        self.sent: list[str] = []

    def send(self, text: str) -> bool:
        if not self.enabled:
            return False
        try:
            r = self.session.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat_id, "text": text[:4000]},
                timeout=10,
            )
            ok = r.status_code == 200
            if not ok:
                log.warning("telegram returned %s", r.status_code)
            else:
                self.sent.append(text)
            return ok
        except requests.RequestException as e:
            log.warning("telegram send failed: %s", e)
            return False


def make_notifier(cfg: Config) -> Notifier:
    return Notifier(enabled=cfg.telegram)
