"""Extensible alert system and structured logging for the CRT trading engine.

Important alert events:
- UNKNOWN order
- position without protection
- exchange disconnect
- stale candles
- risk accounting failure
- unexpected position
- unexpected open order
- trading disabled after safety failure
- daily loss stop
- daily profit stop
- application crash/restart

Credentials and secrets are strictly redacted and never logged.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("ethresearch.crt")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
        datefmt='%Y-%m-%dT%H:%M:%S%z'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class AlertSystem:
    """Extensible alert dispatcher."""

    def __init__(self):
        self._listeners: List[Callable[[Dict[str, Any]], None]] = []
        self._history: List[Dict[str, Any]] = []
        self._max_history = 100

    def add_listener(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        self._listeners.append(listener)

    def dispatch(self, event_type: str, severity: str, message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Dispatch an alert event to all registered listeners and structured logs.
        
        Secrets are never accepted or logged in details.
        """
        clean_details = dict(details or {})
        # Redact any accidental credential keys
        for key in list(clean_details.keys()):
            if any(k in key.lower() for k in ('key', 'secret', 'token', 'auth', 'cookie', 'password')):
                clean_details[key] = '[REDACTED]'

        alert = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event_type': event_type,
            'severity': severity.upper(),
            'message': message,
            'details': clean_details,
        }

        # Log structured message
        log_msg = f"[{severity.upper()}] {event_type}: {message}"
        if severity.upper() in ('CRITICAL', 'FATAL'):
            logger.critical(log_msg)
        elif severity.upper() == 'ERROR':
            logger.error(log_msg)
        elif severity.upper() in ('WARN', 'WARNING'):
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

        # Store in bounded history
        self._history.append(alert)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        # Notify listeners
        for listener in self._listeners:
            try:
                listener(alert)
            except Exception as exc:
                logger.error(f"Alert listener raised: {exc}")

        return alert

    def get_recent_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        return list(reversed(self._history[-limit:]))


# Global default alert dispatcher
ALERTS = AlertSystem()
