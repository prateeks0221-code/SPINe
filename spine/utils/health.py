"""Health check endpoints for spine services."""

from __future__ import annotations

import time
from typing import Any


class HealthChecker:
    """Aggregates health from all spine components."""

    def __init__(self):
        self._checks: dict[str, dict[str, Any]] = {}
        self._start_time = time.time()

    def register(self, name: str, check_fn) -> None:
        self._checks[name] = {"fn": check_fn, "last_status": None, "last_check": 0}

    def check_all(self) -> dict[str, Any]:
        results = {}
        all_healthy = True

        for name, entry in self._checks.items():
            try:
                status = entry["fn"]()
                entry["last_status"] = status
                entry["last_check"] = time.time()
                results[name] = {"healthy": True, **status} if isinstance(status, dict) else {"healthy": bool(status)}
            except Exception as e:
                results[name] = {"healthy": False, "error": str(e)}
                all_healthy = False

        return {
            "healthy": all_healthy,
            "uptime_seconds": time.time() - self._start_time,
            "components": results,
        }
