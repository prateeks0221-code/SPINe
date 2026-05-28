"""Occupancy counter — tracks entry/exit counts per line/zone with analytics."""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

from spine.core.events import EventType, SpineEvent, LineCrossEvent
from spine.core.orchestrator import ModuleBase

logger = logging.getLogger(__name__)


class OccupancyCounter:
    """Maintains real-time occupancy count per zone/line with analytics.

    Tracks:
    - Current occupancy per gate
    - Total entries/exits
    - Per-track crossing history (for unique counting)
    - Crossing rate (entries/minute)
    - Peak occupancy
    """

    def __init__(self, event_bus: Any = None):
        self.event_bus = event_bus
        self._counts: dict[str, int] = {}
        self._total_in: dict[str, int] = {}
        self._total_out: dict[str, int] = {}
        self._peak: dict[str, int] = {}
        self._crossing_log: deque = deque(maxlen=500)  # recent crossings with timestamps
        self._track_crossings: dict[int, list[dict]] = {}  # per-track crossing history

    def on_line_crossed(self, event: SpineEvent) -> None:
        if not isinstance(event, LineCrossEvent):
            return

        line_id = event.line_id
        if line_id not in self._counts:
            self._counts[line_id] = 0
            self._total_in[line_id] = 0
            self._total_out[line_id] = 0
            self._peak[line_id] = 0

        if event.direction == "in":
            self._counts[line_id] += 1
            self._total_in[line_id] += 1
        elif event.direction == "out":
            self._counts[line_id] = max(0, self._counts[line_id] - 1)
            self._total_out[line_id] += 1

        # Track peak
        if self._counts[line_id] > self._peak[line_id]:
            self._peak[line_id] = self._counts[line_id]

        # Log crossing
        crossing_record = {
            "line_id": line_id,
            "track_id": event.track_id,
            "direction": event.direction,
            "timestamp": event.timestamp,
        }
        self._crossing_log.append(crossing_record)

        # Per-track history
        if event.track_id is not None:
            if event.track_id not in self._track_crossings:
                self._track_crossings[event.track_id] = []
            self._track_crossings[event.track_id].append(crossing_record)

        logger.info("Occupancy [%s]: %d (in=%d out=%d peak=%d)",
                     line_id, self._counts[line_id],
                     self._total_in[line_id], self._total_out[line_id],
                     self._peak[line_id])

        if self.event_bus:
            self.event_bus.publish(SpineEvent(
                event_type=EventType.OCCUPANCY_COUNT,
                camera_id=event.camera_id,
                timestamp=event.timestamp,
                data={
                    "line_id": line_id,
                    "count": self._counts[line_id],
                    "total_in": self._total_in[line_id],
                    "total_out": self._total_out[line_id],
                    "peak": self._peak[line_id],
                    "delta": 1 if event.direction == "in" else -1,
                    "rate_per_min": self.get_rate(line_id),
                },
            ))

    def get_count(self, line_id: str) -> int:
        return self._counts.get(line_id, 0)

    def get_all_counts(self) -> dict[str, int]:
        return dict(self._counts)

    def get_rate(self, line_id: str, window_seconds: float = 60.0) -> float:
        """Crossings per minute for a given line."""
        now = time.time()
        recent = [c for c in self._crossing_log
                  if c["line_id"] == line_id and now - c["timestamp"] < window_seconds]
        return len(recent) / (window_seconds / 60.0)

    def get_stats(self, line_id: str) -> dict:
        return {
            "count": self._counts.get(line_id, 0),
            "total_in": self._total_in.get(line_id, 0),
            "total_out": self._total_out.get(line_id, 0),
            "peak": self._peak.get(line_id, 0),
            "rate_per_min": self.get_rate(line_id),
            "unique_tracks": len([t for t, crossings in self._track_crossings.items()
                                  if any(c["line_id"] == line_id for c in crossings)]),
        }

    def reset(self, line_id: str | None = None) -> None:
        if line_id:
            self._counts[line_id] = 0
            self._total_in[line_id] = 0
            self._total_out[line_id] = 0
            self._peak[line_id] = 0
        else:
            self._counts.clear()
            self._total_in.clear()
            self._total_out.clear()
            self._peak.clear()
            self._crossing_log.clear()
            self._track_crossings.clear()
