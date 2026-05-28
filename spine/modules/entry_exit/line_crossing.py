"""Line crossing detection — Pillar 4. Detects when tracks cross defined lines."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

from spine.core.config import LineConfig
from spine.core.events import EventType, LineCrossEvent, SpineEvent
from spine.core.orchestrator import FrameContext, ModuleBase

logger = logging.getLogger(__name__)


class LineCrossingDetector(ModuleBase):
    """Detects when tracked person centroid crosses a defined line segment.

    Uses TWO methods (either triggers crossing):
    1. Segment intersection (classic) — prev→curr path crosses line segment
    2. Side-change detection — centroid's signed distance to line flips sign
       AND centroid is within extended line corridor (within margin of segment endpoints)

    Method 2 catches slow walkers whose per-frame displacement is small.
    """

    def __init__(self, config: dict[str, Any] | None = None, event_bus: Any = None,
                 lines: list[LineConfig] | None = None):
        super().__init__(config)
        self.event_bus = event_bus
        self.lines = lines or []
        self._track_history: dict[int, deque] = {}
        self._track_side: dict[tuple[int, str], float] = {}  # (track_id, line_id) → last signed dist
        self._crossed: dict[tuple[int, str], float] = {}
        self._cooldown = config.get("cooldown", 2.0) if config else 2.0
        self._crossing_count = 0

    def initialize(self) -> None:
        self._initialized = True
        logger.info("LineCrossingDetector ready (%d lines)", len(self.lines))
        for l in self.lines:
            logger.info("  Line '%s': %s → %s, dir_in=%s", l.id, l.points[0], l.points[1], l.direction_in)

    def process(self, ctx: FrameContext) -> None:
        if not self._initialized or not ctx.tracks:
            return

        camera_lines = [l for l in self.lines if l.camera_id == ctx.camera_id]
        if not camera_lines:
            return

        for track in ctx.tracks:
            track_id = track.get("track_id", -1)
            if track_id < 0:
                continue

            bbox = track["bbox"]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2

            if track_id not in self._track_history:
                self._track_history[track_id] = deque(maxlen=30)
            self._track_history[track_id].append((cx, cy, ctx.timestamp))

            history = self._track_history[track_id]
            if len(history) < 2:
                continue

            prev_x, prev_y, _ = history[-2]
            curr_x, curr_y, _ = history[-1]

            for line in camera_lines:
                key = (track_id, line.id)
                last_cross = self._crossed.get(key, 0)
                if ctx.timestamp - last_cross < self._cooldown:
                    continue

                p1 = line.points[0]
                p2 = line.points[1]

                crossed = False

                # Method 1: Segment intersection
                if self._segments_intersect(prev_x, prev_y, curr_x, curr_y,
                                            p1[0], p1[1], p2[0], p2[1]):
                    crossed = True

                # Method 2: Side-change detection (catches slow movement)
                if not crossed:
                    curr_dist = self._signed_distance(cx, cy, p1, p2)
                    prev_dist = self._track_side.get(key, curr_dist)
                    self._track_side[key] = curr_dist

                    # Sign flipped AND point is near the line segment
                    if prev_dist * curr_dist < 0:  # sign change
                        margin = self._line_length(p1, p2) * 0.3  # 30% margin beyond endpoints
                        if self._near_segment(cx, cy, p1, p2, margin):
                            crossed = True

                if crossed:
                    direction = self._get_direction(
                        prev_x, prev_y, curr_x, curr_y,
                        p1, p2, line.direction_in,
                    )

                    self._crossed[key] = ctx.timestamp
                    self._crossing_count += 1

                    logger.info("LINE CROSSED: track=%d line=%s dir=%s pos=(%.0f,%.0f)",
                                track_id, line.id, direction, cx, cy)

                    if self.event_bus:
                        self.event_bus.publish(LineCrossEvent(
                            camera_id=ctx.camera_id,
                            track_id=track_id,
                            timestamp=ctx.timestamp,
                            line_id=line.id,
                            direction=direction,
                        ))

                    if line.alert_on and direction == line.alert_on:
                        self.event_bus.publish(SpineEvent(
                            event_type=EventType.ANOMALY_DETECTED,
                            camera_id=ctx.camera_id,
                            track_id=track_id,
                            timestamp=ctx.timestamp,
                            data={"line_id": line.id, "violation": line.alert_on},
                        ))

    @staticmethod
    def _signed_distance(px, py, p1, p2) -> float:
        """Signed distance from point to infinite line through p1→p2."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1e-6:
            return 0.0
        return (dy * (px - p1[0]) - dx * (py - p1[1])) / length

    @staticmethod
    def _near_segment(px, py, p1, p2, margin) -> bool:
        """Check if point is within extended bounding box of segment + margin."""
        min_x = min(p1[0], p2[0]) - margin
        max_x = max(p1[0], p2[0]) + margin
        min_y = min(p1[1], p2[1]) - margin
        max_y = max(p1[1], p2[1]) + margin
        return min_x <= px <= max_x and min_y <= py <= max_y

    @staticmethod
    def _line_length(p1, p2) -> float:
        return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5

    @staticmethod
    def _segments_intersect(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> bool:
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        A, B = (ax1, ay1), (ax2, ay2)
        C, D = (bx1, by1), (bx2, by2)

        d1 = cross(C, D, A)
        d2 = cross(C, D, B)
        d3 = cross(A, B, C)
        d4 = cross(A, B, D)

        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False

    @staticmethod
    def _get_direction(prev_x, prev_y, curr_x, curr_y, p1, p2, direction_in) -> str:
        dx_move = curr_x - prev_x
        dy_move = curr_y - prev_y

        if direction_in == "up":
            return "in" if dy_move < 0 else "out"
        elif direction_in == "down":
            return "in" if dy_move > 0 else "out"
        elif direction_in == "left":
            return "in" if dx_move < 0 else "out"
        elif direction_in == "right":
            return "in" if dx_move > 0 else "out"
        else:
            # Use cross product with line normal
            dx_line = p2[0] - p1[0]
            dy_line = p2[1] - p1[1]
            normal = (-dy_line, dx_line)
            dot = normal[0] * dx_move + normal[1] * dy_move
            return "in" if dot > 0 else "out"

    def cleanup(self) -> None:
        self._track_history.clear()
        self._crossed.clear()
        self._track_side.clear()
