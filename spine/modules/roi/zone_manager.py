"""ROI Zone Manager — polygon zones, enter/exit/dwell detection. Pillar 5."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from spine.core.config import ZoneConfig
from spine.core.events import EventType, SpineEvent, ZoneEvent
from spine.core.orchestrator import FrameContext, ModuleBase

logger = logging.getLogger(__name__)


class ZoneManager(ModuleBase):
    """Manages polygon zones. Detects zone entry/exit, dwell time, occupancy."""

    def __init__(self, config: dict[str, Any] | None = None, event_bus: Any = None,
                 zones: list[ZoneConfig] | None = None):
        super().__init__(config)
        self.event_bus = event_bus
        self.zones = zones or []
        self._track_zones: dict[int, set[str]] = {}
        self._zone_counts: dict[str, int] = {}
        self._dwell_start: dict[tuple[int, str], float] = {}

    def initialize(self) -> None:
        for zone in self.zones:
            self._zone_counts[zone.id] = 0
        self._initialized = True
        logger.info("ZoneManager ready (%d zones)", len(self.zones))

    def process(self, ctx: FrameContext) -> None:
        if not self._initialized:
            return

        camera_zones = [z for z in self.zones if z.camera_id == ctx.camera_id]
        if not camera_zones:
            return

        current_zone_tracks: dict[str, set[int]] = {z.id: set() for z in camera_zones}

        tracks = ctx.tracks or ctx.detections
        for track in tracks:
            track_id = track.get("track_id", -1)
            if track_id < 0:
                continue

            bbox = track["bbox"]
            cx = (bbox[0] + bbox[2]) / 2
            cy = bbox[3]  # foot point

            prev_zones = self._track_zones.get(track_id, set())
            curr_zones: set[str] = set()

            for zone in camera_zones:
                if self._point_in_polygon(cx, cy, zone.points):
                    curr_zones.add(zone.id)
                    current_zone_tracks[zone.id].add(track_id)

                    if zone.id not in prev_zones:
                        self._on_zone_enter(ctx, track_id, zone)

                    dwell_key = (track_id, zone.id)
                    if dwell_key not in self._dwell_start:
                        self._dwell_start[dwell_key] = ctx.timestamp
                    else:
                        dwell_threshold = zone.analytics.get("dwell_time", {})
                        if isinstance(dwell_threshold, dict):
                            thresh = dwell_threshold.get("threshold", 300)
                        else:
                            thresh = 300
                        dwell = ctx.timestamp - self._dwell_start[dwell_key]
                        if dwell > thresh:
                            if self.event_bus:
                                self.event_bus.publish(SpineEvent(
                                    event_type=EventType.DWELL_EXCEEDED,
                                    camera_id=ctx.camera_id,
                                    track_id=track_id,
                                    timestamp=ctx.timestamp,
                                    data={"zone_id": zone.id, "dwell_seconds": dwell},
                                ))
                            self._dwell_start[dwell_key] = ctx.timestamp

            exited_zones = prev_zones - curr_zones
            for zone_id in exited_zones:
                self._on_zone_exit(ctx, track_id, zone_id)
                self._dwell_start.pop((track_id, zone_id), None)

            self._track_zones[track_id] = curr_zones

        for zone in camera_zones:
            count = len(current_zone_tracks[zone.id])
            self._zone_counts[zone.id] = count

            for alert in zone.alerts:
                self._check_alert(ctx, zone, alert, count)

    def _on_zone_enter(self, ctx: FrameContext, track_id: int, zone: ZoneConfig) -> None:
        if zone.type == "exclusion":
            if self.event_bus:
                self.event_bus.publish(SpineEvent(
                    event_type=EventType.ZONE_UNAUTHORIZED,
                    camera_id=ctx.camera_id,
                    track_id=track_id,
                    timestamp=ctx.timestamp,
                    data={"zone_id": zone.id},
                ))
        elif self.event_bus:
            self.event_bus.publish(ZoneEvent(
                event_type=EventType.ZONE_ENTERED,
                camera_id=ctx.camera_id,
                track_id=track_id,
                timestamp=ctx.timestamp,
                zone_id=zone.id,
            ))

    def _on_zone_exit(self, ctx: FrameContext, track_id: int, zone_id: str) -> None:
        if self.event_bus:
            self.event_bus.publish(ZoneEvent(
                event_type=EventType.ZONE_EXITED,
                camera_id=ctx.camera_id,
                track_id=track_id,
                timestamp=ctx.timestamp,
                zone_id=zone_id,
            ))

    def _check_alert(self, ctx: FrameContext, zone: ZoneConfig, alert: dict, count: int) -> None:
        condition = alert.get("condition", "")
        if "count >" in condition:
            try:
                threshold = int(condition.split(">")[1].strip())
                if count > threshold and self.event_bus:
                    self.event_bus.publish(SpineEvent(
                        event_type=EventType.ZONE_OVERCROWDED,
                        camera_id=ctx.camera_id,
                        timestamp=ctx.timestamp,
                        data={"zone_id": zone.id, "count": count, "threshold": threshold},
                    ))
            except (ValueError, IndexError):
                pass

    @staticmethod
    def _point_in_polygon(x: float, y: float, polygon: list[list[int]]) -> bool:
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi):
                inside = not inside
            j = i
        return inside

    def get_zone_count(self, zone_id: str) -> int:
        return self._zone_counts.get(zone_id, 0)

    def cleanup(self) -> None:
        self._track_zones.clear()
        self._zone_counts.clear()
        self._dwell_start.clear()
