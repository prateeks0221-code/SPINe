"""VibeCheck product adapter — clubs/bars crowd analytics on CV Spine."""

from __future__ import annotations

import logging
from typing import Any

import yaml

from spine.core.adapter import ProductAdapter
from spine.core.config import (
    CameraConfig, DetectionConfig, EntryExitConfig, LineConfig,
    ModulesConfig, PoseConfig, ReIDConfig, ROIConfig, ZoneConfig,
)
from spine.core.events import EventType, SpineEvent

logger = logging.getLogger(__name__)


class VibeCheckAdapter(ProductAdapter):
    """VibeCheck — crowd analytics for nightclubs and bars."""

    product_id = "vibecheck"

    def __init__(self, venue_config_path: str | None = None):
        self._venue_config = {}
        if venue_config_path:
            with open(venue_config_path) as f:
                self._venue_config = yaml.safe_load(f) or {}

        self._vibe_score = 50.0
        self._dance_energy = 0.0
        self._occupancy = 0
        self._capacity = self._venue_config.get("capacity", 200)

    def get_camera_configs(self) -> list[CameraConfig]:
        cameras = self._venue_config.get("cameras", [])
        if not cameras:
            return [
                CameraConfig(
                    camera_id="cam-main",
                    source="rtsp://mediamtx:8554/venue-001",
                    modules=ModulesConfig(
                        person_detector=DetectionConfig(model="yolov8m", confidence=0.35),
                        pose_keypoints=PoseConfig(fps=2.0),
                        entry_exit=EntryExitConfig(enabled=True),
                        roi_zones=ROIConfig(enabled=True),
                        reid=ReIDConfig(enabled=True, gallery_ttl=14400),
                        face=None,
                    ),
                    stream_to_browser=True,
                    anonymize=True,
                )
            ]

        return [
            CameraConfig(
                camera_id=cam["id"],
                source=cam["source"],
                modules=ModulesConfig(
                    person_detector=DetectionConfig(**cam.get("detection", {})),
                    pose_keypoints=PoseConfig(**cam.get("pose", {})) if cam.get("pose") else None,
                    entry_exit=EntryExitConfig(enabled=cam.get("entry_exit", True)),
                    roi_zones=ROIConfig(enabled=cam.get("roi", True)),
                    reid=ReIDConfig(**cam.get("reid", {})) if cam.get("reid") else None,
                    face=None,
                ),
                stream_to_browser=cam.get("stream", True),
                anonymize=True,
            )
            for cam in cameras
        ]

    def get_zone_configs(self) -> list[ZoneConfig]:
        zones = self._venue_config.get("zones", [])
        return [ZoneConfig(**z) for z in zones]

    def get_line_configs(self) -> list[LineConfig]:
        lines = self._venue_config.get("lines", [])
        return [LineConfig(**l) for l in lines]

    def get_face_galleries(self) -> dict[str, Any]:
        return {}

    def on_event(self, event: SpineEvent) -> None:
        if event.event_type == EventType.POSE_DANCE_ENERGY:
            energy = event.data.get("energy", 0)
            self._dance_energy = 0.7 * self._dance_energy + 0.3 * energy
            self._update_vibe()

        elif event.event_type == EventType.OCCUPANCY_COUNT:
            self._occupancy = event.data.get("count", 0)
            self._update_vibe()

        elif event.event_type == EventType.ZONE_OVERCROWDED:
            logger.warning("ALERT: Zone overcrowded — %s", event.data)

    def _update_vibe(self) -> None:
        occ_factor = min(self._occupancy / max(self._capacity, 1), 1.0) * 40
        dance_factor = self._dance_energy * 0.4
        base = 20
        self._vibe_score = min(base + occ_factor + dance_factor, 100.0)

    @property
    def vibe_score(self) -> float:
        return round(self._vibe_score, 1)

    @property
    def vibe_label(self) -> str:
        if self._vibe_score >= 80:
            return "LIT"
        elif self._vibe_score >= 60:
            return "VIBING"
        elif self._vibe_score >= 40:
            return "WARMING UP"
        elif self._vibe_score >= 20:
            return "CHILL"
        return "DEAD"

    def get_state(self) -> dict:
        return {
            "vibe_score": self.vibe_score,
            "vibe_label": self.vibe_label,
            "dance_energy": round(self._dance_energy, 1),
            "occupancy": self._occupancy,
            "capacity": self._capacity,
            "occupancy_pct": round(self._occupancy / max(self._capacity, 1) * 100, 1),
        }
