"""Spine event system — typed events published to event bus."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    PERSON_DETECTED = "person.detected"
    POSE_KEYPOINTS = "pose.keypoints"
    POSE_FALL_DETECTED = "pose.fall_detected"
    POSE_DANCE_ENERGY = "pose.dance_energy"
    POSE_AGGRESSION = "pose.aggression_detected"
    POSE_LOITERING = "pose.loitering"
    REID_MATCH = "reid.match"
    REID_NEW_PERSON = "reid.new_person"
    REID_CROSS_CAMERA = "reid.cross_camera"
    LINE_CROSSED = "line.crossed"
    ZONE_ENTERED = "zone.entered"
    ZONE_EXITED = "zone.exited"
    ZONE_OVERCROWDED = "zone.overcrowded"
    ZONE_UNAUTHORIZED = "zone.unauthorized_entry"
    FACE_DETECTED = "face.detected"
    FACE_RECOGNIZED = "face.recognized"
    FACE_UNKNOWN = "face.unknown"
    FACE_LIVENESS = "face.liveness"
    FACE_DEMOGRAPHICS = "face.demographics"
    TRACK_CREATED = "track.created"
    TRACK_LOST = "track.lost"
    DWELL_EXCEEDED = "dwell.exceeded"
    OCCUPANCY_COUNT = "occupancy.count"
    ANOMALY_DETECTED = "anomaly.detected"


class Detection(BaseModel):
    track_id: int
    bbox: list[int]
    bbox_norm: list[float] = Field(default_factory=list)
    confidence: float
    class_name: str = "person"
    embedding_ref: str | None = None


class SpineEvent(BaseModel):
    event_type: EventType
    camera_id: str
    timestamp: float = Field(default_factory=time.time)
    frame_id: str | None = None
    track_id: int | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    product_id: str | None = None

    def topic(self) -> str:
        base = f"cvspine/{self.product_id or 'global'}/{self.camera_id}"
        category = self.event_type.value.split(".")[0]
        return f"{base}/{category}"


class DetectionEvent(SpineEvent):
    event_type: EventType = EventType.PERSON_DETECTED
    detections: list[Detection] = Field(default_factory=list)
    person_count: int = 0
    inference_ms: float = 0.0
    frame_w: int = 0
    frame_h: int = 0


class PoseEvent(SpineEvent):
    event_type: EventType = EventType.POSE_KEYPOINTS
    keypoints: list[dict[str, Any]] = Field(default_factory=list)
    body_angle: float = 0.0
    arm_raise: str = "none"
    is_standing: bool = True
    movement_magnitude: float = 0.0
    pose_class: str = ""


class LineCrossEvent(SpineEvent):
    event_type: EventType = EventType.LINE_CROSSED
    line_id: str = ""
    direction: str = ""


class ZoneEvent(SpineEvent):
    zone_id: str = ""
    person_count: int = 0


class ReIDEvent(SpineEvent):
    event_type: EventType = EventType.REID_MATCH
    gallery_id: str = ""
    similarity: float = 0.0


class FaceEvent(SpineEvent):
    event_type: EventType = EventType.FACE_RECOGNIZED
    gallery_id: str = ""
    person_name: str = ""
    similarity: float = 0.0
    landmarks: list[dict[str, float]] = Field(default_factory=list)
