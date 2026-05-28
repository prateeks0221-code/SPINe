"""Pydantic config schemas for CV Spine — cameras, zones, lines, modules."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DeviceType(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA_0 = "cuda:0"
    CUDA_1 = "cuda:1"


class DetectionConfig(BaseModel):
    model: str = "yolov8m"
    confidence: float = 0.35
    nms_iou: float = 0.45
    classes: list[int] = Field(default_factory=lambda: [0])
    imgsz: int = 640
    device: str = "auto"
    half: bool = True
    max_detections: int = 100
    min_pixel_area: int = 1200
    aspect_ratio: tuple[float, float] = (0.5, 5.0)


class PoseConfig(BaseModel):
    model: str = "yolov8m-pose"
    fps: float = 5.0
    min_keypoint_conf: float = 0.3
    activity_classifiers: list[str] = Field(
        default_factory=lambda: ["fall", "dance_energy", "aggression"]
    )


class ReIDConfig(BaseModel):
    enabled: bool = True
    model: str = "osnet_x1_0"
    embedding_dim: int = 512
    similarity_threshold: float = 0.75
    gallery_ttl: int = 14400


class FaceConfig(BaseModel):
    enabled: bool = False
    anonymize_before_stream: bool = True
    embedding_storage: str = "volatile"
    embedding_ttl: int = 3600
    gallery_consent_required: bool = True
    gdpr_mode: bool = True


class EntryExitConfig(BaseModel):
    enabled: bool = True
    tracker: str = "bytetrack"


class ROIConfig(BaseModel):
    enabled: bool = True


class ModulesConfig(BaseModel):
    person_detector: DetectionConfig = Field(default_factory=DetectionConfig)
    pose_keypoints: PoseConfig | None = None
    reid: ReIDConfig | None = None
    face: FaceConfig | None = None
    entry_exit: EntryExitConfig = Field(default_factory=EntryExitConfig)
    roi_zones: ROIConfig = Field(default_factory=ROIConfig)


class CameraConfig(BaseModel):
    camera_id: str
    source: str
    modules: ModulesConfig = Field(default_factory=ModulesConfig)
    stream_to_browser: bool = False
    anonymize: bool = True
    target_fps: float = 10.0
    resolution: tuple[int, int] | None = None


class ZoneConfig(BaseModel):
    id: str
    camera_id: str
    type: str = "polygon"
    points: list[list[int]]
    alerts: list[dict[str, Any]] = Field(default_factory=list)
    analytics: dict[str, Any] = Field(default_factory=dict)
    whitelist_gallery: str | None = None


class LineConfig(BaseModel):
    id: str
    camera_id: str
    type: str = "entry_exit"
    points: list[list[int]]
    direction_in: str = "up"
    alert_on: str | None = None


class GalleryConfig(BaseModel):
    id: str
    type: str = "session"
    ttl: int = 86400
    auto_purge: bool = True
    consent: str | None = None


class SpineConfig(BaseModel):
    cameras: list[CameraConfig] = Field(default_factory=list)
    zones: list[ZoneConfig] = Field(default_factory=list)
    lines: list[LineConfig] = Field(default_factory=list)
    galleries: list[GalleryConfig] = Field(default_factory=list)
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    redis_url: str = "redis://localhost:6379"
    db_url: str = "postgresql://localhost:5432/cvspine"
    gpu_device: str = "auto"
