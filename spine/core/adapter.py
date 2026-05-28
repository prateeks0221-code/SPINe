"""ProductAdapter ABC — every product implements this. Spine calls these methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from spine.core.config import CameraConfig, GalleryConfig, LineConfig, ZoneConfig
from spine.core.events import SpineEvent


class ProductAdapter(ABC):
    """
    Every product implements this. Spine calls these methods.
    Product never touches spine internals — only receives events and provides config.
    """

    @property
    @abstractmethod
    def product_id(self) -> str:
        ...

    @abstractmethod
    def get_camera_configs(self) -> list[CameraConfig]:
        ...

    @abstractmethod
    def get_zone_configs(self) -> list[ZoneConfig]:
        ...

    @abstractmethod
    def get_line_configs(self) -> list[LineConfig]:
        ...

    @abstractmethod
    def get_face_galleries(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def on_event(self, event: SpineEvent) -> None:
        ...

    def get_alert_rules(self) -> list[dict]:
        return []

    def get_analytics_config(self) -> dict:
        return {"persist_events": True, "aggregation_interval": 60}
