"""Frame Router / Orchestrator — receives frames, routes to enabled modules, manages pipeline."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from spine.core.adapter import ProductAdapter
from spine.core.config import CameraConfig, SpineConfig
from spine.core.event_bus import EventBus
from spine.core.events import DetectionEvent, Detection, EventType, SpineEvent

logger = logging.getLogger(__name__)


class FrameContext:
    """Shared context dict passed through pipeline for one frame."""

    def __init__(self, frame: np.ndarray, camera_id: str, frame_id: str, timestamp: float):
        self.frame = frame
        self.camera_id = camera_id
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.detections: list[dict[str, Any]] = []
        self.tracks: list[dict[str, Any]] = []
        self.keypoints: list[dict[str, Any]] = []
        self.reid_matches: list[dict[str, Any]] = []
        self.face_results: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {}


class ModuleBase:
    """Base class for spine modules."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._initialized = False

    def initialize(self) -> None:
        self._initialized = True

    def process(self, ctx: FrameContext) -> None:
        raise NotImplementedError

    def cleanup(self) -> None:
        pass


class Orchestrator:
    """Main pipeline orchestrator. Routes frames to enabled modules per camera."""

    def __init__(self, config: SpineConfig, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self._modules: dict[str, dict[str, ModuleBase]] = {}
        self._adapters: dict[str, ProductAdapter] = {}
        self._running = False
        self._frame_count = 0
        self._lock = threading.Lock()

    def register_adapter(self, adapter: ProductAdapter) -> None:
        self._adapters[adapter.product_id] = adapter
        cameras = adapter.get_camera_configs()
        zones = adapter.get_zone_configs()
        lines = adapter.get_line_configs()

        for cam in cameras:
            self.config.cameras.append(cam)
        for zone in zones:
            self.config.zones.append(zone)
        for line in lines:
            self.config.lines.append(line)

        self.event_bus.subscribe(
            f"cvspine/{adapter.product_id}/#",
            adapter.on_event,
        )
        logger.info("Registered adapter: %s (%d cameras)", adapter.product_id, len(cameras))

    def register_module(self, camera_id: str, name: str, module: ModuleBase) -> None:
        if camera_id not in self._modules:
            self._modules[camera_id] = {}
        self._modules[camera_id][name] = module

    def initialize_modules(self) -> None:
        for camera_id, modules in self._modules.items():
            for name, module in modules.items():
                try:
                    module.initialize()
                    logger.info("Module initialized: %s/%s", camera_id, name)
                except Exception:
                    logger.exception("Failed to init module %s/%s", camera_id, name)

    def process_frame(self, frame: np.ndarray, camera_id: str) -> FrameContext:
        self._frame_count += 1
        frame_id = f"f-{camera_id}-{self._frame_count}"
        ctx = FrameContext(
            frame=frame,
            camera_id=camera_id,
            frame_id=frame_id,
            timestamp=time.time(),
        )

        modules = self._modules.get(camera_id, {})

        if "person_detector" in modules:
            modules["person_detector"].process(ctx)

        if "tracker" in modules and ctx.detections:
            modules["tracker"].process(ctx)

        parallel_modules = ["pose_keypoints", "reid", "face"]
        for mod_name in parallel_modules:
            if mod_name in modules and ctx.detections:
                modules[mod_name].process(ctx)

        if "entry_exit" in modules and ctx.tracks:
            modules["entry_exit"].process(ctx)

        if "roi_zones" in modules:
            modules["roi_zones"].process(ctx)

        return ctx

    def start(self) -> None:
        self._running = True
        self.initialize_modules()
        logger.info("Orchestrator started with %d cameras", len(self._modules))

    def stop(self) -> None:
        self._running = False
        for camera_id, modules in self._modules.items():
            for name, module in modules.items():
                module.cleanup()
        logger.info("Orchestrator stopped")

    @property
    def is_running(self) -> bool:
        return self._running
