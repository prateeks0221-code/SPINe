"""CV Spine main entry point — initializes pipeline, loads adapters, runs loop."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

import yaml

from spine.core.config import SpineConfig
from spine.core.event_bus import EventBus, MQTTEventBus
from spine.core.orchestrator import Orchestrator
from spine.utils.frame_dedup import FrameDeduplicator
from spine.utils.frame_grabber import FrameGrabber
from spine.utils.health import HealthChecker
from spine.utils.metrics import start_metrics_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cv-spine")


class SpineRunner:
    """Main runner — manages cameras, orchestrator, and processing loop."""

    def __init__(self, config: SpineConfig):
        self.config = config
        self.event_bus = EventBus()
        self.orchestrator = Orchestrator(config, self.event_bus)
        self.health = HealthChecker()
        self.dedup = FrameDeduplicator()
        self._grabbers: dict[str, FrameGrabber] = {}
        self._running = False

    def setup_infrastructure(self) -> None:
        try:
            self.event_bus = MQTTEventBus(
                broker=self.config.mqtt_broker,
                port=self.config.mqtt_port,
            )
            self.orchestrator.event_bus = self.event_bus
        except Exception:
            logger.warning("Using in-process event bus (MQTT unavailable)")

    def setup_cameras(self) -> None:
        for cam in self.config.cameras:
            grabber = FrameGrabber(
                source=cam.source,
                camera_id=cam.camera_id,
                target_fps=cam.target_fps,
                resolution=cam.resolution,
            )
            self._grabbers[cam.camera_id] = grabber
            self.health.register(f"camera:{cam.camera_id}", grabber.get_health)

    def setup_modules(self) -> None:
        from spine.modules.detection.detector import PersonDetector
        from spine.modules.detection.tracker import MultiObjectTracker

        for cam in self.config.cameras:
            modules_cfg = cam.modules

            det_config = modules_cfg.person_detector.model_dump()
            self.orchestrator.register_module(
                cam.camera_id, "person_detector",
                PersonDetector(config=det_config, event_bus=self.event_bus),
            )

            tracker_config = modules_cfg.entry_exit.model_dump() if modules_cfg.entry_exit else {}
            self.orchestrator.register_module(
                cam.camera_id, "tracker",
                MultiObjectTracker(config=tracker_config, event_bus=self.event_bus),
            )

            if modules_cfg.pose_keypoints:
                from spine.modules.pose.estimator import PoseEstimator
                self.orchestrator.register_module(
                    cam.camera_id, "pose_keypoints",
                    PoseEstimator(config=modules_cfg.pose_keypoints.model_dump(), event_bus=self.event_bus),
                )

            if modules_cfg.reid and modules_cfg.reid.enabled:
                from spine.modules.reid.embedder import ReIDEmbedder
                self.orchestrator.register_module(
                    cam.camera_id, "reid",
                    ReIDEmbedder(config=modules_cfg.reid.model_dump(), event_bus=self.event_bus),
                )

            if modules_cfg.face and modules_cfg.face.enabled:
                from spine.modules.face.detector import FaceDetector
                self.orchestrator.register_module(
                    cam.camera_id, "face",
                    FaceDetector(config=modules_cfg.face.model_dump(), event_bus=self.event_bus),
                )

    def start(self) -> None:
        self._running = True
        self.setup_infrastructure()
        self.setup_cameras()
        self.setup_modules()

        for grabber in self._grabbers.values():
            grabber.start()

        self.orchestrator.start()
        start_metrics_server(9090)
        logger.info("CV Spine started (%d cameras)", len(self._grabbers))

    def run_loop(self) -> None:
        while self._running:
            for camera_id, grabber in self._grabbers.items():
                frame = grabber.get_frame()
                if frame is None:
                    continue

                if not self.dedup.should_process(frame, camera_id):
                    continue

                self.orchestrator.process_frame(frame, camera_id)

            time.sleep(0.001)

    def stop(self) -> None:
        self._running = False
        for grabber in self._grabbers.values():
            grabber.stop()
        self.orchestrator.stop()
        logger.info("CV Spine stopped")


def load_config(config_path: str) -> SpineConfig:
    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f)
        # If it's a spine config (has mqtt_broker or top-level cameras with camera_id), use directly
        if data and "mqtt_broker" in data:
            return SpineConfig(**data)
    # Otherwise return empty spine config (product adapter will populate cameras)
    return SpineConfig()


def main():
    parser = argparse.ArgumentParser(description="CV Spine — Universal CV Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--product", default=None, help="Product adapter: vibecheck, schoolguard, etc.")
    parser.add_argument("--metrics-port", type=int, default=9090)
    args = parser.parse_args()

    config_path = Path(args.config)

    # Auto-detect product from venue config
    product = args.product
    if not product and config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        if "venue_name" in raw or "capacity" in raw:
            product = "vibecheck"

    config = load_config(args.config)
    runner = SpineRunner(config)

    # Register product adapter
    if product == "vibecheck":
        from products.vibecheck.adapter import VibeCheckAdapter
        adapter = VibeCheckAdapter(venue_config_path=str(config_path))
        runner.orchestrator.register_adapter(adapter)
        logger.info("Product: VibeCheck — %s", adapter._venue_config.get("venue_name", "unnamed"))

    def signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        runner.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    runner.start()
    runner.run_loop()


if __name__ == "__main__":
    main()
