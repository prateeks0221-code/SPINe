"""Person Detection module — YOLOv8/v10 wrapper. Pillar 1."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from spine.core.config import DetectionConfig
from spine.core.events import Detection, DetectionEvent, EventType
from spine.core.orchestrator import FrameContext, ModuleBase

logger = logging.getLogger(__name__)


class PersonDetector(ModuleBase):
    """YOLO-based person detector. Emits person.detected events."""

    def __init__(self, config: dict[str, Any] | None = None, event_bus: Any = None):
        super().__init__(config)
        self.det_config = DetectionConfig(**(config or {}))
        self.event_bus = event_bus
        self._model = None

    def initialize(self) -> None:
        from ultralytics import YOLO

        model_name = self.det_config.model
        if not model_name.endswith(".pt"):
            model_name += ".pt"

        logger.info("Loading detection model: %s", model_name)
        self._model = YOLO(model_name)

        if self.det_config.device != "cpu" and self.det_config.half:
            self._model.fuse()

        self._initialized = True
        logger.info("PersonDetector ready (model=%s, device=%s)", self.det_config.model, self.det_config.device)

    def process(self, ctx: FrameContext) -> None:
        if not self._initialized or self._model is None:
            return

        t0 = time.perf_counter()

        results = self._model.predict(
            ctx.frame,
            conf=self.det_config.confidence,
            iou=self.det_config.nms_iou,
            classes=self.det_config.classes,
            imgsz=self.det_config.imgsz,
            device=self.det_config.device if self.det_config.device != "auto" else None,
            half=self.det_config.half,
            max_det=self.det_config.max_detections,
            verbose=False,
        )

        inference_ms = (time.perf_counter() - t0) * 1000
        h, w = ctx.frame.shape[:2]
        detections: list[Detection] = []

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                conf = float(boxes.conf[i].cpu())

                bw = bbox[2] - bbox[0]
                bh = bbox[3] - bbox[1]
                area = bw * bh
                aspect = bh / max(bw, 1)

                if area < self.det_config.min_pixel_area:
                    continue
                if not (self.det_config.aspect_ratio[0] <= aspect <= self.det_config.aspect_ratio[1]):
                    continue

                det = Detection(
                    track_id=-1,
                    bbox=bbox,
                    bbox_norm=[bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h],
                    confidence=conf,
                    class_name="person",
                )
                detections.append(det)

        ctx.detections = [d.model_dump() for d in detections]
        ctx.metadata["inference_ms"] = inference_ms
        ctx.metadata["person_count"] = len(detections)

        if self.event_bus and detections:
            event = DetectionEvent(
                camera_id=ctx.camera_id,
                timestamp=ctx.timestamp,
                frame_id=ctx.frame_id,
                detections=detections,
                person_count=len(detections),
                inference_ms=inference_ms,
                frame_w=w,
                frame_h=h,
            )
            self.event_bus.publish(event)

    def cleanup(self) -> None:
        self._model = None
        self._initialized = False
