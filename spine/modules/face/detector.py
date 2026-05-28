"""Face detection — SCRFD model. Pillar 6."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from spine.core.orchestrator import FrameContext, ModuleBase

logger = logging.getLogger(__name__)


class FaceDetector(ModuleBase):
    """SCRFD-based face detection. Handles angles, occlusion, masks."""

    def __init__(self, config: dict[str, Any] | None = None, event_bus: Any = None):
        super().__init__(config)
        self.event_bus = event_bus
        self._model = None
        self._min_face_size = config.get("min_face_size", 20) if config else 20

    def initialize(self) -> None:
        try:
            from insightface.app import FaceAnalysis
            self._model = FaceAnalysis(
                name="buffalo_sc",
                allowed_modules=["detection"],
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._model.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("SCRFD face detector ready")
        except ImportError:
            logger.warning("insightface not available, face detection disabled")
            self._model = None

        self._initialized = True

    def process(self, ctx: FrameContext) -> None:
        if not self._initialized or self._model is None:
            return

        faces = self._model.get(ctx.frame)
        ctx.face_results = []

        for face in faces:
            bbox = face.bbox.astype(int).tolist()
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w < self._min_face_size or h < self._min_face_size:
                continue

            result = {
                "bbox": bbox,
                "confidence": float(face.det_score),
                "landmarks": face.kps.tolist() if face.kps is not None else [],
            }

            if hasattr(face, "embedding") and face.embedding is not None:
                result["embedding"] = face.embedding

            ctx.face_results.append(result)

    def cleanup(self) -> None:
        self._model = None
