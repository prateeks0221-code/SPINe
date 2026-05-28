"""Face recognition — ArcFace embedding + gallery matching. Pillar 6."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from spine.core.config import FaceConfig
from spine.core.events import EventType, FaceEvent
from spine.core.orchestrator import FrameContext, ModuleBase

logger = logging.getLogger(__name__)


class FaceRecognizer(ModuleBase):
    """ArcFace-based face recognition with gallery matching."""

    def __init__(self, config: dict[str, Any] | None = None, event_bus: Any = None):
        super().__init__(config)
        self.face_config = FaceConfig(**(config or {}))
        self.event_bus = event_bus
        self._model = None
        self._galleries: dict[str, dict[str, np.ndarray]] = {}

    def initialize(self) -> None:
        try:
            from insightface.app import FaceAnalysis
            self._model = FaceAnalysis(
                name="buffalo_l",
                allowed_modules=["recognition"],
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._model.prepare(ctx_id=0)
            logger.info("ArcFace recognizer ready")
        except ImportError:
            logger.warning("insightface not available, face recognition disabled")

        self._initialized = True

    def process(self, ctx: FrameContext) -> None:
        if not self._initialized or not self.face_config.enabled:
            return

        for face_result in ctx.face_results:
            embedding = face_result.get("embedding")
            if embedding is None:
                continue

            embedding = np.array(embedding).flatten()
            match_name, match_gallery, similarity = self._search_galleries(embedding)

            track_id = face_result.get("track_id", -1)

            if match_name:
                if self.event_bus:
                    self.event_bus.publish(FaceEvent(
                        event_type=EventType.FACE_RECOGNIZED,
                        camera_id=ctx.camera_id,
                        track_id=track_id,
                        timestamp=ctx.timestamp,
                        gallery_id=match_gallery,
                        person_name=match_name,
                        similarity=similarity,
                    ))
            else:
                if self.event_bus:
                    self.event_bus.publish(FaceEvent(
                        event_type=EventType.FACE_UNKNOWN,
                        camera_id=ctx.camera_id,
                        track_id=track_id,
                        timestamp=ctx.timestamp,
                        data={"suggested_action": "enroll_or_alert"},
                    ))

    def _search_galleries(self, embedding: np.ndarray) -> tuple[str | None, str, float]:
        best_name = None
        best_gallery = ""
        best_sim = 0.0

        for gallery_name, entries in self._galleries.items():
            for person_name, stored_emb in entries.items():
                sim = float(np.dot(embedding, stored_emb) /
                           (np.linalg.norm(embedding) * np.linalg.norm(stored_emb) + 1e-8))
                if sim > best_sim:
                    best_sim = sim
                    best_name = person_name
                    best_gallery = gallery_name

        if best_sim >= 0.6:
            return best_name, best_gallery, best_sim
        return None, "", best_sim

    def enroll(self, gallery_id: str, person_name: str, embedding: np.ndarray) -> None:
        if gallery_id not in self._galleries:
            self._galleries[gallery_id] = {}
        self._galleries[gallery_id][person_name] = embedding / (np.linalg.norm(embedding) + 1e-8)

    def remove(self, gallery_id: str, person_name: str) -> None:
        if gallery_id in self._galleries:
            self._galleries[gallery_id].pop(person_name, None)

    def cleanup(self) -> None:
        self._model = None
        self._galleries.clear()
