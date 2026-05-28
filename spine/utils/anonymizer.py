"""Anonymizer — privacy-first frame processing. Blurs faces before any frame leaves spine."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Anonymizer:
    """Runs BEFORE any frame leaves spine (MQTT, storage, WebSocket)."""

    def __init__(self, blur_strength: int = 51, method: str = "gaussian",
                 region_expansion: float = 0.2):
        self.blur_strength = blur_strength
        self.method = method
        self.region_expansion = region_expansion
        self._face_detector = None

    def initialize_face_detector(self) -> None:
        try:
            from insightface.app import FaceAnalysis
            self._face_detector = FaceAnalysis(
                name="buffalo_sc",
                allowed_modules=["detection"],
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._face_detector.prepare(ctx_id=0, det_size=(320, 320))
            logger.info("Anonymizer face detector ready")
        except ImportError:
            logger.warning("insightface unavailable, using Haar cascade fallback")
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_detector = cv2.CascadeClassifier(cascade_path)

    def anonymize_frame(self, frame: np.ndarray, face_bboxes: list[list[int]] | None = None,
                        bypass_ids: set[str] | None = None) -> np.ndarray:
        result = frame.copy()

        if face_bboxes is None:
            face_bboxes = self._detect_faces(frame)

        for bbox in face_bboxes:
            x1, y1, x2, y2 = self._expand_bbox(bbox, frame.shape[:2])
            roi = result[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            if self.method == "gaussian":
                blurred = cv2.GaussianBlur(roi, (self.blur_strength, self.blur_strength), 0)
            elif self.method == "pixelate":
                h, w = roi.shape[:2]
                small = cv2.resize(roi, (max(w // 10, 1), max(h // 10, 1)))
                blurred = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            elif self.method == "solid":
                blurred = np.zeros_like(roi)
            else:
                blurred = cv2.GaussianBlur(roi, (self.blur_strength, self.blur_strength), 0)

            result[y1:y2, x1:x2] = blurred

        return result

    def _detect_faces(self, frame: np.ndarray) -> list[list[int]]:
        if self._face_detector is None:
            return []

        if hasattr(self._face_detector, "get"):
            faces = self._face_detector.get(frame)
            return [face.bbox.astype(int).tolist() for face in faces]
        elif hasattr(self._face_detector, "detectMultiScale"):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._face_detector.detectMultiScale(gray, 1.1, 4)
            return [[x, y, x + w, y + h] for (x, y, w, h) in faces]
        return []

    def _expand_bbox(self, bbox: list[int], frame_shape: tuple) -> tuple[int, int, int, int]:
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        exp_x = int(bw * self.region_expansion)
        exp_y = int(bh * self.region_expansion)

        x1 = max(0, x1 - exp_x)
        y1 = max(0, y1 - exp_y)
        x2 = min(w, x2 + exp_x)
        y2 = min(h, y2 + exp_y)
        return x1, y1, x2, y2
