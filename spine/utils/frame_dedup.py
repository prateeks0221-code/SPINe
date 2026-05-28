"""Frame deduplication — pHash-based. Skip inference on static scenes."""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameDeduplicator:
    """Skips inference when frame hasn't changed significantly (static cameras)."""

    def __init__(self, threshold: int = 10, hash_size: int = 16):
        self.threshold = threshold
        self.hash_size = hash_size
        self._last_hash: dict[str, np.ndarray] = {}
        self._skip_count: dict[str, int] = {}
        self._process_count: dict[str, int] = {}

    def should_process(self, frame: np.ndarray, camera_id: str) -> bool:
        current_hash = self._compute_phash(frame)

        if camera_id not in self._last_hash:
            self._last_hash[camera_id] = current_hash
            self._skip_count[camera_id] = 0
            self._process_count[camera_id] = 1
            return True

        distance = self._hamming_distance(current_hash, self._last_hash[camera_id])

        if distance < self.threshold:
            self._skip_count[camera_id] = self._skip_count.get(camera_id, 0) + 1
            return False

        self._last_hash[camera_id] = current_hash
        self._process_count[camera_id] = self._process_count.get(camera_id, 0) + 1
        return True

    def get_stats(self, camera_id: str) -> dict:
        skipped = self._skip_count.get(camera_id, 0)
        processed = self._process_count.get(camera_id, 0)
        total = skipped + processed
        return {
            "camera_id": camera_id,
            "skipped": skipped,
            "processed": processed,
            "skip_rate": skipped / max(total, 1),
        }

    def _compute_phash(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        resized = cv2.resize(gray, (self.hash_size, self.hash_size), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(resized.astype(np.float32))
        dct_low = dct[:8, :8]
        median = np.median(dct_low)
        return (dct_low > median).flatten().astype(np.uint8)

    @staticmethod
    def _hamming_distance(h1: np.ndarray, h2: np.ndarray) -> int:
        return int(np.sum(h1 != h2))
