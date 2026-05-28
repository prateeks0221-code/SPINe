"""Spatial recommendation engine — flow analysis, dead zones, bottlenecks."""

from __future__ import annotations

import numpy as np
from collections import deque


class SpatialRecommender:
    """Analyzes movement patterns to generate spatial insights."""

    def __init__(self, frame_w: int = 1280, frame_h: int = 720, history_size: int = 1000):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self._paths: deque = deque(maxlen=history_size)
        self._flow_vectors: list[tuple[float, float]] = []

    def add_track_point(self, track_id: int, x: float, y: float) -> None:
        self._paths.append({"track_id": track_id, "x": x, "y": y})

    def add_flow(self, dx: float, dy: float) -> None:
        self._flow_vectors.append((dx, dy))
        if len(self._flow_vectors) > 5000:
            self._flow_vectors = self._flow_vectors[-2500:]

    def get_dominant_flow(self) -> dict | None:
        if len(self._flow_vectors) < 50:
            return None
        vectors = np.array(self._flow_vectors)
        mean_flow = vectors.mean(axis=0)
        magnitude = float(np.linalg.norm(mean_flow))
        if magnitude < 0.01:
            return None
        return {
            "type": "high_traffic_path",
            "flow_vector": mean_flow.tolist(),
            "magnitude": magnitude,
            "confidence": min(magnitude * 5, 1.0),
        }

    def get_bottlenecks(self, density_grid: np.ndarray, threshold: float = 0.8) -> list[dict]:
        bottlenecks = []
        h, w = density_grid.shape
        max_val = density_grid.max()
        if max_val == 0:
            return []

        norm = density_grid / max_val
        for gy in range(h):
            for gx in range(w):
                if norm[gy, gx] >= threshold:
                    px = int((gx + 0.5) / w * self.frame_w)
                    py = int((gy + 0.5) / h * self.frame_h)
                    bottlenecks.append({
                        "type": "bottleneck",
                        "position": [px, py],
                        "density": float(norm[gy, gx]),
                    })
        return bottlenecks
