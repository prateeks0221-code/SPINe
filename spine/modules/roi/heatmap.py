"""Density grid / heatmap accumulator for spatial analytics."""

from __future__ import annotations

import numpy as np


class HeatmapAccumulator:
    """Accumulates person positions into NxM grid for density visualization."""

    def __init__(self, grid_w: int = 32, grid_h: int = 18, frame_w: int = 1280, frame_h: int = 720):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.frame_w = frame_w
        self.frame_h = frame_h
        self._grid = np.zeros((grid_h, grid_w), dtype=np.float64)
        self._total_frames = 0

    def update(self, detections: list[dict]) -> None:
        self._total_frames += 1
        for det in detections:
            bbox = det.get("bbox", [0, 0, 0, 0])
            cx = (bbox[0] + bbox[2]) / 2
            cy = bbox[3]  # foot point

            gx = int(cx / self.frame_w * self.grid_w)
            gy = int(cy / self.frame_h * self.grid_h)

            gx = min(max(gx, 0), self.grid_w - 1)
            gy = min(max(gy, 0), self.grid_h - 1)

            self._grid[gy, gx] += 1

    def get_heatmap(self, normalized: bool = True) -> np.ndarray:
        if normalized and self._grid.max() > 0:
            return self._grid / self._grid.max()
        return self._grid.copy()

    def get_hotspots(self, threshold: float = 0.7) -> list[dict]:
        norm = self.get_heatmap(normalized=True)
        hotspots = []
        for gy in range(self.grid_h):
            for gx in range(self.grid_w):
                if norm[gy, gx] >= threshold:
                    hotspots.append({
                        "grid_cell": [gx, gy],
                        "intensity": float(norm[gy, gx]),
                        "pixel_center": [
                            int((gx + 0.5) / self.grid_w * self.frame_w),
                            int((gy + 0.5) / self.grid_h * self.frame_h),
                        ],
                    })
        return hotspots

    def get_dead_zones(self, threshold: float = 0.05) -> list[dict]:
        norm = self.get_heatmap(normalized=True)
        dead = []
        for gy in range(self.grid_h):
            for gx in range(self.grid_w):
                if norm[gy, gx] <= threshold:
                    dead.append({"grid_cell": [gx, gy], "intensity": float(norm[gy, gx])})
        return dead

    def reset(self) -> None:
        self._grid.fill(0)
        self._total_frames = 0
