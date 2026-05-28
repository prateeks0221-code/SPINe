"""Perspective transform — bird's-eye view mapping via homography."""

from __future__ import annotations

import numpy as np
import cv2


class HomographyTransform:
    """Maps image points to real-world coordinates via perspective transform."""

    def __init__(self, src_points: list[list[int]], dst_points: list[list[int]]):
        src = np.array(src_points, dtype=np.float32)
        dst = np.array(dst_points, dtype=np.float32)
        self._matrix, _ = cv2.findHomography(src, dst)
        self._inv_matrix, _ = cv2.findHomography(dst, src)

    def image_to_world(self, x: float, y: float) -> tuple[float, float]:
        pt = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self._matrix)
        return float(transformed[0][0][0]), float(transformed[0][0][1])

    def world_to_image(self, x: float, y: float) -> tuple[float, float]:
        pt = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self._inv_matrix)
        return float(transformed[0][0][0]), float(transformed[0][0][1])

    def get_birds_eye(self, frame: np.ndarray, output_size: tuple[int, int] = (800, 600)) -> np.ndarray:
        return cv2.warpPerspective(frame, self._matrix, output_size)

    def compute_real_distance(self, p1: tuple[float, float], p2: tuple[float, float]) -> float:
        w1 = self.image_to_world(*p1)
        w2 = self.image_to_world(*p2)
        return float(np.sqrt((w1[0] - w2[0])**2 + (w1[1] - w2[1])**2))
