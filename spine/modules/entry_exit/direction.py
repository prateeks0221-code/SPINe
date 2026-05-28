"""Direction classifier — determines entry vs exit from track velocity vector."""

from __future__ import annotations

import numpy as np


def classify_direction(prev_pos: tuple[float, float], curr_pos: tuple[float, float],
                       line_p1: list[int], line_p2: list[int], direction_in: str = "up") -> str:
    dx = curr_pos[0] - prev_pos[0]
    dy = curr_pos[1] - prev_pos[1]

    if direction_in == "up":
        return "in" if dy < 0 else "out"
    elif direction_in == "down":
        return "in" if dy > 0 else "out"
    elif direction_in == "left":
        return "in" if dx < 0 else "out"
    elif direction_in == "right":
        return "in" if dx > 0 else "out"

    line_dx = line_p2[0] - line_p1[0]
    line_dy = line_p2[1] - line_p1[1]
    normal = np.array([-line_dy, line_dx], dtype=np.float64)
    normal /= np.linalg.norm(normal) + 1e-8

    velocity = np.array([dx, dy], dtype=np.float64)
    dot = float(np.dot(velocity, normal))
    return "in" if dot > 0 else "out"
