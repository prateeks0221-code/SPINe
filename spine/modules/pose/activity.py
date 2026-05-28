"""Activity classifiers — fall, dance energy, fight detection from pose data."""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

import numpy as np

from spine.core.events import EventType, SpineEvent
from spine.core.orchestrator import FrameContext, ModuleBase

logger = logging.getLogger(__name__)


class ActivityClassifier(ModuleBase):
    """Derives activity signals from pose keypoints."""

    def __init__(self, config: dict[str, Any] | None = None, event_bus: Any = None):
        super().__init__(config)
        self.event_bus = event_bus
        self._history: dict[int, deque] = {}
        self._loiter_start: dict[int, float] = {}
        self._loiter_threshold = config.get("loiter_seconds", 120) if config else 120

    def initialize(self) -> None:
        self._initialized = True

    def process(self, ctx: FrameContext) -> None:
        if not ctx.keypoints:
            return

        for kp_data in ctx.keypoints:
            track_id = kp_data.get("track_id", -1)
            if track_id < 0:
                continue

            if track_id not in self._history:
                self._history[track_id] = deque(maxlen=30)
            self._history[track_id].append(kp_data)

            self._check_fall(ctx, kp_data, track_id)
            self._check_loitering(ctx, kp_data, track_id)
            self._compute_dance_energy(ctx, kp_data, track_id)

        self._check_aggression(ctx)

    def _check_fall(self, ctx: FrameContext, kp_data: dict, track_id: int) -> None:
        body_angle = kp_data.get("body_angle", 0)
        if body_angle > 60 and not kp_data.get("is_standing", True):
            if self.event_bus:
                self.event_bus.publish(SpineEvent(
                    event_type=EventType.POSE_FALL_DETECTED,
                    camera_id=ctx.camera_id,
                    track_id=track_id,
                    timestamp=ctx.timestamp,
                    data={"body_angle": body_angle},
                ))

    def _check_loitering(self, ctx: FrameContext, kp_data: dict, track_id: int) -> None:
        history = self._history.get(track_id)
        if not history or len(history) < 10:
            return

        recent = list(history)[-10:]
        positions = []
        for h in recent:
            kpts = h.get("keypoints", [])
            hip = next((k for k in kpts if k["name"] == "left_hip"), None)
            if hip:
                positions.append((hip["x"], hip["y"]))

        if len(positions) < 5:
            return

        movement = sum(
            np.sqrt((positions[i][0] - positions[i-1][0])**2 + (positions[i][1] - positions[i-1][1])**2)
            for i in range(1, len(positions))
        )

        if movement < 0.05:
            if track_id not in self._loiter_start:
                self._loiter_start[track_id] = ctx.timestamp
            elif ctx.timestamp - self._loiter_start[track_id] > self._loiter_threshold:
                if self.event_bus:
                    self.event_bus.publish(SpineEvent(
                        event_type=EventType.POSE_LOITERING,
                        camera_id=ctx.camera_id,
                        track_id=track_id,
                        timestamp=ctx.timestamp,
                        data={"duration": ctx.timestamp - self._loiter_start[track_id]},
                    ))
                self._loiter_start[track_id] = ctx.timestamp
        else:
            self._loiter_start.pop(track_id, None)

    def _compute_dance_energy(self, ctx: FrameContext, kp_data: dict, track_id: int) -> None:
        history = self._history.get(track_id)
        if not history or len(history) < 5:
            return

        recent = list(history)[-5:]
        velocities = []
        for i in range(1, len(recent)):
            prev_kpts = {k["name"]: k for k in recent[i-1].get("keypoints", [])}
            curr_kpts = {k["name"]: k for k in recent[i].get("keypoints", [])}
            for joint in ["left_wrist", "right_wrist", "left_ankle", "right_ankle"]:
                if joint in prev_kpts and joint in curr_kpts:
                    dx = curr_kpts[joint]["x"] - prev_kpts[joint]["x"]
                    dy = curr_kpts[joint]["y"] - prev_kpts[joint]["y"]
                    velocities.append(np.sqrt(dx**2 + dy**2))

        if velocities:
            energy = min(float(np.mean(velocities) * 200), 100.0)
            if energy > 20 and self.event_bus:
                self.event_bus.publish(SpineEvent(
                    event_type=EventType.POSE_DANCE_ENERGY,
                    camera_id=ctx.camera_id,
                    track_id=track_id,
                    timestamp=ctx.timestamp,
                    data={"energy": energy},
                ))

    def _check_aggression(self, ctx: FrameContext) -> None:
        if len(ctx.detections) < 2:
            return

        for i in range(len(ctx.detections)):
            for j in range(i + 1, len(ctx.detections)):
                bbox_i = ctx.detections[i].get("bbox", [0, 0, 0, 0])
                bbox_j = ctx.detections[j].get("bbox", [0, 0, 0, 0])

                cx_i = (bbox_i[0] + bbox_i[2]) / 2
                cx_j = (bbox_j[0] + bbox_j[2]) / 2
                cy_i = (bbox_i[1] + bbox_i[3]) / 2
                cy_j = (bbox_j[1] + bbox_j[3]) / 2

                dist = np.sqrt((cx_i - cx_j)**2 + (cy_i - cy_j)**2)
                avg_width = ((bbox_i[2] - bbox_i[0]) + (bbox_j[2] - bbox_j[0])) / 2

                if dist < avg_width * 1.5:
                    tid_i = ctx.detections[i].get("track_id", -1)
                    tid_j = ctx.detections[j].get("track_id", -1)

                    hist_i = self._history.get(tid_i)
                    hist_j = self._history.get(tid_j)
                    if hist_i and hist_j and len(hist_i) > 3 and len(hist_j) > 3:
                        arm_raise_i = hist_i[-1].get("arm_raise", "none")
                        arm_raise_j = hist_j[-1].get("arm_raise", "none")
                        if arm_raise_i != "none" and arm_raise_j != "none":
                            if self.event_bus:
                                self.event_bus.publish(SpineEvent(
                                    event_type=EventType.POSE_AGGRESSION,
                                    camera_id=ctx.camera_id,
                                    timestamp=ctx.timestamp,
                                    data={"track_ids": [tid_i, tid_j], "distance": dist},
                                ))
