"""Pose Keypoints module — YOLO-pose / ViTPose wrapper. Pillar 2."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from spine.core.config import PoseConfig
from spine.core.events import EventType, PoseEvent
from spine.core.orchestrator import FrameContext, ModuleBase

logger = logging.getLogger(__name__)

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


class PoseEstimator(ModuleBase):
    """YOLO-pose keypoint estimator. Emits pose.keypoints events."""

    def __init__(self, config: dict[str, Any] | None = None, event_bus: Any = None):
        super().__init__(config)
        self.pose_config = PoseConfig(**(config or {}))
        self.event_bus = event_bus
        self._model = None
        self._frame_interval = 1.0 / max(self.pose_config.fps, 0.1)
        self._last_process_time = 0.0

    def initialize(self) -> None:
        from ultralytics import YOLO

        model_name = self.pose_config.model
        if not model_name.endswith(".pt"):
            model_name += ".pt"

        self._model = YOLO(model_name)
        self._initialized = True
        logger.info("PoseEstimator ready (model=%s)", self.pose_config.model)

    def process(self, ctx: FrameContext) -> None:
        if not self._initialized or self._model is None:
            return

        now = time.time()
        if now - self._last_process_time < self._frame_interval:
            return
        self._last_process_time = now

        results = self._model.predict(ctx.frame, verbose=False)

        if not results or results[0].keypoints is None:
            return

        kpts_data = results[0].keypoints
        if kpts_data.xy is None:
            return

        h, w = ctx.frame.shape[:2]
        all_keypoints = kpts_data.xy.cpu().numpy()
        all_confs = kpts_data.conf.cpu().numpy() if kpts_data.conf is not None else None

        for person_idx in range(len(all_keypoints)):
            kpts = all_keypoints[person_idx]
            confs = all_confs[person_idx] if all_confs is not None else np.ones(len(kpts))

            keypoints_list = []
            for kp_idx, name in enumerate(COCO_KEYPOINTS):
                if kp_idx < len(kpts):
                    keypoints_list.append({
                        "name": name,
                        "x": float(kpts[kp_idx][0] / w),
                        "y": float(kpts[kp_idx][1] / h),
                        "conf": float(confs[kp_idx]),
                    })

            body_angle = self._compute_body_angle(keypoints_list)
            arm_raise = self._classify_arm_raise(keypoints_list)
            is_standing = body_angle < 45
            movement_mag = 0.0

            track_id = -1
            if person_idx < len(ctx.detections):
                track_id = ctx.detections[person_idx].get("track_id", -1)

            ctx.keypoints.append({
                "track_id": track_id,
                "keypoints": keypoints_list,
                "body_angle": body_angle,
                "arm_raise": arm_raise,
                "is_standing": is_standing,
            })

            if self.event_bus:
                event = PoseEvent(
                    camera_id=ctx.camera_id,
                    timestamp=ctx.timestamp,
                    track_id=track_id,
                    keypoints=keypoints_list,
                    body_angle=body_angle,
                    arm_raise=arm_raise,
                    is_standing=is_standing,
                    movement_magnitude=movement_mag,
                )
                self.event_bus.publish(event)

    def _compute_body_angle(self, kpts: list[dict]) -> float:
        ls = next((k for k in kpts if k["name"] == "left_shoulder"), None)
        rs = next((k for k in kpts if k["name"] == "right_shoulder"), None)
        lh = next((k for k in kpts if k["name"] == "left_hip"), None)
        rh = next((k for k in kpts if k["name"] == "right_hip"), None)

        if not all([ls, rs, lh, rh]):
            return 0.0

        mid_shoulder_y = (ls["y"] + rs["y"]) / 2
        mid_hip_y = (lh["y"] + rh["y"]) / 2
        mid_shoulder_x = (ls["x"] + rs["x"]) / 2
        mid_hip_x = (lh["x"] + rh["x"]) / 2

        dy = mid_hip_y - mid_shoulder_y
        dx = mid_hip_x - mid_shoulder_x

        if abs(dy) < 1e-6:
            return 90.0

        angle = abs(np.degrees(np.arctan2(dx, dy)))
        return float(angle)

    def _classify_arm_raise(self, kpts: list[dict]) -> str:
        ls = next((k for k in kpts if k["name"] == "left_shoulder"), None)
        rs = next((k for k in kpts if k["name"] == "right_shoulder"), None)
        lw = next((k for k in kpts if k["name"] == "left_wrist"), None)
        rw = next((k for k in kpts if k["name"] == "right_wrist"), None)

        if not all([ls, rs, lw, rw]):
            return "none"

        left_raised = lw["y"] < ls["y"] and lw["conf"] > 0.3
        right_raised = rw["y"] < rs["y"] and rw["conf"] > 0.3

        if left_raised and right_raised:
            return "both"
        if left_raised:
            return "left"
        if right_raised:
            return "right"
        return "none"

    def cleanup(self) -> None:
        self._model = None
