"""Multi-object tracker — ByteTrack with trajectory export + Kalman smoothing."""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

import numpy as np

from spine.core.events import EventType, SpineEvent
from spine.core.orchestrator import FrameContext, ModuleBase

logger = logging.getLogger(__name__)


class TrackTrajectory:
    """Per-track trajectory with Kalman-style smoothing and velocity estimation."""

    __slots__ = ("track_id", "positions", "timestamps", "smoothed", "velocities",
                 "first_seen", "last_seen", "total_distance", "is_moving",
                 "_alpha", "_bbox_history")

    def __init__(self, track_id: int, timestamp: float, alpha: float = 0.4):
        self.track_id = track_id
        self.positions: deque[tuple[float, float]] = deque(maxlen=60)
        self.timestamps: deque[float] = deque(maxlen=60)
        self.smoothed: tuple[float, float] = (0.0, 0.0)
        self.velocities: deque[tuple[float, float]] = deque(maxlen=30)
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.total_distance = 0.0
        self.is_moving = False
        self._alpha = alpha  # EMA smoothing factor
        self._bbox_history: deque[list[int]] = deque(maxlen=10)

    def update(self, cx: float, cy: float, timestamp: float, bbox: list[int] | None = None) -> None:
        """Update trajectory with new centroid position."""
        # EMA smoothing
        if self.positions:
            sx = self._alpha * cx + (1 - self._alpha) * self.smoothed[0]
            sy = self._alpha * cy + (1 - self._alpha) * self.smoothed[1]
            self.smoothed = (sx, sy)

            # Velocity (pixels/second)
            dt = timestamp - self.timestamps[-1]
            if dt > 0.001:
                vx = (sx - self.positions[-1][0]) / dt
                vy = (sy - self.positions[-1][1]) / dt
                self.velocities.append((vx, vy))

            # Distance
            dx = cx - self.positions[-1][0]
            dy = cy - self.positions[-1][1]
            self.total_distance += (dx * dx + dy * dy) ** 0.5
        else:
            self.smoothed = (cx, cy)

        self.positions.append((cx, cy))
        self.timestamps.append(timestamp)
        self.last_seen = timestamp
        if bbox:
            self._bbox_history.append(bbox)

        # Moving if recent displacement > threshold
        self.is_moving = self._compute_is_moving()

    def _compute_is_moving(self, window: int = 5, threshold: float = 3.0) -> bool:
        """Check if track has moved significantly in last N frames."""
        if len(self.positions) < window:
            return False
        recent = list(self.positions)[-window:]
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        return (dx * dx + dy * dy) ** 0.5 > threshold

    def get_direction_vector(self, window: int = 8) -> tuple[float, float]:
        """Average movement direction over last N frames. Normalized."""
        if len(self.positions) < 2:
            return (0.0, 0.0)
        n = min(window, len(self.positions))
        recent = list(self.positions)[-n:]
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        mag = (dx * dx + dy * dy) ** 0.5
        if mag < 1e-6:
            return (0.0, 0.0)
        return (dx / mag, dy / mag)

    def get_speed(self, window: int = 5) -> float:
        """Average speed in pixels/frame over last N positions."""
        if len(self.positions) < 2:
            return 0.0
        n = min(window, len(self.positions))
        recent = list(self.positions)[-n:]
        total = sum(
            ((recent[i][0] - recent[i-1][0])**2 + (recent[i][1] - recent[i-1][1])**2)**0.5
            for i in range(1, len(recent))
        )
        return total / (len(recent) - 1)

    @property
    def age_seconds(self) -> float:
        return self.last_seen - self.first_seen

    @property
    def avg_bbox(self) -> list[int] | None:
        if not self._bbox_history:
            return None
        bboxes = np.array(list(self._bbox_history))
        return bboxes.mean(axis=0).astype(int).tolist()


class MultiObjectTracker(ModuleBase):
    """ByteTrack-based MOT with trajectory management.

    Enhancements over basic tracker:
    - Per-track trajectory objects with EMA smoothing
    - Velocity + direction vector computation
    - Track lifecycle events (created, lost)
    - Trajectory export on FrameContext for downstream modules
    """

    def __init__(self, config: dict[str, Any] | None = None, event_bus: Any = None):
        super().__init__(config)
        self.event_bus = event_bus
        self._tracker = None
        self._trajectories: dict[int, TrackTrajectory] = {}
        self._active_ids: set[int] = set()
        self._lost_timeout = config.get("lost_timeout", 5.0) if config else 5.0

    def initialize(self) -> None:
        try:
            from supervision import ByteTrack
            self._tracker = ByteTrack(
                track_activation_threshold=self.config.get("track_thresh", 0.25),
                lost_track_buffer=self.config.get("lost_buffer", 30),
                minimum_matching_threshold=self.config.get("match_thresh", 0.8),
                frame_rate=self.config.get("fps", 10),
            )
            logger.info("ByteTrack initialized")
        except ImportError:
            logger.warning("supervision not available, using basic IOU tracker")
            self._tracker = SimpleTracker()

        self._initialized = True
        logger.info("MultiObjectTracker ready (with trajectory export)")

    def process(self, ctx: FrameContext) -> None:
        if not self._initialized or not ctx.detections:
            return

        bboxes = np.array([d["bbox"] for d in ctx.detections], dtype=np.float32)
        confs = np.array([d["confidence"] for d in ctx.detections], dtype=np.float32)

        current_ids = set()

        if hasattr(self._tracker, "update_with_detections"):
            from supervision import Detections
            sv_dets = Detections(
                xyxy=bboxes,
                confidence=confs,
                class_id=np.zeros(len(bboxes), dtype=int),
            )
            tracked = self._tracker.update_with_detections(sv_dets)

            tracked_bboxes = tracked.xyxy if len(tracked) > 0 else np.empty((0, 4))
            tracked_ids = tracked.tracker_id if tracked.tracker_id is not None else np.arange(len(tracked))

            for ti in range(len(tracked_bboxes)):
                track_id = int(tracked_ids[ti])
                t_bbox = tracked_bboxes[ti]
                current_ids.add(track_id)

                best_idx = -1
                best_iou = 0.3
                for di in range(len(ctx.detections)):
                    if ctx.detections[di].get("track_id", -1) >= 0:
                        continue
                    iou = self._compute_iou(t_bbox, np.array(ctx.detections[di]["bbox"], dtype=np.float32))
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = di

                bbox_list = [int(t_bbox[0]), int(t_bbox[1]), int(t_bbox[2]), int(t_bbox[3])]
                if best_idx >= 0:
                    ctx.detections[best_idx]["track_id"] = track_id
                else:
                    ctx.detections.append({
                        "bbox": bbox_list,
                        "confidence": float(tracked.confidence[ti]) if tracked.confidence is not None else 0.5,
                        "track_id": track_id,
                    })

                # Update trajectory
                cx = (bbox_list[0] + bbox_list[2]) / 2.0
                cy = (bbox_list[1] + bbox_list[3]) / 2.0

                if track_id not in self._trajectories:
                    self._trajectories[track_id] = TrackTrajectory(track_id, ctx.timestamp)
                    if self.event_bus:
                        self.event_bus.publish(SpineEvent(
                            event_type=EventType.TRACK_CREATED,
                            camera_id=ctx.camera_id,
                            track_id=track_id,
                            timestamp=ctx.timestamp,
                            data={"bbox": bbox_list},
                        ))

                self._trajectories[track_id].update(cx, cy, ctx.timestamp, bbox_list)

        else:
            track_ids = self._tracker.update(bboxes, confs)
            for i, tid in enumerate(track_ids):
                if i < len(ctx.detections):
                    ctx.detections[i]["track_id"] = tid
                    current_ids.add(tid)
                    bbox = ctx.detections[i]["bbox"]
                    cx = (bbox[0] + bbox[2]) / 2.0
                    cy = (bbox[1] + bbox[3]) / 2.0
                    if tid not in self._trajectories:
                        self._trajectories[tid] = TrackTrajectory(tid, ctx.timestamp)
                    self._trajectories[tid].update(cx, cy, ctx.timestamp, bbox)

        ctx.tracks = [d for d in ctx.detections if d.get("track_id", -1) >= 0]

        # Publish TRACK_LOST for disappeared tracks
        lost = self._active_ids - current_ids
        for tid in lost:
            traj = self._trajectories.get(tid)
            if traj and ctx.timestamp - traj.last_seen > self._lost_timeout:
                if self.event_bus:
                    self.event_bus.publish(SpineEvent(
                        event_type=EventType.TRACK_LOST,
                        camera_id=ctx.camera_id,
                        track_id=tid,
                        timestamp=ctx.timestamp,
                        data={
                            "age": round(traj.age_seconds, 1),
                            "distance": round(traj.total_distance, 1),
                        },
                    ))
                del self._trajectories[tid]

        self._active_ids = current_ids

        # Export trajectories on context for downstream modules
        ctx.metadata["trajectories"] = self._trajectories

    def get_trajectory(self, track_id: int) -> TrackTrajectory | None:
        return self._trajectories.get(track_id)

    def get_all_trajectories(self) -> dict[int, TrackTrajectory]:
        return dict(self._trajectories)

    @staticmethod
    def _compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def cleanup(self) -> None:
        self._tracker = None
        self._trajectories.clear()
        self._active_ids.clear()


class SimpleTracker:
    """Fallback IOU-based tracker when supervision unavailable."""

    def __init__(self):
        self._next_id = 0
        self._tracks: dict[int, np.ndarray] = {}

    def update(self, bboxes: np.ndarray, confs: np.ndarray) -> list[int]:
        if len(bboxes) == 0:
            return []

        assigned_ids = []
        used_tracks = set()

        for bbox in bboxes:
            best_iou = 0.3
            best_id = -1

            for tid, prev_bbox in self._tracks.items():
                if tid in used_tracks:
                    continue
                iou = self._compute_iou(bbox, prev_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid

            if best_id >= 0:
                assigned_ids.append(best_id)
                used_tracks.add(best_id)
                self._tracks[best_id] = bbox
            else:
                assigned_ids.append(self._next_id)
                self._tracks[self._next_id] = bbox
                self._next_id += 1

        return assigned_ids

    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0
