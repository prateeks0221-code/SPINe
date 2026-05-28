"""Gate Crossing Detector — sophisticated entry/exit detection using trajectory analysis.

Architecture:
    ByteTrack (tracker) → TrackTrajectory (smoothed positions + velocity)
        → GateCrossingDetector (state machine per track×gate)
            → LineCrossEvent (direction: in/out)

State machine per (track_id, gate_id):
    FAR → APPROACHING → IN_ZONE → CROSSED → COOLDOWN → FAR

Detection method (multi-signal fusion):
    1. Signed distance to gate line (perpendicular) — detects side changes
    2. Trajectory direction vector (8-frame window) — determines in/out
    3. Foot position (bottom of bbox) — more accurate than centroid for gates
    4. Smoothed positions from TrackTrajectory — filters jitter

Why this beats simple segment intersection:
    - Slow walkers: centroid moves <5px/frame, segment intersection misses
    - Diagonal approach: trajectory vector gives true direction
    - Occlusion recovery: state machine remembers pre-occlusion side
    - Jitter rejection: EMA smoothing + state hysteresis prevents double-counting
"""

from __future__ import annotations

import logging
import time
from collections import deque
from enum import Enum
from typing import Any

import numpy as np

from spine.core.config import LineConfig
from spine.core.events import EventType, LineCrossEvent, SpineEvent
from spine.core.orchestrator import FrameContext, ModuleBase

logger = logging.getLogger(__name__)


class GateState(str, Enum):
    FAR = "far"
    APPROACHING = "approaching"
    IN_ZONE = "in_zone"
    CROSSED = "crossed"
    COOLDOWN = "cooldown"


class TrackGateState:
    """State machine for one track × one gate."""

    __slots__ = ("track_id", "gate_id", "state", "side", "prev_side",
                 "enter_time", "crossed_time", "approach_dist",
                 "crossing_direction", "foot_positions")

    def __init__(self, track_id: int, gate_id: str):
        self.track_id = track_id
        self.gate_id = gate_id
        self.state = GateState.FAR
        self.side: float = 0.0          # current signed distance
        self.prev_side: float = 0.0     # previous signed distance
        self.enter_time: float = 0.0    # when entered approach zone
        self.crossed_time: float = 0.0  # when crossing confirmed
        self.approach_dist: float = 0.0
        self.crossing_direction: str = ""
        self.foot_positions: deque = deque(maxlen=10)


class GateCrossingDetector(ModuleBase):
    """Sophisticated gate crossing using trajectory analysis + state machine.

    Config:
        approach_radius: float — distance (px) to start tracking approach (default: 120)
        cross_threshold: float — signed distance change to confirm crossing (default: 15)
        cooldown: float — seconds before same track can cross again (default: 3.0)
        use_foot: bool — use bottom of bbox instead of centroid (default: True)
        direction_window: int — frames for direction vector (default: 8)
    """

    def __init__(self, config: dict[str, Any] | None = None, event_bus: Any = None,
                 lines: list[LineConfig] | None = None):
        super().__init__(config)
        self.event_bus = event_bus
        self.lines = lines or []

        cfg = config or {}
        self._approach_radius = float(cfg.get("approach_radius", 120))
        self._cross_threshold = float(cfg.get("cross_threshold", 15))
        self._cooldown = float(cfg.get("cooldown", 3.0))
        self._use_foot = bool(cfg.get("use_foot", True))
        self._direction_window = int(cfg.get("direction_window", 8))

        # State per (track_id, gate_id)
        self._states: dict[tuple[int, str], TrackGateState] = {}
        self._crossing_count = 0
        self._total_in = 0
        self._total_out = 0

    def initialize(self) -> None:
        self._initialized = True
        logger.info("GateCrossingDetector ready (%d gates)", len(self.lines))
        for l in self.lines:
            length = self._line_length(l.points[0], l.points[1])
            logger.info("  Gate '%s': %s→%s len=%.0fpx dir_in=%s",
                        l.id, l.points[0], l.points[1], length, l.direction_in)

    def process(self, ctx: FrameContext) -> None:
        if not self._initialized or not ctx.tracks:
            return

        camera_gates = [l for l in self.lines if l.camera_id == ctx.camera_id]
        if not camera_gates:
            return

        # Get trajectories from tracker (if available)
        trajectories = ctx.metadata.get("trajectories", {})

        for track in ctx.tracks:
            track_id = track.get("track_id", -1)
            if track_id < 0:
                continue

            bbox = track["bbox"]

            # Use foot (bottom center) or centroid
            if self._use_foot:
                px = (bbox[0] + bbox[2]) / 2.0
                py = float(bbox[3])  # bottom of bbox
            else:
                px = (bbox[0] + bbox[2]) / 2.0
                py = (bbox[1] + bbox[3]) / 2.0

            # Get trajectory for direction vector
            traj = trajectories.get(track_id)

            for gate in camera_gates:
                self._process_gate(ctx, track_id, px, py, bbox, gate, traj)

    def _process_gate(self, ctx: FrameContext, track_id: int, px: float, py: float,
                      bbox: list[int], gate: LineConfig, traj: Any) -> None:
        """Run state machine for one track×gate pair."""
        key = (track_id, gate.id)
        p1, p2 = gate.points[0], gate.points[1]

        # Compute signed distance from point to gate line
        signed_dist = self._signed_distance(px, py, p1, p2)
        perp_dist = abs(signed_dist)

        # Project point onto line to check if within gate segment bounds
        proj_ratio = self._projection_ratio(px, py, p1, p2)
        in_segment = -0.3 <= proj_ratio <= 1.3  # 30% margin beyond endpoints

        # Get or create state
        if key not in self._states:
            self._states[key] = TrackGateState(track_id, gate.id)
            self._states[key].side = signed_dist

        state = self._states[key]
        state.prev_side = state.side
        state.side = signed_dist
        state.foot_positions.append((px, py))

        # ── State machine transitions ──

        if state.state == GateState.FAR:
            if perp_dist < self._approach_radius and in_segment:
                state.state = GateState.APPROACHING
                state.enter_time = ctx.timestamp
                state.approach_dist = signed_dist

        elif state.state == GateState.APPROACHING:
            if perp_dist > self._approach_radius * 1.5:
                # Walked away without crossing
                state.state = GateState.FAR
            elif perp_dist < self._cross_threshold * 3:
                state.state = GateState.IN_ZONE

        elif state.state == GateState.IN_ZONE:
            if perp_dist > self._approach_radius * 1.5:
                state.state = GateState.FAR
                return

            # Check for side change (crossing confirmed)
            side_changed = (state.approach_dist * signed_dist) < 0

            if side_changed and perp_dist > self._cross_threshold:
                # ── CROSSING DETECTED ──
                direction = self._determine_direction(
                    state, gate, traj, px, py
                )
                state.state = GateState.CROSSED
                state.crossed_time = ctx.timestamp
                state.crossing_direction = direction
                self._crossing_count += 1

                if direction == "in":
                    self._total_in += 1
                else:
                    self._total_out += 1

                logger.info(
                    "GATE CROSSED: track=%d gate=%s dir=%s | pos=(%.0f,%.0f) "
                    "signed_dist=%.1f approach_dist=%.1f",
                    track_id, gate.id, direction, px, py,
                    signed_dist, state.approach_dist
                )

                if self.event_bus:
                    self.event_bus.publish(LineCrossEvent(
                        camera_id=ctx.camera_id,
                        track_id=track_id,
                        timestamp=ctx.timestamp,
                        line_id=gate.id,
                        direction=direction,
                    ))

                    # Also publish enriched data
                    speed = traj.get_speed() if traj else 0.0
                    self.event_bus.publish(SpineEvent(
                        event_type=EventType.ANOMALY_DETECTED
                        if direction == "out" and gate.alert_on == "out"
                        else EventType.LINE_CROSSED,
                        camera_id=ctx.camera_id,
                        track_id=track_id,
                        timestamp=ctx.timestamp,
                        data={
                            "gate_id": gate.id,
                            "direction": direction,
                            "speed": round(speed, 1),
                            "dwell_approach": round(ctx.timestamp - state.enter_time, 1),
                        },
                    ))

        elif state.state == GateState.CROSSED:
            state.state = GateState.COOLDOWN

        elif state.state == GateState.COOLDOWN:
            if ctx.timestamp - state.crossed_time > self._cooldown:
                state.state = GateState.FAR
                state.approach_dist = signed_dist  # reset approach side

    def _determine_direction(self, state: TrackGateState, gate: LineConfig,
                             traj: Any, px: float, py: float) -> str:
        """Determine crossing direction using multiple signals."""
        dir_in = gate.direction_in

        # Signal 1: Trajectory direction vector (strongest signal)
        if traj and hasattr(traj, "get_direction_vector"):
            dvx, dvy = traj.get_direction_vector(self._direction_window)
            if abs(dvx) > 0.1 or abs(dvy) > 0.1:
                if dir_in == "left":
                    return "in" if dvx < -0.1 else "out"
                elif dir_in == "right":
                    return "in" if dvx > 0.1 else "out"
                elif dir_in == "up":
                    return "in" if dvy < -0.1 else "out"
                elif dir_in == "down":
                    return "in" if dvy > 0.1 else "out"

        # Signal 2: Foot position displacement (last N positions)
        if len(state.foot_positions) >= 3:
            fps = list(state.foot_positions)
            dx = fps[-1][0] - fps[0][0]
            dy = fps[-1][1] - fps[0][1]
            if dir_in == "left":
                return "in" if dx < 0 else "out"
            elif dir_in == "right":
                return "in" if dx > 0 else "out"
            elif dir_in == "up":
                return "in" if dy < 0 else "out"
            elif dir_in == "down":
                return "in" if dy > 0 else "out"

        # Signal 3: Signed distance change (fallback)
        if state.approach_dist > 0 and state.side < 0:
            return "in" if dir_in in ("left", "up") else "out"
        elif state.approach_dist < 0 and state.side > 0:
            return "in" if dir_in in ("right", "down") else "out"

        return "in"  # default

    # ── Geometry helpers ──

    @staticmethod
    def _signed_distance(px: float, py: float, p1: list[int], p2: list[int]) -> float:
        """Signed perpendicular distance from point to infinite line through p1→p2."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1e-6:
            return 0.0
        return (dy * (px - p1[0]) - dx * (py - p1[1])) / length

    @staticmethod
    def _projection_ratio(px: float, py: float, p1: list[int], p2: list[int]) -> float:
        """Project point onto line segment. Returns 0.0 at p1, 1.0 at p2."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        len_sq = dx * dx + dy * dy
        if len_sq < 1e-6:
            return 0.5
        return ((px - p1[0]) * dx + (py - p1[1]) * dy) / len_sq

    @staticmethod
    def _line_length(p1: list[int], p2: list[int]) -> float:
        return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5

    # ── Stats ──

    @property
    def stats(self) -> dict:
        return {
            "total_crossings": self._crossing_count,
            "total_in": self._total_in,
            "total_out": self._total_out,
            "active_states": len(self._states),
        }

    def cleanup(self) -> None:
        self._states.clear()
