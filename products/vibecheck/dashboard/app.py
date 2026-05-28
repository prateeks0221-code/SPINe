"""VibeCheck Real-Time Dashboard — Optimized for speed."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import threading
import signal
import sys
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from pathlib import Path

import cv2
import numpy as np

# Fix: ensure project root on path
PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("supervision").setLevel(logging.WARNING)
logger = logging.getLogger("dashboard")

app = FastAPI(title="CV Spine — VibeCheck Dashboard")


# ══════════════════════════════════════════════
# GLOBAL STATE
# ══════════════════════════════════════════════
class DashState:
    def __init__(self):
        self.lock = threading.RLock()  # Reentrant: event handlers can nest
        self.vibe_score = 0.0
        self.vibe_label = "LOADING..."
        self.person_count = 0
        self.total_entries = 0
        self.total_exits = 0
        self.dance_energy = 0.0
        self.fps = 0.0
        self.inference_ms = 0.0
        self.pose_ms = 0.0
        self.active_tracks = 0
        self.peak_count = 0
        self.capacity = 200
        self.alerts: deque = deque(maxlen=30)
        self.count_hist: deque = deque(maxlen=180)
        self.vibe_hist: deque = deque(maxlen=180)
        self.energy_hist: deque = deque(maxlen=180)
        self.fps_hist: deque = deque(maxlen=60)
        self.heatmap: list = []
        self.frames_processed = 0
        self.events_total = 0
        self.unique_visitors = 0
        self.reid_matches = 0
        self.uptime_start = time.time()
        # Video frame — double-buffered JPEG bytes
        self.frame_jpeg: bytes = b""
        self.frame_ready = threading.Event()
        # Real CV-driven metrics
        self.avg_movement_speed = 0.0       # from track displacement per frame
        self.arm_raise_ratio = 0.0          # % of people with arms raised
        self.standing_ratio = 1.0           # % of people standing
        self.avg_body_angle = 0.0           # from pose estimator
        self.crowd_density = 0.0            # people per zone area
        self.per_track_energy: dict = {}    # track_id → energy from pose
        # Line config for live editing
        self.line_configs: list = []
        self.lines_dirty = False
        self.frame_w: int = 1222
        self.frame_h: int = 888

S = DashState()


# ══════════════════════════════════════════════
# CV SPINE WORKER — OPTIMIZED PIPELINE
# ══════════════════════════════════════════════
def spine_worker():
    from spine.core.event_bus import EventBus
    from spine.core.orchestrator import Orchestrator, FrameContext
    from spine.core.config import SpineConfig, ZoneConfig, LineConfig
    from spine.core.events import EventType, SpineEvent
    from spine.modules.detection.detector import PersonDetector
    from spine.modules.detection.tracker import MultiObjectTracker
    from spine.modules.pose.estimator import PoseEstimator
    from spine.modules.pose.activity import ActivityClassifier
    from spine.modules.reid.embedder import ReIDEmbedder
    from spine.modules.roi.zone_manager import ZoneManager
    from spine.modules.roi.heatmap import HeatmapAccumulator
    from spine.modules.entry_exit.gate_detector import GateCrossingDetector
    from spine.modules.entry_exit.counter import OccupancyCounter
    from spine.utils.frame_grabber import FrameGrabber

    # Load venue config
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "venues.yaml"
    with open(cfg_path) as f:
        venue = yaml.safe_load(f)

    S.capacity = venue.get("capacity", 200)
    cam = venue["cameras"][0]
    cam_id = cam["id"]

    # Event bus
    bus = EventBus()

    def handle(event: SpineEvent):
        with S.lock:
            S.events_total += 1
            et = event.event_type
            if et == EventType.POSE_DANCE_ENERGY:
                tid = event.track_id if hasattr(event, 'track_id') else -1
                energy_val = event.data.get("energy", 0)
                if tid >= 0:
                    S.per_track_energy[tid] = energy_val
                # Aggregate: mean energy across all tracked persons
                if S.per_track_energy:
                    S.dance_energy = sum(S.per_track_energy.values()) / len(S.per_track_energy)
                else:
                    S.dance_energy = 0.6 * S.dance_energy + 0.4 * energy_val
            elif et == EventType.POSE_FALL_DETECTED:
                S.alerts.appendleft({"type": "FALL", "sev": "critical", "msg": f"Fall — Track #{event.track_id}", "t": time.strftime("%H:%M:%S")})
            elif et == EventType.POSE_AGGRESSION:
                S.alerts.appendleft({"type": "FIGHT", "sev": "critical", "msg": f"Aggression — {event.data.get('track_ids', [])}", "t": time.strftime("%H:%M:%S")})
            elif et == EventType.LINE_CROSSED:
                direction = getattr(event, 'direction', None) or event.data.get("direction", "")
                if direction == "in":
                    S.total_entries += 1
                elif direction == "out":
                    S.total_exits += 1
            elif et == EventType.ZONE_OVERCROWDED:
                S.alerts.appendleft({"type": "CROWD", "sev": "warning", "msg": f"Overcrowded — {event.data.get('zone_id')}", "t": time.strftime("%H:%M:%S")})
            elif et == EventType.REID_NEW_PERSON:
                S.unique_visitors += 1
            elif et == EventType.REID_MATCH:
                S.reid_matches += 1

    for pattern in ["person.detected", "pose.dance_energy", "pose.fall_detected",
                    "pose.aggression_detected", "line.crossed", "zone.overcrowded",
                    "reid.new_person", "reid.match"]:
        bus.subscribe(pattern, handle)

    # Modules
    orch = Orchestrator(SpineConfig(), bus)
    det_cfg = cam.get("detection", {})
    pose_cfg = cam.get("pose", {})
    reid_cfg = cam.get("reid", {})

    detector = PersonDetector(config=det_cfg, event_bus=bus)
    tracker = MultiObjectTracker(config={}, event_bus=bus)
    pose_est = PoseEstimator(config=pose_cfg, event_bus=bus)
    activity = ActivityClassifier(config={}, event_bus=bus)
    reid = ReIDEmbedder(config=reid_cfg, event_bus=bus)

    zones_raw = venue.get("zones", [])
    zone_cfgs = [ZoneConfig(**z) for z in zones_raw]
    zone_mgr = ZoneManager(config={}, event_bus=bus, zones=zone_cfgs)

    lines_raw = venue.get("lines", [])
    line_cfgs = [LineConfig(**l) for l in lines_raw]
    line_det = GateCrossingDetector(config={
        "approach_radius": 120,
        "cross_threshold": 15,
        "cooldown": 2.5,
        "use_foot": True,
        "direction_window": 8,
    }, event_bus=bus, lines=line_cfgs)
    S.line_configs = lines_raw
    occ = OccupancyCounter(event_bus=bus)
    bus.subscribe("line.crossed", occ.on_line_crossed)

    heatmap = HeatmapAccumulator(grid_w=32, grid_h=18)

    # Register all
    orch.register_module(cam_id, "person_detector", detector)
    orch.register_module(cam_id, "tracker", tracker)
    orch.register_module(cam_id, "pose_keypoints", pose_est)
    orch.register_module(cam_id, "activity", activity)
    orch.register_module(cam_id, "reid", reid)
    orch.register_module(cam_id, "roi_zones", zone_mgr)
    orch.register_module(cam_id, "entry_exit", line_det)

    orch.start()

    # Frame grabber
    grabber = FrameGrabber(source=cam["source"], camera_id=cam_id, target_fps=15.0)
    if not grabber.start():
        logger.error("Camera failed!")
        return
    logger.info("Pipeline running — %s", cam_id)

    # Wait for first frame to get actual resolution
    actual_w, actual_h = 1222, 888  # default from image properties
    for _ in range(100):
        test_frame = grabber.get_frame()
        if test_frame is not None:
            actual_h, actual_w = test_frame.shape[:2]
            logger.info("Camera resolution: %dx%d", actual_w, actual_h)
            break
        time.sleep(0.05)

    # Store frame dimensions for line editor scaling
    S.frame_w = actual_w
    S.frame_h = actual_h

    # ── Drawing constants ──
    SKEL = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    JOINT_COLORS = {
        0:(139,92,246), 1:(139,92,246), 2:(139,92,246), 3:(139,92,246), 4:(139,92,246),
        5:(6,214,160), 6:(6,214,160), 7:(59,130,246), 8:(59,130,246), 9:(236,72,153), 10:(236,72,153),
        11:(245,158,11), 12:(245,158,11), 13:(34,197,94), 14:(34,197,94), 15:(6,182,212), 16:(6,182,212),
    }
    BONE_COLORS = {
        (5,6):(6,214,160), (5,7):(59,130,246), (7,9):(59,130,246), (6,8):(236,72,153), (8,10):(236,72,153),
        (5,11):(245,158,11), (6,12):(245,158,11), (11,12):(245,158,11),
        (11,13):(34,197,94), (13,15):(34,197,94), (12,14):(6,182,212), (14,16):(6,182,212),
    }

    # Track trail history
    track_trails: dict[int, deque] = {}
    frame_n = 0
    fps_t = time.time()
    fps_c = 0

    while True:
        frame = grabber.get_frame()
        if frame is None:
            time.sleep(0.005)
            continue

        frame_n += 1
        fps_c += 1
        t0 = time.perf_counter()

        if frame_n == 1:
            logger.info("First frame: %dx%d | Lines: %d | Zones: %d",
                        frame.shape[1], frame.shape[0], len(line_det.lines), len(zone_cfgs))

        # ── DETECTION (every frame) ──
        ctx = FrameContext(frame, cam_id, f"f-{frame_n}", time.time())
        detector.process(ctx)
        det_ms = (time.perf_counter() - t0) * 1000

        # ── TRACKING ──
        if ctx.detections:
            tracker.process(ctx)

        # ── POSE (every 2nd frame for speed) ──
        pose_ms = 0
        if frame_n % 2 == 0 and ctx.detections:
            tp = time.perf_counter()
            pose_est.process(ctx)
            activity.process(ctx)
            pose_ms = (time.perf_counter() - tp) * 1000

            # ── Extract real CV signals from pose data ──
            if ctx.keypoints:
                arm_raised_count = 0
                standing_count = 0
                total_body_angle = 0.0
                for kp_data in ctx.keypoints:
                    if kp_data.get("arm_raise", "none") != "none":
                        arm_raised_count += 1
                    if kp_data.get("is_standing", True):
                        standing_count += 1
                    total_body_angle += kp_data.get("body_angle", 0)
                n = len(ctx.keypoints)
                with S.lock:
                    S.arm_raise_ratio = arm_raised_count / max(n, 1)
                    S.standing_ratio = standing_count / max(n, 1)
                    S.avg_body_angle = total_body_angle / max(n, 1)

        # ── Track movement speed (every frame) ──
        if ctx.tracks and len(ctx.tracks) > 0:
            speeds = []
            for trk in ctx.tracks:
                tid = trk.get("track_id", -1)
                if tid in track_trails and len(track_trails[tid]) >= 2:
                    p1 = track_trails[tid][-2]
                    p2 = track_trails[tid][-1]
                    spd = ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) ** 0.5
                    speeds.append(spd)
            if speeds:
                with S.lock:
                    S.avg_movement_speed = sum(speeds) / len(speeds)

        # ── REID (every 5th frame) ──
        if frame_n % 5 == 0 and ctx.detections:
            reid.process(ctx)
            # Sync unique visitor count from gallery (more accurate than event counting)
            with S.lock:
                S.unique_visitors = reid.get_unique_count()
                S.reid_matches = len(ctx.reid_matches)

        # ── Check for line config reload ──
        if S.lines_dirty:
            with S.lock:
                S.lines_dirty = False
                new_lines = S.line_configs
            if new_lines:
                try:
                    line_cfgs = [LineConfig(**l) for l in new_lines]
                    line_det.lines = line_cfgs
                    line_det._states.clear()
                    line_det._crossing_count = 0
                    occ.reset()
                    logger.info("Gates reloaded: %d gates", len(line_cfgs))
                except Exception as e:
                    logger.error("Gate reload failed: %s", e)

        # ── ZONES + LINES ──
        zone_mgr.process(ctx)
        line_det.process(ctx)

        # Debug: log track positions vs gates every 90 frames
        if frame_n % 90 == 0 and ctx.tracks and line_det.lines:
            for trk in ctx.tracks[:2]:
                bbox = trk["bbox"]
                cx, cy = (bbox[0]+bbox[2])//2, bbox[3]  # foot position
                for gate in line_det.lines:
                    sd = line_det._signed_distance(cx, cy, gate.points[0], gate.points[1])
                    key = (trk.get("track_id", -1), gate.id)
                    gs = line_det._states.get(key)
                    state_str = gs.state.value if gs else "none"
                    logger.debug("Track #%d foot=(%d,%d) gate=%s sd=%.1f state=%s | crossings=%d",
                                 trk.get("track_id", -1), cx, cy, gate.id,
                                 sd, state_str, line_det._crossing_count)

        # ── HEATMAP ──
        heatmap.update(ctx.detections)

        total_ms = (time.perf_counter() - t0) * 1000

        # ── Update state ──
        h, w = frame.shape[:2]
        pc = len(ctx.detections)

        with S.lock:
            S.person_count = pc
            S.inference_ms = round(det_ms, 1)
            S.pose_ms = round(pose_ms, 1)
            S.active_tracks = len(ctx.tracks)
            if pc > S.peak_count:
                S.peak_count = pc
            S.frames_processed = frame_n

        # FPS + history (1Hz)
        now = time.time()
        if now - fps_t >= 1.0:
            S.fps = round(fps_c / (now - fps_t), 1)
            fps_c = 0
            fps_t = now
            with S.lock:
                S.count_hist.append(pc)

                # ── Real CV-driven vibe score ──
                # Components:
                #   1. Occupancy factor (0-25): people present relative to capacity
                #   2. Movement energy (0-35): actual dance energy from limb velocities
                #   3. Arm raise factor (0-20): social engagement signal
                #   4. Activity diversity (0-20): standing + moving vs idle
                occ_pct = min(pc / max(S.capacity, 1), 1.0)
                occ_factor = occ_pct * 25

                energy_factor = min(S.dance_energy, 100) * 0.35

                arm_factor = S.arm_raise_ratio * 20

                activity_factor = 0
                if pc > 0:
                    # More standing + moving = more vibe
                    activity_factor = S.standing_ratio * 10 + min(S.avg_movement_speed * 50, 10)

                vibe = occ_factor + energy_factor + arm_factor + activity_factor
                S.vibe_score = max(0, min(vibe, 100))

                if S.vibe_score >= 80: S.vibe_label = "LIT 🔥"
                elif S.vibe_score >= 60: S.vibe_label = "VIBING ✨"
                elif S.vibe_score >= 40: S.vibe_label = "WARMING UP 🌡️"
                elif S.vibe_score >= 20: S.vibe_label = "CHILL 😎"
                else: S.vibe_label = "DEAD 💀"
                S.vibe_hist.append(S.vibe_score)
                S.energy_hist.append(S.dance_energy)
                S.fps_hist.append(S.fps)

        # Heatmap snapshot
        if frame_n % 30 == 0:
            S.heatmap = heatmap.get_heatmap(normalized=True).tolist()

        # ══════════════════════════════════════
        # RENDER ANNOTATED FRAME
        # ══════════════════════════════════════
        vis = frame.copy()

        # ── Zone fills (translucent) ──
        overlay = vis.copy()
        for zc in zone_cfgs:
            if zc.camera_id != cam_id:
                continue
            pts = np.array(zc.points, np.int32)
            color = (6, 214, 160) if zc.type != "exclusion" else (60, 60, 239)
            cv2.fillPoly(overlay, [pts], (*color, ))
            cv2.polylines(vis, [pts], True, color, 2)
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1])) - 12
            cv2.putText(vis, zc.id.upper(), (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        vis = cv2.addWeighted(overlay, 0.12, vis, 0.88, 0)

        # ── Entry/exit lines ──
        for lc in line_det.lines:
            if lc.camera_id != cam_id:
                continue
            p1, p2 = tuple(lc.points[0]), tuple(lc.points[1])
            # Glow effect + thick line for visibility
            cv2.line(vis, p1, p2, (80, 50, 4), 6, cv2.LINE_AA)
            cv2.line(vis, p1, p2, (245, 158, 11), 3, cv2.LINE_AA)
            # Endpoints
            cv2.circle(vis, p1, 6, (6, 214, 160), -1, cv2.LINE_AA)
            cv2.circle(vis, p2, 6, (239, 68, 68), -1, cv2.LINE_AA)
            # Direction arrow + label
            mx, my = (p1[0]+p2[0])//2, (p1[1]+p2[1])//2
            arrows = {"up": "▲ IN", "down": "▼ IN", "left": "◀ IN", "right": "▶ IN"}
            arrow_txt = arrows.get(lc.direction_in, "● IN")
            cv2.putText(vis, f"{arrow_txt} | {lc.id}", (mx-60, my-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 158, 11), 2, cv2.LINE_AA)

        # ── Track trails ──
        for det in ctx.detections:
            tid = det.get("track_id", -1)
            if tid < 0:
                continue
            bbox = det["bbox"]
            cx, cy = (bbox[0]+bbox[2])//2, bbox[3]
            if tid not in track_trails:
                track_trails[tid] = deque(maxlen=40)
            track_trails[tid].append((cx, cy))

            trail = list(track_trails[tid])
            for i in range(1, len(trail)):
                alpha = i / len(trail)
                thickness = max(1, int(alpha * 3))
                color = (int(139 * alpha), int(92 * alpha), int(246 * alpha))
                cv2.line(vis, trail[i-1], trail[i], color, thickness, cv2.LINE_AA)

        # ── Detection boxes + labels ──
        for det in ctx.detections:
            bbox = det["bbox"]
            tid = det.get("track_id", -1)
            conf = det.get("confidence", 0)
            x1, y1, x2, y2 = bbox

            # Gradient-style box (top brighter)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (139, 92, 246), 2, cv2.LINE_AA)
            # Corner accents
            cl = 12
            cv2.line(vis, (x1, y1), (x1+cl, y1), (6, 214, 160), 3, cv2.LINE_AA)
            cv2.line(vis, (x1, y1), (x1, y1+cl), (6, 214, 160), 3, cv2.LINE_AA)
            cv2.line(vis, (x2, y1), (x2-cl, y1), (6, 214, 160), 3, cv2.LINE_AA)
            cv2.line(vis, (x2, y1), (x2, y1+cl), (6, 214, 160), 3, cv2.LINE_AA)
            cv2.line(vis, (x1, y2), (x1+cl, y2), (6, 214, 160), 3, cv2.LINE_AA)
            cv2.line(vis, (x1, y2), (x1, y2-cl), (6, 214, 160), 3, cv2.LINE_AA)
            cv2.line(vis, (x2, y2), (x2-cl, y2), (6, 214, 160), 3, cv2.LINE_AA)
            cv2.line(vis, (x2, y2), (x2, y2-cl), (6, 214, 160), 3, cv2.LINE_AA)

            # Label pill
            label = f"#{tid}  {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis, (x1, y1-th-10), (x1+tw+10, y1), (139, 92, 246), -1)
            cv2.putText(vis, label, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

        # ── Pose skeletons with glow ──
        for kp_data in ctx.keypoints:
            kpts = kp_data.get("keypoints", [])
            pts = {}
            for i, kp in enumerate(kpts):
                if kp.get("conf", 0) > 0.25:
                    px, py = int(kp["x"] * w), int(kp["y"] * h)
                    pts[i] = (px, py)

            # Bones with color
            for (i, j) in SKEL:
                if i in pts and j in pts:
                    color = BONE_COLORS.get((i,j), (6, 182, 212))
                    # Glow pass
                    cv2.line(vis, pts[i], pts[j], tuple(c//3 for c in color), 5, cv2.LINE_AA)
                    cv2.line(vis, pts[i], pts[j], color, 2, cv2.LINE_AA)

            # Joints
            for idx, pt in pts.items():
                color = JOINT_COLORS.get(idx, (6, 214, 160))
                cv2.circle(vis, pt, 5, tuple(c//3 for c in color), -1, cv2.LINE_AA)
                cv2.circle(vis, pt, 3, color, -1, cv2.LINE_AA)

        # ── HUD panel (top-left) ──
        hud_h = 130
        hud_overlay = vis.copy()
        cv2.rectangle(hud_overlay, (0, 0), (300, hud_h), (6, 6, 12), -1)
        vis = cv2.addWeighted(hud_overlay, 0.75, vis, 0.25, 0)
        cv2.line(vis, (0, hud_h), (300, hud_h), (42, 42, 68), 1)
        cv2.line(vis, (300, 0), (300, hud_h), (42, 42, 68), 1)

        # Vibe score big
        vibe_color = (68, 68, 239) if S.vibe_score >= 80 else (246, 92, 139) if S.vibe_score >= 60 else (11, 158, 245) if S.vibe_score >= 40 else (246, 130, 59)
        cv2.putText(vis, f"{S.vibe_score:.0f}", (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.6, vibe_color, 3, cv2.LINE_AA)
        cv2.putText(vis, "VIBE", (110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 160), 1, cv2.LINE_AA)
        cv2.putText(vis, S.vibe_label.split(" ")[0], (110, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 214, 6), 2, cv2.LINE_AA)

        # Stats
        y_off = 72
        stats = [
            (f"People: {S.person_count}", (6, 182, 212)),
            (f"Energy: {S.dance_energy:.0f}%", (6, 214, 160)),
            (f"FPS: {S.fps}  Det: {det_ms:.0f}ms  Pose: {pose_ms:.0f}ms", (120, 120, 160)),
            (f"In: {S.total_entries}  Out: {S.total_exits}  Tracks: {S.active_tracks}", (120, 120, 160)),
        ]
        for txt, color in stats:
            cv2.putText(vis, txt, (12, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            y_off += 16

        # ── Mini vibe bar (bottom of frame) ──
        bar_h = 4
        bar_w = int(w * min(S.vibe_score / 100, 1.0))
        bar_y = h - bar_h
        # Gradient bar
        if bar_w > 0:
            for x in range(bar_w):
                ratio = x / w
                r = int(59 + ratio * 180)
                g = int(130 - ratio * 60)
                b = int(246 - ratio * 100)
                cv2.line(vis, (x, bar_y), (x, h), (b, g, r), 1)

        # ── Encode JPEG (quality 80 for speed vs quality balance) ──
        _, jpeg = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
        S.frame_jpeg = jpeg.tobytes()
        S.frame_ready.set()

        # Throttle to ~20fps max render
        elapsed = time.perf_counter() - t0
        target = 1.0 / 20
        if elapsed < target:
            time.sleep(target - elapsed)


# Start worker
threading.Thread(target=spine_worker, daemon=True).start()


# ══════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return (Path(__file__).parent / "index.html").read_text()


# ── MJPEG stream (fastest for video) ──
def mjpeg_generator():
    while True:
        S.frame_ready.wait(timeout=1.0)
        S.frame_ready.clear()
        if S.frame_jpeg:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + S.frame_jpeg + b"\r\n")


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


# ── KPI WebSocket ──
@app.websocket("/ws/kpi")
async def kpi_ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            with S.lock:
                data = {
                    "vs": round(S.vibe_score, 1),
                    "vl": S.vibe_label,
                    "pc": S.person_count,
                    "pk": S.peak_count,
                    "cap": S.capacity,
                    "occ": round(S.person_count / max(S.capacity, 1) * 100, 1),
                    "de": round(S.dance_energy, 1),
                    "ei": S.total_entries,
                    "eo": S.total_exits,
                    "at": S.active_tracks,
                    "fps": S.fps,
                    "dms": S.inference_ms,
                    "pms": S.pose_ms,
                    "fn": S.frames_processed,
                    "ev": S.events_total,
                    "uv": S.unique_visitors,
                    "rm": S.reid_matches,
                    "up": int(time.time() - S.uptime_start),
                    "al": [dict(a) for a in list(S.alerts)[:12]],
                    "ch": list(S.count_hist),
                    "vh": list(S.vibe_hist),
                    "eh": list(S.energy_hist),
                    "fh": list(S.fps_hist),
                    "hm": S.heatmap,
                    # Real CV signals
                    "spd": round(S.avg_movement_speed, 1),
                    "arm": round(S.arm_raise_ratio * 100, 1),
                    "std": round(S.standing_ratio * 100, 1),
                    "ba": round(S.avg_body_angle, 1),
                    "lc": S.line_configs,
                    "fw": S.frame_w,
                    "fh": S.frame_h,
                }
            await ws.send_json(data)
            await asyncio.sleep(0.4)
    except WebSocketDisconnect:
        pass


# ── Line config API (live editor) ──
from fastapi import Request
from pydantic import BaseModel as PydanticBase

class LineUpdate(PydanticBase):
    lines: list[dict]

@app.get("/api/lines")
async def get_lines():
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "venues.yaml"
    with open(cfg_path) as f:
        venue = yaml.safe_load(f)
    return {"lines": venue.get("lines", []), "frame_w": S.frame_w, "frame_h": S.frame_h}

@app.post("/api/lines")
async def save_lines(req: Request):
    body = await req.json()
    new_lines = body.get("lines", [])
    editor_w = body.get("frame_w", S.frame_w)
    editor_h = body.get("frame_h", S.frame_h)

    # Scale coordinates from editor space to actual frame space
    sx = S.frame_w / max(editor_w, 1)
    sy = S.frame_h / max(editor_h, 1)
    for line in new_lines:
        if "points" in line:
            line["points"] = [
                [int(p[0] * sx), int(p[1] * sy)] for p in line["points"]
            ]

    cfg_path = Path(__file__).resolve().parents[1] / "config" / "venues.yaml"
    with open(cfg_path) as f:
        venue = yaml.safe_load(f)

    venue["lines"] = new_lines

    with open(cfg_path, "w") as f:
        yaml.dump(venue, f, default_flow_style=False, sort_keys=False)

    # Signal worker to reload line configs
    with S.lock:
        S.line_configs = new_lines
        S.lines_dirty = True
        S.total_entries = 0
        S.total_exits = 0

    logger.info("Lines updated (scaled %dx%d→%dx%d): %s", editor_w, editor_h, S.frame_w, S.frame_h, new_lines)
    return {"ok": True, "lines": new_lines}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")
