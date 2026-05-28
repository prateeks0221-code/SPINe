<p align="center">
  <img src="https://img.shields.io/badge/CV--SPINE-Universal%20Vision%20Pipeline-8b5cf6?style=for-the-badge&logo=eye&logoColor=white" alt="CV-SPINE"/>
</p>

<h1 align="center">CV-SPINE</h1>
<p align="center"><strong>Universal Computer Vision Pipeline with Product Adapters</strong></p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/YOLOv8-Detection%20%2B%20Pose-00FFFF?style=flat-square"/>
  <img src="https://img.shields.io/badge/ByteTrack-MOT-06d6a0?style=flat-square"/>
  <img src="https://img.shields.io/badge/FastAPI-Dashboard-009688?style=flat-square&logo=fastapi"/>
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square"/>
</p>

<p align="center">
  <em>One spine. Many products. Real-time CV inference with modular architecture.</em>
</p>

---

## What is CV-SPINE?

CV-SPINE is a **modular computer vision pipeline** that provides 6 core pillars as reusable building blocks. Product layers (VibeCheck, SchoolGuard, RetailIQ, etc.) sit on top via an **Adapter pattern** — zero coupling, infinite extensibility.

```
 Product Layer     VibeCheck  |  SchoolGuard  |  RetailIQ  |  Your Product
                   ──────────────────────────────────────────────────────────
 Adapter Layer     ProductAdapter ABC  →  maps config + handles events
                   ──────────────────────────────────────────────────────────
 SPINE Core        Detection │ Pose │ ReID │ Entry/Exit │ ROI │ Face
                   ──────────────────────────────────────────────────────────
 Infrastructure    MQTT  │  Redis  │  TimescaleDB  │  Qdrant  │  MediaMTX
```

---

## 6 Pillars

| Pillar | Module | Model | What It Does |
|--------|--------|-------|-------------|
| **Detection** | `PersonDetector` | YOLOv8m | Real-time person detection with NMS, size/aspect filters |
| **Pose** | `PoseEstimator` + `ActivityClassifier` | YOLOv8m-pose | 17-keypoint skeleton, fall detection, dance energy, aggression |
| **ReID** | `ReIDEmbedder` | OSNet / ONNX stub | Body-based re-identification, EMA embedding updates, unique visitor counting |
| **Entry/Exit** | `GateCrossingDetector` + `OccupancyCounter` | ByteTrack trajectories | State-machine gate crossing with trajectory analysis |
| **ROI** | `ZoneManager` + `HeatmapAccumulator` | Geometry | Polygon zones, dwell time, overcrowding alerts, spatial heatmaps |
| **Face** | `FaceDetector` + `FaceRecognizer` | SCRFD + ArcFace | Face detection, recognition, demographics, liveness (stubs) |

---

## Architecture Highlights

### Trajectory-Based Gate Crossing (Entry/Exit)

Not your typical line-crossing. The gate detector uses a **5-state machine** per track:

```
FAR  →  APPROACHING  →  IN_ZONE  →  CROSSED  →  COOLDOWN  →  FAR
```

- **EMA-smoothed trajectories** from ByteTrack (filters jitter)
- **Foot position** (bbox bottom) instead of centroid
- **Multi-signal direction**: trajectory vector > foot displacement > signed distance
- **Perpendicular distance + projection ratio** for robust proximity detection
- Catches slow walkers that simple segment intersection misses

### ReID with Feature Retention

- **EMA embedding updates** — each sighting refines stored identity (alpha=0.3)
- **Multi-sample confirmation** — 3 samples before registering new person
- **Sticky track-to-gallery mapping** — no re-matching jitter
- **Deterministic ONNX stub** — same crop always produces same embedding (hash-based)

### Real-Time CV-Driven KPIs

Vibe Score is computed from **actual CV signals**, not dummy formulas:

| Component | Weight | Source |
|-----------|--------|--------|
| Occupancy | 0-25 | person count / capacity |
| Movement Energy | 0-35 | wrist + ankle velocity from pose keypoints |
| Arm Raise | 0-20 | % of people with arms above shoulders |
| Activity | 0-20 | standing ratio + movement speed |

### Event Bus

Typed event system with MQTT/Redis backends (graceful fallback to in-process):

```python
bus.subscribe("line.crossed", handler)
bus.subscribe("pose.dance_energy", handler)
bus.subscribe("cvspine/vibecheck/#", product_handler)
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- conda (recommended) or venv
- A camera source (IP webcam, USB, RTSP, or video file)

### 1. Clone

```bash
git clone https://github.com/prateeks0221-code/SPINe.git
cd SPINe
```

### 2. Create Environment

```bash
conda create -n spine python=3.13 -y
conda activate spine
pip install -e .
```

### 3. Configure Camera

Edit `products/vibecheck/config/venues.yaml`:

```yaml
venue_name: "My Space"
capacity: 10

cameras:
  - id: "cam-main"
    source: "https://192.168.1.100:4343/video"  # IP Webcam, RTSP, USB (0), or file path
    detection:
      model: "yolov8m"
      confidence: 0.35
    pose:
      model: "yolov8m-pose"
      fps: 2.0
    reid:
      enabled: true
      gallery_ttl: 14400
      similarity_threshold: 0.65
      ema_alpha: 0.3
      min_samples: 3

zones:
  - id: "main-area"
    camera_id: "cam-main"
    type: "polygon"
    points: [[0, 200], [900, 200], [900, 888], [0, 888]]
    analytics:
      heatmap: true
      dwell: true

lines:
  - id: "main-entrance"
    camera_id: "cam-main"
    type: "entry_exit"
    points: [[1060, 200], [1060, 720]]  # Adjust to your door position
    direction_in: "left"                 # Direction person moves when ENTERING
```

**Camera sources:**
| Type | Source Value |
|------|-------------|
| IP Webcam (Android) | `https://192.168.x.x:4343/video` |
| USB Camera | `0` (first cam), `1` (second) |
| RTSP | `rtsp://user:pass@ip:554/stream` |
| Video File | `/path/to/video.mp4` |

### 4. Run

**Linux/WSL:**
```bash
chmod +x run.sh
./run.sh
```

**Windows (direct):**
```bash
python -m products.vibecheck.dashboard.app
```

### 5. Open Dashboard

Navigate to **http://localhost:8765**

---

## Dashboard Features

### Live Video with CV Overlays
- Colored skeleton bones with glow effect
- Track trails (fading purple)
- Corner-accented detection boxes
- Zone polygon fills (translucent)
- Gate lines with direction arrows + endpoint markers

### Real-Time KPI Cards
- **Vibe Score** — CV-driven composite metric with breakdown
- **Dance Energy** — aggregate limb velocity from pose keypoints
- **People Count** — with occupancy ring and peak tracking
- **Entry/Exit Flow** — gate crossing counts
- **Unique Visitors** — ReID gallery-based (not event counting)
- **Pipeline Health** — per-module latency

### Interactive Gate Line Editor
Click **"Edit Gate Lines"** on the video feed to:
1. Click two points on the frame to draw a gate line
2. Set direction (which way is "entering")
3. Save — pipeline hot-reloads instantly, no restart needed

### Sparkline Charts
- People count (3-min history)
- Vibe + Energy dual chart
- FPS performance

### Spatial Heatmap
32x18 density grid showing where people spend time.

---

## Project Structure

```
CV-SPIN/
  spine/                          # Core pipeline (product-agnostic)
    core/
      config.py                   # Pydantic schemas for all config
      events.py                   # Typed event system (24 event types)
      event_bus.py                # MQTT + Redis + in-process pub/sub
      orchestrator.py             # Frame router, module pipeline
      adapter.py                  # ProductAdapter ABC
    modules/
      detection/
        detector.py               # YOLOv8 person detector
        tracker.py                # ByteTrack MOT + TrackTrajectory
      pose/
        estimator.py              # YOLOv8-pose keypoints
        activity.py               # Fall, dance energy, aggression
      reid/
        embedder.py               # Body ReID with EMA + multi-sample
        gallery.py                # Qdrant vector DB / in-memory
      entry_exit/
        gate_detector.py          # State-machine gate crossing
        counter.py                # Occupancy with analytics
        line_crossing.py          # Legacy simple crossing (backup)
      roi/
        zone_manager.py           # Polygon zones, dwell, alerts
        heatmap.py                # 32x18 density accumulator
        homography.py             # Bird's-eye perspective transform
        spatial_recommend.py      # Flow analysis, bottlenecks
      face/
        detector.py               # SCRFD face detection
        recognizer.py             # ArcFace recognition
        demographics.py           # Age/gender (stub)
        liveness.py               # Anti-spoofing (stub)
    utils/
      frame_grabber.py            # Multi-protocol video ingest
      anonymizer.py               # Face blur for privacy
      frame_dedup.py              # pHash deduplication
      gpu_scheduler.py            # Priority-based GPU queue
      model_registry.py           # Central model catalog
      metrics.py                  # Prometheus metrics
      health.py                   # Health check aggregator
    infra/
      docker-compose.yml          # MQTT, Redis, TimescaleDB, Qdrant
  products/
    vibecheck/                    # Example product: nightclub/venue analytics
      adapter.py                  # VibeCheckAdapter (maps venue config)
      config/venues.yaml          # Venue-specific camera + zone config
      dashboard/
        app.py                    # FastAPI + MJPEG + WebSocket dashboard
        index.html                # Dark-themed real-time UI
  run.sh                          # Launch script (WSL/Linux)
  pyproject.toml                  # Dependencies + build config
```

---

## Building Your Own Product

Create a new adapter by implementing `ProductAdapter`:

```python
from spine.core.adapter import ProductAdapter

class MyProductAdapter(ProductAdapter):
    @property
    def product_id(self) -> str:
        return "my-product"

    def get_camera_configs(self) -> list:
        # Return CameraConfig objects from your config format
        ...

    def get_zone_configs(self) -> list:
        # Define zones specific to your use case
        ...

    def get_line_configs(self) -> list:
        # Define entry/exit gates
        ...

    def on_event(self, event):
        # React to spine events (detections, crossings, etc.)
        ...
```

Then register it in your dashboard/app and the entire spine pipeline works for your product.

---

## Infrastructure (Optional)

For production, start the backing services:

```bash
cd spine/infra
docker-compose up -d
```

This gives you:
- **Mosquitto** (MQTT) — event streaming
- **Redis** — fast pub/sub + caching
- **TimescaleDB** — time-series analytics storage
- **Qdrant** — vector DB for ReID galleries

The spine runs fine **without these** — it falls back to in-process event bus and in-memory galleries.

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `ultralytics` | YOLOv8 detection + pose |
| `supervision` | ByteTrack MOT |
| `opencv-python` | Video I/O + rendering |
| `fastapi` + `uvicorn` | Dashboard server |
| `pydantic` | Config validation |
| `numpy` | Array ops |
| `torch` | Model inference backend |
| `paho-mqtt` | MQTT event bus (optional) |
| `redis` | Redis event bus (optional) |
| `qdrant-client` | Vector DB for ReID (optional) |

---

## Intel MKL Fix (WSL/Linux)

If you hit `Intel oneMKL FATAL ERROR`, the `run.sh` script handles it automatically. For manual runs:

```bash
export LD_PRELOAD=$(python -c "import torch; print(torch.__file__.replace('__init__.py', 'lib/libtorch_cpu.so'))")
export MKL_SERVICE_FORCE_INTEL=1
python -m products.vibecheck.dashboard.app
```

---

## License

MIT

---

<p align="center">
  <strong>Built with CV-SPINE</strong> — One pipeline, infinite products.
</p>
