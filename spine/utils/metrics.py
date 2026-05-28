"""Prometheus metrics for CV Spine observability."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    FRAMES_PROCESSED = Counter("cvspine_frames_processed_total", "Total frames processed",
                               ["camera_id", "product_id"])
    DETECTIONS_COUNT = Counter("cvspine_detections_total", "Total person detections",
                               ["camera_id"])
    INFERENCE_LATENCY = Histogram("cvspine_inference_seconds", "Model inference latency",
                                  ["model_name"], buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1])
    GPU_UTILIZATION = Gauge("cvspine_gpu_utilization", "GPU utilization percent", ["device"])
    QUEUE_DEPTH = Gauge("cvspine_queue_depth", "Inference queue depth", ["priority"])
    ACTIVE_TRACKS = Gauge("cvspine_active_tracks", "Currently active tracks", ["camera_id"])
    CAMERA_FPS = Gauge("cvspine_camera_fps", "Camera actual FPS", ["camera_id"])
    EVENTS_PUBLISHED = Counter("cvspine_events_published_total", "Events published to bus",
                               ["event_type"])

    def start_metrics_server(port: int = 9090) -> None:
        start_http_server(port)
        logger.info("Metrics server started on :%d", port)

except ImportError:
    logger.warning("prometheus_client not installed, metrics disabled")

    class _Stub:
        def labels(self, *a, **kw): return self
        def inc(self, *a, **kw): pass
        def dec(self, *a, **kw): pass
        def set(self, *a, **kw): pass
        def observe(self, *a, **kw): pass

    FRAMES_PROCESSED = _Stub()
    DETECTIONS_COUNT = _Stub()
    INFERENCE_LATENCY = _Stub()
    GPU_UTILIZATION = _Stub()
    QUEUE_DEPTH = _Stub()
    ACTIVE_TRACKS = _Stub()
    CAMERA_FPS = _Stub()
    EVENTS_PUBLISHED = _Stub()

    def start_metrics_server(port: int = 9090) -> None:
        pass
