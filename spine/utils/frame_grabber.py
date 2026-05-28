"""Frame Grabber — multi-protocol video ingest with auto-reconnect."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameGrabber:
    """Multi-protocol video reader. Always latest frame, no buffer bloat."""

    def __init__(self, source: str, camera_id: str, target_fps: float = 10.0,
                 resolution: tuple[int, int] | None = None):
        self.source = source
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.resolution = resolution
        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._fps_actual = 0.0
        self._frame_count = 0
        self._reconnect_attempts = 0
        self._max_reconnect = 10
        self._last_frame_time = 0.0

    def start(self) -> bool:
        if not self._connect():
            return False
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        logger.info("FrameGrabber started: %s (%s)", self.camera_id, self.source)
        return True

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self._cap:
            self._cap.release()
        logger.info("FrameGrabber stopped: %s", self.camera_id)

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def is_alive(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    @property
    def fps(self) -> float:
        return self._fps_actual

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def _connect(self) -> bool:
        try:
            if self.source.startswith("rtsp://") or self.source.startswith("rtmp://"):
                self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            elif self.source.isdigit():
                self._cap = cv2.VideoCapture(int(self.source))
            else:
                self._cap = cv2.VideoCapture(self.source)

            if not self._cap.isOpened():
                logger.error("Cannot open source: %s", self.source)
                return False

            self._reconnect_attempts = 0
            return True
        except Exception:
            logger.exception("Connection failed: %s", self.source)
            return False

    def _read_loop(self) -> None:
        frame_interval = 1.0 / self.target_fps
        fps_counter = 0
        fps_start = time.time()

        while self._running:
            if self._cap is None or not self._cap.isOpened():
                if not self._reconnect():
                    break
                continue

            ret, frame = self._cap.read()
            if not ret:
                if not self._reconnect():
                    break
                continue

            if self.resolution:
                frame = cv2.resize(frame, self.resolution)

            with self._lock:
                self._frame = frame
                self._frame_count += 1
                self._last_frame_time = time.time()

            fps_counter += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                self._fps_actual = fps_counter / elapsed
                fps_counter = 0
                fps_start = time.time()

            sleep_time = frame_interval - (time.time() - self._last_frame_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _reconnect(self) -> bool:
        self._reconnect_attempts += 1
        if self._reconnect_attempts > self._max_reconnect:
            logger.error("Max reconnect attempts reached: %s", self.camera_id)
            self._running = False
            return False

        wait = min(2 ** self._reconnect_attempts, 30)
        logger.warning("Reconnecting %s (attempt %d, wait %ds)",
                      self.camera_id, self._reconnect_attempts, wait)
        time.sleep(wait)

        if self._cap:
            self._cap.release()
        return self._connect()

    def get_health(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "source": self.source,
            "alive": self.is_alive,
            "fps": round(self._fps_actual, 1),
            "frames": self._frame_count,
            "reconnects": self._reconnect_attempts,
            "last_frame_age": time.time() - self._last_frame_time if self._last_frame_time else -1,
        }
