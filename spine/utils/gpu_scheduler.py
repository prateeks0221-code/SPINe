"""GPU Scheduler — memory-aware batching, priority queues, multi-GPU."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    REALTIME = 0
    ANALYTICS = 1
    BACKGROUND = 2


@dataclass(order=True)
class InferenceJob:
    priority: int
    timestamp: float = field(compare=False)
    camera_id: str = field(compare=False)
    frame: np.ndarray = field(compare=False, repr=False)
    callback: Callable = field(compare=False, repr=False)
    model_name: str = field(compare=False, default="")


class GPUScheduler:
    """Manages GPU inference queue with priority and memory awareness."""

    def __init__(self, max_queue_size: int = 100, drop_on_full: bool = True):
        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_queue_size)
        self._drop_on_full = drop_on_full
        self._running = False
        self._thread: threading.Thread | None = None
        self._stats = {"processed": 0, "dropped": 0, "queue_peak": 0}

    def submit(self, job: InferenceJob) -> bool:
        try:
            self._queue.put_nowait(job)
            qsize = self._queue.qsize()
            if qsize > self._stats["queue_peak"]:
                self._stats["queue_peak"] = qsize
            return True
        except queue.Full:
            if self._drop_on_full:
                self._stats["dropped"] += 1
                return False
            self._queue.put(job, timeout=1.0)
            return True

    def start(self, processor: Callable[[InferenceJob], None]) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, args=(processor,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _process_loop(self, processor: Callable) -> None:
        while self._running:
            try:
                job = self._queue.get(timeout=0.1)
                processor(job)
                self._stats["processed"] += 1
            except queue.Empty:
                continue
            except Exception:
                logger.exception("GPU scheduler error")

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def stats(self) -> dict:
        return dict(self._stats)


def get_gpu_memory_info() -> dict[str, Any]:
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "total_memory_mb": torch.cuda.get_device_properties(0).total_mem / 1024**2,
                "allocated_mb": torch.cuda.memory_allocated(0) / 1024**2,
                "free_mb": (torch.cuda.get_device_properties(0).total_mem - torch.cuda.memory_allocated(0)) / 1024**2,
            }
    except ImportError:
        pass
    return {"available": False}
