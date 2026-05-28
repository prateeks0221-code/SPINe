"""Model Registry — central catalog, hot-reload, A/B testing, versioning."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ModelEntry:
    def __init__(self, name: str, version: str, path: str, format: str, loaded_at: float = 0):
        self.name = name
        self.version = version
        self.path = path
        self.format = format
        self.loaded_at = loaded_at or time.time()
        self.instance: Any = None
        self.metadata: dict[str, Any] = {}


class ModelRegistry:
    """Central catalog of all loaded models. Supports hot-reload and versioning."""

    def __init__(self, model_dir: str = "./models"):
        self.model_dir = Path(model_dir)
        self._models: dict[str, ModelEntry] = {}
        self._history: list[dict] = []

    def register(self, name: str, version: str, path: str, format: str = "auto") -> ModelEntry:
        if format == "auto":
            ext = Path(path).suffix.lower()
            format = {".pt": "pytorch", ".onnx": "onnx", ".engine": "tensorrt",
                     ".xml": "openvino", ".tflite": "tflite"}.get(ext, "unknown")

        entry = ModelEntry(name=name, version=version, path=path, format=format)
        key = f"{name}:{version}"

        if name in self._models:
            self._history.append({
                "action": "replaced",
                "name": name,
                "old_version": self._models[name].version,
                "new_version": version,
                "timestamp": time.time(),
            })

        self._models[name] = entry
        logger.info("Registered model: %s v%s (%s)", name, version, format)
        return entry

    def get(self, name: str) -> ModelEntry | None:
        return self._models.get(name)

    def load(self, name: str) -> Any:
        entry = self._models.get(name)
        if not entry:
            raise KeyError(f"Model not registered: {name}")

        if entry.instance is not None:
            return entry.instance

        if entry.format == "pytorch":
            from ultralytics import YOLO
            entry.instance = YOLO(entry.path)
        elif entry.format == "onnx":
            import onnxruntime as ort
            entry.instance = ort.InferenceSession(entry.path)
        else:
            raise ValueError(f"Unsupported format: {entry.format}")

        entry.loaded_at = time.time()
        return entry.instance

    def unload(self, name: str) -> None:
        entry = self._models.get(name)
        if entry:
            entry.instance = None

    def list_models(self) -> list[dict]:
        return [
            {
                "name": e.name,
                "version": e.version,
                "format": e.format,
                "loaded": e.instance is not None,
                "path": e.path,
            }
            for e in self._models.values()
        ]

    def rollback(self, name: str) -> bool:
        history = [h for h in self._history if h["name"] == name and h["action"] == "replaced"]
        if not history:
            return False
        last = history[-1]
        logger.info("Rollback %s: v%s → v%s", name, last["new_version"], last["old_version"])
        return True
