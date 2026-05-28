"""ReID Embedder — body-based re-identification with persistent feature retention. Pillar 3."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from spine.core.config import ReIDConfig
from spine.core.events import EventType, ReIDEvent
from spine.core.orchestrator import FrameContext, ModuleBase

logger = logging.getLogger(__name__)


class ReIDEmbedder(ModuleBase):
    """Body-based ReID using OSNet/FastReID.

    Unique visitor accuracy via:
    - EMA (exponential moving average) embedding updates per gallery entry
    - Multi-sample confirmation before registering new person
    - Track-to-gallery sticky mapping (track keeps matched gallery ID)
    - Cosine similarity with adaptive threshold
    """

    def __init__(self, config: dict[str, Any] | None = None, event_bus: Any = None):
        super().__init__(config)
        self.reid_config = ReIDConfig(**(config or {}))
        self.event_bus = event_bus
        self._model = None

        # Gallery: gallery_id → averaged embedding
        self._gallery: dict[str, np.ndarray] = {}
        # Gallery metadata
        self._gallery_meta: dict[str, dict] = {}

        # Track → gallery sticky mapping (survives across frames)
        self._track_gallery: dict[int, str] = {}

        # Candidate buffer: track_id → list of embeddings (before promoting to gallery)
        self._candidates: dict[int, list[np.ndarray]] = {}
        self._candidate_ts: dict[int, float] = {}

        # Config
        self._ema_alpha = float((config or {}).get("ema_alpha", 0.3))
        self._min_samples = int((config or {}).get("min_samples", 3))
        self._sim_threshold = float((config or {}).get("similarity_threshold", 0.65))

    def initialize(self) -> None:
        try:
            import torchreid
            self._model = torchreid.utils.FeatureExtractor(
                model_name=self.reid_config.model,
                device="cuda" if self.reid_config.model != "cpu" else "cpu",
            )
            logger.info("ReID model loaded: %s", self.reid_config.model)
        except ImportError:
            logger.warning("torchreid not available, using ONNX fallback stub")
            self._model = self._create_onnx_stub()

        self._initialized = True

    def process(self, ctx: FrameContext) -> None:
        if not self._initialized or not ctx.detections:
            return

        h, w = ctx.frame.shape[:2]

        for det in ctx.detections:
            track_id = det.get("track_id", -1)
            if track_id < 0:
                continue

            bbox = det["bbox"]
            x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(w, bbox[2]), min(h, bbox[3])
            crop = ctx.frame[y1:y2, x1:x2]

            if crop.size == 0 or crop.shape[0] < 32 or crop.shape[1] < 16:
                continue

            embedding = self._extract_embedding(crop)
            if embedding is None:
                continue

            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            emb_ref = f"emb-{track_id}-{int(ctx.timestamp)}"
            det["embedding_ref"] = emb_ref

            # ── Sticky mapping: if track already matched to gallery, update EMA ──
            if track_id in self._track_gallery:
                gid = self._track_gallery[track_id]
                if gid in self._gallery:
                    self._update_gallery_ema(gid, embedding, ctx.timestamp)
                    ctx.reid_matches.append({
                        "track_id": track_id,
                        "gallery_id": gid,
                        "similarity": self._cosine_similarity(embedding, self._gallery[gid]),
                    })
                    continue

            # ── Search gallery for match ──
            match_id, similarity = self._match_gallery(embedding)

            if match_id:
                # Found existing person — bind track, update EMA
                self._track_gallery[track_id] = match_id
                self._update_gallery_ema(match_id, embedding, ctx.timestamp)
                ctx.reid_matches.append({
                    "track_id": track_id,
                    "gallery_id": match_id,
                    "similarity": similarity,
                })
                if self.event_bus:
                    self.event_bus.publish(ReIDEvent(
                        event_type=EventType.REID_MATCH,
                        camera_id=ctx.camera_id,
                        track_id=track_id,
                        timestamp=ctx.timestamp,
                        gallery_id=match_id,
                        similarity=similarity,
                    ))
            else:
                # ── Candidate buffer: collect min_samples before promoting ──
                if track_id not in self._candidates:
                    self._candidates[track_id] = []
                    self._candidate_ts[track_id] = ctx.timestamp

                self._candidates[track_id].append(embedding)

                if len(self._candidates[track_id]) >= self._min_samples:
                    # Promote: average all candidate embeddings
                    avg_emb = np.mean(self._candidates[track_id], axis=0)
                    avg_norm = np.linalg.norm(avg_emb)
                    if avg_norm > 0:
                        avg_emb = avg_emb / avg_norm

                    # Double-check against gallery with averaged embedding
                    re_match_id, re_sim = self._match_gallery(avg_emb)
                    if re_match_id:
                        # Was actually existing person — just needed more samples
                        self._track_gallery[track_id] = re_match_id
                        self._update_gallery_ema(re_match_id, avg_emb, ctx.timestamp)
                        if self.event_bus:
                            self.event_bus.publish(ReIDEvent(
                                event_type=EventType.REID_MATCH,
                                camera_id=ctx.camera_id,
                                track_id=track_id,
                                timestamp=ctx.timestamp,
                                gallery_id=re_match_id,
                                similarity=re_sim,
                            ))
                    else:
                        # Genuinely new person
                        gallery_id = f"person-{track_id}-{int(ctx.timestamp)}"
                        self._gallery[gallery_id] = avg_emb
                        self._gallery_meta[gallery_id] = {
                            "first_seen": self._candidate_ts[track_id],
                            "last_seen": ctx.timestamp,
                            "camera_id": ctx.camera_id,
                            "track_id": track_id,
                            "sample_count": self._min_samples,
                        }
                        self._track_gallery[track_id] = gallery_id
                        if self.event_bus:
                            self.event_bus.publish(ReIDEvent(
                                event_type=EventType.REID_NEW_PERSON,
                                camera_id=ctx.camera_id,
                                track_id=track_id,
                                timestamp=ctx.timestamp,
                                gallery_id=gallery_id,
                                similarity=0.0,
                            ))

                    # Clear candidate buffer
                    del self._candidates[track_id]
                    del self._candidate_ts[track_id]

    def _update_gallery_ema(self, gallery_id: str, new_emb: np.ndarray, ts: float) -> None:
        """Update gallery embedding with exponential moving average."""
        old = self._gallery[gallery_id]
        alpha = self._ema_alpha
        updated = alpha * new_emb + (1 - alpha) * old
        # Re-normalize
        norm = np.linalg.norm(updated)
        if norm > 0:
            updated = updated / norm
        self._gallery[gallery_id] = updated

        if gallery_id in self._gallery_meta:
            self._gallery_meta[gallery_id]["last_seen"] = ts
            self._gallery_meta[gallery_id]["sample_count"] = \
                self._gallery_meta[gallery_id].get("sample_count", 1) + 1

    def _extract_embedding(self, crop: np.ndarray) -> np.ndarray | None:
        try:
            if self._model is None:
                return None

            if hasattr(self._model, "__call__"):
                import cv2
                resized = cv2.resize(crop, (128, 256))
                features = self._model([resized])
                if hasattr(features, "numpy"):
                    return features.numpy().flatten()
                return np.array(features).flatten()
            return None
        except Exception:
            logger.debug("Embedding extraction failed", exc_info=True)
            return None

    def _match_gallery(self, embedding: np.ndarray) -> tuple[str | None, float]:
        if not self._gallery:
            return None, 0.0

        best_id = None
        best_sim = 0.0

        for gid, gemb in self._gallery.items():
            sim = self._cosine_similarity(embedding, gemb)
            if sim > best_sim:
                best_sim = sim
                best_id = gid

        if best_sim >= self._sim_threshold:
            return best_id, best_sim
        return None, best_sim

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _create_onnx_stub(self):
        """Deterministic stub: same crop → same embedding (hash-based)."""
        class Stub:
            def __call__(self, imgs):
                results = []
                for img in imgs:
                    # Hash pixel data for deterministic output
                    h = hash(img.tobytes()) % (2**32)
                    rng = np.random.RandomState(h)
                    emb = rng.randn(512).astype(np.float32)
                    emb /= np.linalg.norm(emb)
                    results.append(emb)
                return np.array(results)
        return Stub()

    def purge_expired(self, current_time: float) -> int:
        expired = [
            gid for gid, meta in self._gallery_meta.items()
            if current_time - meta.get("last_seen", meta["first_seen"]) > self.reid_config.gallery_ttl
        ]
        for gid in expired:
            del self._gallery[gid]
            del self._gallery_meta[gid]
            # Clean track mappings pointing to expired gallery
            dead_tracks = [t for t, g in self._track_gallery.items() if g == gid]
            for t in dead_tracks:
                del self._track_gallery[t]
        return len(expired)

    def get_unique_count(self) -> int:
        """Total unique visitors = gallery size."""
        return len(self._gallery)

    def get_gallery_summary(self) -> list[dict]:
        """Summary of all known persons."""
        return [
            {"gallery_id": gid, **meta}
            for gid, meta in self._gallery_meta.items()
        ]

    def cleanup(self) -> None:
        self._model = None
        self._gallery.clear()
        self._gallery_meta.clear()
        self._track_gallery.clear()
        self._candidates.clear()
        self._candidate_ts.clear()
