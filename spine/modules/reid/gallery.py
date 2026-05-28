"""Gallery manager — Qdrant vector DB integration for persistent ReID."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class GalleryManager:
    """Manages ReID galleries via Qdrant vector DB (or in-memory fallback)."""

    def __init__(self, qdrant_url: str = "http://localhost:6333", collection: str = "reid_embeddings"):
        self._qdrant_url = qdrant_url
        self._collection = collection
        self._client = None
        self._fallback: dict[str, dict] = {}

    def connect(self) -> bool:
        try:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(url=self._qdrant_url)
            collections = self._client.get_collections().collections
            names = [c.name for c in collections]
            if self._collection not in names:
                from qdrant_client.models import Distance, VectorParams
                self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
                )
            logger.info("Qdrant connected: %s", self._qdrant_url)
            return True
        except Exception:
            logger.warning("Qdrant unavailable, using in-memory gallery")
            return False

    def add(self, gallery_id: str, embedding: np.ndarray, metadata: dict[str, Any]) -> None:
        if self._client:
            from qdrant_client.models import PointStruct
            import uuid
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={"gallery_id": gallery_id, **metadata},
            )
            self._client.upsert(collection_name=self._collection, points=[point])
        else:
            self._fallback[gallery_id] = {"embedding": embedding, "metadata": metadata}

    def search(self, embedding: np.ndarray, top_k: int = 5, threshold: float = 0.75) -> list[dict]:
        if self._client:
            results = self._client.search(
                collection_name=self._collection,
                query_vector=embedding.tolist(),
                limit=top_k,
                score_threshold=threshold,
            )
            return [
                {"gallery_id": r.payload.get("gallery_id", ""), "score": r.score, **r.payload}
                for r in results
            ]
        else:
            matches = []
            for gid, data in self._fallback.items():
                sim = float(np.dot(embedding, data["embedding"]) /
                           (np.linalg.norm(embedding) * np.linalg.norm(data["embedding"]) + 1e-8))
                if sim >= threshold:
                    matches.append({"gallery_id": gid, "score": sim, **data["metadata"]})
            matches.sort(key=lambda x: x["score"], reverse=True)
            return matches[:top_k]

    def delete(self, gallery_id: str) -> None:
        if self._client:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            self._client.delete(
                collection_name=self._collection,
                points_selector=Filter(
                    must=[FieldCondition(key="gallery_id", match=MatchValue(value=gallery_id))]
                ),
            )
        else:
            self._fallback.pop(gallery_id, None)

    def purge_expired(self, max_age_seconds: float, current_time: float) -> int:
        if not self._client:
            expired = [
                gid for gid, data in self._fallback.items()
                if current_time - data["metadata"].get("enrolled_at", 0) > max_age_seconds
            ]
            for gid in expired:
                del self._fallback[gid]
            return len(expired)
        return 0
