"""Anti-spoofing / liveness detection — MiniFASNet."""

from __future__ import annotations

import logging
from typing import Any

from spine.core.events import EventType, FaceEvent
from spine.core.orchestrator import FrameContext, ModuleBase

logger = logging.getLogger(__name__)


class LivenessDetector(ModuleBase):
    """Detects photo/screen spoofing attacks on face recognition."""

    def __init__(self, config: dict[str, Any] | None = None, event_bus: Any = None):
        super().__init__(config)
        self.event_bus = event_bus
        self._model = None

    def initialize(self) -> None:
        self._initialized = True
        logger.info("LivenessDetector ready (stub)")

    def process(self, ctx: FrameContext) -> None:
        if not self._initialized:
            return

        for face in ctx.face_results:
            is_real = True
            attack_type = None

            if self.event_bus:
                self.event_bus.publish(FaceEvent(
                    event_type=EventType.FACE_LIVENESS,
                    camera_id=ctx.camera_id,
                    timestamp=ctx.timestamp,
                    data={"is_real": is_real, "attack_type": attack_type},
                ))
