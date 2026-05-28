"""Age/gender estimation — MiVOLO body-aware demographics."""

from __future__ import annotations

import logging
from typing import Any

from spine.core.events import EventType, SpineEvent
from spine.core.orchestrator import FrameContext, ModuleBase

logger = logging.getLogger(__name__)


class DemographicsEstimator(ModuleBase):
    """Estimates age range and presenting gender from face/body."""

    def __init__(self, config: dict[str, Any] | None = None, event_bus: Any = None):
        super().__init__(config)
        self.event_bus = event_bus
        self._model = None

    def initialize(self) -> None:
        self._initialized = True
        logger.info("DemographicsEstimator ready (stub — model load on demand)")

    def process(self, ctx: FrameContext) -> None:
        if not self._initialized:
            return

        for face in ctx.face_results:
            age_range = face.get("age", "unknown")
            gender = face.get("gender", "unknown")

            if self.event_bus:
                self.event_bus.publish(SpineEvent(
                    event_type=EventType.FACE_DEMOGRAPHICS,
                    camera_id=ctx.camera_id,
                    timestamp=ctx.timestamp,
                    data={"age_range": age_range, "gender_presenting": gender},
                ))
