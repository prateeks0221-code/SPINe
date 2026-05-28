"""Event bus — MQTT + Redis pub/sub abstraction. Spine publishes, products subscribe."""

from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from typing import Any, Callable

from spine.core.events import SpineEvent

logger = logging.getLogger(__name__)

EventHandler = Callable[[SpineEvent], None]


class EventBus:
    """Dual-backend event bus: in-process for dev, MQTT/Redis for production."""

    def __init__(self, mqtt_client: Any | None = None, redis_client: Any | None = None):
        self._mqtt = mqtt_client
        self._redis = redis_client
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._lock = threading.RLock()  # Reentrant: handlers can publish events

    def publish(self, event: SpineEvent) -> None:
        topic = event.topic()
        payload = event.model_dump_json()

        if self._mqtt:
            qos = 2 if "alert" in event.event_type.value else (1 if "reid" in event.event_type.value or "line" in event.event_type.value or "face" in event.event_type.value else 0)
            self._mqtt.publish(topic, payload, qos=qos)

        if self._redis:
            self._redis.publish(topic, payload)

        with self._lock:
            for pattern, handlers in self._handlers.items():
                if self._topic_matches(pattern, topic) or self._topic_matches(pattern, event.event_type.value):
                    for handler in handlers:
                        try:
                            handler(event)
                        except Exception:
                            logger.exception("Handler error for %s", topic)

    def subscribe(self, pattern: str, handler: EventHandler) -> None:
        with self._lock:
            self._handlers[pattern].append(handler)

    def unsubscribe(self, pattern: str, handler: EventHandler) -> None:
        with self._lock:
            if pattern in self._handlers:
                self._handlers[pattern] = [h for h in self._handlers[pattern] if h is not handler]

    @staticmethod
    def _topic_matches(pattern: str, topic: str) -> bool:
        if pattern == topic:
            return True
        if pattern.endswith("/#"):
            return topic.startswith(pattern[:-2])
        if pattern.endswith("/*"):
            prefix = pattern[:-2]
            rest = topic[len(prefix) + 1:] if topic.startswith(prefix + "/") else ""
            return "/" not in rest and len(rest) > 0
        if "*" in pattern:
            return topic.startswith(pattern.split("*")[0])
        return False


class MQTTEventBus(EventBus):
    """Production event bus backed by MQTT broker."""

    def __init__(self, broker: str = "localhost", port: int = 1883, client_id: str = "cv-spine"):
        try:
            import paho.mqtt.client as mqtt
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
            client.connect(broker, port)
            client.loop_start()
            super().__init__(mqtt_client=client)
            logger.info("MQTT connected: %s:%d", broker, port)
        except Exception:
            logger.warning("MQTT unavailable, falling back to in-process bus")
            super().__init__()


class RedisEventBus(EventBus):
    """Production event bus backed by Redis pub/sub."""

    def __init__(self, url: str = "redis://localhost:6379"):
        try:
            import redis
            client = redis.from_url(url)
            client.ping()
            super().__init__(redis_client=client)
            logger.info("Redis connected: %s", url)
        except Exception:
            logger.warning("Redis unavailable, falling back to in-process bus")
            super().__init__()
