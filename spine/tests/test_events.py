"""Tests for spine event system."""

import time

from spine.core.event_bus import EventBus
from spine.core.events import DetectionEvent, Detection, EventType, SpineEvent


def test_event_creation():
    event = SpineEvent(
        event_type=EventType.PERSON_DETECTED,
        camera_id="cam-01",
        timestamp=time.time(),
    )
    assert event.event_type == EventType.PERSON_DETECTED
    assert event.camera_id == "cam-01"


def test_detection_event():
    det = Detection(track_id=1, bbox=[10, 20, 100, 300], confidence=0.85)
    event = DetectionEvent(
        camera_id="cam-01",
        detections=[det],
        person_count=1,
        inference_ms=3.2,
        frame_w=1280,
        frame_h=720,
    )
    assert event.person_count == 1
    assert event.detections[0].track_id == 1


def test_event_bus_subscribe():
    bus = EventBus()
    received = []

    def handler(event):
        received.append(event)

    bus.subscribe("person.detected", handler)
    event = SpineEvent(
        event_type=EventType.PERSON_DETECTED,
        camera_id="cam-01",
        product_id="test",
    )
    bus.publish(event)
    assert len(received) == 1


def test_event_topic():
    event = SpineEvent(
        event_type=EventType.LINE_CROSSED,
        camera_id="cam-lobby",
        product_id="vibecheck",
    )
    topic = event.topic()
    assert "vibecheck" in topic
    assert "cam-lobby" in topic
