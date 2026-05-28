"""Tests for frame deduplication."""

import numpy as np

from spine.utils.frame_dedup import FrameDeduplicator


def test_identical_frames_skipped():
    dedup = FrameDeduplicator(threshold=5)
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    assert dedup.should_process(frame, "cam1") is True
    assert dedup.should_process(frame, "cam1") is False


def test_different_frames_processed():
    dedup = FrameDeduplicator(threshold=5)
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 255
    assert dedup.should_process(frame1, "cam1") is True
    assert dedup.should_process(frame2, "cam1") is True
