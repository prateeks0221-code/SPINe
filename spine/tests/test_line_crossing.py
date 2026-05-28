"""Tests for line crossing detection."""

from spine.modules.entry_exit.line_crossing import LineCrossingDetector


def test_segments_intersect():
    assert LineCrossingDetector._segments_intersect(0, 0, 10, 10, 0, 10, 10, 0)
    assert not LineCrossingDetector._segments_intersect(0, 0, 5, 5, 6, 6, 10, 10)


def test_parallel_no_intersect():
    assert not LineCrossingDetector._segments_intersect(0, 0, 10, 0, 0, 5, 10, 5)
