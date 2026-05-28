"""Tests for ROI zone manager."""

from spine.modules.roi.zone_manager import ZoneManager


def test_point_in_polygon():
    assert ZoneManager._point_in_polygon(5, 5, [[0, 0], [10, 0], [10, 10], [0, 10]])
    assert not ZoneManager._point_in_polygon(15, 5, [[0, 0], [10, 0], [10, 10], [0, 10]])


def test_point_in_triangle():
    triangle = [[0, 0], [10, 0], [5, 10]]
    assert ZoneManager._point_in_polygon(5, 3, triangle)
    assert not ZoneManager._point_in_polygon(0, 10, triangle)
