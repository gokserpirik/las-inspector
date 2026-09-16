"""Tests for LasHeader derived properties."""

from las_inspector import LasHeader


def test_area_2d():
    h = LasHeader(min_x=0.0, max_x=10.0, min_y=0.0, max_y=5.0)
    assert h.area_2d == 50.0


def test_point_density():
    h = LasHeader(point_count=200, min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0)
    assert h.point_density == 2.0


def test_point_density_zero_area_returns_zero():
    h = LasHeader(point_count=100, min_x=5.0, max_x=5.0, min_y=0.0, max_y=10.0)
    assert h.area_2d == 0.0
    assert h.point_density == 0.0


def test_point_density_negative_area_returns_zero():
    h = LasHeader(point_count=100, min_x=10.0, max_x=0.0, min_y=0.0, max_y=10.0)
    assert h.point_density == 0.0


def test_ranges():
    h = LasHeader(min_x=1.0, max_x=4.0, min_y=2.0, max_y=8.0, min_z=-1.0, max_z=3.0)
    assert h.x_range == 3.0
    assert h.y_range == 6.0
    assert h.z_range == 4.0
