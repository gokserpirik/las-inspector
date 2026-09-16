"""Tests for compute_stats QC logic."""

from conftest import make_las, make_point
from las_inspector import compute_stats


def _points(n, **kwargs):
    return [make_point(x=float(i), y=float(i), **kwargs) for i in range(n)]


def test_counts_and_ranges():
    pts = _points(8, classification=2, intensity=50, gps_time=1000.0)
    pts += [make_point(classification=5, intensity=200, flag_withheld=True, gps_time=1010.0)]
    pts += [make_point(classification=1, intensity=150, flag_synthetic=True, flag_keypoint=True)]
    las = make_las(pts)  # 10 pts over 100 m²
    stats, _ = compute_stats(las)
    assert stats.total_points == 10
    assert stats.classification_counts == {2: 8, 5: 1, 1: 1}
    assert stats.return_counts == {1: 10}
    assert stats.withheld_count == 1
    assert stats.synthetic_count == 1
    assert stats.keypoint_count == 1
    assert stats.intensity_range == (50, 200)
    assert stats.has_gps_time is True
    assert stats.gps_time_range == (1000.0, 1010.0)


def test_no_gps_time():
    las = make_las(_points(5, classification=2))
    stats, _ = compute_stats(las)
    assert stats.has_gps_time is False
    assert stats.gps_time_range is None


def test_withheld_warning_above_threshold():
    pts = _points(94, classification=2) + _points(6, classification=2, flag_withheld=True)
    _, warnings = compute_stats(make_las(pts))
    withheld = [w for w in warnings if w.category == "withheld"]
    assert withheld and withheld[0].severity == "warning"


def test_withheld_info_below_threshold():
    pts = _points(99, classification=2) + [make_point(classification=2, flag_withheld=True)]
    _, warnings = compute_stats(make_las(pts))
    withheld = [w for w in warnings if w.category == "withheld"]
    assert withheld and withheld[0].severity == "info"


def test_synthetic_warning():
    pts = _points(98, classification=2) + _points(2, classification=2, flag_synthetic=True)
    _, warnings = compute_stats(make_las(pts))
    synthetic = [w for w in warnings if w.category == "synthetic"]
    assert synthetic and synthetic[0].severity == "warning"


def test_few_ground_points_warns():
    pts = _points(100, classification=5)  # no ground at all
    _, warnings = compute_stats(make_las(pts))
    assert any(w.category == "classification" for w in warnings)


def test_low_density_warns(las_file):
    # fixture: 200 pts over 100 m² -> 2 pts/m²
    _, warnings = compute_stats(las_file)
    density = [w for w in warnings if w.category == "density"]
    assert density and "Low point density" in density[0].message


def test_healthy_file_has_no_warnings():
    pts = _points(200, classification=2, return_number=2, number_of_returns=2)
    # 200 pts over 10 m² -> 20 pts/m², all ground, no flags
    las = make_las(pts, max_x=5.0, max_y=2.0)
    _, warnings = compute_stats(las)
    assert warnings == []


def test_empty_file_does_not_crash():
    las = make_las([])
    stats, warnings = compute_stats(las)
    assert stats.total_points == 0
    assert stats.intensity_range == (0, 0)
    assert any(w.category == "classification" for w in warnings)
