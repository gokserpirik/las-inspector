"""Tests for the LAS/LAZ binary reader."""

import pytest

from conftest import build_las_bytes, pack_point
from las_inspector import read_las


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_las(str(tmp_path / "nope.las"))


def test_bad_signature_raises(tmp_path):
    path = tmp_path / "bad.las"
    path.write_bytes(build_las_bytes([pack_point()], signature=b"BAD!"))
    with pytest.raises(ValueError, match="Not a valid LAS file"):
        read_las(str(path))


def test_unsupported_version_raises(tmp_path):
    path = tmp_path / "v2.las"
    path.write_bytes(build_las_bytes([pack_point()], version=(2, 0)))
    with pytest.raises(ValueError, match="Unsupported LAS version"):
        read_las(str(path))


def test_valid_file_parses(sample_las_path):
    las = read_las(str(sample_las_path))
    assert las.filename == "sample.las"
    assert las.header.point_count == 5
    assert las.header.version_minor == 2
    assert len(las.points) == 5
    assert [p.classification for p in las.points] == [2, 2, 2, 2, 5]
    assert las.points[0].x == pytest.approx(0.0)
    assert las.points[3].x == pytest.approx(3.0)


def test_flags_and_returns_parsed(tmp_path):
    blob = pack_point(return_number=2, number_of_returns=3, synthetic=True, withheld=True)
    path = tmp_path / "flags.las"
    path.write_bytes(build_las_bytes([blob]))
    (rec,) = read_las(str(path)).points
    assert rec.return_number == 2
    assert rec.number_of_returns == 3
    assert rec.flag_synthetic is True
    assert rec.flag_withheld is True
    assert rec.flag_keypoint is False


def test_max_points_limits_parsing(tmp_path):
    blobs = [pack_point(x=float(i)) for i in range(10)]
    path = tmp_path / "many.las"
    path.write_bytes(build_las_bytes(blobs))
    las = read_las(str(path), max_points=3)
    assert len(las.points) == 3
    assert las.header.point_count == 3


def test_laz_without_vlr_raises(tmp_path):
    path = tmp_path / "fake.laz"
    path.write_bytes(build_las_bytes([pack_point()]))
    with pytest.raises(ValueError, match="LASzip VLR"):
        read_las(str(path))
