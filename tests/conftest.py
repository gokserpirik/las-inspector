"""Shared fixtures and binary builders for las-inspector tests."""

import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from las_inspector import LasFile, LasHeader, PointRecord  # noqa: E402

HEADER_SIZE = 227
RECORD_LEN = 20
SCALE = 0.01


def make_point(
    x=0.0,
    y=0.0,
    z=0.0,
    classification=2,
    intensity=100,
    return_number=1,
    number_of_returns=1,
    flag_synthetic=False,
    flag_keypoint=False,
    flag_withheld=False,
    gps_time=0.0,
) -> PointRecord:
    return PointRecord(
        x=x,
        y=y,
        z=z,
        classification=classification,
        intensity=intensity,
        return_number=return_number,
        number_of_returns=number_of_returns,
        flag_synthetic=flag_synthetic,
        flag_keypoint=flag_keypoint,
        flag_withheld=flag_withheld,
        gps_time=gps_time,
    )


def make_las(
    points,
    min_x=0.0,
    max_x=10.0,
    min_y=0.0,
    max_y=10.0,
    min_z=0.0,
    max_z=5.0,
    filename="test.las",
) -> LasFile:
    header = LasHeader(
        point_count=len(points),
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        min_z=min_z,
        max_z=max_z,
    )
    return LasFile(header=header, points=list(points), filename=filename)


def pack_point(
    x=1.0,
    y=2.0,
    z=3.0,
    intensity=100,
    return_number=1,
    number_of_returns=1,
    classification=2,
    synthetic=False,
    keypoint=False,
    withheld=False,
) -> bytes:
    """Pack one LAS 1.2 PDRF-0 record (20 bytes)."""
    xi, yi, zi = (round(v / SCALE) for v in (x, y, z))
    b14 = (return_number & 0x07) | ((number_of_returns & 0x07) << 3)
    b15 = (classification & 0x1F) | (0x20 if synthetic else 0) | (0x40 if keypoint else 0) | (
        0x80 if withheld else 0
    )
    return struct.pack("<3iHBB4s", xi, yi, zi, intensity, b14, b15, b"\x00" * 4)


def build_las_bytes(point_blobs, version=(1, 2), signature=b"LASF") -> bytes:
    """Assemble a minimal LAS header + point records."""
    buf = bytearray(HEADER_SIZE)
    struct.pack_into("4s", buf, 0, signature)
    struct.pack_into("<H", buf, 6, 0)  # global encoding
    struct.pack_into("BB", buf, 24, *version)
    struct.pack_into("<H", buf, 94, HEADER_SIZE)
    struct.pack_into("<I", buf, 96, HEADER_SIZE)  # offset to point data
    struct.pack_into("<I", buf, 100, 0)  # no VLRs
    struct.pack_into("B", buf, 104, 0)  # PDRF 0
    struct.pack_into("<H", buf, 105, RECORD_LEN)
    struct.pack_into("<I", buf, 107, len(point_blobs))  # legacy count
    struct.pack_into(
        "<12d",
        buf,
        131,
        SCALE,
        SCALE,
        SCALE,
        0.0,
        0.0,
        0.0,
        11.0,
        0.0,  # max_x, min_x
        12.0,
        0.0,  # max_y, min_y
        6.0,
        0.0,  # max_z, min_z
    )
    return bytes(buf) + b"".join(point_blobs)


@pytest.fixture
def las_file() -> LasFile:
    """200 ground points over a 10x10 m area (density 2 pts/m²)."""
    points = [
        make_point(
            x=(i % 20) * 0.5,
            y=(i // 20) * 1.0,
            z=float(i % 5),
            classification=2,
            intensity=100 + i,
            gps_time=123456.0 + i,
        )
        for i in range(200)
    ]
    return make_las(points)


@pytest.fixture
def sample_las_path(tmp_path) -> Path:
    """A small valid .las file on disk (5 points)."""
    blobs = [pack_point(x=float(i), classification=2 if i < 4 else 5) for i in range(5)]
    path = tmp_path / "sample.las"
    path.write_bytes(build_las_bytes(blobs))
    return path
