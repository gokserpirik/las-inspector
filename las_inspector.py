#!/usr/bin/env python3
"""
las-inspector — Automated LiDAR QC reports from your terminal.

Usage:
    python las_inspector.py input.laz --output report.html
    python las_inspector.py survey.laz --output qc.html --density

Reads any LAS/LAZ file and produces a client-ready quality report:
  - Point statistics (count, density, returns, classification breakdown)
  - Spatial bounds and CRS detection
  - QC flags (withheld points, synthetic points, overlap zones)
  - Density heatmap (PNG)
  - Self-contained HTML report
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import struct
import sys
import textwrap
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Iterator, Optional

try:
    import numpy as np
except ImportError:
    sys.exit("Missing dependency: numpy (pip install numpy)")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore

HAS_LAZ = False
try:
    import lazrs  # noqa: F811

    HAS_LAZ = True
except ImportError:
    pass

# ────────────────────────────────────────────────────────────────
# 1. Data structures
# ────────────────────────────────────────────────────────────────


@dataclass
class LasHeader:
    version_minor: int = 0
    point_format_id: int = 0
    point_record_length: int = 0
    point_count: int = 0
    x_scale: float = 0.01
    y_scale: float = 0.01
    z_scale: float = 0.01
    x_offset: float = 0.0
    y_offset: float = 0.0
    z_offset: float = 0.0
    max_x: float = 0.0
    min_x: float = 0.0
    max_y: float = 0.0
    min_y: float = 0.0
    max_z: float = 0.0
    min_z: float = 0.0
    crs_wkt: str = ""

    @property
    def x_range(self) -> float:
        return self.max_x - self.min_x

    @property
    def y_range(self) -> float:
        return self.max_y - self.min_y

    @property
    def z_range(self) -> float:
        return self.max_z - self.min_z

    @property
    def area_2d(self) -> float:
        return self.x_range * self.y_range

    @property
    def point_density(self) -> float:
        return self.point_count / self.area_2d if self.area_2d > 0 else 0.0


@dataclass
class PointRecord:
    x: float
    y: float
    z: float
    classification: int = 0
    intensity: int = 0
    return_number: int = 1
    number_of_returns: int = 1
    flag_synthetic: bool = False
    flag_keypoint: bool = False
    flag_withheld: bool = False
    gps_time: float = 0.0


@dataclass
class LasFile:
    header: LasHeader = field(default_factory=LasHeader)
    points: list[PointRecord] = field(default_factory=list)
    filename: str = ""


@dataclass
class QcWarning:
    severity: str  # "info" | "warning" | "error"
    category: str
    message: str


CLASS_NAMES = {
    0: "Created / never classified",
    1: "Unclassified",
    2: "Ground",
    3: "Low Vegetation",
    4: "Medium Vegetation",
    5: "High Vegetation",
    6: "Building",
    7: "Low Point (noise)",
    8: "Model Key-Point (mass point)",
    9: "Water",
    10: "Rail",
    11: "Road Surface",
    12: "Overlap / Reserved",
    13: "Wire – Guard (Shield)",
    14: "Wire – Conductor (Phase)",
    15: "Transmission Tower",
    16: "Wire-Structure Connector",
    17: "Bridge Deck",
    18: "High Noise",
}

# ────────────────────────────────────────────────────────────────
# 2. LAS/LAZ binary reader
# ────────────────────────────────────────────────────────────────


def _read_uint8(f: BinaryIO) -> int:
    return struct.unpack("B", f.read(1))[0]


def _read_uint16(f: BinaryIO) -> int:
    return struct.unpack("<H", f.read(2))[0]


def _read_uint32(f: BinaryIO) -> int:
    return struct.unpack("<I", f.read(4))[0]


def _read_uint64(f: BinaryIO) -> int:
    return struct.unpack("<Q", f.read(8))[0]


def _read_float64(f: BinaryIO) -> float:
    return struct.unpack("<d", f.read(8))[0]


def _read_char(f: BinaryIO, n: int) -> bytes:
    return f.read(n)


def _unpack_bitfield(byte: int) -> dict:
    """Unpack classification / flag byte (LAS 1.2+ format 0-5)."""
    return {
        "synthetic": bool(byte & 0x20),
        "keypoint": bool(byte & 0x40),
        "withheld": bool(byte & 0x80),
        "classification": byte & 0x1F,
    }


def read_las(path: str, max_points: int | None = None) -> LasFile:
    """Parse a LAS 1.2–1.4 or LAZ file into a LasFile dataclass."""
    path = os.path.abspath(path)
    raw = Path(path).read_bytes()
    is_laz = path.lower().endswith(".laz")

    if is_laz and not HAS_LAZ:
        sys.exit(
            "LAZ support requires the 'lazrs' package.\n"
            "  pip install lazrs\n"
            "  (or convert to LAS with: laszip input.laz output.las)"
        )

    buf = io.BytesIO(raw)

    # ── Public Header Block ──
    sig = _read_char(buf, 4).decode("ascii")
    if sig != "LASF":
        raise ValueError(f"Not a valid LAS file (signature: {sig!r})")

    buf.seek(6)
    global_encoding = _read_uint16(buf)
    # Allow gps_time_type detection from global_encoding bit 0
    gps_time_type = "adjusted_standard" if (global_encoding & 0x01) else "gps_week"

    buf.seek(24)
    version_major = _read_uint8(buf)
    version_minor = _read_uint8(buf)
    if version_major != 1:
        raise ValueError(f"Unsupported LAS version: {version_major}.{version_minor}")

    buf.seek(94)
    header_size = _read_uint16(buf)

    buf.seek(96)
    offset_to_point_data = _read_uint32(buf)
    num_variable_length_records = _read_uint32(buf)
    point_format_id = _read_uint8(buf)
    point_record_length = _read_uint16(buf)
    legacy_point_count = _read_uint32(buf)

    buf.seek(131)
    x_scale = _read_float64(buf)
    y_scale = _read_float64(buf)
    z_scale = _read_float64(buf)
    x_offset = _read_float64(buf)
    y_offset = _read_float64(buf)
    z_offset = _read_float64(buf)
    max_x = _read_float64(buf)
    min_x = _read_float64(buf)
    max_y = _read_float64(buf)
    min_y = _read_float64(buf)
    max_z = _read_float64(buf)
    min_z = _read_float64(buf)

    # LAS 1.4 has 64-bit point count in the header
    point_count = legacy_point_count
    if version_minor >= 4:
        buf.seek(247)
        point_count_64 = _read_uint64(buf)
        if point_count_64 > 0:
            point_count = point_count_64

    # Normalize point format ID to strip compression flags (e.g. 134 -> 6)
    base_point_format_id = point_format_id & 0x3F

    # Source filename for the report
    source_filename = os.path.basename(path)

    # ── Variable Length Records (VLRs) — scan for CRS & LASzip VLR ──
    crs_wkt = ""
    laszip_vlr_data = None
    buf.seek(header_size)
    for _ in range(num_variable_length_records):
        _reserved = _read_uint16(buf)
        user_id = _read_char(buf, 16).decode("ascii", errors="ignore").strip("\x00").strip()
        record_id = _read_uint16(buf)
        record_length_after_header = _read_uint16(buf)
        _description = _read_char(buf, 32)  # Skip standard 32-byte description field in VLR header
        data = _read_char(buf, record_length_after_header)
        if user_id == "LASF_Projection" and record_id == 2112:
            try:
                wkt_bytes = data.split(b"\x00")[0]
                crs_wkt = wkt_bytes.decode("utf-8", errors="ignore").strip()
            except Exception:
                pass
        elif user_id == "LASF_Projection" and record_id == 34735:
            try:
                crs_wkt = data.decode("utf-8", errors="ignore").strip("\x00").strip()
            except Exception:
                pass
        elif (user_id == "laszip encoded" or user_id == "laszip") and record_id == 22204:
            laszip_vlr_data = data

    # ── Point Data Records ──
    use_14bit_format = version_minor >= 4 and base_point_format_id >= 6
    byte_count = point_record_length if point_record_length > 0 else _guess_record_len(
        base_point_format_id
    )

    points_buf: BinaryIO
    if is_laz:
        if not laszip_vlr_data:
            raise ValueError("LAZ file is missing the LASzip VLR metadata required for decompression.")

        buf.seek(offset_to_point_data)
        import lazrs  # noqa: F811

        points_to_decompress = point_count
        if max_points is not None:
            points_to_decompress = min(point_count, max_points)
        points_to_decompress = max(0, points_to_decompress)

        decompress_size = points_to_decompress * byte_count
        decompressed_points = bytearray(decompress_size)

        decompressor = lazrs.LasZipDecompressor(buf, laszip_vlr_data)
        decompressor.decompress_many(decompressed_points)
        points_buf = io.BytesIO(decompressed_points)
    else:
        buf.seek(offset_to_point_data)
        points_buf = buf

    header = LasHeader(
        version_minor=version_minor,
        point_format_id=base_point_format_id,
        point_record_length=point_record_length,
        point_count=point_count,
        x_scale=x_scale,
        y_scale=y_scale,
        z_scale=z_scale,
        x_offset=x_offset,
        y_offset=y_offset,
        z_offset=z_offset,
        max_x=max_x,
        min_x=min_x,
        max_y=max_y,
        min_y=min_y,
        max_z=max_z,
        min_z=min_z,
        crs_wkt=crs_wkt,
    )

    points: list[PointRecord] = []
    limit = max_points if max_points is not None else point_count if point_count else sys.maxsize

    for _ in range(limit if limit < sys.maxsize else 10_000_000):
        chunk = points_buf.read(byte_count)
        if len(chunk) < byte_count:
            break

        if use_14bit_format:
            # LAS 1.4 PDRF 6-10 have a different bit layout:
            # X/Y/Z are always 3 × 4-byte integers (0-11)
            xi, yi, zi = struct.unpack_from("<3i", chunk, 0)
            x = xi * x_scale + x_offset
            y = yi * y_scale + y_offset
            z = zi * z_scale + z_offset

            # Bytes 12-13: intensity (uint16)
            intensity = struct.unpack_from("<H", chunk, 12)[0]

            # Byte 14: return_number (bits 0-3), number_of_returns (bits 4-7)
            b14 = chunk[14]
            return_number = b14 & 0x0F
            number_of_returns = (b14 >> 4) & 0x0F

            # Byte 15: classification flags (bits 0-3) and others
            b15 = chunk[15]
            flag_synthetic = bool(b15 & 0x01)
            flag_keypoint = bool(b15 & 0x02)
            flag_withheld = bool(b15 & 0x04)

            # Byte 16: classification (uint8)
            classification = chunk[16]

            rec = PointRecord(
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
            )

            # GPS time at byte 22 for formats 6-10 (double, 8 bytes)
            if base_point_format_id in (6, 7, 8, 9, 10):
                if len(chunk) >= 30:
                    rec.gps_time = struct.unpack_from("<d", chunk, 22)[0]
        else:
            # LAS 1.2 PDRF 0-5: standard 3×int32 xyz
            xi, yi, zi = struct.unpack_from("<3i", chunk, 0)
            x = xi * x_scale + x_offset
            y = yi * y_scale + y_offset
            z = zi * z_scale + z_offset

            # Byte 12: intensity (uint16)
            intensity = struct.unpack_from("<H", chunk, 12)[0]

            # Byte 14: return_number (bits 0-2), number_of_returns (bits 3-5)
            b14 = chunk[14]
            return_number = b14 & 0x07
            number_of_returns = (b14 >> 3) & 0x07

            # Byte 15: classification flags
            b15 = chunk[15]
            flags = _unpack_bitfield(b15)

            rec = PointRecord(
                x=x,
                y=y,
                z=z,
                classification=flags["classification"],
                intensity=intensity,
                return_number=return_number,
                number_of_returns=number_of_returns,
                flag_synthetic=flags["synthetic"],
                flag_keypoint=flags["keypoint"],
                flag_withheld=flags["withheld"],
            )

            # Byte 20: GPS time (optional, for formats 1, 3, 4, 5)
            if base_point_format_id in (1, 3, 4, 5):
                if len(chunk) >= 28:
                    rec.gps_time = struct.unpack_from("<d", chunk, 20)[0]

        points.append(rec)
        if len(points) >= limit:
            break

    header.point_count = len(points)
    return LasFile(header=header, points=points, filename=source_filename)


def _guess_record_len(fmt_id: int) -> int:
    return {0: 20, 1: 28, 2: 26, 3: 34, 4: 57, 5: 63, 6: 30, 7: 36, 8: 38, 9: 59, 10: 67}.get(
        fmt_id, 30
    )


# ────────────────────────────────────────────────────────────────
# 3. Statistics
# ────────────────────────────────────────────────────────────────


@dataclass
class LasStats:
    total_points: int = 0
    point_density: float = 0.0
    classification_counts: dict[int, int] = field(default_factory=dict)
    return_counts: dict[int, int] = field(default_factory=dict)
    withheld_count: int = 0
    synthetic_count: int = 0
    keypoint_count: int = 0
    has_gps_time: bool = False
    gps_time_range: tuple[float, float] | None = None
    intensity_range: tuple[int, int] = (0, 0)
    overlaps_detected: bool = False


def _detect_overlap_cells(
    pts: list[PointRecord],
    grid: int = 32,
    min_points: int = 1000,
    gap_seconds: float = 120.0,
    min_pass_fraction: float = 0.10,
) -> str | None:
    """Detect flight-line overlap via 2D spatial binning (np.histogram2d).

    Two independent signals are checked on populated bins:
      * multi-pass cells: GPS times inside a bin split into >= 2 distinct passes, i.e. by a
        gap > ``gap_seconds`` (flight-line turnaround) with each pass holding at least
        ``min_pass_fraction`` of the cell's points (and >= 3 points), so one stray
        timestamp is not enough
      * density spikes: bins whose count exceeds 2x the median populated-bin count
        (bimodal density, typical of overlapping swaths)
    Returns a warning message, or None if no overlap is evident / data too small.
    """
    if len(pts) < min_points:
        return None

    xs = np.fromiter((p.x for p in pts), dtype=np.float64, count=len(pts))
    ys = np.fromiter((p.y for p in pts), dtype=np.float64, count=len(pts))
    if np.ptp(xs) <= 0 or np.ptp(ys) <= 0:
        return None

    hist, x_edges, y_edges = np.histogram2d(xs, ys, bins=grid)
    populated = hist[hist >= 5]
    if populated.size < 4:
        return None

    median = float(np.median(populated))
    spike_cells = int(np.count_nonzero(hist > 2.0 * median))
    spike_frac = spike_cells / populated.size

    multipass_frac = 0.0
    gps = np.fromiter((p.gps_time for p in pts), dtype=np.float64, count=len(pts))
    if np.count_nonzero(gps > 0) > len(pts) * 0.5:
        xi = np.clip(np.digitize(xs, x_edges) - 1, 0, grid - 1)
        yi = np.clip(np.digitize(ys, y_edges) - 1, 0, grid - 1)
        cell = xi * grid + yi
        order = np.lexsort((gps, cell))
        c_sorted, g_sorted = cell[order], gps[order]
        new_seg = np.ones(len(g_sorted), dtype=bool)
        new_seg[1:] = (c_sorted[1:] != c_sorted[:-1]) | (np.diff(g_sorted) > gap_seconds)
        seg_id = np.cumsum(new_seg) - 1
        seg_count = np.bincount(seg_id)
        seg_cell = c_sorted[new_seg]
        cell_count = np.bincount(cell, minlength=grid * grid)
        qualifies = seg_count >= np.maximum(3, min_pass_fraction * cell_count[seg_cell])
        passes_per_cell = np.bincount(seg_cell[qualifies], minlength=grid * grid)
        multipass_cells = int(np.count_nonzero(passes_per_cell >= 2))
        multipass_frac = multipass_cells / populated.size

    if multipass_frac > 0.05:
        return (
            f"Overlap detected: {multipass_frac * 100:.0f}% of populated cells are covered by "
            f"2+ distinct passes (GPS gap > {gap_seconds:.0f}s)"
        )
    if spike_frac > 0.10:
        return (
            f"Density anomaly: {spike_frac * 100:.0f}% of populated cells exceed 2x the "
            f"median density (possible overlapping swaths)"
        )
    return None


def compute_stats(las: LasFile) -> tuple[LasStats, list[QcWarning]]:
    hdr = las.header
    pts = las.points
    stats = LasStats(total_points=hdr.point_count, point_density=hdr.point_density)

    warnings: list[QcWarning] = []
    classifications: Counter[int] = Counter()
    returns: Counter[int] = Counter()
    intensities: list[int] = []
    gps_times: list[float] = []
    withheld = 0
    synthetic = 0
    keypoint = 0

    for p in pts:
        classifications[p.classification] += 1
        returns[p.return_number] += 1
        intensities.append(p.intensity)
        if p.flag_withheld:
            withheld += 1
        if p.flag_synthetic:
            synthetic += 1
        if p.flag_keypoint:
            keypoint += 1
        if p.gps_time > 0:
            gps_times.append(p.gps_time)

        # Overlap detection: approx by checking many points with high number_of_returns
        # in close spatial area — simple proxy: flag if >5% are return 1-of-1
        # Overlap detection is performed after the loop via spatial binning
        # (see _detect_overlap_cells).

    stats.classification_counts = dict(classifications)
    stats.return_counts = dict(returns)
    stats.withheld_count = withheld
    stats.synthetic_count = synthetic
    stats.keypoint_count = keypoint

    if intensities:
        stats.intensity_range = (min(intensities), max(intensities))
    if gps_times:
        stats.has_gps_time = True
        stats.gps_time_range = (min(gps_times), max(gps_times))

    # ── QC checks ──
    withheld_pct = (withheld / len(pts)) * 100 if pts else 0
    if withheld_pct > 5:
        warnings.append(
            QcWarning("warning", "withheld", f"{withheld_pct:.1f}% points flagged 'withheld'")
        )
    elif withheld_pct > 0:
        warnings.append(
            QcWarning("info", "withheld", f"{withheld_pct:.1f}% points flagged 'withheld'")
        )

    synthetic_pct = (synthetic / len(pts)) * 100 if pts else 0
    if synthetic_pct > 1:
        warnings.append(
            QcWarning("warning", "synthetic", f"{synthetic_pct:.1f}% points flagged 'synthetic'")
        )

    if 2 not in classifications or classifications[2] < len(pts) * 0.01:
        warnings.append(
            QcWarning("warning", "classification", "Very few ground-classified points (<1%)")
        )

    if hdr.point_density < 10:
        warnings.append(
            QcWarning("warning", "density", f"Low point density: {hdr.point_density:.0f} pts/m²")
        )
    elif hdr.point_density > 1000:
        warnings.append(QcWarning("info", "density", f"High density: {hdr.point_density:.0f} pts/m²"))

    # Overlap heuristic: spatial binning (multi-pass GPS cells / density spikes) first,
    # then the original fallback (high density + many 1st returns).
    overlap_msg = _detect_overlap_cells(pts)
    if overlap_msg:
        stats.overlaps_detected = True
        warnings.append(QcWarning("info", "overlap", overlap_msg))
    elif stats.return_counts.get(1, 0) > len(pts) * 0.7 and hdr.point_density > 200:
        stats.overlaps_detected = True
        warnings.append(
            QcWarning("info", "overlap", "Possible overlap zone detected (high density + many 1st returns)")
        )

    return stats, warnings


# ────────────────────────────────────────────────────────────────
# 4. Density heatmap
# ────────────────────────────────────────────────────────────────


def generate_density_heatmap(las: LasFile, output_path: str, grid_size: int = 128) -> Optional[str]:
    """Write a density heatmap PNG. Returns output path or None if unavailable."""
    if plt is None:
        return None
    if len(las.points) < 100:
        return None

    xs = np.array([p.x for p in las.points], dtype=np.float64)
    ys = np.array([p.y for p in las.points], dtype=np.float64)
    zs = np.array([p.z for p in las.points], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 2D density heatmap
    hist, x_edges, y_edges = np.histogram2d(xs, ys, bins=grid_size)
    hist = np.rot90(hist)
    hist = np.ma.masked_where(hist == 0, hist)

    ax = axes[0]
    im = ax.pcolormesh(x_edges, y_edges, hist, cmap="viridis", shading="auto")
    ax.set_title("Point Density (pts/bin)")
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label="Points per bin")

    # Elevation scatter (colored by Z)
    ax2 = axes[1]
    # Downsample for scatter if too many points
    if len(xs) > 50_000:
        idx = np.random.choice(len(xs), 50_000, replace=False)
        sx, sy, sz = xs[idx], ys[idx], zs[idx]
    else:
        sx, sy, sz = xs, ys, zs
    sc = ax2.scatter(sx, sy, c=sz, s=1, cmap="terrain", alpha=0.6)
    ax2.set_title("Elevation Map")
    ax2.set_xlabel("Easting")
    ax2.set_ylabel("Northing")
    ax2.set_aspect("equal")
    fig.colorbar(sc, ax=ax2, label="Elevation (m)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ────────────────────────────────────────────────────────────────
# 5. HTML Report Generator
# ────────────────────────────────────────────────────────────────


def _severity_icon(severity: str) -> str:
    return {"info": "ℹ️", "warning": "⚠️", "error": "🚫"}.get(severity, "ℹ️")


def _fmt_num(n: int | float) -> str:
    if isinstance(n, float):
        if n >= 1_000_000:
            return f"{n / 1_000_000:.2f}M"
        elif n >= 1_000:
            return f"{n / 1_000:.1f}k"
        return f"{n:.2f}"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return f"{n:,}"


def _cls_name(c: int) -> str:
    return CLASS_NAMES.get(c, f"Class {c}")


def _detect_linear_unit(crs_wkt: str) -> tuple[str, float]:
    """Return (unit_label, metres_per_unit) from CRS WKT text. Defaults to metres."""
    import re

    text = (crs_wkt or "").lower()
    if re.search(r"us[\s_-]*survey[\s_-]*(foot|feet)|foot_us|ftus", text):
        return "ft", 0.3048006096
    if re.search(r"foot|feet|\bft\b|\"ft\"", text):
        return "ft", 0.3048
    return "m", 1.0


def _build_3d_payload(las: LasFile, max_3d_points: int = 150_000) -> dict:
    """Pack a compact, centroid-referenced binary payload for the WebGL viewer.

    * Uniform deterministic stride subsample (<= ``max_3d_points`` points).
    * float64 bbox/centroid are kept in metadata; XYZ is quantised to uint16 against
      the float64 bbox, so the browser reconstructs ``(world - centroid)`` in float64
      before casting to Float32 (no WebGL jitter on large projected coordinates).
    * Binary layout (8 bytes/point), Base64 encoded in ``data_b64``:
        xyz  : ``<u2`` x3 interleaved  @ offset 0
        int. : ``u1``                  @ offset 6*n
        class: ``u1``                  @ offset 7*n
    """
    pts = las.points
    n_total = len(pts)
    unit, metres_per_unit = _detect_linear_unit(las.header.crs_wkt)
    tol_default = round(0.005 / metres_per_unit, 6)
    base: dict = {
        "n_points": 0,
        "n_total": n_total,
        "stride": 1,
        "unit": unit,
        "tol_default": tol_default,
        "tol_max": round(0.25 / metres_per_unit, 6),
        "bytes_per_point": 8,
        "data_b64": "",
    }
    if n_total == 0 or max_3d_points <= 0:
        return base

    stride = max(1, math.ceil(n_total / max_3d_points))
    sel = pts[::stride]
    n = len(sel)

    xyz = np.empty((n, 3), dtype=np.float64)
    xyz[:, 0] = np.fromiter((p.x for p in sel), dtype=np.float64, count=n)
    xyz[:, 1] = np.fromiter((p.y for p in sel), dtype=np.float64, count=n)
    xyz[:, 2] = np.fromiter((p.z for p in sel), dtype=np.float64, count=n)
    intensity = np.fromiter((p.intensity for p in sel), dtype=np.float64, count=n)
    classification = np.fromiter((p.classification for p in sel), dtype=np.int64, count=n)

    centroid = xyz.mean(axis=0)
    bmin = xyz.min(axis=0)
    bmax = xyz.max(axis=0)
    span = bmax - bmin
    span_safe = np.where(span > 0, span, 1.0)

    q = np.rint((xyz - bmin) / span_safe * 65535.0).astype("<u2")
    imax = max(1.0, float(intensity.max()))
    i8 = np.rint(intensity / imax * 255.0).astype("u1")
    c8 = np.clip(classification, 0, 255).astype("u1")

    data = q.tobytes() + i8.tobytes() + c8.tobytes()
    base.update(
        {
            "n_points": n,
            "stride": stride,
            "bbox_min": bmin.tolist(),
            "bbox_max": bmax.tolist(),
            "centroid": centroid.tolist(),
            "min_z": float(bmin[2]),
            "max_z": float(bmax[2]),
            "intensity_max": imax,
            "layout": {"xyz_u16": 0, "intensity_u8": 6 * n, "class_u8": 7 * n},
            "data_b64": base64.b64encode(data).decode("ascii"),
        }
    )
    return base


_WEBGL_CSS = r"""
  /* 3D WebGL section */
  section.webgl-section { background: #151d30; border: 1px solid #24314f; border-radius: 16px;
    padding: 1.5rem; box-shadow: 0 4px 10px rgba(0,0,0,0.15); margin-bottom: 1.5rem; }
  section.webgl-section h2 { font-family: 'Outfit', sans-serif; font-size: 1.1rem; font-weight: 700;
    color: #f8fafc; margin-bottom: 0.8rem; padding-bottom: 0.6rem; border-bottom: 1px solid #24314f; }
  .subsample-badge { display: inline-block; background: rgba(56,189,248,0.1); border: 1px solid #38bdf8;
    color: #7dd3fc; border-radius: 99px; padding: 0.2rem 0.8rem; font-size: 0.78rem; margin-bottom: 0.6rem; }
  .offline-note { font-size: 0.78rem; color: #94a3b8; margin-bottom: 0.8rem; }
  .webgl-offline { background: rgba(245,158,11,0.08); border-left: 4px solid #f59e0b; border-radius: 10px;
    padding: 0.85rem 1rem; font-size: 0.85rem; color: #fcd34d; margin-bottom: 0.8rem; }
  .webgl-offline[hidden] { display: none; }
  .webgl-controls { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 0.6rem 1rem;
    margin-bottom: 1rem; }
  .webgl-controls label { display: flex; flex-direction: column; font-size: 0.7rem; font-weight: 600;
    color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; gap: 0.2rem; }
  .webgl-controls label output { color: #e2e8f0; text-transform: none; letter-spacing: 0; font-weight: 500; }
  .webgl-controls select, .webgl-controls input[type=range] { width: 100%; }
  .webgl-controls select { background: #0f172a; color: #e2e8f0; border: 1px solid #334155; border-radius: 8px;
    padding: 0.35rem; }
  .webgl-controls .check { flex-direction: row; align-items: center; gap: 0.5rem; text-transform: none; }
  #webgl-canvas-host { width: 100%; height: 520px; background: #0b0f19; border: 1px solid #334155;
    border-radius: 12px; overflow: hidden; }
  #webgl-canvas-host canvas { display: block; }
  .webgl-legend { font-size: 0.75rem; color: #94a3b8; margin: 0.5rem 0 1rem; }
  #profile-canvas { width: 100%; height: 260px; background: #0b0f19; border: 1px solid #334155;
    border-radius: 12px; display: block; }
  #profile-readout { font-size: 0.8rem; color: #cbd5e1; margin-top: 0.4rem; min-height: 1.3rem;
    font-variant-numeric: tabular-nums; }
  .webgl-subhead { font-family: 'Outfit', sans-serif; font-size: 0.95rem; color: #f8fafc; margin: 1.1rem 0 0.4rem; }
"""

_WEBGL_BODY = r"""
  <section class="section webgl-section" id="webgl-section">
    <h2>🧊 3D Point Cloud &amp; Corridor Profile</h2>
    <div class="subsample-badge">__BADGE__</div>
    <div class="offline-note">The 2D QC tables, charts and density heatmap in this report are 100% offline.
      This interactive 3D WebGL viewer loads Three.js from the jsDelivr CDN and needs an internet connection.</div>
    <div id="webgl-offline" class="webgl-offline" hidden>⚠ The 3D viewer could not load Three.js from the
      jsDelivr CDN (are you offline?). All 2D QC content in this report remains fully available.</div>
    <div class="webgl-controls">
      <label>Color mode
        <select id="c-mode">
          <option value="0">Elevation (Z)</option>
          <option value="1">ASPRS Classification</option>
          <option value="2">Intensity</option>
          <option value="3">Reference Datum ΔZ (Elevation Deviation)</option>
        </select>
      </label>
      <label>Min Z clip <output id="o-minz"></output><input type="range" id="c-minz"></label>
      <label>Max Z clip <output id="o-maxz"></output><input type="range" id="c-maxz"></label>
      <label>Point size <output id="o-psize"></output><input type="range" id="c-psize" min="1" max="8" step="0.5" value="2"></label>
      <label>Reference datum Z <output id="o-datum"></output><input type="range" id="c-datum"></label>
      <label>Tolerance (green ±) <output id="o-tol"></output><input type="range" id="c-tol"></label>
    </div>
    <div class="webgl-legend" id="dz-legend">
      Reference Datum ΔZ: signed vertical deviation Δz = z − z<sub>datum</sub> against an adjustable horizontal
      reference plane (slab flatness / grading checks). It is not a Scan-vs-BIM mesh comparison.
      <span id="dz-legend-live"></span>
    </div>
    <div id="webgl-canvas-host"></div>

    <h3 class="webgl-subhead">Corridor cross-section (station vs. elevation)</h3>
    <div class="webgl-controls">
      <label>Azimuth (° from +X, 0–180) <output id="o-azim"></output><input type="range" id="c-azim" min="0" max="180" step="1" value="0"></label>
      <label>Corridor width w <output id="o-w"></output><input type="range" id="c-w"></label>
      <label>Corridor length L <output id="o-l"></output><input type="range" id="c-l"></label>
      <label>Offset X <output id="o-ox"></output><input type="range" id="c-ox"></label>
      <label>Offset Y <output id="o-oy"></output><input type="range" id="c-oy"></label>
      <label class="check"><input type="checkbox" id="c-hl" checked> Highlight corridor points</label>
      <label class="check"><input type="checkbox" id="c-iso"> Isolate corridor points</label>
    </div>
    <canvas id="profile-canvas"></canvas>
    <div id="profile-readout">Hover the profile for station, elevation and world coordinates.</div>
  </section>
"""

_WEBGL_IMPORTMAP = r"""
<script type="importmap">
{ "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
} }
</script>
"""

_WEBGL_JS = r"""
<script type="module">
const offlineBanner = document.getElementById('webgl-offline');
const P = JSON.parse(document.getElementById('payload3d').textContent);
const CLS = JSON.parse(document.getElementById('payload3d-cls').textContent);
const $ = (id) => document.getElementById(id);

(async function main() {
  if (!P.n_points) {
    offlineBanner.hidden = false;
    offlineBanner.textContent = 'No points available for the 3D view.';
    return;
  }
  let THREE, OrbitControls;
  try {
    THREE = await import('three');
    ({ OrbitControls } = await import('three/addons/controls/OrbitControls.js'));
  } catch (e) {
    offlineBanner.hidden = false;
    return;
  }

  // ── Decode quantised binary payload (float64 math, then Float32 centroid-centred) ──
  const n = P.n_points, cen = P.centroid, bmin = P.bbox_min, bmax = P.bbox_max;
  const span = [0, 1, 2].map((k) => bmax[k] - bmin[k]);
  const bin = atob(P.data_b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const q = new Uint16Array(bytes.buffer, P.layout.xyz_u16, n * 3);
  const pos = new Float32Array(n * 3);
  const aInt = new Float32Array(n), aCls = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) {
      const w = bmin[k] + (span[k] > 0 ? (q[3 * i + k] / 65535) * span[k] : 0);
      pos[3 * i + k] = w - cen[k];
    }
    aInt[i] = bytes[P.layout.intensity_u8 + i] / 255;
    aCls[i] = bytes[P.layout.class_u8 + i];
  }

  const extXY = Math.max(span[0], span[1], 1e-6);
  const extent = Math.max(span[0], span[1], span[2], 1e-6);
  const diag = Math.hypot(span[0], span[1]) || 1;
  const zc = cen[2];
  const bboxC = [0, 1, 2].map((k) => (bmin[k] + bmax[k]) / 2 - cen[k]);
  const fmt = (v, d = 3) => Number(v).toFixed(d);

  // ── Three.js scene ──
  const host = $('webgl-canvas-host');
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(host.clientWidth, host.clientHeight);
  renderer.setClearColor(0x0b0f19, 1);
  host.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, host.clientWidth / host.clientHeight, extent * 0.001, extent * 50);
  camera.up.set(0, 0, 1);
  camera.position.set(bboxC[0] + extent * 0.6, bboxC[1] - extent * 1.0, bboxC[2] + extent * 0.9);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(bboxC[0], bboxC[1], bboxC[2]);
  controls.update();

  const clsColors = [];
  for (let i = 0; i < 32; i++) {
    const hex = CLS[String(i)] || '#9e9e9e';
    const c = new THREE.Color(hex);
    clsColors.push(new THREE.Vector3(c.r, c.g, c.b));
  }

  const uniforms = {
    uPointSize: { value: 2.0 },
    uMinZ: { value: -1e9 }, uMaxZ: { value: 1e9 },
    uZLo: { value: P.min_z - zc }, uZHi: { value: P.max_z - zc },
    uMode: { value: 0 },
    uDatum: { value: 0 }, uTol: { value: P.tol_default },
    uCorrCenter: { value: new THREE.Vector2(0, 0) },
    uCorrDir: { value: new THREE.Vector2(1, 0) },
    uHalfW: { value: 1 }, uHalfL: { value: 1 },
    uHighlight: { value: 1 }, uIsolate: { value: 0 },
    uClsColors: { value: clsColors },
  };

  const material = new THREE.ShaderMaterial({
    uniforms,
    vertexShader: `
      attribute float aIntensity;
      attribute float aClass;
      uniform float uPointSize, uMinZ, uMaxZ, uZLo, uZHi, uDatum, uTol, uHalfW, uHalfL;
      uniform int uMode, uHighlight, uIsolate;
      uniform vec3 uClsColors[32];
      uniform vec2 uCorrCenter, uCorrDir;
      varying vec3 vColor;

      vec3 ramp(float t) {
        t = clamp(t, 0.0, 1.0);
        vec3 a = vec3(0.10, 0.20, 0.75), b = vec3(0.10, 0.75, 0.80),
             c = vec3(0.25, 0.85, 0.30), d = vec3(0.98, 0.90, 0.20), e = vec3(0.90, 0.20, 0.15);
        if (t < 0.25) return mix(a, b, t / 0.25);
        if (t < 0.5)  return mix(b, c, (t - 0.25) / 0.25);
        if (t < 0.75) return mix(c, d, (t - 0.5) / 0.25);
        return mix(d, e, (t - 0.75) / 0.25);
      }

      void main() {
        // GPU Z clipping
        if (position.z < uMinZ || position.z > uMaxZ) {
          gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
          gl_PointSize = 0.0;
          vColor = vec3(0.0);
          return;
        }
        // Corridor membership: |d_perp| <= w/2 and |d_par| <= L/2
        vec2 d = position.xy - uCorrCenter;
        float par = dot(d, uCorrDir);
        float perp = dot(d, vec2(-uCorrDir.y, uCorrDir.x));
        bool inside = abs(perp) <= uHalfW && abs(par) <= uHalfL;
        if (uIsolate == 1 && !inside) {
          gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
          gl_PointSize = 0.0;
          vColor = vec3(0.0);
          return;
        }

        vec3 col;
        if (uMode == 0) {
          col = ramp((position.z - uZLo) / max(uZHi - uZLo, 1e-9));
        } else if (uMode == 1) {
          int idx = int(clamp(aClass, 0.0, 31.0));
          col = uClsColors[0];
          for (int i = 0; i < 32; i++) { if (i == idx) col = uClsColors[i]; }
        } else if (uMode == 2) {
          col = vec3(aIntensity);
        } else {
          float dz = position.z - uDatum;
          float g = uTol, r = 4.0 * uTol;
          vec3 green = vec3(0.15, 0.80, 0.35), red = vec3(0.92, 0.15, 0.15), blue = vec3(0.15, 0.30, 0.95);
          float a = abs(dz);
          if (a <= g) col = green;
          else col = mix(green, dz > 0.0 ? red : blue, clamp((a - g) / max(r - g, 1e-9), 0.0, 1.0));
        }
        if (uHighlight == 1 && inside) col = mix(col, vec3(1.0, 0.9, 0.2), 0.55);

        vColor = col;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = uPointSize;
      }`,
    fragmentShader: `
      varying vec3 vColor;
      void main() {
        if (length(gl_PointCoord - vec2(0.5)) > 0.5) discard;
        gl_FragColor = vec4(vColor, 1.0);
      }`,
  });

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geom.setAttribute('aIntensity', new THREE.BufferAttribute(aInt, 1));
  geom.setAttribute('aClass', new THREE.BufferAttribute(aCls, 1));
  const points = new THREE.Points(geom, material);
  points.frustumCulled = false;
  scene.add(points);

  // Corridor outline
  const outlineGeom = new THREE.BufferGeometry();
  outlineGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(12), 3));
  const outline = new THREE.LineLoop(outlineGeom, new THREE.LineBasicMaterial({ color: 0xfacc15 }));
  outline.frustumCulled = false;
  scene.add(outline);

  const render = () => renderer.render(scene, camera);
  controls.addEventListener('change', render);

  // ── Controls ──
  function setup(id, min, max, step, value) {
    const el = $(id);
    el.min = min; el.max = max; el.step = step; el.value = value;
    return el;
  }
  const zStep = Math.max((P.max_z - P.min_z) / 1000, 1e-6);
  const zMin = setup('c-minz', P.min_z, P.max_z, zStep, P.min_z);
  const zMax = setup('c-maxz', P.min_z, P.max_z, zStep, P.max_z);
  const datum = setup('c-datum', P.min_z, P.max_z, zStep, zc);
  const tol = setup('c-tol', P.tol_default / 10, P.tol_max, P.tol_default / 10, P.tol_default);
  const azim = $('c-azim');
  const alongY = span[1] > span[0];
  azim.value = alongY ? 90 : 0;  // default: corridor along the longer horizontal axis
  const cw = setup('c-w', extXY / 200, extXY, extXY / 400, extXY / 10);
  const cl = setup('c-l', extXY / 50, diag * 1.2, diag / 400, Math.max(span[0], span[1]) * 0.9);
  const ox = setup('c-ox', -extXY / 2, extXY / 2, extXY / 400, 0);
  const oy = setup('c-oy', -extXY / 2, extXY / 2, extXY / 400, 0);
  const mode = $('c-mode');

  function updateScene() {
    const lo = Math.min(+zMin.value, +zMax.value), hi = Math.max(+zMin.value, +zMax.value);
    uniforms.uMinZ.value = lo - zc;
    uniforms.uMaxZ.value = hi - zc;
    uniforms.uPointSize.value = +$('c-psize').value;
    uniforms.uDatum.value = +datum.value - zc;
    uniforms.uTol.value = +tol.value;
    uniforms.uMode.value = +mode.value;
    uniforms.uHighlight.value = $('c-hl').checked ? 1 : 0;
    uniforms.uIsolate.value = $('c-iso').checked ? 1 : 0;
    $('o-minz').textContent = fmt(lo); $('o-maxz').textContent = fmt(hi);
    $('o-psize').textContent = $('c-psize').value;
    $('o-datum').textContent = fmt(datum.value) + ' ' + P.unit;
    $('o-tol').textContent = '±' + fmt(tol.value, 4) + ' ' + P.unit;
    $('dz-legend-live').textContent =
      ' Blue < −' + fmt(4 * tol.value, 4) + ' ' + P.unit + ' · Green ±' + fmt(tol.value, 4) + ' ' + P.unit +
      ' · Red > +' + fmt(4 * tol.value, 4) + ' ' + P.unit +
      ' (defaults: −20 mm / ±5 mm / +20 mm in metres).';
    updateCorridor();
  }

  let corr = { cx: 0, cy: 0, dx: 1, dy: 0, hw: 1, hl: 1 };
  function updateCorridor() {
    const th = (+azim.value * Math.PI) / 180;
    corr = { cx: +ox.value, cy: +oy.value, dx: Math.cos(th), dy: Math.sin(th), hw: +cw.value / 2, hl: +cl.value / 2 };
    uniforms.uCorrCenter.value.set(corr.cx, corr.cy);
    uniforms.uCorrDir.value.set(corr.dx, corr.dy);
    uniforms.uHalfW.value = corr.hw;
    uniforms.uHalfL.value = corr.hl;
    $('o-azim').textContent = azim.value + '°';
    $('o-w').textContent = fmt(cw.value) + ' ' + P.unit;
    $('o-l').textContent = fmt(cl.value) + ' ' + P.unit;
    $('o-ox').textContent = fmt(ox.value); $('o-oy').textContent = fmt(oy.value);
    const z = +datum.value - zc;
    const px = -corr.dy, py = corr.dx;
    const arr = outlineGeom.attributes.position.array;
    const cs = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
    cs.forEach(([a, b], i) => {
      arr[3 * i] = corr.cx + corr.dx * a * corr.hl + px * b * corr.hw;
      arr[3 * i + 1] = corr.cy + corr.dy * a * corr.hl + py * b * corr.hw;
      arr[3 * i + 2] = z;
    });
    outlineGeom.attributes.position.needsUpdate = true;
    render();
    scheduleProfile();
  }

  // ── 2D stationing profile: station s vs elevation z ──
  const pc = $('profile-canvas');
  const pctx = pc.getContext('2d');
  let prof = { px: null, py: null, idx: null, count: 0 };
  let pending = false;
  function scheduleProfile() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; drawProfile(); });
  }
  function drawProfile() {
    const dpr = window.devicePixelRatio || 1;
    const W = pc.clientWidth, H = pc.clientHeight;
    pc.width = W * dpr; pc.height = H * dpr;
    pctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    pctx.clearRect(0, 0, W, H);
    const lo = uniforms.uMinZ.value, hi = uniforms.uMaxZ.value;
    const L = corr.hl * 2;
    const sArr = new Float32Array(n), zArr = new Float32Array(n), ids = new Int32Array(n);
    let m = 0, zlo = Infinity, zhi = -Infinity;
    for (let i = 0; i < n; i++) {
      const x = pos[3 * i] - corr.cx, y = pos[3 * i + 1] - corr.cy, z = pos[3 * i + 2];
      if (z < lo || z > hi) continue;
      const par = x * corr.dx + y * corr.dy;        // d_parallel (signed)
      const perp = -x * corr.dy + y * corr.dx;      // d_perpendicular (signed)
      if (Math.abs(perp) > corr.hw || Math.abs(par) > corr.hl) continue;
      sArr[m] = par + corr.hl;                       // stationing (0 .. L)
      zArr[m] = z; ids[m] = i; m++;
      if (z < zlo) zlo = z;
      if (z > zhi) zhi = z;
    }
    const ml = 64, mr = 12, mt = 12, mb = 42;
    pctx.fillStyle = '#94a3b8'; pctx.font = '11px Inter, sans-serif';
    if (!m) {
      pctx.fillText('No points inside the corridor — adjust offset / width / length.', ml, H / 2);
      prof = { px: null, py: null, idx: null, count: 0 };
      return;
    }
    if (zhi - zlo < 1e-9) { zlo -= 0.5; zhi += 0.5; }
    const pad = (zhi - zlo) * 0.05; zlo -= pad; zhi += pad;
    const X = (s) => ml + (s / L) * (W - ml - mr);
    const Y = (z) => H - mb - ((z - zlo) / (zhi - zlo)) * (H - mt - mb);
    // axes
    pctx.strokeStyle = '#334155'; pctx.lineWidth = 1;
    pctx.strokeRect(ml, mt, W - ml - mr, H - mt - mb);
    for (let t = 0; t <= 4; t++) {
      const s = (L * t) / 4, zz = zlo + ((zhi - zlo) * t) / 4;
      pctx.fillText(fmt(s, 1), Math.min(X(s) - 12, W - mr - 30), H - 26);
      pctx.fillText(fmt(zz + zc, 2), 4, Y(zz) + 4);
      pctx.beginPath(); pctx.moveTo(ml, Y(zz)); pctx.lineTo(W - mr, Y(zz));
      pctx.strokeStyle = '#1e293b'; pctx.stroke();
    }
    pctx.fillText('Station s (' + P.unit + ')', W / 2 - 40, H - 8);
    // datum line
    const dz = +datum.value - zc;
    if (dz > zlo && dz < zhi) {
      pctx.strokeStyle = '#f59e0b'; pctx.setLineDash([5, 4]);
      pctx.beginPath(); pctx.moveTo(ml, Y(dz)); pctx.lineTo(W - mr, Y(dz)); pctx.stroke();
      pctx.setLineDash([]);
    }
    // points
    const px = new Float32Array(m), py = new Float32Array(m);
    pctx.fillStyle = '#38bdf8';
    for (let k = 0; k < m; k++) {
      px[k] = X(sArr[k]); py[k] = Y(zArr[k]);
      pctx.fillRect(px[k] - 1, py[k] - 1, 2, 2);
    }
    prof = { px, py, idx: ids.subarray(0, m), count: m, sArr, zArr };
    $('profile-readout').textContent = m.toLocaleString() + ' points in corridor — hover for readout.';
  }
  pc.addEventListener('mousemove', (ev) => {
    if (!prof.count) return;
    const r = pc.getBoundingClientRect();
    const mx = ev.clientX - r.left, my = ev.clientY - r.top;
    let best = -1, bd = 12 * 12;
    for (let k = 0; k < prof.count; k++) {
      const dx = prof.px[k] - mx, dy = prof.py[k] - my, d2 = dx * dx + dy * dy;
      if (d2 < bd) { bd = d2; best = k; }
    }
    if (best < 0) return;
    const i = prof.idx[best];
    const E = pos[3 * i] + cen[0], N = pos[3 * i + 1] + cen[1], Z = pos[3 * i + 2] + cen[2];
    $('profile-readout').textContent =
      'Station s = ' + fmt(prof.sArr[best], 3) + ' ' + P.unit + '  |  Elevation z = ' + fmt(Z, 3) +
      '  |  E = ' + fmt(E, 3) + '  N = ' + fmt(N, 3);
  });

  ['c-mode', 'c-minz', 'c-maxz', 'c-psize', 'c-datum', 'c-tol', 'c-azim', 'c-w', 'c-l', 'c-ox', 'c-oy', 'c-hl', 'c-iso']
    .forEach((id) => $(id).addEventListener('input', updateScene));
  window.addEventListener('resize', () => {
    renderer.setSize(host.clientWidth, host.clientHeight);
    camera.aspect = host.clientWidth / host.clientHeight;
    camera.updateProjectionMatrix();
    render(); scheduleProfile();
  });
  updateScene();
})();
</script>
"""


def generate_html_report(
    las: LasFile,
    stats: LasStats,
    warnings: list[QcWarning],
    density_png_path: str | None = None,
    filename: str = "",
    max_3d_points: int = 150_000,
) -> str:
    """Return a self-contained HTML string."""
    hdr = las.header
    crs_label = hdr.crs_wkt[:80] + "…" if len(hdr.crs_wkt) > 80 else hdr.crs_wkt or "Not detected"
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Classification color palette (muted, accessible)
    CLS_COLORS = {
        0: "#94a3b8", 1: "#64748b", 2: "#10b981", 3: "#84cc16",
        4: "#f59e0b", 5: "#ef4444", 6: "#ec4899", 7: "#4b5563",
        8: "#14b8a6", 9: "#3b82f6", 10: "#8b5cf6", 11: "#6b7280",
        12: "#06b6d4", 13: "#6366f1", 14: "#4f46e5", 15: "#78350f",
        16: "#db2777", 17: "#0d9488", 18: "#f87171",
    }

    # Classification table rows
    cls_rows = ""
    sorted_cls = sorted(stats.classification_counts.items(), key=lambda x: -x[1])
    for code, count in sorted_cls:
        pct = (count / stats.total_points) * 100
        color = CLS_COLORS.get(code, "#9e9e9e")
        cls_rows += f"""
        <tr>
          <td style="font-weight:600;">{code}</td>
          <td><span class="cls-badge"><span class="cls-dot" style="background:{color}"></span>{_cls_name(code)}</span></td>
          <td>{_fmt_num(count)}</td>
          <td style="font-variant-numeric:tabular-nums;">{pct:.1f}%</td>
        </tr>"""

    # Return table rows
    ret_rows = ""
    for n in sorted(stats.return_counts):
        pct = (stats.return_counts[n] / stats.total_points) * 100
        ret_rows += f"""
        <tr>
          <td style="font-weight:600;">{n}</td>
          <td>{_fmt_num(stats.return_counts[n])}</td>
          <td style="font-variant-numeric:tabular-nums;">{pct:.1f}%</td>
          <td><div class="bar-row"><div class="bar-track"><div class="bar-fill" style="width:{pct}%"></div></div></div></td>
        </tr>"""

    # QC warnings
    warn_cards = ""
    for w in warnings:
        warn_cards += f"""
        <div class="qcw {w.severity}">
          <span class="icon">{_severity_icon(w.severity)}</span>
          <span class="msg"><strong>{w.category}</strong> — {w.message}</span>
        </div>"""
    if not warn_cards:
        warn_cards = '<div class="qcw ok"><span class="icon">✅</span><span class="msg">No issues detected.</span></div>'

    # 3D WebGL viewer (centroid-centred, u16-quantised payload)
    payload = _build_3d_payload(las, max_3d_points=max_3d_points)
    badge = (
        f"Rendering {payload['n_points']:,} of {payload['n_total']:,} points "
        f"(uniform stride subsample)"
    )
    webgl_section = _WEBGL_BODY.replace("__BADGE__", badge)
    webgl_scripts = (
        _WEBGL_IMPORTMAP
        + '<script id="payload3d" type="application/json">'
        + json.dumps(payload)
        + "</script>\n"
        + '<script id="payload3d-cls" type="application/json">'
        + json.dumps({str(k): v for k, v in CLS_COLORS.items()})
        + "</script>\n"
        + _WEBGL_JS
    )

    # Density image embed (inline base64)
    density_img = ""
    if density_png_path and os.path.exists(density_png_path):
        import base64

        with open(density_png_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        density_img = f'<img src="data:image/png;base64,{b64}" alt="Density heatmap" style="max-width:100%;border:1px solid #334155;border-radius:12px;box-shadow: 0 4px 20px rgba(0,0,0,0.3);">'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<link rel="icon" href="data:,">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Automated Quality Control (QC) report for LiDAR Point Cloud data generated by las-inspector.">
<title>las-inspector QC Report - {filename}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@600;700;800&display=swap" rel="stylesheet">
<style>
  /* Premium Dark Theme */
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0b0f19;
    color: #e2e8f0;
    padding: 2rem;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
  }}
  .container {{ max-width: 1100px; margin: 0 auto; }}

  /* Header structure */
  header.topbar {{
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
  }}
  header.topbar h1 {{
    font-family: 'Outfit', sans-serif;
    font-size: 1.75rem;
    font-weight: 800;
    background: linear-gradient(to right, #38bdf8, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: flex;
    align-items: center;
    gap: 0.6rem;
  }}
  header.topbar .meta {{
    font-size: 0.85rem;
    color: #94a3b8;
    display: flex;
    gap: 1.5rem;
    align-items: center;
    flex-wrap: wrap;
  }}
  .topbar-actions {{
    display: flex;
    align-items: center;
    gap: 1.25rem;
    flex-wrap: wrap;
  }}
  .gh-btn {{
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: #0f172a;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 0.35rem 0.75rem;
    font-size: 0.78rem;
    font-weight: 600;
    text-decoration: none;
    transition: background 0.2s ease, border-color 0.2s ease, color 0.2s ease;
    white-space: nowrap;
  }}
  .gh-btn:hover {{
    background: #1e293b;
    border-color: #38bdf8;
    color: #38bdf8;
  }}

  /* Hero Cards */
  main.hero {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1.25rem;
    margin-bottom: 2rem;
  }}
  .hero-item {{
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s ease, border-color 0.2s ease;
  }}
  .hero-item:hover {{
    transform: translateY(-2px);
    border-color: #3b82f6;
  }}
  .hero-item .hl {{
    font-size: 0.72rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
  }}
  .hero-item .hv {{
    font-size: 1.85rem;
    font-weight: 700;
    color: #f8fafc;
    line-height: 1.1;
  }}
  .hero-item .hu {{
    font-size: 0.78rem;
    color: #64748b;
    font-weight: 500;
    margin-top: 0.2rem;
  }}

  /* Layout grids */
  .main-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    margin-bottom: 2rem;
  }}

  /* Section Styles */
  section.section {{
    background: #151d30;
    border: 1px solid #24314f;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
    margin-bottom: 1.5rem;
  }}
  section.section:last-child {{ margin-bottom: 0; }}
  
  section.section h2 {{
    font-family: 'Outfit', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 1.25rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #24314f;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }}

  /* Tables styling */
  table.bounds-table, table.data-table {{
    width: 100%;
    font-size: 0.85rem;
    border-collapse: collapse;
  }}
  table.bounds-table th, table.data-table thead th {{
    text-align: left;
    font-weight: 600;
    color: #94a3b8;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 0.6rem 0.5rem;
    border-bottom: 1.5px solid #24314f;
  }}
  table.bounds-table td, table.data-table tbody td {{
    padding: 0.7rem 0.5rem;
    border-top: 1px solid #1e293b;
    color: #cbd5e1;
  }}
  table.bounds-table tr:first-child td {{ border-top: none; }}
  table.bounds-table .axis {{ font-weight: 700; color: #38bdf8; }}

  /* QC Warnings */
  .qcw {{
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    padding: 0.85rem 1rem;
    border-radius: 12px;
    margin-bottom: 0.6rem;
    font-size: 0.85rem;
    line-height: 1.5;
    transition: transform 0.2s ease;
  }}
  .qcw:hover {{ transform: translateX(2px); }}
  .qcw.info {{ background: rgba(59, 130, 246, 0.08); border-left: 4px solid #3b82f6; }}
  .qcw.warning {{ background: rgba(245, 158, 11, 0.08); border-left: 4px solid #f59e0b; }}
  .qcw.ok {{ background: rgba(16, 185, 129, 0.08); border-left: 4px solid #10b981; }}
  .qcw .icon {{ font-size: 1.1rem; flex-shrink: 0; line-height: 1; }}
  .qcw .msg {{ flex: 1; }}
  .qcw .msg strong {{ color: #f8fafc; }}

  /* Badges & distribution bars */
  .cls-badge {{ display: inline-flex; align-items: center; gap: 0.4rem; font-weight: 500; }}
  .cls-dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}
  .bar-row {{ display: flex; align-items: center; gap: 0.6rem; }}
  .bar-track {{ flex: 1; height: 6px; background: #1e293b; border-radius: 99px; overflow: hidden; min-width: 60px; }}
  .bar-fill {{ height: 100%; border-radius: 99px; background: linear-gradient(90deg, #3b82f6, #38bdf8); }}

  /* CRS box */
  .crs-box {{
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 0.85rem;
    font-family: 'SF Mono', monospace;
    font-size: 0.75rem;
    color: #94a3b8;
    line-height: 1.5;
    word-break: break-all;
    max-height: 80px;
    overflow-y: auto;
  }}

  /* Grid details */
  .mini-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; }}
  .mini-stat {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 0.85rem; text-align: center; }}
  .mini-stat .ml {{ font-size: 0.65rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; }}
  .mini-stat .mv {{ font-size: 1.25rem; font-weight: 700; color: #f8fafc; margin-top: 0.2rem; }}

  /* Density & Visualizer section */
  section.density-section {{
    background: #151d30;
    border: 1px solid #24314f;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
    margin-top: 1.5rem;
  }}
  .density-placeholder {{
    padding: 2.5rem;
    color: #64748b;
    font-size: 0.9rem;
    background: #0f172a;
    border: 1px dashed #334155;
    border-radius: 12px;
    text-align: center;
  }}

  /* Footer */
  footer.footer {{
    text-align: center;
    color: #64748b;
    font-size: 0.78rem;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid #1e293b;
  }}

  /* Responsive styling */
  @media (max-width: 860px) {{
    body {{ padding: 1rem; }}
    main.hero {{ grid-template-columns: 1fr 1fr; }}
    .main-grid {{ grid-template-columns: 1fr; }}
    header.topbar {{ flex-direction: column; align-items: flex-start; gap: 0.85rem; }}
    .topbar-actions {{ width: 100%; justify-content: space-between; }}
  }}
  @media (max-width: 480px) {{
    main.hero {{ grid-template-columns: 1fr; }}
    .mini-grid {{ grid-template-columns: 1fr; }}
  }}
{_WEBGL_CSS}
</style>
</head>
<body>
<div class="container">

  <!-- Top Bar -->
  <header class="topbar">
    <h1>🔍 las-inspector</h1>
    <div class="topbar-actions">
      <div class="meta">
        <span>📄 {filename}</span>
        <span>📅 {now}</span>
        <span>⚙ LAS 1.{hdr.version_minor} · fmt {hdr.point_format_id}</span>
      </div>
      <a href="https://github.com/gokserpirik/las-inspector" target="_blank" rel="noopener noreferrer" class="gh-btn" aria-label="View project on GitHub">
        <svg width="15" height="15" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
          <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8z"/>
        </svg>
        <span>GitHub</span>
      </a>
    </div>
  </header>

  <!-- Hero Stats -->
  <main class="hero">
    <div class="hero-item">
      <div class="hl">Total Points</div>
      <div class="hv">{_fmt_num(stats.total_points)}</div>
    </div>
    <div class="hero-item">
      <div class="hl">Point Density</div>
      <div class="hv">{stats.point_density:.0f}</div>
      <div class="hu">pts/m²</div>
    </div>
    <div class="hero-item">
      <div class="hl">Area</div>
      <div class="hv">{hdr.area_2d:.1f}</div>
      <div class="hu">m²</div>
    </div>
    <div class="hero-item">
      <div class="hl">Classifications</div>
      <div class="hv">{len(stats.classification_counts)}</div>
      <div class="hu">unique classes</div>
    </div>
  </main>

  <!-- Two-Column Grid -->
  <div class="main-grid">

    <!-- Left Column -->
    <div class="left-col">

      <!-- Spatial Bounds -->
      <section class="section">
        <h2>📐 Spatial Bounds</h2>
        <table class="bounds-table">
          <thead>
            <tr><th>Dim</th><th>Min</th><th>Max</th><th>Range</th></tr>
          </thead>
          <tbody>
            <tr><td class="axis">X</td><td>{hdr.min_x:.3f}</td><td>{hdr.max_x:.3f}</td><td>{hdr.x_range:.3f}</td></tr>
            <tr><td class="axis">Y</td><td>{hdr.min_y:.3f}</td><td>{hdr.max_y:.3f}</td><td>{hdr.y_range:.3f}</td></tr>
            <tr><td class="axis">Z</td><td>{hdr.min_z:.3f}</td><td>{hdr.max_z:.3f}</td><td>{hdr.z_range:.3f}</td></tr>
          </tbody>
        </table>
      </section>

      <!-- CRS -->
      <section class="section">
        <h2>📍 CRS / Projection</h2>
        <div class="crs-box">{crs_label}</div>
      </section>

      <!-- Quality Flags -->
      <section class="section">
        <h2>⚠ Quality Flags</h2>
        <div class="mini-grid">
          <div class="mini-stat"><div class="ml">Withheld</div><div class="mv">{_fmt_num(stats.withheld_count)}</div></div>
          <div class="mini-stat"><div class="ml">Synthetic</div><div class="mv">{_fmt_num(stats.synthetic_count)}</div></div>
          <div class="mini-stat"><div class="ml">Keypoints</div><div class="mv">{_fmt_num(stats.keypoint_count)}</div></div>
        </div>
        <div style="margin-top: 1rem;">
          {warn_cards}
        </div>
      </section>

    </div>

    <!-- Right Column -->
    <div class="right-col">

      <!-- Classification Breakdown -->
      <section class="section">
        <h2>🏷 Classification Breakdown</h2>
        <table class="data-table">
          <thead><tr><th>Code</th><th>Name</th><th>Count</th><th>%</th></tr></thead>
          <tbody>
            {cls_rows}
          </tbody>
        </table>
      </section>

      <!-- Return Analysis -->
      <section class="section">
        <h2>↩ Return Analysis</h2>
        <table class="data-table">
          <thead><tr><th>Return</th><th>Count</th><th>%</th><th>Distribution</th></tr></thead>
          <tbody>
            {ret_rows}
          </tbody>
        </table>
      </section>

    </div>

  </div>

  {webgl_section}

  <!-- Density & Elevation -->
  <section class="section density-section">
    <h2>🗺 Density & Elevation</h2>
    <div style="text-align:center;">
      {density_img or '<div class="density-placeholder">No visualization — run with <strong>--density</strong></div>'}
    </div>
  </section>

  <!-- Footer -->
  <footer class="footer">
    Generated by <a href="https://github.com/gokserpirik/las-inspector" target="_blank" rel="noopener noreferrer" style="color:#38bdf8;text-decoration:none;font-weight:600;">las-inspector</a> &middot; {now}
  </footer>

</div>
{webgl_scripts}
</body>
</html>"""

    return html


# ────────────────────────────────────────────────────────────────
# 6. CLI
# ────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="las-inspector",
        description="Automated LiDAR QC reports from your terminal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python las_inspector.py input.laz --output report.html
              python las_inspector.py survey.laz --density
        """),
    )
    parser.add_argument("input", help="Path to LAS/LAZ file")
    parser.add_argument("--output", "-o", default=None, help="Output HTML report path")
    parser.add_argument("--density", action="store_true", help="Generate density heatmap (PNG)")
    parser.add_argument("--max-points", type=int, default=None, help="Max points to read (for testing)")
    parser.add_argument(
        "--max-3d-points",
        type=int,
        default=150_000,
        help="Max points embedded in the interactive 3D viewer (uniform stride subsample, default 150000)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}", file=sys.stderr)
        return 1

    # Parse
    print(f"📂 Reading {input_path} ...")
    try:
        las = read_las(input_path, max_points=args.max_points)
    except (ValueError, struct.error, IOError) as e:
        print(f"❌ Read error: {e}", file=sys.stderr)
        return 1

    hdr = las.header
    print(f"   Points: {hdr.point_count:,}  |  Format: {hdr.point_format_id}  |  "
          f"LAS 1.{hdr.version_minor}")
    print(f"   Bounds: {hdr.min_x:.2f} {hdr.min_y:.2f} → {hdr.max_x:.2f} {hdr.max_y:.2f}")

    # Stats + QC
    stats, warnings = compute_stats(las)
    print(f"   Density: {stats.point_density:.0f} pts/m²  |  "
          f"Classes: {len(stats.classification_counts)}")

    if warnings:
        for w in warnings:
            icon = _severity_icon(w.severity)
            print(f"   {icon} [{w.severity}] {w.category}: {w.message}")

    # Density PNG
    # Generate qc_summary.txt (like main.py)
    summary_path = os.path.splitext(os.path.basename(input_path))[0] + "_qc_summary.txt"
    try:
        with open(summary_path, "w") as f:
            f.write(f"QC Report: {os.path.basename(input_path)}\n")
            f.write(f"Total Points: {hdr.point_count:,}\n")
            f.write(f"Point Density: {stats.point_density:.2f} pts/m²\n")
            f.write(f"Bounds: X={hdr.min_x:.3f}..{hdr.max_x:.3f}, "
                    f"Y={hdr.min_y:.3f}..{hdr.max_y:.3f}, "
                    f"Z={hdr.min_z:.3f}..{hdr.max_z:.3f}\n")
            f.write(f"CRS: {hdr.crs_wkt[:60] if hdr.crs_wkt else 'Not detected'}\n")
            f.write(f"Classifications: {len(stats.classification_counts)}\n")
            f.write(f"Withheld points: {stats.withheld_count}\n")
            f.write(f"Synthetic points: {stats.synthetic_count}\n")
            n_err = sum(1 for w in warnings if w.severity == "error")
            n_warn = sum(1 for w in warnings if w.severity == "warning")
            verdict = "FAILED" if n_err or n_warn else "PASSED"
            f.write(f"QC Verdict: {verdict}\n")
    except IOError as e:
        print(f"❌ Write error ({summary_path}): {e}", file=sys.stderr)
        return 1
    print(f"📝 Summary → {summary_path}")

    # Density PNG
    density_png = None
    if args.density:
        density_path = os.path.splitext(args.output or "report.html")[0] + "_density.png"
        print(f"🗺 Generating density heatmap → {density_path} ...")
        result = generate_density_heatmap(las, density_path)
        if result:
            density_png = result

    # HTML report
    output_path = args.output or "qc_report.html"
    print(f"📄 Generating report → {output_path} ...")
    html = generate_html_report(
        las,
        stats,
        warnings,
        density_png_path=density_png,
        filename=las.filename,
        max_3d_points=args.max_3d_points,
    )
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
    except IOError as e:
        print(f"❌ Write error ({output_path}): {e}", file=sys.stderr)
        return 1

    print(f"✅ Done — report written to {output_path}")
    if not warnings:
        print("   ✅ No issues detected.")
    else:
        n_err = sum(1 for w in warnings if w.severity == "error")
        n_warn = sum(1 for w in warnings if w.severity == "warning")
        parts = []
        if n_err:
            parts.append(f"{n_err} error(s)")
        if n_warn:
            parts.append(f"{n_warn} warning(s)")
        print(f"   {' + '.join(parts) or 'All clear'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
