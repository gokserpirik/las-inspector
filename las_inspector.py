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
import io
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
        # Real overlap detection needs spatial binning; this is a placeholder.

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

    # Simple overlap heuristic: if many points are return 1-of-1 and area coverage
    # is high relative to point count, it may indicate overlapping flight lines.
    if stats.return_counts.get(1, 0) > len(pts) * 0.7 and hdr.point_density > 200:
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


def generate_html_report(
    las: LasFile,
    stats: LasStats,
    warnings: list[QcWarning],
    density_png_path: str | None = None,
    filename: str = "",
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
    header.topbar {{ flex-direction: column; align-items: flex-start; gap: 0.75rem; }}
  }}
  @media (max-width: 480px) {{
    main.hero {{ grid-template-columns: 1fr; }}
    .mini-grid {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<div class="container">

  <!-- Top Bar -->
  <header class="topbar">
    <h1>🔍 las-inspector</h1>
    <div class="meta">
      <span>📄 {filename}</span>
      <span>📅 {now}</span>
      <span>⚙ LAS 1.{hdr.version_minor} · fmt {hdr.point_format_id}</span>
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

  <!-- Density & Elevation -->
  <section class="section density-section">
    <h2>🗺 Density & Elevation</h2>
    <div style="text-align:center;">
      {density_img or '<div class="density-placeholder">No visualization — run with <strong>--density</strong></div>'}
    </div>
  </section>

  <!-- Footer -->
  <footer class="footer">
    Generated by <strong>las-inspector</strong> &middot; {now}
  </footer>

</div>
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
    html = generate_html_report(las, stats, warnings, density_png_path=density_png, filename=las.filename)
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
