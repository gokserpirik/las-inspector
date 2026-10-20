"""Tests for the density heatmap and HTML report."""

import base64
import json
import re

import numpy as np
from conftest import make_las, make_point
from las_inspector import (
    _build_3d_payload,
    _detect_linear_unit,
    compute_stats,
    generate_density_heatmap,
    generate_html_report,
)


def test_heatmap_written_to_disk(las_file, tmp_path):
    out = tmp_path / "density.png"
    result = generate_density_heatmap(las_file, str(out), grid_size=8)
    assert result == str(out)
    assert out.stat().st_size > 0


def test_heatmap_none_for_small_files():
    las = make_las([make_point() for _ in range(10)])
    assert generate_density_heatmap(las, "never.png") is None


def test_html_report_contents(las_file):
    stats, warnings = compute_stats(las_file)
    html = generate_html_report(las_file, stats, warnings, filename="test.las")
    assert "test.las" in html
    assert "Low point density" in html
    assert "Ground" in html
    assert "pts/m²" in html


def test_html_report_clean_verdict():
    pts = [make_point(x=float(i), y=float(i), classification=2) for i in range(200)]
    las = make_las(pts, max_x=5.0, max_y=2.0)
    stats, warnings = compute_stats(las)
    assert warnings == []
    html = generate_html_report(las, stats, warnings, filename="clean.las")
    assert "No issues detected" in html


def test_html_report_embeds_density_image(las_file, tmp_path):
    stats, warnings = compute_stats(las_file)
    png = tmp_path / "d.png"
    generate_density_heatmap(las_file, str(png), grid_size=8)
    html = generate_html_report(las_file, stats, warnings, density_png_path=str(png))
    assert "data:image/png;base64," in html


def _rd_points(n):
    # Dutch RD New style large projected coordinates
    return [
        make_point(
            x=155000.123 + (i % 50) * 0.37,
            y=463000.456 + (i // 50) * 0.41,
            z=2.5 + (i % 7) * 0.013,
            classification=2 if i % 2 else 6,
            intensity=i % 1000,
        )
        for i in range(n)
    ]


def test_payload_centroid_and_byte_lengths():
    las = make_las(_rd_points(1000))
    p = _build_3d_payload(las, max_3d_points=400)
    assert p["n_total"] == 1000
    assert p["stride"] == 3
    n = p["n_points"]
    assert n == len(range(0, 1000, 3))
    raw = base64.b64decode(p["data_b64"])
    assert len(raw) == 8 * n  # 6 B xyz(u16) + 1 B intensity + 1 B class
    assert p["layout"] == {"xyz_u16": 0, "intensity_u8": 6 * n, "class_u8": 7 * n}

    xs = np.array([pt.x for pt in las.points[::3]])
    expected_cx = xs.mean()
    assert abs(p["centroid"][0] - expected_cx) < 1e-9  # float64 centroid preserved

    q = np.frombuffer(raw, dtype="<u2", count=3 * n).reshape(n, 3)
    bmin, bmax = np.array(p["bbox_min"]), np.array(p["bbox_max"])
    world = bmin + q / 65535.0 * (bmax - bmin)
    centered = (world - np.array(p["centroid"])).astype(np.float32)
    true_centered = np.array([[pt.x, pt.y, pt.z] for pt in las.points[::3]]) - np.array(p["centroid"])
    tol = (bmax - bmin) / 65535.0 / 2 + 1e-4
    assert np.all(np.abs(centered - true_centered) <= tol)
    assert np.abs(centered).max() < 100  # small local values: no float32 jitter
    assert p["min_z"] == bmin[2] and p["max_z"] == bmax[2]


def test_payload_empty_and_unit_detection():
    p = _build_3d_payload(make_las([]))
    assert p["n_points"] == 0 and p["data_b64"] == ""
    assert _detect_linear_unit('PROJCS["x",UNIT["US survey foot",0.3048006096]]')[0] == "ft"
    assert _detect_linear_unit('PROJCS["x",UNIT["metre",1]]') == ("m", 1.0)
    assert _detect_linear_unit("") == ("m", 1.0)


def test_html_report_has_webgl_and_corridor_controls(las_file):
    stats, warnings = compute_stats(las_file)
    html = generate_html_report(las_file, stats, warnings, filename="t.las", max_3d_points=100)
    assert "webgl-section" in html
    assert "Rendering 100 of 200 points (uniform stride subsample)" in html
    assert 'type="importmap"' in html and "cdn.jsdelivr.net/npm/three" in html
    assert "gl_Position = vec4(2.0, 2.0, 2.0, 1.0);" in html
    for uniform in ("uMinZ", "uMaxZ", "uPointSize", "uTol"):
        assert uniform in html
    assert "Reference Datum ΔZ (Elevation Deviation)" in html
    assert "Scan-vs-BIM" in html  # only in the "not a Scan-vs-BIM comparison" disclaimer
    for ctl in ("c-azim", "c-w", "c-l", "c-ox", "c-oy", "profile-canvas", "c-iso"):
        assert ctl in html
    assert html.index("webgl-section") < html.index('class="section density-section"')
    payload = json.loads(re.search(r'id="payload3d" type="application/json">(.*?)</script>', html, re.S).group(1))
    assert payload["n_points"] == 100
