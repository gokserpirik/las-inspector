"""Tests for the density heatmap and HTML report."""

from conftest import make_las, make_point
from las_inspector import compute_stats, generate_density_heatmap, generate_html_report


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
