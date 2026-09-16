"""Tests for the CLI entry point and the legacy main.py script."""

import subprocess
import sys
from pathlib import Path

from las_inspector import main

ROOT = Path(__file__).resolve().parent.parent


def test_cli_missing_file_returns_1(capsys):
    assert main(["missing.laz"]) == 1
    captured = capsys.readouterr()
    assert "not found" in (captured.out + captured.err).lower()


def test_cli_garbage_file_returns_1(tmp_path, capsys, monkeypatch):
    bad = tmp_path / "bad.las"
    bad.write_bytes(b"this is not a LAS file")
    monkeypatch.chdir(tmp_path)
    assert main([str(bad)]) == 1


def test_cli_end_to_end(sample_las_path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out.html"
    assert main([str(sample_las_path), "--output", str(out)]) == 0
    assert out.stat().st_size > 0
    summaries = list(tmp_path.glob("*_qc_summary.txt"))
    assert len(summaries) == 1
    assert "QC Verdict" in summaries[0].read_text()


def test_legacy_main_py_missing_file(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "main.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert "not found" in proc.stderr.lower()
