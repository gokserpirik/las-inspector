#!/usr/bin/env python3
"""Derive a small committed sample tile from a large local LAS/LAZ file.

Usage:
    python scripts/make_sample.py las2018.laz --output data/sample.laz --stride 22

Takes every Nth point (preserving header, CRS and VLRs via laspy) so the
sample stays representative while remaining small enough to commit.
"""

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Path to the large source LAS/LAZ file")
    parser.add_argument("--output", default="data/sample.laz", help="Sample output path")
    parser.add_argument("--stride", type=int, default=22, help="Keep every Nth point")
    parser.add_argument("--max-mb", type=float, default=5.0, help="Size budget in MB")
    args = parser.parse_args(argv)

    import laspy

    print(f"Reading {args.input} ...")
    try:
        las = laspy.read(args.input)
    except Exception as e:
        print(f"Read error: {e}", file=sys.stderr)
        return 1

    sub = laspy.LasData(las.header)
    # .copy() to get a C-contiguous array (strided views fail LAZ compression)
    sub.points = las.points[:: args.stride].copy()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.write(out)

    size_mb = out.stat().st_size / 1_000_000
    print(f"Wrote {len(sub.points):,} / {len(las.points):,} points -> {out} ({size_mb:.2f} MB)")
    if size_mb > args.max_mb:
        print(f"Over budget ({args.max_mb} MB) — rerun with a larger --stride.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
