#!/usr/bin/env python3
"""Benchmark las_inspector.read_las against laspy on header/stats extraction.

Both contenders do the same work: decompress + parse every point, then
derive point count, bounds, point density and the classification breakdown.

Usage:
    python scripts/benchmark.py las2018.laz --repeat 3
"""

import argparse
import statistics
import sys
import timeit


def bench_las_inspector(path: str, number: int) -> list[float]:
    from las_inspector import compute_stats, read_las

    def workload():
        las = read_las(path)
        compute_stats(las)

    return timeit.repeat(workload, number=1, repeat=number)


def bench_laspy(path: str, number: int) -> list[float]:
    import numpy as np
    import laspy

    def workload():
        las = laspy.read(path)
        h = las.header
        area = (h.maxs[0] - h.mins[0]) * (h.maxs[1] - h.mins[1])
        _density = h.point_count / area if area > 0 else 0.0
        _classes, _counts = np.unique(las.classification, return_counts=True)
        _returns, _rcounts = np.unique(las.return_number, return_counts=True)

    return timeit.repeat(workload, number=1, repeat=number)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Path to a LAS/LAZ file")
    parser.add_argument("--repeat", type=int, default=3, help="Timing repetitions")
    args = parser.parse_args(argv)

    print(f"File: {args.input} (repeat={args.repeat})")
    ours = bench_las_inspector(args.input, args.repeat)
    theirs = bench_laspy(args.input, args.repeat)

    mean_ours, mean_theirs = statistics.mean(ours), statistics.mean(theirs)
    print(f"las_inspector.read_las + compute_stats : {mean_ours:.2f}s (best {min(ours):.2f}s)")
    print(f"laspy.read + numpy stats               : {mean_theirs:.2f}s (best {min(theirs):.2f}s)")
    print(f"speedup: {mean_theirs / mean_ours:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
