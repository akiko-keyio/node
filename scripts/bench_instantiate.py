"""Benchmark node.instantiate() for trop_eval with sweep.

Usage (from trop-system/exps/05_compression):
    set PYTHONPATH=Y:\node\src
    uv run python Y:\node\scripts\bench_instantiate.py [--degree-max 40] [--profile]
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import pstats
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree-max", type=int, default=40)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--skip-repr", action="store_true",
                        help="Skip repr SHA (can be very slow on large graphs)")
    args = parser.parse_args()

    import trop_system  # noqa: F401
    from trop_system import node

    basis_order = ("ahsh", "poly")
    sweep = {
        "trop_ls.basis": list(basis_order),
        "trop_ls.degree": list(range(1, args.degree_max + 1)),
    }

    if args.profile:
        pr = cProfile.Profile()
        pr.enable()

    t0 = time.perf_counter()
    eval_result = node.instantiate("trop_eval", sweep=sweep)
    t1 = time.perf_counter()

    if args.profile:
        pr.disable()

    wall = t1 - t0
    print(f"instantiate wall: {wall:.3f} s")

    # Compatibility fingerprints (fast)
    print(f"hash(node):       {hash(eval_result)}")
    print(f"node._hash:       {eval_result._hash:#x}")

    # Count total nodes in execution graph
    from node.core import build_graph
    order, _ = build_graph(eval_result, None)
    print(f"total nodes:      {len(order)}")

    # Leaf-level hash sample: check a few inner item hashes
    if eval_result._items is not None:
        sample_hashes = [
            n._hash for n in eval_result._items.flat[:5]
            if hasattr(n, "_hash")
        ]
        print(f"item_hash[:5]:    {[f'{h:#x}' for h in sample_hashes]}")

    if not args.skip_repr:
        t2 = time.perf_counter()
        repr_sha = hashlib.sha256(repr(eval_result).encode()).hexdigest()[:16]
        t3 = time.perf_counter()
        print(f"repr SHA256[:16]: {repr_sha}  ({t3 - t2:.1f}s)")

    if args.profile:
        print("\n--- cProfile top 20 (cumulative) ---")
        stats = pstats.Stats(pr)
        stats.sort_stats("cumulative")
        stats.print_stats(20)


if __name__ == "__main__":
    main()
