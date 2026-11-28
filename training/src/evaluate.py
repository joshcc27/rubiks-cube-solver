"""Measure how well a trained network solves scrambles of different lengths.

For each scramble length and beam width, reports the share of cubes solved,
the average solution length and the average time per cube. A 100-move scramble
is effectively a random cube.

    uv run python -m src.evaluate checkpoints/policy.pt --depths 5 10 15 20 100 --widths 1 64 512
"""

import argparse
import time

import torch

from src.cube import apply_sequence, is_solved, random_walks
from src.model import load
from src.search import beam_search, network_policy


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--depths", type=int, nargs="+", default=[5, 10, 15, 20, 100])
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 64, 512])
    parser.add_argument("--cubes", type=int, default=50, help="scrambles per length")
    parser.add_argument("--max-depth", type=int, default=60, help="longest solution searched for")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    policy = network_policy(load(args.checkpoint))
    generator = torch.Generator().manual_seed(args.seed)
    print(f"{'scramble':>8} {'width':>6} {'solved':>7} {'avg moves':>9} {'avg s':>6}")
    for depth in args.depths:
        states = random_walks(args.cubes, depth, generator)[0][-1]
        for width in args.widths:
            lengths = []
            start = time.perf_counter()
            for state in states:
                path = beam_search(policy, state, width, args.max_depth)
                if path is not None:
                    assert is_solved(apply_sequence(state, path)[None])[0]
                    lengths.append(len(path))
            seconds = (time.perf_counter() - start) / len(states)
            rate = len(lengths) / len(states)
            avg = f"{sum(lengths) / len(lengths):9.1f}" if lengths else f"{'-':>9}"
            print(f"{depth:8d} {width:6d} {rate:7.0%} {avg} {seconds:6.2f}", flush=True)


if __name__ == "__main__":
    main()
