"""Solve one cube with a trained network.

The cube is given either as a scramble or as the 54-letter facelet string the
page uses (cube-model.js state.join('')):

    uv run python -m src.solve checkpoints/policy.pt "R U R' U' F2 D L"
    uv run python -m src.solve checkpoints/policy.pt UUUUUUUUURRR...
"""

import argparse
import time

import torch

from src.cube import FACES, MOVES, NUM_STICKERS, apply_sequence, from_facelets, solved
from src.model import load
from src.search import beam_search, network_policy


def relabel_by_centres(facelets: str) -> str:
    """Name each colour after the face whose centre shows it (as in cube-model.js)."""
    rename = {facelets[f * 9 + 4]: face for f, face in enumerate(FACES)}
    return "".join(rename[c] for c in facelets)


def parse_cube(text: str) -> torch.Tensor:
    if len(text) == NUM_STICKERS and set(text) <= set(FACES):
        return from_facelets(relabel_by_centres(text))
    return apply_sequence(solved(1)[0], text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint")
    parser.add_argument("cube", help='a scramble like "R U R\'" or a 54-letter facelet string')
    parser.add_argument("--width", type=int, default=512, help="beam width (1 = greedy)")
    parser.add_argument("--max-depth", type=int, default=60)
    args = parser.parse_args()

    model = load(args.checkpoint)
    start = time.perf_counter()
    path = beam_search(network_policy(model), parse_cube(args.cube), args.width, args.max_depth)
    seconds = time.perf_counter() - start
    if path is None:
        print(f"no solution within {args.max_depth} moves ({seconds:.1f} s)")
    else:
        print(f"{' '.join(MOVES[m] for m in path)}\n{len(path)} moves, {seconds:.1f} s")


if __name__ == "__main__":
    main()
