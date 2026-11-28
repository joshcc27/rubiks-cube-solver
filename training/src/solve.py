"""Solve one cube with a trained network.

The cube is given either as a scramble, typed the way the page accepts it
(slice turns and moves without spaces included), or as the 54-letter facelet
string the page uses (cube-model.js state.join('')):

    uv run python -m src.solve checkpoints/policy.pt "R U R' U' F2 D L"
    uv run python -m src.solve checkpoints/policy.pt "M2 E S' RUR'U'"
    uv run python -m src.solve checkpoints/policy.pt UUUUUUUUURRR...

A 54-letter string made only of U R F D L B is read as facelets.
"""

import argparse
import time

import torch

from src.cube import FACES, MOVES, NUM_STICKERS, from_facelets
from src.facelets import (
    SOLVED_FACELETS,
    apply_to_facelets,
    check_facelets,
    parse_sequence,
    relabel_by_centres,
)
from src.model import load
from src.search import beam_search, network_policy


def parse_cube(text: str) -> torch.Tensor:
    """The cube described by `text`, with its centres home. Raises ValueError if unreadable."""
    text = text.strip()
    if len(text) == NUM_STICKERS and set(text) <= set(FACES):
        check_facelets(text)
        facelets = text
    else:
        facelets = apply_to_facelets(SOLVED_FACELETS, parse_sequence(text))
    return from_facelets(relabel_by_centres(facelets))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint")
    parser.add_argument("cube", help='a scramble like "R U R\'" or a 54-letter facelet string')
    parser.add_argument("--width", type=int, default=512, help="beam width (1 = greedy)")
    parser.add_argument("--max-depth", type=int, default=60)
    args = parser.parse_args()
    try:
        cube = parse_cube(args.cube)
    except ValueError as err:
        parser.error(str(err))

    model = load(args.checkpoint)
    start = time.perf_counter()
    path = beam_search(network_policy(model), cube, args.width, args.max_depth)
    seconds = time.perf_counter() - start
    if path is None:
        print(f"no solution within {args.max_depth} moves ({seconds:.1f} s)")
    else:
        print(f"{' '.join(MOVES[m] for m in path)}\n{len(path)} moves, {seconds:.1f} s")


if __name__ == "__main__":
    main()
