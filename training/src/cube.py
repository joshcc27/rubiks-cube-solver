"""Batched cube model in PyTorch, built from moves.json.

moves.json is written from cube-model.js by scripts/export-moves.mjs, so the
moves here are exactly the moves the page uses. A state is a row of 54 colour
indices (0..5 for U R F D L B) in the same sticker order as cube-model.js.
Only the 18 face turns are used, so the centres never move.
"""

import json
from functools import cache
from pathlib import Path

import torch

MOVES_PATH = Path(__file__).resolve().parents[1] / "moves.json"
_DATA = json.loads(MOVES_PATH.read_text())

FACES: str = _DATA["faces"]  # "URFDLB"
MOVES: list[str] = _DATA["moves"].split()  # X, X', X2 for each face in FACES order
NUM_MOVES = len(MOVES)
NUM_STICKERS = 54
CENTRES = [f * 9 + 4 for f in range(6)]
NON_CENTRES = [i for i in range(NUM_STICKERS) if i not in CENTRES]

# Index 18 stands for "no previous move", used at the start of a sequence.
START = NUM_MOVES


def _invert(move: str) -> str:
    if move.endswith("2"):
        return move
    return move[0] if move.endswith("'") else move + "'"


INVERSE = [MOVES.index(_invert(m)) for m in MOVES]
FACE_OF_MOVE = [FACES.index(m[0]) for m in MOVES]


def _allowed_after() -> torch.Tensor:
    """allowed[p, m] says whether move m may follow move p.

    Two turns of the same face in a row are never needed, and two turns of
    opposite faces commute, so only one order is kept: U before D, R before L,
    F before B. In FACES order the opposite of face f is f + 3.
    """
    allowed = torch.ones(NUM_MOVES + 1, NUM_MOVES, dtype=torch.bool)
    for p in range(NUM_MOVES):
        fp = FACE_OF_MOVE[p]
        for m in range(NUM_MOVES):
            f = FACE_OF_MOVE[m]
            allowed[p, m] = f != fp and f + 3 != fp
    return allowed


class Tables:
    """The move tables on one device."""

    def __init__(self, device: torch.device):
        self.source = torch.tensor(_DATA["source"], dtype=torch.long, device=device)
        self.inverse = torch.tensor(INVERSE, dtype=torch.long, device=device)
        self.allowed = _allowed_after().to(device)
        self.solved = torch.arange(6, dtype=torch.uint8, device=device).repeat_interleave(9)


@cache
def tables(device: str | torch.device = "cpu") -> Tables:
    return Tables(torch.device(device))


def solved(n: int, device: str | torch.device = "cpu") -> torch.Tensor:
    return tables(device).solved.expand(n, NUM_STICKERS).clone()


def apply_moves(states: torch.Tensor, moves: torch.Tensor) -> torch.Tensor:
    """Apply moves[i] to states[i]. states is (n, 54), moves is (n,)."""
    return torch.gather(states, 1, tables(states.device).source[moves])


def all_children(states: torch.Tensor) -> torch.Tensor:
    """Every state one move away: (n, 54) -> (n, 18, 54)."""
    source = tables(states.device).source
    return states[:, source]


def is_solved(states: torch.Tensor) -> torch.Tensor:
    return (states == tables(states.device).solved).all(dim=-1)


def from_facelets(facelets: str) -> torch.Tensor:
    """'UUUUUUUUURRR...' -> a (54,) state."""
    if len(facelets) != NUM_STICKERS:
        raise ValueError(f"expected 54 facelets, got {len(facelets)}")
    return torch.tensor([FACES.index(c) for c in facelets], dtype=torch.uint8)


def to_facelets(state: torch.Tensor) -> str:
    return "".join(FACES[int(c)] for c in state)


def parse_moves(seq: str) -> list[int]:
    return [MOVES.index(m) for m in seq.split()]


def apply_sequence(state: torch.Tensor, seq: str | list[int]) -> torch.Tensor:
    moves = parse_moves(seq) if isinstance(seq, str) else seq
    out = state[None]
    for m in moves:
        out = apply_moves(out, torch.tensor([m], device=state.device))
    return out[0]


def random_walks(
    n: int,
    length: int,
    generator: torch.Generator | None = None,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scramble n solved cubes with `length` random moves each.

    Moves that would undo or reorder the previous move are skipped (see
    _allowed_after). Returns the state after every move, shape (length, n, 54),
    and the moves themselves, shape (length, n).
    """
    t = tables(device)
    states = solved(n, device)
    prev = torch.full((n,), START, dtype=torch.long, device=device)
    all_states = torch.empty(length, n, NUM_STICKERS, dtype=torch.uint8, device=device)
    all_moves = torch.empty(length, n, dtype=torch.long, device=device)
    for step in range(length):
        weights = t.allowed[prev].float()
        moves = torch.multinomial(weights, 1, generator=generator).squeeze(1)
        states = apply_moves(states, moves)
        all_states[step] = states
        all_moves[step] = moves
        prev = moves
    return all_states, all_moves
