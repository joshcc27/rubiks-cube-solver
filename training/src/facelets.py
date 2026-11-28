"""Cubes written the way the page writes them: typed moves and facelet strings.

The page (cube-model.js) accepts slice turns and moves typed without spaces,
and slice turns carry the centres around. These helpers read the same input
and give the facelet string the page would show. relabel_by_centres then puts
the centres home, which is the only layout the network knows.
"""

import re

from src.cube import CENTRES, EVERY_SOURCE, FACES, NUM_STICKERS

SOLVED_FACELETS = "".join(face * 9 for face in FACES)

_TOKEN = re.compile(r"\s*([URFDLBMES])(2'|'2|'|2)?")
_PRIMES = re.compile("[’‘´`′]")


def parse_sequence(text: str, max_moves: int = 500) -> list[str]:
    """Read typed moves the same way as cube-model.js parseSequence.

    Accepts "R U R' U'", "RUR'U'", slice turns, typographic apostrophes, and
    "2'" or "'2" for a double turn. Raises ValueError with a readable message.
    """
    src = _PRIMES.sub("'", text).strip()
    moves: list[str] = []
    pos = 0
    while pos < len(src):
        m = _TOKEN.match(src, pos)
        if m is None:
            bad = src[pos:].split()[0]
            raise ValueError(
                f'"{bad}" isn\'t a move. Use the letters U D R L F B M E S, '
                "optionally followed by ' or 2."
            )
        suffix = "'" if m.group(2) == "'" else "2" if m.group(2) else ""
        moves.append(m.group(1) + suffix)
        if len(moves) > max_moves:
            raise ValueError(f"That's more than {max_moves} moves.")
        pos = m.end()
    return moves


def apply_to_facelets(facelets: str, moves: list[str]) -> str:
    """Turn a facelet string with any of the page's moves, slice turns included."""
    for move in moves:
        source = EVERY_SOURCE[move]
        facelets = "".join(facelets[i] for i in source)
    return facelets


def check_facelets(facelets: str) -> None:
    """Raise ValueError if this can't be a cube's facelet string.

    Checks the length, the letters, nine stickers of each colour and six
    different centres. A string that passes can still describe a cube that no
    turns can reach, such as one with a single twisted corner; the search then
    finds no solution.
    """
    if len(facelets) != NUM_STICKERS or set(facelets) - set(FACES):
        raise ValueError(f"A facelet string is {NUM_STICKERS} letters from {' '.join(FACES)}.")
    wrong = {face: facelets.count(face) for face in FACES if facelets.count(face) != 9}
    if wrong:
        counts = ", ".join(f"{n} {face}" for face, n in wrong.items())
        raise ValueError(f"Each colour must appear 9 times, but this has {counts}.")
    if len({facelets[i] for i in CENTRES}) != len(FACES):
        raise ValueError("The six centres must all be different colours.")


def relabel_by_centres(facelets: str) -> str:
    """Name each colour after the face whose centre shows it, as cube-model.js does."""
    rename = {facelets[i]: face for i, face in zip(CENTRES, FACES, strict=True)}
    return "".join(rename[c] for c in facelets)
