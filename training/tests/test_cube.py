import json

import torch

from src.cube import (
    INVERSE,
    MOVES,
    MOVES_PATH,
    apply_sequence,
    from_facelets,
    is_solved,
    random_walks,
    solved,
    tables,
    to_facelets,
)
from src.solve import parse_cube

SOLVED = solved(1)[0]
START = apply_sequence(SOLVED, "R U2 F' L D B2")


def test_solved_state_matches_the_page():
    assert to_facelets(SOLVED) == "".join(f * 9 for f in "URFDLB")
    assert is_solved(SOLVED[None])[0]
    assert not is_solved(START[None])[0]


def test_examples_from_cube_model_js_are_reproduced():
    examples = json.loads(MOVES_PATH.read_text())["examples"]
    assert examples
    for ex in examples:
        assert to_facelets(apply_sequence(SOLVED, ex["moves"])) == ex["state"], ex["moves"]


def test_move_identities():
    for i, move in enumerate(MOVES):
        assert torch.equal(apply_sequence(START, [i, INVERSE[i]]), START), move
        assert torch.equal(apply_sequence(START, [i] * 4), START), move
        if move.endswith("2"):
            quarter = MOVES.index(move[0])
            assert torch.equal(apply_sequence(START, [i]), apply_sequence(START, [quarter] * 2))


def test_each_label_undoes_the_last_move_of_the_walk():
    gen = torch.Generator().manual_seed(1)
    states, moves = random_walks(64, 30, gen)
    t = tables("cpu")
    previous = torch.cat([solved(64)[None], states[:-1]])
    for step in range(30):
        undone = torch.gather(states[step], 1, t.source[t.inverse[moves[step]]])
        assert torch.equal(undone, previous[step])


def test_walks_skip_redundant_moves():
    gen = torch.Generator().manual_seed(2)
    _, moves = random_walks(256, 30, gen)
    faces = [m[0] for m in MOVES]
    for walk in moves.T.tolist():
        for a, b in zip(walk, walk[1:], strict=False):
            assert faces[a] != faces[b]
            assert (faces[a], faces[b]) not in {("D", "U"), ("L", "R"), ("B", "F")}


def test_parse_cube_reads_scrambles_and_relabels_facelets():
    assert torch.equal(parse_cube("R U2 F' L D B2"), START)
    # Facelets of the page's cube after the slice move M, which equals R L' after relabelling.
    after_m = "UBUUBUUBURRRRRRRRRFUFFUFFUFDFDDFDDFDLLLLLLLLLBDBBDBBDB"
    assert torch.equal(parse_cube(after_m), apply_sequence(SOLVED, "R L'"))
    assert torch.equal(from_facelets(to_facelets(START)), START)
