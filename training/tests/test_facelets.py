import json

import pytest

from src.cube import MOVES_PATH, apply_sequence, solved
from src.facelets import (
    SOLVED_FACELETS,
    apply_to_facelets,
    check_facelets,
    parse_sequence,
    relabel_by_centres,
)
from src.solve import parse_cube

SOLVED = solved(1)[0]


def test_every_move_example_from_cube_model_js_is_reproduced():
    examples = json.loads(MOVES_PATH.read_text())["every_move_examples"]
    assert examples
    for ex in examples:
        assert apply_to_facelets(SOLVED_FACELETS, ex["moves"].split()) == ex["state"], ex["moves"]


def test_typed_moves_are_read_like_the_page():
    assert parse_sequence("R U R' U'") == ["R", "U", "R'", "U'"]
    assert parse_sequence("RUR'U'") == ["R", "U", "R'", "U'"]
    assert parse_sequence("  M2 E' S \n F2' B'2 ") == ["M2", "E'", "S", "F2", "B2"]
    assert parse_sequence("R’ U‘") == ["R'", "U'"]
    assert parse_sequence("   ") == []


@pytest.mark.parametrize(
    ("text", "message"),
    [("R u", '"u" isn\'t a move'), ("R x U", '"x" isn\'t a move'), ("R3", '"3" isn\'t a move')],
)
def test_typed_moves_that_are_not_moves_give_a_readable_error(text, message):
    with pytest.raises(ValueError, match=message):
        parse_sequence(text)


def test_too_many_moves_is_an_error():
    with pytest.raises(ValueError, match="more than 5 moves"):
        parse_sequence("R " * 6, max_moves=5)


def test_parse_cube_reads_slice_turns_and_compact_moves():
    # After relabelling, M is R L' (as the page's tests check).
    assert parse_cube("M").equal(apply_sequence(SOLVED, "R L'"))
    assert parse_cube("RUR'U'").equal(apply_sequence(SOLVED, "R U R' U'"))
    assert parse_cube("R2'").equal(apply_sequence(SOLVED, "R2"))
    slices = apply_to_facelets(SOLVED_FACELETS, parse_sequence("M U E'"))
    assert parse_cube("M U E'").tolist() == ["URFDLB".index(c) for c in relabel_by_centres(slices)]


def test_bad_facelet_strings_are_rejected():
    with pytest.raises(ValueError, match="9 times"):
        check_facelets("U" * 10 + SOLVED_FACELETS[10:])
    two_u_centres = list(SOLVED_FACELETS)
    two_u_centres[13] = "U"  # the R centre shows U
    two_u_centres[0] = "R"  # and a U sticker shows R, so there are still nine of each
    with pytest.raises(ValueError, match="centres"):
        check_facelets("".join(two_u_centres))
    with pytest.raises(ValueError, match="54 letters"):
        check_facelets("UUU")
    check_facelets(apply_to_facelets(SOLVED_FACELETS, parse_sequence("M2 R U")))
    with pytest.raises(ValueError, match="isn't a move"):
        parse_cube("R x")
