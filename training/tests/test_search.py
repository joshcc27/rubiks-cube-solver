import math

import torch

from src.cube import NUM_MOVES, apply_sequence, is_solved, random_walks, solved, tables
from src.model import INPUT_SIZE, PolicyNet, encode
from src.search import beam_search, network_policy
from src.train import Config, train

SOLVED = solved(1)[0]


def uniform(states: torch.Tensor) -> torch.Tensor:
    return torch.full((len(states), NUM_MOVES), -math.log(NUM_MOVES))


def test_encode_and_model_shapes():
    states = solved(5)
    x = encode(states)
    assert x.shape == (5, INPUT_SIZE)
    assert torch.all(x.sum(dim=1) == 48)
    assert PolicyNet(width=32, blocks=1).eval()(states).shape == (5, NUM_MOVES)


def test_solved_cube_needs_no_moves():
    assert beam_search(uniform, SOLVED, width=4, max_depth=5) == []


def test_beam_wide_enough_to_be_exhaustive_solves_every_two_move_state():
    allowed = tables("cpu").allowed
    for a in range(NUM_MOVES):
        for b in range(NUM_MOVES):
            if not allowed[a, b]:
                continue
            state = apply_sequence(SOLVED, [a, b])
            path = beam_search(uniform, state, width=NUM_MOVES, max_depth=2)
            assert path is not None and len(path) == 2
            assert is_solved(apply_sequence(state, path)[None])[0]


def test_short_training_run_learns_to_undo_short_scrambles(tmp_path):
    cfg = Config(
        width=128, blocks=1, walks=64, depth=3, minutes=5, max_steps=300, out=str(tmp_path / "p.pt")
    )
    policy = network_policy(train(cfg, log=False))
    gen = torch.Generator().manual_seed(3)
    states = random_walks(50, 3, gen)[0][-1]
    solved_greedily = sum(beam_search(policy, s, width=1, max_depth=3) is not None for s in states)
    assert solved_greedily >= 40
