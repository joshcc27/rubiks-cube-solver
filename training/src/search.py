"""Solve a cube by beam search guided by the policy network.

The network gives a probability for each move from a state. Beam search keeps
the `width` move sequences with the highest total log-probability, extends each
by every allowed move, and repeats until one reaches the solved state. With
width 1 this is the greedy solver: always take the network's top move.
"""

from collections.abc import Callable
from functools import cache

import torch

from src.cube import NUM_MOVES, START, all_children, is_solved, tables
from src.model import PolicyNet

# Maps a batch of states (n, 54) to log-probabilities over moves (n, 18).
LogPolicy = Callable[[torch.Tensor], torch.Tensor]


def network_policy(model: PolicyNet) -> LogPolicy:
    model.eval()

    @torch.no_grad()
    def log_policy(states: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(model(states), dim=1)

    return log_policy


@cache
def _hash_weights(device: torch.device) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randint(-(2**62), 2**62, (54,), generator=generator).to(device)


def _hash(states: torch.Tensor) -> torch.Tensor:
    """One int64 per state, used to drop duplicate states from the beam."""
    return (states.long() * _hash_weights(states.device)).sum(dim=1)


def beam_search(
    log_policy: LogPolicy, start: torch.Tensor, width: int, max_depth: int
) -> list[int] | None:
    """Return a list of move indices that solves `start`, or None if none is found."""
    t = tables(start.device)
    if is_solved(start[None])[0]:
        return []
    states = start[None]
    scores = torch.zeros(1, device=start.device)
    paths = torch.empty(1, 0, dtype=torch.long, device=start.device)
    last = torch.full((1,), START, dtype=torch.long, device=start.device)

    for _ in range(max_depth):
        children = all_children(states).reshape(-1, states.shape[1])
        child_scores = (scores[:, None] + log_policy(states)).masked_fill(
            ~t.allowed[last], float("-inf")
        )
        child_scores = child_scores.flatten()
        parent = torch.arange(states.shape[0], device=start.device).repeat_interleave(NUM_MOVES)
        move = torch.arange(NUM_MOVES, device=start.device).repeat(states.shape[0])

        # Check every allowed child for solved, not only the ones the beam keeps.
        hit = (is_solved(children) & torch.isfinite(child_scores)).nonzero()
        if len(hit):
            i = int(hit[0])
            return [*paths[parent[i]].tolist(), int(move[i])]

        # Keep the best-scoring copy of each distinct state, then the top `width`.
        order = child_scores.argsort(descending=True)
        order = order[torch.isfinite(child_scores[order])]
        _, inverse = torch.unique(_hash(children[order]), return_inverse=True)
        first = torch.full((int(inverse.max()) + 1,), len(order), device=start.device)
        first = first.scatter_reduce(
            0, inverse, torch.arange(len(order), device=start.device), "amin"
        )
        keep = order[first.sort().values[:width]]

        states = children[keep]
        scores = child_scores[keep]
        paths = torch.cat([paths[parent[keep]], move[keep, None]], dim=1)
        last = move[keep]
    return None
