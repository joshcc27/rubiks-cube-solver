"""The policy network: a cube state in, a score for each of the 18 moves out."""

import torch
from torch import nn

from src.cube import NON_CENTRES, NUM_MOVES

INPUT_SIZE = len(NON_CENTRES) * 6  # one-hot colour for each of the 48 moving stickers


def encode(states: torch.Tensor) -> torch.Tensor:
    """(n, 54) colour indices -> (n, 288) floats. Centres never move, so they are left out."""
    idx = torch.tensor(NON_CENTRES, device=states.device)
    one_hot = nn.functional.one_hot(states[:, idx].long(), 6)
    return one_hot.flatten(1).float()


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + self.layers(x))


class PolicyNet(nn.Module):
    """Fully connected network with residual blocks.

    Outputs one logit per move in cube.MOVES. Trained so that softmax(logits)
    is the probability that each move is the next step towards solved.
    """

    def __init__(self, width: int = 512, blocks: int = 4):
        super().__init__()
        self.config = {"width": width, "blocks": blocks}
        self.stem = nn.Sequential(nn.Linear(INPUT_SIZE, width), nn.BatchNorm1d(width), nn.ReLU())
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(blocks)])
        self.head = nn.Linear(width, NUM_MOVES)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.from_encoded(encode(states))

    def from_encoded(self, x: torch.Tensor) -> torch.Tensor:
        """Logits from the (n, 288) encoding. The browser model starts here."""
        logits: torch.Tensor = self.head(self.blocks(self.stem(x)))
        return logits


def save(model: PolicyNet, path: str, **extra: object) -> None:
    torch.save({"config": model.config, "state_dict": model.state_dict(), **extra}, path)


def load(path: str, device: str | torch.device = "cpu") -> PolicyNet:
    checkpoint = torch.load(path, map_location=device)
    model = PolicyNet(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
