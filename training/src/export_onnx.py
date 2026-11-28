"""Export a trained network to ONNX for the solver in the page.

The exported model takes the one-hot encoding from model.encode, shape (n, 288)
float32, and returns log-probabilities over cube.MOVES, shape (n, 18). The page
builds the encoding itself (solver.js), so the model is only matrix maths.

    uv run python -m src.export_onnx checkpoints/policy.pt ../models/policy.onnx
"""

import argparse
from pathlib import Path

import torch
from torch import nn

from src.cube import random_walks
from src.model import PolicyNet, encode, load

INPUT_NAME = "onehot"
OUTPUT_NAME = "log_probs"


class LogPolicy(nn.Module):
    def __init__(self, net: PolicyNet):
        super().__init__()
        self.net = net

    def forward(self, onehot: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.net.from_encoded(onehot), dim=1)


def export(model: PolicyNet, path: str | Path) -> None:
    model.eval()
    example = encode(random_walks(4, 10, torch.Generator().manual_seed(0))[0][-1])
    torch.onnx.export(
        LogPolicy(model).eval(),
        (example,),
        str(path),
        dynamo=True,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_shapes={"onehot": {0: torch.export.Dim("batch", min=1, max=1 << 16)}},
        external_data=False,
        verbose=False,  # its progress messages include emoji, which a Windows console can't print
    )


def check(model: PolicyNet, path: str | Path, n: int = 64) -> float:
    """Largest difference between PyTorch and ONNX Runtime outputs on random cubes."""
    import onnxruntime as ort

    states = random_walks(n, 40, torch.Generator().manual_seed(1))[0][-1]
    onehot = encode(states)
    with torch.no_grad():
        expected = torch.log_softmax(model.eval()(states), dim=1)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (got,) = session.run([OUTPUT_NAME], {INPUT_NAME: onehot.numpy()})
    return float((torch.from_numpy(got) - expected).abs().max())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint")
    parser.add_argument("out")
    args = parser.parse_args()

    model = load(args.checkpoint)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    export(model, args.out)
    diff = check(model, args.out)
    size = Path(args.out).stat().st_size / 1e6
    print(f"wrote {args.out} ({size:.1f} MB), max difference from PyTorch {diff:.2e}")


if __name__ == "__main__":
    main()
