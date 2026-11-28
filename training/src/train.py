"""Train the policy network on random scrambles.

Each example is a state reached by a random walk from solved, labelled with the
move that undoes the walk's last move. Over many walks the network learns which
moves tend to lead back towards solved. Training data is generated on the fly,
so there is no dataset to store.

    uv run python -m src.train --minutes 30
"""

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from src.cube import random_walks, tables
from src.model import PolicyNet, save


@dataclass
class Config:
    width: int = 512  # hidden layer size
    blocks: int = 4  # residual blocks
    walks: int = 256  # random walks per batch; each gives `depth` examples
    depth: int = 26  # length of each walk
    lr: float = 1e-3
    minutes: float = 30.0  # time budget; the learning rate decays to zero over it
    max_steps: int | None = None  # optional step budget, whichever runs out first
    device: str = "auto"
    threads: int | None = None  # CPU threads; None keeps PyTorch's default
    seed: int = 0
    out: str = "checkpoints/policy.pt"
    log_seconds: float = 30.0
    save_minutes: float = 5.0


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def make_batch(
    walks: int, depth: int, generator: torch.Generator, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """States, labels (the move that undoes the last move) and walk depths, all flattened."""
    states, moves = random_walks(walks, depth, generator, device)
    labels = tables(device).inverse[moves]
    depths = torch.arange(1, depth + 1, device=device)[:, None].expand(depth, walks)
    return states.flatten(0, 1), labels.flatten(), depths.flatten()


def train(cfg: Config, log: bool = True) -> PolicyNet:
    device = pick_device(cfg.device)
    if cfg.threads is not None:
        torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    model = PolicyNet(cfg.width, cfg.blocks).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)

    if log:
        params = sum(p.numel() for p in model.parameters())
        print(
            f"device {device}, {params / 1e6:.2f}M parameters, {cfg.walks * cfg.depth} examples per step"
        )

    start = last_log = last_save = time.perf_counter()
    step = examples = 0
    # Running totals since the last log line: loss, and accuracy by walk depth.
    loss_sum = 0.0
    batches = 0
    correct = torch.zeros(cfg.depth + 1)
    seen = torch.zeros(cfg.depth + 1)

    while True:
        elapsed = time.perf_counter() - start
        progress = elapsed / (cfg.minutes * 60)
        if cfg.max_steps is not None:
            progress = max(progress, step / cfg.max_steps)
        if progress >= 1:
            break
        lr = cfg.lr * 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimiser.param_groups:
            group["lr"] = lr

        model.train()
        states, labels, depths = make_batch(cfg.walks, cfg.depth, generator, device)
        logits = model(states)
        loss = nn.functional.cross_entropy(logits, labels)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

        step += 1
        examples += len(labels)
        loss_sum += loss.item()
        batches += 1
        hits = (logits.argmax(dim=1) == labels).cpu().float()
        correct.index_add_(0, depths.cpu(), hits)
        seen.index_add_(0, depths.cpu(), torch.ones_like(hits))

        now = time.perf_counter()
        if log and now - last_log >= cfg.log_seconds:
            acc = correct / seen.clamp(min=1)
            buckets = " ".join(
                f"d{d}:{acc[d]:.2f}" for d in (1, 3, 5, 8, 12, 16, 20) if d <= cfg.depth
            )
            print(
                f"{elapsed / 60:6.1f} min  step {step:6d}  {examples / 1e6:7.1f}M ex  "
                f"loss {loss_sum / batches:.3f}  lr {lr:.1e}  {examples / elapsed:6.0f} ex/s  "
                f"acc {buckets}",
                flush=True,
            )
            last_log = now
            loss_sum, batches = 0.0, 0
            correct.zero_()
            seen.zero_()
        if now - last_save >= cfg.save_minutes * 60:
            save(model, cfg.out, step=step, examples=examples)
            last_save = now

    save(model, cfg.out, step=step, examples=examples)
    if log:
        print(f"saved {cfg.out} after {step} steps, {examples / 1e6:.1f}M examples")
    model.eval()
    return model


def main() -> None:
    defaults = Config()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for name, value in vars(defaults).items():
        flag = "--" + name.replace("_", "-")
        kind = type(value) if value is not None else int
        parser.add_argument(flag, type=kind, default=value)
    cfg = Config(**vars(parser.parse_args()))
    train(cfg)


if __name__ == "__main__":
    main()
