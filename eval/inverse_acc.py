"""Inverse dynamics accuracy evaluation.

Measures accuracy of the inverse dynamics head on held-out
(z_t, z_{t+gap}) pairs. Validates that the encoder encodes
action-relevant information.

Target: > 40% (chance is 1/72 ≈ 1.4%).
If near chance: encoder is ignoring actions — stop and debug.

Usage:
    uv run python eval/inverse_acc.py \
        --checkpoint checkpoints/vizdoom/predictor_best.pt \
        --config configs/vizdoom_predictor.yaml
"""

import argparse

import torch
from torch.utils.data import Dataset


def compute_inverse_accuracy(
    inverse_head: torch.nn.Module,
    encoder: torch.nn.Module,
    dataset: Dataset,
    gap: int = 4,
    n_samples: int = 1000,
    device: str = "cuda",
) -> float:
    """Accuracy of inverse dynamics head on held-out data.

    Args:
        inverse_head: Trained inverse dynamics MLP
        encoder: Frozen/trained encoder
        dataset: Dataset providing frame sequences
        gap: Frame gap for (z_t, z_{t+gap}) pairs (default 4)
        n_samples: Number of samples to evaluate
        device: Computation device

    Returns:
        Accuracy as fraction (target > 0.4)
    """
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Inverse dynamics accuracy")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    args = parser.parse_args()
    raise NotImplementedError("Full eval pipeline not yet implemented")


if __name__ == "__main__":
    main()
