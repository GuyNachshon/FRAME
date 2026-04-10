"""Action sensitivity evaluation.

Measures whether the predictor produces different outputs for different actions.
Hold visual context fixed, vary action between 'forward' and 'back',
compute cosine distance between predicted next-frame token distributions.

Target: > 0.1. If near zero, FiLM conditioning is not working.

Usage:
    uv run python eval/action_sensitivity.py \
        --checkpoint checkpoints/vizdoom/predictor_best.pt \
        --config configs/vizdoom_predictor.yaml
"""

import argparse

import torch


def compute_action_sensitivity(
    predictor: torch.nn.Module,
    tokenizer_encoder: torch.nn.Module,
    context_frames: torch.Tensor,
    action_a: torch.Tensor,
    action_b: torch.Tensor,
    device: str = "cuda",
) -> float:
    """Cosine distance between predictions under two different actions.

    Args:
        predictor: Trained predictor model
        tokenizer_encoder: Frozen tokenizer encoder
        context_frames: (B, T, 3, H, W) context frames
        action_a: (B, 72) first action (e.g., forward)
        action_b: (B, 72) second action (e.g., back)
        device: Computation device

    Returns:
        Mean cosine distance (target > 0.1)
    """
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Action sensitivity eval")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    args = parser.parse_args()
    raise NotImplementedError("Full eval pipeline not yet implemented")


if __name__ == "__main__":
    main()
