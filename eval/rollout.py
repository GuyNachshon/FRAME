"""Long-horizon rollout stability evaluation.

Measures per-step token entropy over autoregressive rollouts.
Detects three collapse signatures:
  1. Mean-field collapse: entropy drops to near zero (scene → average color)
  2. Attractor loop: entropy oscillates (2-3 repeating frames)
  3. Entropy explosion: entropy spikes (pure noise)

Usage:
    uv run python eval/rollout.py \
        --checkpoint checkpoints/vizdoom/predictor_best.pt \
        --config configs/vizdoom_predictor.yaml
"""

import argparse

import torch


def compute_rollout_stability(
    predictor: torch.nn.Module,
    tokenizer_encoder: torch.nn.Module,
    tokenizer_decoder: torch.nn.Module,
    initial_frames: torch.Tensor,
    actions: torch.Tensor,
    n_steps: int = 30,
    device: str = "cuda",
) -> list[float]:
    """Per-step token entropy over a long-horizon rollout.

    Args:
        predictor: Trained predictor model
        tokenizer_encoder: Frozen encoder
        tokenizer_decoder: Frozen decoder
        initial_frames: (B, T_context, 3, H, W) initial context
        actions: (B, n_steps, 72) actions for each rollout step
        n_steps: Number of autoregressive steps
        device: Computation device

    Returns:
        List of mean token entropies per step (length n_steps).
        Target: < 0.5 nats increase over 30 steps.
    """
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Rollout stability eval")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n_steps", type=int, default=30)
    args = parser.parse_args()
    raise NotImplementedError("Full eval pipeline not yet implemented")


if __name__ == "__main__":
    main()
