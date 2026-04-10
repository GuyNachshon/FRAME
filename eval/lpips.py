"""LPIPS evaluation for tokenizer reconstructions.

Measures perceptual distance between original and reconstructed frames.
Target: < 0.15. If above this, fix the tokenizer before proceeding.

Usage:
    uv run python eval/lpips.py \
        --checkpoint checkpoints/vizdoom/tokenizer_best.pt \
        --data data/vizdoom/raw/vizdoom_data.hdf5
"""

import argparse

import torch


def compute_lpips(original: torch.Tensor, reconstructed: torch.Tensor,
                  device: str = "cuda") -> float:
    """Compute mean LPIPS distance on tokenizer reconstructions.

    Args:
        original: (N, 3, H, W) original frames
        reconstructed: (N, 3, H, W) reconstructed frames
        device: Computation device

    Returns:
        Mean LPIPS distance (lower is better, target < 0.15)
    """
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute LPIPS")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_samples", type=int, default=1000)
    args = parser.parse_args()
    raise NotImplementedError("Full eval pipeline not yet implemented")


if __name__ == "__main__":
    main()
