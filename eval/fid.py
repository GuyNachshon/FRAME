"""FID computation for tokenizer and predictor evaluation.

Uses clean-fid for robust FID computation.

Usage:
    uv run python eval/fid.py \
        --real_dir data/vizdoom/raw/frames/ \
        --fake_dir outputs/reconstructions/
"""

import argparse

import torch


def compute_fid(real_dir: str, fake_dir: str,
                device: str = "cuda") -> float:
    """Compute FID between real and generated/reconstructed frames.

    Args:
        real_dir: Directory of real frames (PNG/JPG)
        fake_dir: Directory of generated frames (PNG/JPG)
        device: Computation device

    Returns:
        FID score (lower is better, target < 50 for tokenizer)
    """
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute FID")
    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--fake_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    fid = compute_fid(args.real_dir, args.fake_dir, args.device)
    print(f"FID: {fid:.2f}")


if __name__ == "__main__":
    main()
