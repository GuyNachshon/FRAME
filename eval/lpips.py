"""LPIPS measurement for tokenizer reconstructions.

Measures perceptual distance between original and reconstructed frames.
Target: < 0.15. If above this, fix the tokenizer before proceeding.

Usage:
    uv run python -m eval.lpips \
        --checkpoint checkpoints/vizdoom/tokenizer/tokenizer_best.pt \
        --data data/vizdoom/raw/vizdoom_data.hdf5
"""

import argparse

import lpips
import torch
from torch.utils.data import DataLoader

from data.vizdoom.dataset import ViZDoomFrameDataset
from tokenizer.encoder import CNNEncoder
from tokenizer.vq import VectorQuantizer
from tokenizer.decoder import CNNDecoder


def load_tokenizer(
    ckpt_path: str, device: torch.device,
) -> tuple[CNNEncoder, VectorQuantizer, CNNDecoder]:
    """Load tokenizer from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]["model"]

    encoder = CNNEncoder(
        channels=cfg["encoder_channels"],
        codebook_dim=cfg["codebook_dim"],
    ).to(device)
    vq = VectorQuantizer(
        n_codes=cfg["codebook_size"],
        code_dim=cfg["codebook_dim"],
        ema_decay=cfg["ema_decay"],
    ).to(device)
    decoder = CNNDecoder(codebook_dim=cfg["codebook_dim"]).to(device)

    encoder.load_state_dict(ckpt["encoder"])
    vq.load_state_dict(ckpt["vq"])
    decoder.load_state_dict(ckpt["decoder"])

    encoder.requires_grad_(False)
    vq.requires_grad_(False)
    decoder.requires_grad_(False)
    return encoder, vq, decoder


@torch.no_grad()
def compute_lpips(
    encoder: CNNEncoder,
    vq: VectorQuantizer,
    decoder: CNNDecoder,
    dataset: ViZDoomFrameDataset,
    n_samples: int = 1000,
    batch_size: int = 32,
    device: str = "cuda",
) -> float:
    """Compute mean LPIPS distance on tokenizer reconstructions.

    Args:
        encoder: Trained encoder
        vq: Trained VQ
        decoder: Trained decoder
        dataset: Frame dataset
        n_samples: Number of samples to measure
        batch_size: Batch size
        device: Computation device

    Returns:
        Mean LPIPS distance (lower is better, target < 0.15)
    """
    loss_fn = lpips.LPIPS(net="alex").to(device)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    total_lpips = 0.0
    n_done = 0

    for frames in loader:
        if n_done >= n_samples:
            break

        frames = frames.to(device)  # (B, 3, 128, 128) in [0, 1]

        # Reconstruct
        z = encoder(frames)
        z_q, _, _ = vq(z)
        recon = decoder(z_q)

        # LPIPS expects inputs in [-1, 1]
        frames_scaled = frames * 2 - 1
        recon_scaled = recon.clamp(0, 1) * 2 - 1

        dist = loss_fn(frames_scaled, recon_scaled)  # (B, 1, 1, 1)
        total_lpips += dist.sum().item()
        n_done += frames.shape[0]

    return total_lpips / n_done


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute LPIPS")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_samples", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    encoder, vq, decoder = load_tokenizer(args.checkpoint, device)
    dataset = ViZDoomFrameDataset(args.data)

    print(f"Computing LPIPS on {args.n_samples} samples...")
    score = compute_lpips(
        encoder, vq, decoder, dataset,
        n_samples=args.n_samples, device=args.device,
    )
    gate = "PASS" if score < 0.15 else "FAIL"
    print(f"LPIPS: {score:.4f} (target < 0.15) [{gate}]")


if __name__ == "__main__":
    main()
