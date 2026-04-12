"""Inverse dynamics accuracy measurement.

Measures accuracy of the inverse dynamics head on held-out
(z_t, z_{t+gap}) pairs. Validates that the encoder encodes
action-relevant information.

Target: > 40% (chance is 1/72 = 1.4%, but for 8-class keyboard: 1/8 = 12.5%).
If near chance: encoder is ignoring actions — stop and debug.

Usage:
    uv run python -m eval.inverse_acc \
        --checkpoint checkpoints/vizdoom/predictor/predictor_best.pt \
        --tokenizer_checkpoint checkpoints/vizdoom/tokenizer/tokenizer_best.pt \
        --data data/vizdoom/raw/vizdoom_data.hdf5
"""

import argparse

import torch
from torch.utils.data import DataLoader

from data.vizdoom.dataset import ViZDoomSequenceDataset
from predictor.inverse_dynamics import InverseDynamicsHead
from predictor.transformer import CausalTransformerPredictor
from tokenizer.encoder import CNNEncoder
from tokenizer.vq import VectorQuantizer


def _load_models(pred_ckpt: str, tok_ckpt: str, device: torch.device) -> tuple:
    """Load predictor (for token_embed) and inverse dynamics head."""
    # Predictor
    ckpt = torch.load(pred_ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]["model"]

    predictor = CausalTransformerPredictor(
        n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
        d_model=cfg["d_model"], d_ffn=cfg["d_ffn"],
        codebook_size=cfg["codebook_size"],
        tokens_per_frame=cfg["tokens_per_frame"],
        action_dim=cfg["action_dim"],
        action_embed_dim=cfg["action_embed_dim"],
        film_layers=cfg["film_layers"],
    ).to(device)
    predictor.load_state_dict(ckpt["predictor"])
    predictor.requires_grad_(False)

    inverse_head = InverseDynamicsHead(
        latent_dim=cfg["d_model"], action_dim=cfg["action_dim"],
    ).to(device)
    inverse_head.load_state_dict(ckpt["inverse_head"])
    inverse_head.requires_grad_(False)

    # Tokenizer
    tok = torch.load(tok_ckpt, map_location=device, weights_only=False)
    tok_cfg = tok["config"]["model"]

    encoder = CNNEncoder(
        channels=tok_cfg["encoder_channels"],
        codebook_dim=tok_cfg["codebook_dim"],
    ).to(device)
    vq = VectorQuantizer(
        n_codes=tok_cfg["codebook_size"],
        code_dim=tok_cfg["codebook_dim"],
        ema_decay=tok_cfg["ema_decay"],
    ).to(device)

    encoder.load_state_dict(tok["encoder"])
    vq.load_state_dict(tok["vq"])
    encoder.requires_grad_(False)
    vq.requires_grad_(False)

    gap = ckpt["config"]["loss"]["inverse_dynamics_gap"]

    return predictor, inverse_head, encoder, vq, gap


@torch.no_grad()
def compute_inverse_accuracy(
    predictor: CausalTransformerPredictor,
    inverse_head: InverseDynamicsHead,
    encoder: CNNEncoder,
    vq: VectorQuantizer,
    dataset: ViZDoomSequenceDataset,
    gap: int = 4,
    n_samples: int = 1000,
    device: str = "cuda",
) -> float:
    """Accuracy of inverse dynamics head on held-out data.

    Args:
        predictor: Trained predictor (used for token_embed)
        inverse_head: Trained inverse dynamics MLP
        encoder: Frozen tokenizer encoder
        vq: Frozen VQ
        dataset: Sequence dataset
        gap: Frame gap for (z_t, z_{t+gap}) pairs
        n_samples: Number of samples
        device: Computation device

    Returns:
        Accuracy as fraction (target > 0.4)
    """
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    correct = 0
    total = 0

    for frames, actions in loader:
        if total >= n_samples:
            break

        frames = frames.to(device)    # (B, T, 3, H, W)
        actions = actions.to(device)  # (B, T, 8)

        B, T = frames.shape[:2]
        if T <= gap:
            continue

        # Tokenize frame t and frame t+gap
        frame_t = frames[:, 0]        # (B, 3, H, W)
        frame_t4 = frames[:, min(gap, T - 1)]

        z_t = encoder(frame_t)
        _, _, idx_t = vq(z_t)
        z_t4 = encoder(frame_t4)
        _, _, idx_t4 = vq(z_t4)

        # Get latent representations via token embedding
        emb_t = predictor.token_embed(idx_t.reshape(B, -1)).mean(dim=1)  # (B, D)
        emb_t4 = predictor.token_embed(idx_t4.reshape(B, -1)).mean(dim=1)

        # Predict action
        logits = inverse_head(emb_t, emb_t4)  # (B, action_dim)
        pred = logits[:, :8].argmax(dim=1)  # keyboard actions only (first 8)

        # Ground truth action at frame t
        gt = actions[:, 0].argmax(dim=1)  # (B,)

        correct += (pred == gt).sum().item()
        total += B

    return correct / total if total > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Inverse dynamics accuracy")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer_checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_samples", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    predictor, inverse_head, encoder, vq, gap = _load_models(
        args.checkpoint, args.tokenizer_checkpoint, device,
    )
    dataset = ViZDoomSequenceDataset(args.data, seq_len=max(9, gap + 2))

    print(f"Computing inverse dynamics accuracy on {args.n_samples} samples (gap={gap})...")
    acc = compute_inverse_accuracy(
        predictor, inverse_head, encoder, vq, dataset,
        gap=gap, n_samples=args.n_samples, device=args.device,
    )
    gate = "PASS" if acc > 0.4 else "FAIL"
    print(f"Inverse dynamics accuracy: {acc:.4f} (target > 0.4) [{gate}]")


if __name__ == "__main__":
    main()
