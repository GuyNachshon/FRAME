"""Action sensitivity measurement.

Measures whether the predictor produces different outputs for different actions.
Hold visual context fixed, vary action between 'forward' and 'back',
compute cosine distance between predicted next-frame token distributions.

Target: > 0.1. If near zero, FiLM conditioning is not working.

Usage:
    uv run python -m eval.action_sensitivity \
        --checkpoint checkpoints/vizdoom/predictor/predictor_best.pt \
        --tokenizer_checkpoint checkpoints/vizdoom/tokenizer/tokenizer_best.pt \
        --data data/vizdoom/raw/vizdoom_data.hdf5
"""

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.vizdoom.dataset import ViZDoomSequenceDataset
from predictor.transformer import CausalTransformerPredictor
from predictor.scene_state import PersistentSceneState
from predictor.gru_state import GRUContinuousState
from tokenizer.encoder import CNNEncoder
from tokenizer.vq import VectorQuantizer


def _load_predictor(ckpt_path: str, device: torch.device) -> tuple:
    """Load predictor and components from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]["model"]

    predictor = CausalTransformerPredictor(
        n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
        d_model=cfg["d_model"], d_ffn=cfg["d_ffn"],
        codebook_size=cfg["codebook_size"],
        tokens_per_frame=cfg["tokens_per_frame"],
        action_dim=cfg["action_dim"],
        action_embed_dim=cfg["action_embed_dim"],
        film_layers=cfg["film_layers"],
        action_dropout=0.0,  # no dropout at inference
    ).to(device)
    predictor.load_state_dict(ckpt["predictor"])
    predictor.requires_grad_(False)

    scene_state = PersistentSceneState(
        dim=cfg["scene_state_dim"], alpha=cfg["scene_state_alpha"],
    ).to(device)
    if "scene_state" in ckpt:
        scene_state.load_state_dict(ckpt["scene_state"])

    gru = GRUContinuousState(
        input_dim=cfg["d_model"], hidden_dim=cfg["gru_dim"],
    ).to(device)
    gru.load_state_dict(ckpt["gru_state"])
    gru.requires_grad_(False)

    return predictor, scene_state, gru, cfg


def _load_tokenizer(ckpt_path: str, device: torch.device) -> tuple:
    """Load frozen tokenizer."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]["model"]

    encoder = CNNEncoder(
        channels=cfg["encoder_channels"], codebook_dim=cfg["codebook_dim"],
    ).to(device)
    vq = VectorQuantizer(
        n_codes=cfg["codebook_size"], code_dim=cfg["codebook_dim"],
        ema_decay=cfg["ema_decay"],
    ).to(device)

    encoder.load_state_dict(ckpt["encoder"])
    vq.load_state_dict(ckpt["vq"])
    encoder.requires_grad_(False)
    vq.requires_grad_(False)
    return encoder, vq


@torch.no_grad()
def compute_action_sensitivity(
    predictor: CausalTransformerPredictor,
    scene_state: PersistentSceneState,
    gru: GRUContinuousState,
    encoder: CNNEncoder,
    vq: VectorQuantizer,
    dataset: ViZDoomSequenceDataset,
    n_samples: int = 100,
    device: str = "cuda",
) -> float:
    """Cosine distance between predictions under forward vs backward actions.

    Args:
        predictor: Trained predictor
        scene_state: Scene state module
        gru: GRU state module
        encoder: Frozen tokenizer encoder
        vq: Frozen VQ
        dataset: Sequence dataset for context frames
        n_samples: Number of samples to measure
        device: Computation device

    Returns:
        Mean cosine distance (target > 0.1)
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    total_dist = 0.0
    n_done = 0

    for frames, _ in loader:
        if n_done >= n_samples:
            break

        frames = frames.to(device)  # (1, T, 3, H, W)
        B, T = frames.shape[:2]
        T_context = T - 1

        # Tokenize context
        flat = frames[:, :T_context].reshape(T_context, 3, 128, 128)
        z = encoder(flat)
        _, _, indices = vq(z)
        context_tokens = indices.reshape(1, T_context, -1)

        scene_tok = scene_state.update(torch.zeros(1, 512, device=device))
        gru_hidden = gru.reset(1).to(device)

        # Forward action
        action_fwd = torch.zeros(1, 72, device=device)
        action_fwd[0, 0] = 1.0  # forward
        logits_fwd, _ = predictor(context_tokens, action_fwd, scene_tok, gru_hidden)

        # Backward action
        action_bwd = torch.zeros(1, 72, device=device)
        action_bwd[0, 1] = 1.0  # back
        logits_bwd, _ = predictor(context_tokens, action_bwd, scene_tok, gru_hidden)

        # Cosine distance on flattened logits
        flat_fwd = logits_fwd.reshape(1, -1)
        flat_bwd = logits_bwd.reshape(1, -1)
        cos_sim = F.cosine_similarity(flat_fwd, flat_bwd, dim=1).item()
        total_dist += 1.0 - cos_sim
        n_done += 1

    return total_dist / n_done


def main() -> None:
    parser = argparse.ArgumentParser(description="Action sensitivity measurement")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer_checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_samples", type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    predictor, scene_state, gru, _ = _load_predictor(args.checkpoint, device)
    encoder, vq = _load_tokenizer(args.tokenizer_checkpoint, device)
    dataset = ViZDoomSequenceDataset(args.data, seq_len=9)

    print(f"Computing action sensitivity on {args.n_samples} samples...")
    score = compute_action_sensitivity(
        predictor, scene_state, gru, encoder, vq, dataset,
        n_samples=args.n_samples, device=args.device,
    )
    gate = "PASS" if score > 0.1 else "FAIL"
    print(f"Action sensitivity: {score:.4f} (target > 0.1) [{gate}]")


if __name__ == "__main__":
    main()
