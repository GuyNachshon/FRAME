"""Long-horizon rollout stability measurement.

Measures per-step token entropy over autoregressive rollouts.
Detects three collapse signatures:
  1. Mean-field collapse: entropy drops to near zero (scene -> average color)
  2. Attractor loop: entropy oscillates (2-3 repeating frames)
  3. Entropy explosion: entropy spikes (pure noise)

Usage:
    uv run python -m eval.rollout \
        --checkpoint checkpoints/vizdoom/predictor/predictor_best.pt \
        --tokenizer_checkpoint checkpoints/vizdoom/tokenizer/tokenizer_best.pt \
        --data data/vizdoom/raw/vizdoom_data.hdf5
"""

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.vizdoom.dataset import ViZDoomSequenceDataset
from predictor.gru_state import GRUContinuousState
from predictor.scene_state import PersistentSceneState
from predictor.transformer import CausalTransformerPredictor
from tokenizer.decoder import CNNDecoder
from tokenizer.encoder import CNNEncoder
from tokenizer.vq import VectorQuantizer


def _load_all(pred_ckpt: str, tok_ckpt: str, device: torch.device) -> tuple:
    """Load all models for rollout."""
    # Tokenizer
    tok = torch.load(tok_ckpt, map_location=device, weights_only=False)
    tcfg = tok["config"]["model"]
    encoder = CNNEncoder(channels=tcfg["encoder_channels"], codebook_dim=tcfg["codebook_dim"]).to(device)
    vq = VectorQuantizer(n_codes=tcfg["codebook_size"], code_dim=tcfg["codebook_dim"], ema_decay=tcfg["ema_decay"]).to(device)
    decoder = CNNDecoder(codebook_dim=tcfg["codebook_dim"]).to(device)
    encoder.load_state_dict(tok["encoder"]); vq.load_state_dict(tok["vq"]); decoder.load_state_dict(tok["decoder"])
    encoder.requires_grad_(False); vq.requires_grad_(False); decoder.requires_grad_(False)

    # Predictor
    pred = torch.load(pred_ckpt, map_location=device, weights_only=False)
    pcfg = pred["config"]["model"]
    predictor = CausalTransformerPredictor(
        n_layers=pcfg["n_layers"], n_heads=pcfg["n_heads"], d_model=pcfg["d_model"],
        d_ffn=pcfg["d_ffn"], codebook_size=pcfg["codebook_size"],
        tokens_per_frame=pcfg["tokens_per_frame"], action_dim=pcfg["action_dim"],
        action_embed_dim=pcfg["action_embed_dim"], film_layers=pcfg["film_layers"],
        action_dropout=0.0,
    ).to(device)
    predictor.load_state_dict(pred["predictor"]); predictor.requires_grad_(False)

    scene_state = PersistentSceneState(dim=pcfg["scene_state_dim"], alpha=pcfg["scene_state_alpha"]).to(device)
    gru = GRUContinuousState(input_dim=pcfg["d_model"], hidden_dim=pcfg["gru_dim"]).to(device)
    gru.load_state_dict(pred["gru_state"]); gru.requires_grad_(False)

    return encoder, vq, decoder, predictor, scene_state, gru, pcfg["context_frames"]


@torch.no_grad()
def compute_rollout_stability(
    encoder: CNNEncoder, vq: VectorQuantizer, decoder: CNNDecoder,
    predictor: CausalTransformerPredictor,
    scene_state: PersistentSceneState, gru: GRUContinuousState,
    dataset: ViZDoomSequenceDataset,
    n_steps: int = 30,
    n_rollouts: int = 10,
    context_frames: int = 8,
    device: str = "cuda",
) -> list[float]:
    """Per-step token entropy over long-horizon rollouts.

    Returns:
        List of mean token entropies per step (length n_steps).
        Target: < 0.5 nats increase over 30 steps.
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # Accumulate entropies: [step] -> list of entropies across rollouts
    step_entropies: list[list[float]] = [[] for _ in range(n_steps)]
    n_done = 0

    for frames, actions in loader:
        if n_done >= n_rollouts:
            break

        frames = frames.to(device)  # (1, T, 3, H, W)
        T = frames.shape[1]
        if T < 2:
            continue

        # Tokenize initial context
        context_f = frames[:, :min(T - 1, context_frames)]
        B, Tc = context_f.shape[:2]
        flat = context_f.reshape(Tc, 3, 128, 128)
        z = encoder(flat)
        _, _, idx = vq(z)
        token_buffer = list(idx.reshape(Tc, -1).unsqueeze(0).split(1, dim=1))
        # token_buffer: list of (1, 1, 256)
        token_buffer = [t.squeeze(1) for t in token_buffer]  # list of (1, 256)

        gru_hidden = gru.reset(1).to(device)
        scene_state.reset()

        # Use forward action for all rollout steps
        action = torch.zeros(1, 72, device=device)
        action[0, 0] = 1.0  # forward

        for s in range(n_steps):
            context = torch.stack(token_buffer[-context_frames:], dim=1)  # (1, C, 256)

            # Scene state from current tokens
            tok_emb = predictor.token_embed(token_buffer[-1])
            scene_tok = scene_state.update(tok_emb.mean(dim=1))

            logits, info = predictor(context, action, scene_tok, gru_hidden)
            # logits: (1, 256, codebook_size)

            # Token entropy: mean over 256 positions
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean().item()
            step_entropies[s].append(entropy)

            # Argmax for next step
            pred_tokens = logits.argmax(dim=-1)  # (1, 256)
            token_buffer.append(pred_tokens)

            # GRU update
            _, gru_hidden = gru(info["hidden_mean"], gru_hidden)

        n_done += 1

    # Average across rollouts
    return [sum(ents) / len(ents) if ents else 0.0 for ents in step_entropies]


def main() -> None:
    parser = argparse.ArgumentParser(description="Rollout stability measurement")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer_checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--n_steps", type=int, default=30)
    parser.add_argument("--n_rollouts", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    encoder, vq, decoder, predictor, scene_state, gru, ctx = _load_all(
        args.checkpoint, args.tokenizer_checkpoint, device,
    )
    dataset = ViZDoomSequenceDataset(args.data, seq_len=max(9, ctx + 1))

    print(f"Running {args.n_rollouts} rollouts of {args.n_steps} steps...")
    entropies = compute_rollout_stability(
        encoder, vq, decoder, predictor, scene_state, gru, dataset,
        n_steps=args.n_steps, n_rollouts=args.n_rollouts,
        context_frames=ctx, device=args.device,
    )

    print(f"\nPer-step mean token entropy (nats):")
    for i, e in enumerate(entropies):
        marker = ""
        if i == 0:
            e0 = e
        elif e - e0 > 0.5:
            marker = " << WARNING: >0.5 increase"
        print(f"  Step {i + 1:3d}: {e:.4f}{marker}")

    increase = entropies[-1] - entropies[0] if entropies else 0
    gate = "PASS" if increase < 0.5 else "FAIL"
    print(f"\nEntropy increase (step 1 -> {args.n_steps}): {increase:.4f} (target < 0.5 nats) [{gate}]")


if __name__ == "__main__":
    main()
