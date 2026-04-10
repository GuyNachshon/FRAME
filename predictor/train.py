"""Stage 2 training loop: transformer predictor.

AdamW lr=1e-4, warmup 2k, cosine decay, batch 4, grad clip 1.0.
Includes scheduled sampling, inverse dynamics loss (lambda=0.1),
FiLM conditioning, and all auxiliary components.

Requires a trained, frozen tokenizer checkpoint.

Usage:
    uv run python -m predictor.train \
        --config configs/vizdoom_predictor.yaml \
        --tokenizer_checkpoint checkpoints/vizdoom/tokenizer/tokenizer_best.pt
"""

import argparse
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from torch.utils.data import DataLoader

from data.vizdoom.dataset import ViZDoomSequenceDataset
from predictor.gru_state import GRUContinuousState
from predictor.inverse_dynamics import InverseDynamicsHead
from predictor.sampling import ScheduledSamplingScheduler
from predictor.scene_state import PersistentSceneState
from predictor.transformer import CausalTransformerPredictor
from tokenizer.decoder import CNNDecoder
from tokenizer.encoder import CNNEncoder
from tokenizer.vq import VectorQuantizer

CHECKPOINT_DIR = Path("checkpoints")


def _load_frozen_tokenizer(
    ckpt_path: str, device: torch.device,
) -> tuple[CNNEncoder, VectorQuantizer, CNNDecoder]:
    """Load a trained tokenizer and freeze all weights."""
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

    # Freeze
    for model in (encoder, vq, decoder):
        for p in model.parameters():
            p.requires_grad = False
        model.requires_grad_(False)

    return encoder, vq, decoder


@torch.no_grad()
def _tokenize_frames(
    encoder: CNNEncoder, vq: VectorQuantizer, frames: torch.Tensor,
) -> torch.LongTensor:
    """Tokenize a batch of frame sequences.

    Args:
        encoder: Frozen encoder
        vq: Frozen VQ
        frames: (B, T, 3, H, W) float32

    Returns:
        indices: (B, T, 256) token indices
    """
    B, T, C, H, W = frames.shape
    flat = frames.reshape(B * T, C, H, W)
    z = encoder(flat)
    _, _, indices = vq(z)  # (B*T, 16, 16)
    indices = indices.reshape(B, T, -1)  # (B, T, 256)
    return indices


def _cosine_lr_warmup(
    step: int, total_steps: int, lr: float, lr_min: float,
    warmup_steps: int,
) -> float:
    """Cosine decay with linear warmup."""
    if step < warmup_steps:
        return lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * progress))


def save_checkpoint(
    path: Path,
    predictor: nn.Module,
    scene_state: nn.Module,
    gru_state: nn.Module,
    inverse_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict,
    best_pred_loss: float,
) -> None:
    """Save all predictor state for resume."""
    torch.save({
        "step": step,
        "predictor": predictor.state_dict(),
        "scene_state": scene_state.state_dict(),
        "gru_state": gru_state.state_dict(),
        "inverse_head": inverse_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "best_pred_loss": best_pred_loss,
    }, path)


def train_predictor(config_path: str, tokenizer_checkpoint: str | None = None,
                    data_path: str | None = None,
                    resume: str | None = None) -> None:
    """Train transformer predictor (Stage 2).

    Requires a trained, frozen tokenizer checkpoint.

    Day 1 check (mandatory at 24h):
      - Action sensitivity > 0.1
      - Inverse dynamics accuracy > 40%
      If both near zero/chance: stop and debug FiLM.

    Args:
        config_path: Path to YAML config file
        tokenizer_checkpoint: Path to frozen tokenizer checkpoint
        data_path: Path to HDF5 data (overrides config)
        resume: Path to checkpoint to resume from
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = config["model"]
    train_cfg = config["training"]
    loss_cfg = config["loss"]
    ss_cfg = config["scheduled_sampling"]

    # Load frozen tokenizer
    assert tokenizer_checkpoint is not None, (
        "Provide --tokenizer_checkpoint with a trained tokenizer"
    )
    tok_encoder, tok_vq, tok_decoder = _load_frozen_tokenizer(
        tokenizer_checkpoint, device,
    )
    print(f"Loaded frozen tokenizer from {tokenizer_checkpoint}")

    # Build predictor components
    predictor = CausalTransformerPredictor(
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        d_model=model_cfg["d_model"],
        d_ffn=model_cfg["d_ffn"],
        codebook_size=model_cfg["codebook_size"],
        tokens_per_frame=model_cfg["tokens_per_frame"],
        action_dim=model_cfg["action_dim"],
        action_embed_dim=model_cfg["action_embed_dim"],
        film_layers=model_cfg["film_layers"],
        action_dropout=model_cfg["action_dropout"],
    ).to(device)

    scene_state = PersistentSceneState(
        dim=model_cfg["scene_state_dim"],
        alpha=model_cfg["scene_state_alpha"],
    ).to(device)

    gru = GRUContinuousState(
        input_dim=model_cfg["d_model"],
        hidden_dim=model_cfg["gru_dim"],
    ).to(device)

    inverse_head = InverseDynamicsHead(
        latent_dim=model_cfg["d_model"],
        action_dim=model_cfg["action_dim"],
    ).to(device)

    ss_scheduler = ScheduledSamplingScheduler(
        max_p=ss_cfg["max_p"],
        ramp_steps=ss_cfg["ramp_steps"],
    )

    # Log param counts
    pred_p = sum(p.numel() for p in predictor.parameters())
    gru_p = sum(p.numel() for p in gru.parameters())
    inv_p = sum(p.numel() for p in inverse_head.parameters())
    total_p = pred_p + gru_p + inv_p
    print(f"Predictor: {pred_p:,} params")
    print(f"GRU: {gru_p:,} params")
    print(f"Inverse dynamics: {inv_p:,} params")
    print(f"Total trainable: {total_p:,} ({total_p / 1e6:.1f}M)")

    # Optimizer: all trainable predictor components
    all_params = (
        list(predictor.parameters())
        + list(gru.parameters())
        + list(inverse_head.parameters())
    )
    optimizer = torch.optim.AdamW(
        all_params, lr=train_cfg["lr"], weight_decay=0.01,
    )

    # Resume
    start_step = 0
    best_pred_loss = float("inf")
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        predictor.load_state_dict(ckpt["predictor"])
        scene_state.load_state_dict(ckpt["scene_state"])
        gru.load_state_dict(ckpt["gru_state"])
        inverse_head.load_state_dict(ckpt["inverse_head"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        best_pred_loss = ckpt.get("best_pred_loss", float("inf"))
        print(f"Resumed from step {start_step}")

    # Dataset
    data_cfg = config["data"]
    if data_path is None:
        data_path = f"data/{config['domain']}/raw/vizdoom_data.hdf5"
    dataset = ViZDoomSequenceDataset(data_path, seq_len=data_cfg["seq_len"])
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # wandb
    wandb_cfg = config.get("wandb", {})
    date_str = datetime.now().strftime("%Y%m%d")
    run_name = f"{config['domain']}_predictor_{date_str}_baseline"
    wandb.init(
        project=wandb_cfg.get("project", "frame-vizdoom"),
        name=run_name,
        config=config,
        tags=wandb_cfg.get("tags", []),
    )

    # Checkpoint dir
    ckpt_dir = CHECKPOINT_DIR / config["domain"] / "predictor"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    total_steps = train_cfg["total_steps"]
    inv_gap = loss_cfg["inverse_dynamics_gap"]
    inv_lambda = loss_cfg["inverse_dynamics_lambda"]

    predictor.train()
    gru.train()
    inverse_head.train()

    data_iter = iter(loader)
    step = start_step

    print(f"\nTraining predictor for {total_steps} steps...")
    print(f"  Batch size: {train_cfg['batch_size']}")
    print(f"  Context frames: {model_cfg['context_frames']}")
    print(f"  Inv. dynamics gap: {inv_gap}, lambda: {inv_lambda}")
    print(f"  Device: {device}\n")

    while step < total_steps:
        # Get batch
        try:
            frames, actions_seq = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            frames, actions_seq = next(data_iter)

        frames = frames.to(device)        # (B, seq_len, 3, H, W)
        actions_seq = actions_seq.to(device)  # (B, seq_len, 8)

        B, T_total = frames.shape[:2]
        T_context = T_total - 1  # last frame is the target

        # Tokenize all frames (frozen tokenizer)
        all_tokens = _tokenize_frames(tok_encoder, tok_vq, frames)
        # (B, T_total, 256)

        context_tokens = all_tokens[:, :T_context, :]  # (B, T_context, 256)
        target_tokens = all_tokens[:, -1, :]            # (B, 256)

        # Action for current step: last context action
        # Extend actions to 72-dim (8 keyboard -> 72 with zeros for mouse)
        action_8 = actions_seq[:, T_context - 1, :]  # (B, 8)
        action_72 = torch.zeros(B, 72, device=device)
        action_72[:, :8] = action_8

        # LR schedule
        lr = _cosine_lr_warmup(
            step, total_steps, train_cfg["lr"],
            train_cfg["lr_min"], train_cfg["warmup_steps"],
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Scene state update (from mean of context token embeddings)
        with torch.no_grad():
            ctx_flat = context_tokens.reshape(B, -1).float()
            scene_input = torch.zeros(B, model_cfg["d_model"], device=device)
            scene_input[:, :min(ctx_flat.shape[1], model_cfg["d_model"])] = \
                ctx_flat[:, :model_cfg["d_model"]]
        scene_tok = scene_state.update(scene_input)

        # GRU state (initialized per sequence)
        gru_hidden = gru.reset(B).to(device)

        # Forward
        logits, info = predictor(
            context_tokens, action_72, scene_tok, gru_hidden,
        )
        # logits: (B, 256, codebook_size)

        # Main loss: cross-entropy over token predictions
        pred_loss = F.cross_entropy(
            logits.reshape(-1, predictor.codebook_size),
            target_tokens.reshape(-1),
        )

        # Inverse dynamics loss (if sequence is long enough)
        inv_loss = torch.tensor(0.0, device=device)
        if T_total > inv_gap:
            # Use token embeddings at frames t and t+gap as latent reps
            z_t = predictor.token_embed(all_tokens[:, 0, :]).mean(dim=1)
            z_t4 = predictor.token_embed(
                all_tokens[:, min(inv_gap, T_total - 1), :]
            ).mean(dim=1)
            inv_logits = inverse_head(z_t, z_t4)

            # Target action: action at frame t
            inv_target_8 = actions_seq[:, 0, :]  # (B, 8)
            inv_target = inv_target_8.argmax(dim=1)  # (B,)
            inv_loss = F.cross_entropy(inv_logits, inv_target)

        # Total loss
        total_loss = pred_loss + inv_lambda * inv_loss

        # GRU update
        gru_out, gru_hidden = gru(info["hidden_mean"].detach(), gru_hidden)

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(all_params, train_cfg["grad_clip"])
        optimizer.step()

        step += 1

        # Logging
        if step % train_cfg["log_every"] == 0:
            # Token accuracy
            with torch.no_grad():
                pred_indices = logits.argmax(dim=-1)
                accuracy = (pred_indices == target_tokens).float().mean().item()

            ss_p = ss_scheduler.get_p(step)
            log_dict = {
                "train/total_loss": total_loss.item(),
                "train/prediction_loss": pred_loss.item(),
                "train/inverse_dynamics_loss": inv_loss.item(),
                "train/token_accuracy": accuracy,
                "train/scheduled_sampling_p": ss_p,
                "train/lr": lr,
            }
            wandb.log(log_dict, step=step)

            if step % (train_cfg["log_every"] * 10) == 0:
                print(
                    f"Step {step:7d}/{total_steps} | "
                    f"loss={total_loss.item():.4f} "
                    f"pred={pred_loss.item():.4f} "
                    f"inv={inv_loss.item():.4f} "
                    f"acc={accuracy:.3f} "
                    f"ss_p={ss_p:.3f} "
                    f"lr={lr:.2e}"
                )

        # Checkpoint
        if step % train_cfg["save_every"] == 0:
            with torch.no_grad():
                pred_indices = logits.argmax(dim=-1)
                accuracy = (pred_indices == target_tokens).float().mean().item()
                if T_total > inv_gap:
                    inv_pred = inv_logits.argmax(dim=-1)
                    inv_acc = (inv_pred == inv_target).float().mean().item()
                else:
                    inv_acc = 0.0

            wandb.log({
                "checkpoint/token_accuracy": accuracy,
                "checkpoint/inverse_dynamics_acc": inv_acc,
            }, step=step)
            print(f"  [checkpoint] step {step}: acc={accuracy:.3f}, inv_acc={inv_acc:.3f}")

            ckpt_path = ckpt_dir / f"predictor_{step:07d}.pt"
            save_checkpoint(
                ckpt_path, predictor, scene_state, gru, inverse_head,
                optimizer, step, config, best_pred_loss,
            )
            # Save best by prediction loss
            if pred_loss.item() < best_pred_loss:
                best_pred_loss = pred_loss.item()
                best_path = ckpt_dir / "predictor_best.pt"
                save_checkpoint(
                    best_path, predictor, scene_state, gru, inverse_head,
                    optimizer, step, config, best_pred_loss,
                )
                print(f"  [best] new best pred_loss={best_pred_loss:.4f} at step {step}")

    # Final checkpoint
    save_checkpoint(
        ckpt_dir / f"predictor_{step:07d}.pt",
        predictor, scene_state, gru, inverse_head,
        optimizer, step, config, best_pred_loss,
    )

    wandb.finish()
    print(f"\nTraining complete. Final step: {step}")
    print(f"Checkpoints in: {ckpt_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train transformer predictor")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to predictor config YAML")
    parser.add_argument("--tokenizer_checkpoint", type=str, required=True,
                        help="Path to frozen tokenizer checkpoint")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to HDF5 data (overrides config)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train_predictor(args.config, args.tokenizer_checkpoint, args.data,
                    args.resume)


if __name__ == "__main__":
    main()
