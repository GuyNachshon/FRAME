"""Stage 2 training loop: transformer predictor.

AdamW lr=1e-4, warmup 2k, cosine decay, batch 4, grad clip 1.0.
Includes scheduled sampling, inverse dynamics loss (lambda=0.1),
FiLM conditioning, and all auxiliary components.

Requires a trained, frozen tokenizer checkpoint. Supports multi-GPU via accelerate.

Usage:
    accelerate launch -m predictor.train \
        --config configs/vizdoom_predictor.yaml \
        --tokenizer_checkpoint checkpoints/vizdoom/tokenizer/tokenizer_best.pt
"""

import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
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
    _, _, indices = vq(z)
    indices = indices.reshape(B, T, -1)
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


def _unwrap(model: nn.Module) -> nn.Module:
    """Unwrap DDP/accelerate wrapper to get raw module."""
    return model.module if hasattr(model, "module") else model


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
        "predictor": _unwrap(predictor).state_dict(),
        "scene_state": scene_state.state_dict(),
        "gru_state": _unwrap(gru_state).state_dict(),
        "inverse_head": _unwrap(inverse_head).state_dict(),
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
    """
    accelerator = Accelerator(log_with="wandb")
    is_main = accelerator.is_main_process

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    train_cfg = config["training"]
    loss_cfg = config["loss"]
    ss_cfg = config["scheduled_sampling"]

    # Load frozen tokenizer (on accelerator device)
    assert tokenizer_checkpoint is not None, (
        "Provide --tokenizer_checkpoint with a trained tokenizer"
    )
    tok_encoder, tok_vq, tok_decoder = _load_frozen_tokenizer(
        tokenizer_checkpoint, accelerator.device,
    )
    if is_main:
        print(f"Loaded frozen tokenizer from {tokenizer_checkpoint}")

    # Build predictor components (CPU, accelerate handles placement)
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
    )

    scene_state = PersistentSceneState(
        dim=model_cfg["scene_state_dim"],
        alpha=model_cfg["scene_state_alpha"],
    ).to(accelerator.device)

    gru = GRUContinuousState(
        input_dim=model_cfg["d_model"],
        hidden_dim=model_cfg["gru_dim"],
    )

    inverse_head = InverseDynamicsHead(
        latent_dim=model_cfg["d_model"],
        action_dim=model_cfg["action_dim"],
    )

    ss_scheduler = ScheduledSamplingScheduler(
        max_p=ss_cfg["max_p"],
        ramp_steps=ss_cfg["ramp_steps"],
    )

    if is_main:
        pred_p = sum(p.numel() for p in predictor.parameters())
        gru_p = sum(p.numel() for p in gru.parameters())
        inv_p = sum(p.numel() for p in inverse_head.parameters())
        total_p = pred_p + gru_p + inv_p
        print(f"Predictor: {pred_p:,} params")
        print(f"GRU: {gru_p:,} params")
        print(f"Inverse dynamics: {inv_p:,} params")
        print(f"Total trainable: {total_p:,} ({total_p / 1e6:.1f}M)")
        print(f"GPUs: {accelerator.num_processes}")

    # Optimizer
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
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        predictor.load_state_dict(ckpt["predictor"])
        gru.load_state_dict(ckpt["gru_state"])
        inverse_head.load_state_dict(ckpt["inverse_head"])
        scene_state.load_state_dict(ckpt["scene_state"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        best_pred_loss = ckpt.get("best_pred_loss", float("inf"))
        if is_main:
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

    # Prepare with accelerate
    # Scene state is NOT wrapped (EMA buffer, not DDP-compatible)
    predictor, gru, inverse_head, optimizer, loader = accelerator.prepare(
        predictor, gru, inverse_head, optimizer, loader,
    )

    # wandb (main process only)
    if is_main:
        import wandb
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
    if is_main:
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
    t_train_start = time.time()

    if is_main:
        print(f"\nTraining predictor for {total_steps:,} steps...")
        print(f"  Batch size: {train_cfg['batch_size']} x {accelerator.num_processes} GPUs "
              f"= {train_cfg['batch_size'] * accelerator.num_processes} effective")
        print(f"  Context frames: {model_cfg['context_frames']}")
        print(f"  Inv. dynamics gap: {inv_gap}, lambda: {inv_lambda}")
        print(f"  Device: {accelerator.device}")
        print(f"  Log every {train_cfg['log_every']} steps, "
              f"save every {train_cfg['save_every']} steps")
        print()

    while step < total_steps:
        try:
            frames, actions_seq = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            frames, actions_seq = next(data_iter)

        # frames already on device via accelerate

        B, T_total = frames.shape[:2]
        T_context = T_total - 1

        # Tokenize (frozen, on device)
        all_tokens = _tokenize_frames(tok_encoder, tok_vq, frames)
        context_tokens = all_tokens[:, :T_context, :]
        target_tokens = all_tokens[:, -1, :]

        action_8 = actions_seq[:, T_context - 1, :]
        action_72 = torch.zeros(B, 72, device=accelerator.device)
        action_72[:, :8] = action_8

        # LR schedule
        lr = _cosine_lr_warmup(
            step, total_steps, train_cfg["lr"],
            train_cfg["lr_min"], train_cfg["warmup_steps"],
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Scene state
        with torch.no_grad():
            ctx_flat = context_tokens.reshape(B, -1).float()
            scene_input = torch.zeros(B, model_cfg["d_model"], device=accelerator.device)
            scene_input[:, :min(ctx_flat.shape[1], model_cfg["d_model"])] = \
                ctx_flat[:, :model_cfg["d_model"]]
        scene_tok = scene_state.update(scene_input)

        gru_hidden = _unwrap(gru).reset(B).to(accelerator.device)

        # Forward
        logits, info = predictor(
            context_tokens, action_72, scene_tok, gru_hidden,
        )

        pred_loss = F.cross_entropy(
            logits.reshape(-1, _unwrap(predictor).codebook_size),
            target_tokens.reshape(-1),
        )

        inv_loss = torch.tensor(0.0, device=accelerator.device)
        if T_total > inv_gap:
            z_t = _unwrap(predictor).token_embed(all_tokens[:, 0, :]).mean(dim=1)
            z_t4 = _unwrap(predictor).token_embed(
                all_tokens[:, min(inv_gap, T_total - 1), :]
            ).mean(dim=1)
            inv_logits = inverse_head(z_t, z_t4)
            inv_target_8 = actions_seq[:, 0, :]
            inv_target = inv_target_8.argmax(dim=1)
            inv_loss = F.cross_entropy(inv_logits, inv_target)

        total_loss = pred_loss + inv_lambda * inv_loss

        gru_out, gru_hidden = gru(info["hidden_mean"].detach(), gru_hidden)

        optimizer.zero_grad()
        accelerator.backward(total_loss)
        accelerator.clip_grad_norm_(all_params, train_cfg["grad_clip"])
        optimizer.step()

        step += 1

        if is_main:
            # Live progress
            if step % 10 == 0:
                elapsed = time.time() - t_train_start
                steps_done = step - start_step
                steps_per_sec = steps_done / elapsed if elapsed > 0 else 0
                eta_sec = (total_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
                eta_m, eta_s = divmod(int(eta_sec), 60)
                eta_h, eta_m = divmod(eta_m, 60)
                pct = step / total_steps * 100
                sys.stdout.write(
                    f"\r  [{pct:5.1f}%] step {step:>7,}/{total_steps:,} | "
                    f"{steps_per_sec:.1f} it/s | "
                    f"loss={total_loss.item():.3f} "
                    f"pred={pred_loss.item():.3f} "
                    f"inv={inv_loss.item():.3f} | "
                    f"ETA {eta_h}h{eta_m:02d}m"
                )
                sys.stdout.flush()

            if step % train_cfg["log_every"] == 0:
                with torch.no_grad():
                    pred_indices = logits.argmax(dim=-1)
                    accuracy = (pred_indices == target_tokens).float().mean().item()

                ss_p = ss_scheduler.get_p(step)
                wandb.log({
                    "train/total_loss": total_loss.item(),
                    "train/prediction_loss": pred_loss.item(),
                    "train/inverse_dynamics_loss": inv_loss.item(),
                    "train/token_accuracy": accuracy,
                    "train/scheduled_sampling_p": ss_p,
                    "train/lr": lr,
                }, step=step)

                if step % (train_cfg["log_every"] * 10) == 0:
                    elapsed = time.time() - t_train_start
                    steps_per_sec = (step - start_step) / elapsed if elapsed > 0 else 0
                    print(
                        f"\n  Step {step:>7,}/{total_steps:,} "
                        f"({steps_per_sec:.1f} it/s) | "
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
                    inv_acc = 0.0
                    if T_total > inv_gap:
                        inv_pred = inv_logits.argmax(dim=-1)
                        inv_acc = (inv_pred == inv_target).float().mean().item()

                wandb.log({
                    "checkpoint/token_accuracy": accuracy,
                    "checkpoint/inverse_dynamics_acc": inv_acc,
                }, step=step)
                print(f"\n  [checkpoint] step {step}: acc={accuracy:.3f}, inv_acc={inv_acc:.3f}")

                ckpt_path = ckpt_dir / f"predictor_{step:07d}.pt"
                save_checkpoint(
                    ckpt_path, predictor, scene_state, gru, inverse_head,
                    optimizer, step, config, best_pred_loss,
                )
                if pred_loss.item() < best_pred_loss:
                    best_pred_loss = pred_loss.item()
                    best_path = ckpt_dir / "predictor_best.pt"
                    save_checkpoint(
                        best_path, predictor, scene_state, gru, inverse_head,
                        optimizer, step, config, best_pred_loss,
                    )
                    print(f"  [best] new best pred_loss={best_pred_loss:.4f} at step {step}")

    if is_main:
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
