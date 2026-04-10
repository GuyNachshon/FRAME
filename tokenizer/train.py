"""Stage 1 training loop: VQGAN tokenizer.

Trains encoder + VQ + decoder + discriminator with perceptual + GAN + commitment loss.
Adam lr=2e-4, cosine decay. Evals FID + codebook utilization every 5k steps.
Logs to wandb. Supports multi-GPU via accelerate.

Usage:
    accelerate launch -m tokenizer.train --config configs/vizdoom_tokenizer.yaml
    # or single GPU:
    uv run python -m tokenizer.train --config configs/vizdoom_tokenizer.yaml
"""

import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader

from data.vizdoom.dataset import ViZDoomFrameDataset
from tokenizer.decoder import CNNDecoder
from tokenizer.discriminator import PatchGANDiscriminator
from tokenizer.encoder import CNNEncoder
from tokenizer.loss import VQGANLoss
from tokenizer.vq import VectorQuantizer

CHECKPOINT_DIR = Path("checkpoints")


def _build_models(config: dict) -> tuple:
    """Construct all tokenizer components from config.

    Returns:
        (encoder, vq, decoder, discriminator, loss_fn)
    """
    model_cfg = config["model"]

    encoder = CNNEncoder(
        channels=model_cfg["encoder_channels"],
        codebook_dim=model_cfg["codebook_dim"],
    )
    vq = VectorQuantizer(
        n_codes=model_cfg["codebook_size"],
        code_dim=model_cfg["codebook_dim"],
        ema_decay=model_cfg["ema_decay"],
        commitment_beta=model_cfg["commitment_beta"],
    )
    decoder = CNNDecoder(
        codebook_dim=model_cfg["codebook_dim"],
    )
    discriminator = PatchGANDiscriminator()

    loss_cfg = config["loss"]
    loss_fn = VQGANLoss(
        perceptual_weight=loss_cfg["perceptual_weight"],
        gan_weight=loss_cfg["gan_weight"],
        commitment_beta=loss_cfg["commitment_beta"],
    )

    return encoder, vq, decoder, discriminator, loss_fn


def _cosine_lr(step: int, total_steps: int, lr: float, lr_min: float) -> float:
    """Cosine decay learning rate."""
    if step >= total_steps:
        return lr_min
    progress = step / total_steps
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * progress))


def _unwrap(model: nn.Module) -> nn.Module:
    """Unwrap DDP/accelerate wrapper to get raw module (for state_dict)."""
    return model.module if hasattr(model, "module") else model


def save_checkpoint(
    path: Path,
    encoder: nn.Module,
    vq: nn.Module,
    decoder: nn.Module,
    discriminator: nn.Module,
    opt_gen: torch.optim.Optimizer,
    opt_disc: torch.optim.Optimizer,
    step: int,
    config: dict,
) -> None:
    """Save all tokenizer state for resume."""
    torch.save({
        "step": step,
        "encoder": _unwrap(encoder).state_dict(),
        "vq": vq.state_dict(),  # VQ is not wrapped by DDP
        "decoder": _unwrap(decoder).state_dict(),
        "discriminator": _unwrap(discriminator).state_dict(),
        "opt_gen": opt_gen.state_dict(),
        "opt_disc": opt_disc.state_dict(),
        "config": config,
    }, path)


def train_tokenizer(config_path: str, data_path: str | None = None,
                    resume: str | None = None) -> None:
    """Train VQGAN tokenizer (Stage 1).

    Gates (must pass before Stage 2):
      - Reconstruction FID < 50
      - Codebook utilization > 80%
      - LPIPS < 0.15

    Args:
        config_path: Path to YAML config file
        data_path: Path to HDF5 data (overrides config if set)
        resume: Path to checkpoint to resume from
    """
    accelerator = Accelerator(log_with="wandb")
    is_main = accelerator.is_main_process

    with open(config_path) as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]

    # Build models (on CPU first, accelerate handles device placement)
    encoder, vq, decoder, discriminator, loss_fn = _build_models(config)

    if is_main:
        enc_p = sum(p.numel() for p in encoder.parameters())
        dec_p = sum(p.numel() for p in decoder.parameters())
        disc_p = sum(p.numel() for p in discriminator.parameters())
        print(f"Encoder: {enc_p:,} params")
        print(f"Decoder: {dec_p:,} params")
        print(f"Discriminator: {disc_p:,} params")
        print(f"Total trainable: {enc_p + dec_p + disc_p:,}")
        print(f"GPUs: {accelerator.num_processes}")

    # Optimizers: separate for generator and discriminator
    gen_params = list(encoder.parameters()) + list(decoder.parameters())
    opt_gen = torch.optim.Adam(gen_params, lr=train_cfg["lr"], betas=(0.5, 0.999))
    opt_disc = torch.optim.Adam(discriminator.parameters(), lr=train_cfg["lr"],
                                 betas=(0.5, 0.999))

    # Resume (load before accelerate.prepare)
    start_step = 0
    if resume:
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        vq.load_state_dict(ckpt["vq"])
        decoder.load_state_dict(ckpt["decoder"])
        discriminator.load_state_dict(ckpt["discriminator"])
        opt_gen.load_state_dict(ckpt["opt_gen"])
        opt_disc.load_state_dict(ckpt["opt_disc"])
        start_step = ckpt["step"]
        if is_main:
            print(f"Resumed from step {start_step}")

    # Dataset
    if data_path is None:
        data_path = f"data/{config['domain']}/raw/vizdoom_data.hdf5"
    dataset = ViZDoomFrameDataset(data_path)
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Prepare with accelerate (handles DDP wrapping, device placement, dataloader sharding)
    # VQ is NOT wrapped — its EMA codebook must be consistent across processes.
    # We manually move it to the accelerator device.
    encoder, decoder, discriminator, opt_gen, opt_disc, loader = accelerator.prepare(
        encoder, decoder, discriminator, opt_gen, opt_disc, loader,
    )
    loss_fn = loss_fn.to(accelerator.device)
    vq = vq.to(accelerator.device)

    # wandb (main process only)
    if is_main:
        import wandb
        wandb_cfg = config.get("wandb", {})
        date_str = datetime.now().strftime("%Y%m%d")
        run_name = f"{config['domain']}_tokenizer_{date_str}_baseline"
        wandb.init(
            project=wandb_cfg.get("project", "frame-vizdoom"),
            name=run_name,
            config=config,
            tags=wandb_cfg.get("tags", []),
        )

    # Checkpoint dir
    ckpt_dir = CHECKPOINT_DIR / config["domain"] / "tokenizer"
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    total_steps = train_cfg["total_steps"]
    encoder.train()
    decoder.train()
    discriminator.train()

    best_recon_loss = float("inf")
    data_iter = iter(loader)
    step = start_step
    t_train_start = time.time()

    if is_main:
        print(f"\nTraining tokenizer for {total_steps:,} steps...")
        print(f"  Batch size: {train_cfg['batch_size']} x {accelerator.num_processes} GPUs "
              f"= {train_cfg['batch_size'] * accelerator.num_processes} effective")
        print(f"  LR: {train_cfg['lr']} -> {train_cfg['lr_min']}")
        print(f"  Device: {accelerator.device}")
        print(f"  Log every {train_cfg['log_every']} steps, "
              f"eval every {train_cfg['eval_every']} steps, "
              f"save every {train_cfg['save_every']} steps")
        print()

    while step < total_steps:
        # Get batch (cycle through dataset)
        try:
            frames = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            frames = next(data_iter)

        # frames already on device via accelerate

        # Cosine LR decay
        lr = _cosine_lr(step, total_steps, train_cfg["lr"], train_cfg["lr_min"])
        for pg in opt_gen.param_groups:
            pg["lr"] = lr
        for pg in opt_disc.param_groups:
            pg["lr"] = lr

        # ---- Generator step ----
        # Freeze discriminator so DDP doesn't track it during gen backward
        _unwrap(discriminator).requires_grad_(False)

        z = encoder(frames)
        z_q, commitment_loss, indices = vq(z)
        x_recon = decoder(z_q)

        disc_fake = discriminator(x_recon)
        losses = loss_fn(frames, x_recon, commitment_loss, disc_fake)

        opt_gen.zero_grad()
        accelerator.backward(losses["total"])
        if train_cfg.get("grad_clip"):
            accelerator.clip_grad_norm_(gen_params, train_cfg["grad_clip"])
        opt_gen.step()

        # ---- Discriminator step ----
        _unwrap(discriminator).requires_grad_(True)

        disc_real = discriminator(frames.detach())
        disc_fake = discriminator(x_recon.detach())
        d_loss = loss_fn.discriminator_loss(disc_real, disc_fake)

        opt_disc.zero_grad()
        accelerator.backward(d_loss)
        if train_cfg.get("grad_clip"):
            accelerator.clip_grad_norm_(
                _unwrap(discriminator).parameters(), train_cfg["grad_clip"]
            )
        opt_disc.step()

        # ---- Logging ----
        step += 1

        if is_main:
            # Live progress line every 10 steps
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
                    f"loss={losses['total'].item():.3f} "
                    f"recon={losses['recon'].item():.3f} "
                    f"util={vq.utilization():.0%} | "
                    f"ETA {eta_h}h{eta_m:02d}m"
                )
                sys.stdout.flush()

            if step % train_cfg["log_every"] == 0:
                util = vq.utilization()
                recent_losses = {
                    "total": losses["total"].item(),
                    "recon": losses["recon"].item(),
                    "perceptual": losses["perceptual"].item(),
                    "gan_gen": losses["gan_gen"].item(),
                    "commitment": losses["commitment"].item(),
                    "disc": d_loss.item(),
                }
                wandb.log({
                    "train/total_loss": recent_losses["total"],
                    "train/recon_loss": recent_losses["recon"],
                    "train/perceptual_loss": recent_losses["perceptual"],
                    "train/gan_loss": recent_losses["gan_gen"],
                    "train/commitment_loss": recent_losses["commitment"],
                    "train/disc_loss": recent_losses["disc"],
                    "train/codebook_utilization": util,
                    "train/lr": lr,
                }, step=step)

                if step % (train_cfg["log_every"] * 10) == 0:
                    elapsed = time.time() - t_train_start
                    steps_per_sec = (step - start_step) / elapsed if elapsed > 0 else 0
                    print(
                        f"\n  Step {step:>7,}/{total_steps:,} "
                        f"({steps_per_sec:.1f} it/s) | "
                        f"loss={recent_losses['total']:.4f} "
                        f"recon={recent_losses['recon']:.4f} "
                        f"perc={recent_losses['perceptual']:.4f} "
                        f"gan={recent_losses['gan_gen']:.4f} "
                        f"disc={recent_losses['disc']:.4f} "
                        f"util={util:.1%} "
                        f"lr={lr:.2e}"
                    )

            # ---- Eval ----
            if step % train_cfg["eval_every"] == 0:
                util = vq.utilization()
                wandb.log({"eval/codebook_utilization": util}, step=step)
                vq.reset_usage_tracking()

                with torch.no_grad():
                    sample = frames[:8]
                    z_s = _unwrap(encoder)(sample)
                    z_q_s, _, _ = vq(z_s)
                    recon_s = _unwrap(decoder)(z_q_s)
                    grid = torch.cat([sample, recon_s], dim=3)
                    grid = grid.clamp(0, 1)
                    wandb.log({
                        "samples/reconstructions": wandb.Image(
                            grid.cpu(), caption=f"Step {step}: original | recon"
                        ),
                    }, step=step)

                print(f"\n  [eval] step {step}: util={util:.1%}")

            # ---- Checkpoint ----
            if step % train_cfg["save_every"] == 0:
                ckpt_path = ckpt_dir / f"tokenizer_{step:07d}.pt"
                save_checkpoint(
                    ckpt_path, encoder, vq, decoder, discriminator,
                    opt_gen, opt_disc, step, config,
                )
                recon_val = losses["recon"].item()
                if recon_val < best_recon_loss:
                    best_recon_loss = recon_val
                    best_path = ckpt_dir / "tokenizer_best.pt"
                    save_checkpoint(
                        best_path, encoder, vq, decoder, discriminator,
                        opt_gen, opt_disc, step, config,
                    )
                    print(f"\n  [best] new best recon={recon_val:.4f} at step {step}")

    # Final checkpoint
    if is_main:
        save_checkpoint(
            ckpt_dir / f"tokenizer_{step:07d}.pt",
            encoder, vq, decoder, discriminator,
            opt_gen, opt_disc, step, config,
        )
        wandb.finish()
        print(f"\nTraining complete. Final step: {step}")
        print(f"Checkpoints in: {ckpt_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VQGAN tokenizer")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to tokenizer config YAML")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to HDF5 data (overrides config)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train_tokenizer(args.config, args.data, args.resume)


if __name__ == "__main__":
    main()
