"""Stage 1 training loop: VQGAN tokenizer.

Trains encoder + VQ + decoder + discriminator with perceptual + GAN + commitment loss.
Adam lr=2e-4, cosine decay. Evals FID + codebook utilization every 5k steps.
Logs to wandb.

Usage:
    uv run python -m tokenizer.train --config configs/vizdoom_tokenizer.yaml
"""

import argparse
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import wandb
import yaml
from torch.utils.data import DataLoader

from data.vizdoom.dataset import ViZDoomFrameDataset
from tokenizer.decoder import CNNDecoder
from tokenizer.discriminator import PatchGANDiscriminator
from tokenizer.encoder import CNNEncoder
from tokenizer.loss import VQGANLoss
from tokenizer.vq import VectorQuantizer

CHECKPOINT_DIR = Path("checkpoints")


def _build_models(config: dict, device: torch.device) -> tuple:
    """Construct all tokenizer components from config.

    Returns:
        (encoder, vq, decoder, discriminator, loss_fn)
    """
    model_cfg = config["model"]

    encoder = CNNEncoder(
        channels=model_cfg["encoder_channels"],
        codebook_dim=model_cfg["codebook_dim"],
    ).to(device)

    vq = VectorQuantizer(
        n_codes=model_cfg["codebook_size"],
        code_dim=model_cfg["codebook_dim"],
        ema_decay=model_cfg["ema_decay"],
        commitment_beta=model_cfg["commitment_beta"],
    ).to(device)

    decoder = CNNDecoder(
        codebook_dim=model_cfg["codebook_dim"],
    ).to(device)

    discriminator = PatchGANDiscriminator().to(device)

    loss_cfg = config["loss"]
    loss_fn = VQGANLoss(
        perceptual_weight=loss_cfg["perceptual_weight"],
        gan_weight=loss_cfg["gan_weight"],
        commitment_beta=loss_cfg["commitment_beta"],
    ).to(device)

    return encoder, vq, decoder, discriminator, loss_fn


def _cosine_lr(step: int, total_steps: int, lr: float, lr_min: float) -> float:
    """Cosine decay learning rate."""
    if step >= total_steps:
        return lr_min
    progress = step / total_steps
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * progress))


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
        "encoder": encoder.state_dict(),
        "vq": vq.state_dict(),
        "decoder": decoder.state_dict(),
        "discriminator": discriminator.state_dict(),
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
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cfg = config["training"]

    # Build models
    encoder, vq, decoder, discriminator, loss_fn = _build_models(config, device)

    # Log param counts
    enc_p = sum(p.numel() for p in encoder.parameters())
    dec_p = sum(p.numel() for p in decoder.parameters())
    disc_p = sum(p.numel() for p in discriminator.parameters())
    print(f"Encoder: {enc_p:,} params")
    print(f"Decoder: {dec_p:,} params")
    print(f"Discriminator: {disc_p:,} params")
    print(f"Total trainable: {enc_p + dec_p + disc_p:,}")

    # Optimizers: separate for generator and discriminator
    gen_params = list(encoder.parameters()) + list(decoder.parameters())
    opt_gen = torch.optim.Adam(gen_params, lr=train_cfg["lr"], betas=(0.5, 0.999))
    opt_disc = torch.optim.Adam(discriminator.parameters(), lr=train_cfg["lr"],
                                 betas=(0.5, 0.999))

    # Resume
    start_step = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        vq.load_state_dict(ckpt["vq"])
        decoder.load_state_dict(ckpt["decoder"])
        discriminator.load_state_dict(ckpt["discriminator"])
        opt_gen.load_state_dict(ckpt["opt_gen"])
        opt_disc.load_state_dict(ckpt["opt_disc"])
        start_step = ckpt["step"]
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

    # wandb
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
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    total_steps = train_cfg["total_steps"]
    encoder.train()
    decoder.train()
    discriminator.train()

    best_recon_loss = float("inf")
    data_iter = iter(loader)
    step = start_step

    print(f"\nTraining tokenizer for {total_steps} steps...")
    print(f"  Batch size: {train_cfg['batch_size']}")
    print(f"  LR: {train_cfg['lr']} -> {train_cfg['lr_min']}")
    print(f"  Device: {device}\n")

    while step < total_steps:
        # Get batch (cycle through dataset)
        try:
            frames = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            frames = next(data_iter)

        frames = frames.to(device)  # (B, 3, 128, 128)

        # Cosine LR decay
        lr = _cosine_lr(step, total_steps, train_cfg["lr"], train_cfg["lr_min"])
        for pg in opt_gen.param_groups:
            pg["lr"] = lr
        for pg in opt_disc.param_groups:
            pg["lr"] = lr

        # ---- Generator step ----
        z = encoder(frames)
        z_q, commitment_loss, indices = vq(z)
        x_recon = decoder(z_q)

        disc_fake = discriminator(x_recon)
        losses = loss_fn(frames, x_recon, commitment_loss, disc_fake)

        opt_gen.zero_grad()
        losses["total"].backward()
        if train_cfg.get("grad_clip"):
            nn.utils.clip_grad_norm_(gen_params, train_cfg["grad_clip"])
        opt_gen.step()

        # ---- Discriminator step ----
        disc_real = discriminator(frames)
        disc_fake = discriminator(x_recon.detach())
        d_loss = loss_fn.discriminator_loss(disc_real, disc_fake)

        opt_disc.zero_grad()
        d_loss.backward()
        if train_cfg.get("grad_clip"):
            nn.utils.clip_grad_norm_(
                discriminator.parameters(), train_cfg["grad_clip"]
            )
        opt_disc.step()

        # ---- Logging ----
        step += 1

        if step % train_cfg["log_every"] == 0:
            util = vq.utilization()
            log_dict = {
                "train/total_loss": losses["total"].item(),
                "train/recon_loss": losses["recon"].item(),
                "train/perceptual_loss": losses["perceptual"].item(),
                "train/gan_loss": losses["gan_gen"].item(),
                "train/commitment_loss": losses["commitment"].item(),
                "train/disc_loss": d_loss.item(),
                "train/codebook_utilization": util,
                "train/lr": lr,
            }
            wandb.log(log_dict, step=step)

            if step % (train_cfg["log_every"] * 10) == 0:
                print(
                    f"Step {step:7d}/{total_steps} | "
                    f"loss={losses['total'].item():.4f} "
                    f"recon={losses['recon'].item():.4f} "
                    f"perc={losses['perceptual'].item():.4f} "
                    f"gan={losses['gan_gen'].item():.4f} "
                    f"disc={d_loss.item():.4f} "
                    f"util={util:.1%} "
                    f"lr={lr:.2e}"
                )

        # ---- Eval ----
        if step % train_cfg["eval_every"] == 0:
            util = vq.utilization()
            wandb.log({
                "eval/codebook_utilization": util,
            }, step=step)
            vq.reset_usage_tracking()

            # Log sample reconstructions
            with torch.no_grad():
                sample = frames[:8]
                z_s = encoder(sample)
                z_q_s, _, _ = vq(z_s)
                recon_s = decoder(z_q_s)
                # Concat original and recon side by side
                grid = torch.cat([sample, recon_s], dim=3)  # (8, 3, 128, 256)
                grid = grid.clamp(0, 1)
                wandb.log({
                    "samples/reconstructions": wandb.Image(
                        grid.cpu(), caption=f"Step {step}: original | recon"
                    ),
                }, step=step)

            print(f"  [eval] step {step}: util={util:.1%}")

        # ---- Checkpoint ----
        if step % train_cfg["save_every"] == 0:
            ckpt_path = ckpt_dir / f"tokenizer_{step:07d}.pt"
            save_checkpoint(
                ckpt_path, encoder, vq, decoder, discriminator,
                opt_gen, opt_disc, step, config,
            )
            # Save best by reconstruction loss
            recon_val = losses["recon"].item()
            if recon_val < best_recon_loss:
                best_recon_loss = recon_val
                best_path = ckpt_dir / "tokenizer_best.pt"
                save_checkpoint(
                    best_path, encoder, vq, decoder, discriminator,
                    opt_gen, opt_disc, step, config,
                )
                print(f"  [best] new best recon={recon_val:.4f} at step {step}")

    # Final checkpoint
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
