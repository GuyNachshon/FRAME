---
license: mit
tags:
  - world-model
  - vqgan
  - tokenizer
  - vizdoom
  - frame
datasets:
  - guychuk/frame-vizdoom-data
library_name: pytorch
pipeline_tag: image-to-image
---

# FRAME ViZDoom Tokenizer

VQGAN tokenizer for the [FRAME](https://github.com/GuyNachshon/FRAME) project — a real-time interactive neural world model for FPS games.

## Model Description

This tokenizer converts 128×128 RGB game frames into 256 discrete tokens (16×16 spatial grid, 1024-code vocabulary) and back. It's Stage 1 of FRAME's two-stage architecture: the tokenizer converts pixels to tokens, then a causal transformer predicts next-frame tokens conditioned on player actions.

| Component | Architecture | Params |
|---|---|---|
| Encoder | CNN with ResBlocks, 3 downsample stages | 7.2M |
| VQ Bottleneck | 1024 codes × 256-dim, EMA updates | — |
| Decoder | CNN with ResBlocks, 3 upsample stages | 7.2M |
| Discriminator | PatchGAN, 3 layers | 2.8M |
| **Total trainable** | | **17.2M** |

## Training

- **Data:** 50,000 frames from ViZDoom `basic` scenario, 128×128 RGB, 15fps
- **Loss:** Perceptual (VGG-16) + PatchGAN hinge + VQ commitment (β=0.25)
- **Optimizer:** Adam, lr=2e-4, cosine decay to 1e-5
- **Steps:** 100,000
- **Hardware:** 2×RTX 4090, ~4.7 hours
- **wandb:** [training run](https://wandb.ai/guy-na8/frame-vizdoom/runs/kub4wyrl)

## Evaluation

| Metric | Value | Target | Status |
|---|---|---|---|
| Reconstruction loss | 0.005 | low | Pass |
| LPIPS | 0.0388 | < 0.15 | **Pass** (4× under) |
| Codebook utilization | 89.6% | > 80% | **Pass** |
| Perceptual loss | 0.457 | low | Healthy |

## Usage

```python
import torch
from tokenizer.encoder import CNNEncoder
from tokenizer.vq import VectorQuantizer
from tokenizer.decoder import CNNDecoder

# Load checkpoint
ckpt = torch.load("tokenizer_best.pt", map_location="cuda")
cfg = ckpt["config"]["model"]

encoder = CNNEncoder(channels=cfg["encoder_channels"], codebook_dim=cfg["codebook_dim"]).cuda()
vq = VectorQuantizer(n_codes=cfg["codebook_size"], code_dim=cfg["codebook_dim"]).cuda()
decoder = CNNDecoder(codebook_dim=cfg["codebook_dim"]).cuda()

encoder.load_state_dict(ckpt["encoder"])
vq.load_state_dict(ckpt["vq"])
decoder.load_state_dict(ckpt["decoder"])

# Encode frame to tokens
frame = torch.rand(1, 3, 128, 128).cuda()  # [0, 1] RGB
z = encoder(frame)
z_q, _, indices = vq(z)  # indices: (1, 16, 16), values in [0, 1024)

# Decode tokens back to frame
reconstructed = decoder(z_q)  # (1, 3, 128, 128)
```

## Files

- `tokenizer_best.pt` — best checkpoint by reconstruction loss (recommended)
- `tokenizer_0100000.pt` — final checkpoint at 100k steps

## Citation

Part of the FRAME project by [Oz Labs](https://github.com/GuyNachshon/FRAME).

```
FRAME: Fast Recurrent Action-Masked Egocentric World Model
Oz Labs, 2026
```
