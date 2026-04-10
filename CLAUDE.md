# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FRAME (Fast Recurrent Action-Masked Egocentric World Model) is a real-time interactive neural world model trained on egocentric game footage. Users navigate hallucinated environments via keyboard at ≥15fps on a single GPU. No game engine — pure neural network imagination.

all relevant information for project is under @docs. follow @docs/BEST_PRACTICES.md when creating files/working with gh etc..

**Demo targets:** ViZDoom (scaffold, v1) → CS:GO dust2 (headline, v1) → ternary weights on laptop CPU (paper, v2).

## Commands

```bash
# Run (UV for all Python execution)
uv run python main.py

# Run inference shell (stub mode for profiling)
uv run python -m inference.loop --stub --profile --headless

# Data collection
uv run python data/vizdoom/collect.py --frames 50000 --output data/vizdoom/raw/ --resolution 128 --fps 15 --random_action_prob 0.15

# Training (via shell scripts)
bash scripts/train_tokenizer.sh vizdoom
bash scripts/train_predictor.sh vizdoom

# Demo
bash scripts/run_demo.sh --domain vizdoom --checkpoint checkpoints/vizdoom/predictor_best.pt

# Evaluation
uv run python eval/fid.py
uv run python eval/lpips.py
uv run python eval/action_sensitivity.py
uv run python eval/inverse_acc.py
```

## Architecture (Two-Stage Pipeline)

For deepdive: @docs/architecture

**Stage 1 — VQGAN Tokenizer** (~17M trainable params, train first, then freeze):
- CNN encoder (ResBlocks + downsampling): 128×128 RGB → 16×16 feature map → VQ bottleneck (1024 codes, 256-dim, EMA) → CNN decoder
- PatchGAN discriminator (2.8M params)
- Loss: perceptual (VGG-16, 7.6M frozen) + PatchGAN hinge + commitment (β=0.25)
- Located in `tokenizer/`

**Stage 2 — Causal Transformer Predictor** (~30M params total):
- Transformer: 28.5M — 8 layers, 8 heads, 512-dim, FFN 2048, spatially parallel masked prediction (all 256 next-frame tokens at once — not raster-order AR)
- GRU fast memory: 1.6M — 512-dim, per-frame update, full gradient
- Inverse dynamics head: 0.3M — MLP [1024→256→72], predicts action from (z_t, z_{t+4})
- FiLM action conditioning on layers 2,4,6,8 — γ(a)·LayerNorm(x) + β(a)
- EMA scene state (α=0.95, slow/global, no gradient)
- Scheduled sampling: linear ramp 0→0.5 over 100k steps
- Located in `predictor/`
- If capacity is insufficient, first knob: d_model=768 (~65M) or n_layers=12 (~45M)

**Inference loop** (`inference/`): keyboard → action encoding (72-dim one-hot: 8 keys + 8×8 mouse bins) → predict → VQGAN decode → pygame display

## Three Failure Modes (Everything in the Architecture Exists to Prevent One of These)

| Failure | Component that fixes it |
|---|---|
| Blur | VQGAN with perceptual + GAN loss (not L2) |
| Temporal collapse | EMA scene state + GRU two-tiered memory |
| Action dropout (model ignores input) | FiLM conditioning + inverse dynamics head |

## Training Gates (Mandatory — Do Not Skip)

**Tokenizer:** FID < 50, LPIPS < 0.15, codebook utilization > 80%
**Predictor (24h checkpoint):** action sensitivity > 0.1, inverse dynamics accuracy > 40%. If both near zero/chance: stop training, debug FiLM conditioning.
**Demo:** ≥30s navigation without collapse, visible action response, no blur, 360° consistency.

## Compute Rules

- Use RTX 4090 ($0.34/hr) for all debugging. Only A100 ($1.19/hr) for full training runs.
- Hard budget cap: evaluate before spending >$200 on any single run.

## Datasets

- **ViZDoom** (Phase 1): collected locally via `data/vizdoom/collect.py`, ~10GB, 128×128 RGB, 15fps
- **CS:GO** (Phase 2): `TeaPearce/CounterStrike_Deathmatch` on HuggingFace, `dataset_dm_expert_dust2` subset, 24GB HDF5, pre-aligned action labels
