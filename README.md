# FRAME
### Fast Recurrent Action-Masked Egocentric World Model

> Train on game footage. Hand someone a keyboard. They navigate a world that doesn't exist.

---

## What this is

FRAME is a small-scale interactive neural world model trained on egocentric game footage. At inference time, a user navigates a hallucinated environment in real-time via keyboard — the model generates the next frame conditioned on their action, continuously, at ≥15fps on a single GPU.

No game engine. No physics simulation. Just a neural network's imagination, navigable in real-time.

**Demo target:** ViZDoom (scaffold) → CS:GO dust2 (headline).

---

## Why it's interesting

Most world models either:
- Predict in latent space only — nothing to show
- Use diffusion decoders — too slow for real-time interaction
- Require industrial-scale compute — Matrix-Game 3.0 needs 8 GPUs

FRAME targets a gap: a pixel-generative, keyboard-interactive world model trainable on a single A100, built by a small team, with a working demo in 6 weeks.

The v2 research contribution: the first ternary-weight (`{-1, 0, 1}`) interactive world model predictor. If ternary matches fp32 on this task, the demo runs on a laptop CPU with no GPU.

---

## Three failure modes we explicitly target

| Failure | Cause | FRAME's fix |
|---|---|---|
| **Blur** | L2 reconstruction loss averages possible futures | VQGAN with perceptual + GAN loss |
| **Collapse** | Temporal drift accumulates beyond context window | Two-tiered memory: EMA scene state + GRU |
| **Action dropout** | Model ignores keyboard input | FiLM conditioning + inverse dynamics auxiliary head |

---

## Architecture

```
Frame t (128×128 RGB)
    │
    ▼
┌─────────────────────────────────────┐
│  VQGAN Tokenizer  [Stage 1, frozen] │  → 256 discrete tokens (16×16, codebook 1024)
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Persistent scene state             │  → 512-dim EMA vector (slow, global memory, α=0.95)
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Causal Transformer  [Stage 2]      │  → 8 layers · 8 heads · 512-dim · ~95M params
│  + GRU continuous state             │  → 512-dim fast memory (per-frame, full gradient)
│  + FiLM action conditioning         │  → layers 2,4,6,8 — γ(a)·LayerNorm(x) + β(a)
│  + Inverse dynamics head            │  → predict action from (z_t, z_{t+4})
│  Spatially parallel masked pred.    │  → all 256 next-frame tokens at once (not raster)
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  VQGAN Decoder  [reused, frozen]    │  → 128×128 RGB frame
└─────────────────────────────────────┘
    │
    ▼
Interactive inference loop  →  keyboard · predict · display · ≥15fps
```

**Action space:** 8 keyboard keys (one-hot) + 8×8 mouse bins = 72-dim one-hot vector.

---

## Repository structure

```
frame/
│
├── data/
│   ├── vizdoom/
│   │   ├── collect.py          # ViZDoom data collection (mixed policy + 15% random)
│   │   └── dataset.py          # PyTorch dataset: (frame_t, action_t) sequences
│   └── csgo/
│       ├── loader.py           # TeaPearce HDF5 loader
│       └── dataset.py          # PyTorch dataset: aligned (frame, action) sequences
│
├── tokenizer/
│   ├── encoder.py              # CNN encoder: 128×128 → 16×16 feature map
│   ├── vq.py                   # VQ bottleneck: EMA codebook (1024 codes, 256-dim)
│   ├── decoder.py              # CNN decoder: 16×16 → 128×128 RGB
│   ├── discriminator.py        # PatchGAN discriminator
│   ├── loss.py                 # Perceptual (VGG) + GAN + commitment losses
│   └── train.py                # Stage 1 training loop
│
├── predictor/
│   ├── transformer.py          # Causal transformer (8L, 8H, 512-dim)
│   ├── film.py                 # FiLM conditioning: γ(a), β(a) per layer
│   ├── scene_state.py          # Persistent scene state (EMA update)
│   ├── gru_state.py            # GRU continuous state (fast memory)
│   ├── inverse_dynamics.py     # Auxiliary head: (z_t, z_{t+4}) → action logits
│   ├── sampling.py             # Scheduled sampling scheduler
│   └── train.py                # Stage 2 training loop
│
├── inference/
│   ├── loop.py                 # Real-time inference loop (keyboard → frame)
│   ├── keyboard.py             # Keyboard capture → 72-dim one-hot action
│   ├── display.py              # Frame display (pygame)
│   └── stub.py                 # Random-noise stub for shell profiling
│
├── eval/
│   ├── fid.py                  # FID computation on reconstructions / rollouts
│   ├── lpips.py                # LPIPS on tokenizer reconstructions
│   ├── action_sensitivity.py   # Cosine distance under different actions
│   ├── inverse_acc.py          # Inverse dynamics accuracy on held-out pairs
│   └── rollout.py              # Long-horizon rollout stability (entropy per step)
│
├── configs/
│   ├── vizdoom_tokenizer.yaml
│   ├── vizdoom_predictor.yaml
│   ├── csgo_tokenizer.yaml
│   └── csgo_predictor.yaml
│
├── scripts/
│   ├── train_tokenizer.sh
│   ├── train_predictor.sh
│   ├── run_demo.sh
│   └── runpod_setup.sh
│
├── docs/
│   ├── FRAME_architecture_v1.1.docx   # Full architecture specification
│   └── FRAME_plan.md                  # Project plan and timeline
│
├── requirements.txt
├── Dockerfile
└── README.md                          # this file
```

---

## Quickstart

### 1. Environment

```bash
git clone https://github.com/ozlabs/frame
cd frame
pip install -r requirements.txt
```

```bash
# requirements.txt
torch>=2.2.0
torchvision>=0.17.0
einops>=0.7.0
pygame>=2.5.0
wandb>=0.16.0
h5py>=3.10.0
lpips>=0.1.4
clean-fid>=0.1.35
vizdoom>=1.2.0
accelerate>=0.27.0
```

### 2. Build the inference shell first

Before training anything:

```bash
python inference/loop.py --stub
```

This runs the full keyboard → display pipeline on random token indices. Target: 30fps on stubs. Profile every step. Fix all demo-layer bugs before introducing a real model.

### 3. Collect ViZDoom data

```bash
python data/vizdoom/collect.py \
  --frames 50000 \
  --output data/vizdoom/raw/ \
  --resolution 128 \
  --fps 15 \
  --random_action_prob 0.15
```

### 4. Train tokenizer

```bash
bash scripts/train_tokenizer.sh vizdoom
# Runs ~100k steps on 2×A100. Target: FID < 50, utilization > 80%, LPIPS < 0.15
# Checkpoints saved every 5k steps with metrics logged to wandb
```

**Do not proceed to step 5 until all three tokenizer gates pass.**

### 5. Train predictor

```bash
bash scripts/train_predictor.sh vizdoom
# Runs ~200k steps on 2×A100
# MANDATORY: run eval/action_sensitivity.py and eval/inverse_acc.py at 24h checkpoint
# If action sensitivity < 0.1 or inverse dynamics accuracy near chance: stop and debug
```

### 6. Run the demo

```bash
bash scripts/run_demo.sh --domain vizdoom --checkpoint checkpoints/vizdoom/predictor_best.pt
```

---

## Training gates (mandatory — do not skip)

### Tokenizer
```
FID    < 50    → reconstruction quality
LPIPS  < 0.15  → perceptual sharpness (critical: if above this, fix decoder before continuing)
Util   > 80%   → codebook coverage (below 50%: enable dead code reset)
```

### Predictor — Day 1 check (run at 24h checkpoint)
```
Action sensitivity   > 0.1    → cosine dist between 'forward' vs 'back' predictions
Inverse dyn. acc.    > 40%    → action prediction from (z_t, z_{t+4}) held-out pairs
```

If both are near zero/chance at 24h: **stop training, debug FiLM conditioning.** Common cause: action tokens not reaching FiLM layers, or action dropout rate accidentally set too high.

### Demo
```
Navigation duration  ≥ 30s   → without visual collapse
Action response      visible  → directional keys produce correct motion
Visual quality       sharp    → no pervasive blur or block artifacts
360° consistency     pass     → room looks the same after full turn
```

---

## Datasets

| Domain | Dataset | Size | Action labels | Status |
|---|---|---|---|---|
| ViZDoom | Collected locally via `data/vizdoom/collect.py` | ~10GB | From engine API | Phase 1 |
| CS:GO | [`TeaPearce/CounterStrike_Deathmatch`](https://huggingface.co/datasets/TeaPearce/CounterStrike_Deathmatch) — `dataset_dm_expert_dust2` | 24GB | Pre-aligned HDF5 | Phase 2 |

---

## Compute

| Run | Hardware | Duration | Cost |
|---|---|---|---|
| ViZDoom tokenizer | 2×A100 | ~12h | ~$30 |
| Debugging runs | 2×RTX 4090 | ~3 days | ~$50 |
| ViZDoom predictor | 2×A100 | ~3 days | ~$180 |
| CS:GO tokenizer | 2×A100 | ~6h | ~$15 |
| CS:GO predictor | 2×A100 | ~2 days | ~$120 |
| v1.5 retraining | 2×A100 | ~1 day | ~$60 |
| **Total v1** | | | **~$455** |

> **Rule:** use RTX 4090 ($0.34/hr) for all debugging. Only switch to A100 ($1.19/hr) for full training runs. Hard budget cap: evaluate before spending >$200 on any single run.

---

## Roadmap

### v1 — Interactive ViZDoom + CS:GO demo
- [x] Architecture design
- [x] Project plan
- [ ] Inference shell (random noise → display)
- [ ] ViZDoom data collection
- [ ] VQGAN tokenizer
- [ ] Transformer predictor + FiLM + scene state + GRU + inverse dynamics head
- [ ] ViZDoom interactive demo
- [ ] CS:GO tokenizer
- [ ] CS:GO predictor fine-tune
- [ ] CS:GO interactive demo

### v1.5 — Stronger action conditioning
- [ ] Counterfactual action ranking loss (InfoNCE-style)
- [ ] Self-forcing multi-step rollout training
- [ ] Decoder fine-tune on predicted tokens

### v2 — Research contributions
- [ ] Ternary predictor weights (BitNet b1.58 style) — novel, publishable
- [ ] Uncertainty heatmap overlay (token entropy, real-time)
- [ ] Depth auxiliary head
- [ ] Delta-token context-aware tokenizer (Δ-IRIS style)
- [ ] Paper writeup

### v3 — Future directions
- [ ] Multi-future branching (fork KV-cache, show N parallel futures)
- [ ] Flow-matching decoder
- [ ] Latent action interface (Genie-style, unlabeled video)

---

## Key references

| Paper | Relevance |
|---|---|
| [IRIS (ICLR 2022)](https://arxiv.org/abs/2209.00588) | VQ tokenizer + AR transformer baseline |
| [REM (ICML 2024)](https://arxiv.org/abs/2402.05643) | Parallel observation prediction |
| [Δ-IRIS (ICML 2024)](https://huggingface.co/papers/2406.19320) | Delta token efficiency (v2) |
| [GameNGen (Google 2024)](https://arxiv.org/abs/2408.14837) | ViZDoom interactive WM — our baseline |
| [DIAMOND (NeurIPS 2024)](https://arxiv.org/abs/2405.12399) | CS:GO diffusion WM — fidelity ceiling |
| [DreamerV3 (2023)](https://arxiv.org/abs/2301.04104) | Auxiliary objectives, state design |
| [Matrix-Game 3.0 (2025)](https://matrix-game-v3.github.io/) | Persistent memory at scale |
| [ICM (Pathak et al. 2017)](https://arxiv.org/abs/1705.05363) | Inverse dynamics head grounding |
| [BitNet b1.58](https://arxiv.org/abs/2402.17764) | Ternary weight training (v2) |
| [TeaPearce CS:GO dataset](https://arxiv.org/abs/2104.04258) | CS:GO action-labeled gameplay data |

---

## What success looks like

**Week 6:** a human navigates a hallucinated ViZDoom world for 30 seconds with a keyboard. Running on a single A100. Sharp, coherent, responsive.

**Week 8:** same demo, CS:GO dust2. That's the headline.

**Week 12:** ternary predictor matches fp32. Demo runs on a laptop CPU. That's the paper.

---

*Oz Labs · 2026*