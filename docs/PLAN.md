# FRAME — Project Plan
**Fast Recurrent Action-Masked Egocentric World Model**
*Oz Labs · April 2026 · Working Draft*

---

## What we're building

A small-scale interactive game world model. Train on egocentric game footage. At inference: hand someone a keyboard, they navigate a hallucinated environment in real-time. The model generates the next frame conditioned on their action, continuously, at ≥15fps on a single GPU.

**The demo moment:** a user presses forward and the hallucinated world moves. They turn left and it turns. They haven't loaded a game — they're navigating a neural network's imagination.

**Three things that would make it a failure:**
- Blurry output
- World collapses after a few seconds
- Model ignores keyboard input

Everything in the architecture exists to prevent one of those three.

---

## What we're not building (v1)

- A diffusion world model (too slow for real-time)
- A latent-only model (nothing to show)
- A ternary model (v2 research contribution)
- An uncertainty heatmap (v2 demo feature)
- A CS:GO world model (v2 target — ViZDoom is the scaffold)

---

## The research contribution

**v1 claim:** minimal sufficient architecture for real-time interactive FPS world modeling at small-team scale. We identify which components are load-bearing for the three failure modes and which are eliminable.

**v2 claim:** first ternary-weight interactive world model. Does `{-1, 0, 1}` prediction match fp32 on this task? Either finding is publishable.

---

## Dataset

### Phase 1 — ViZDoom (scaffold)

**Why ViZDoom:**
- Python API, deterministic engine, discrete action space
- Existing world model literature to compare against (GameNGen)
- Fast to collect, fast to iterate
- Forgiving visual domain — find architectural failure modes here, not on CS:GO

**Data collection:**
- Run ViZDoom with mixed policy: 85% scripted/random agent, 15% pure random actions
- Record `(frame_t, action_t)` pairs at 15fps
- Target: ~50k frames for tokenizer, ~200k frame-sequences for predictor
- Resolution: 128×128 RGB
- Action space: `{forward, back, left, right, turn-left, turn-right, shoot, noop}` — 8 keys one-hot

**Action diversity augmentation:** 15% of training sequences have randomly injected actions replacing scripted actions. Forces the model to learn what happens under unusual action sequences — users will immediately do things no scripted agent ever did.

### Phase 2 — CS:GO (target)

**Dataset:** [`TeaPearce/CounterStrike_Deathmatch`](https://huggingface.co/datasets/TeaPearce/CounterStrike_Deathmatch) on HuggingFace

**Subset to use:** `dataset_dm_expert_dust2` — 190 HDF5 files, 24GB, expert deathmatch on dust2, pre-aligned action labels.

**Data format (already labeled):**
```
frame_i_x          → RGB screenshot
frame_i_y          → [keys_pressed_onehot, Lclick, Rclick, mouse_x_onehot, mouse_y_onehot]
frame_i_helperarr  → [kill_flag, death_flag]
```

**Why this dataset:**
- Action labels are pre-aligned with frames — no optical flow inference needed
- Expert gameplay → clean, coherent trajectories
- 24GB is manageable (not the 700GB full scrape)
- Has been used for world modeling in prior work (DIAMOND NeurIPS 2024) — baseline to compare against

**Why CS:GO is hard:**
- Fast camera motion, dynamic lighting, other players, HUD elements
- Action→visual coupling is noisy (pressing W in a firefight has weak signal)
- Nobody has shipped a working CS:GO world model — that's the novelty, but also the risk

**Mouse handling:** discretize into 8×8 = 64 bins (one-hot). Total action vocab: 72 tokens (8 keyboard + 64 mouse).

---

## Architecture summary

Five components, two training stages.

```
[Frame t, 128×128 RGB]
        ↓
[Stage 1] VQGAN Tokenizer          → 256 discrete tokens (16×16 grid, codebook 1024)
        ↓ (freeze)
[Stage 2] Persistent scene state   → 512-dim EMA vector (slow, global memory)
        ↓
[Stage 2] Causal transformer       → 8 layers, 512-dim, ~30M params
          + GRU continuous state   → 512-dim fast memory (per-frame update)
          + FiLM action cond.      → on layers 2,4,6,8 — unavoidable action signal
          + Inverse dynamics head  → predict action from (z_t, z_{t+4}) — anti-action-dropout
        ↓
[Stage 1] VQGAN Decoder            → 128×128 RGB frame
        ↓
[Inference] Keyboard → action → predict → display at ≥15fps
```

**Key decisions and why:**
| Decision | Choice | Why |
|---|---|---|
| Tokenizer | VQGAN (perceptual + GAN loss) | L2 alone = blur. Non-negotiable. |
| Prediction | Spatially parallel masked (all 256 tokens at once) | 15× faster than raster-order AR |
| Action conditioning | FiLM on alternating layers | Can't be routed around by attention |
| Temporal memory | EMA scene state + GRU | Two timescales — global and per-frame |
| Anti-action-dropout | Inverse dynamics auxiliary head | Forces latent to encode controllable dynamics |
| Train/inference gap | Scheduled sampling (0→0.5 over 100k steps) | Model learns to recover from its own errors |
| Pixel output | VQGAN decoder (reused from stage 1) | Fast, sharp, no extra training needed |

---

## Timeline

### Week 1 — Infrastructure
- [ ] Set up RunPod environment (persistent volume, Docker image, wandb)
- [ ] Build inference shell: keyboard → random token indices → VQGAN decode → display at 30fps on stubs
- [ ] Profile every step of the inference loop (encode / predict / decode latencies)
- [ ] Write ViZDoom data collection script (mixed policy, 15fps, HDF5 output)
- [ ] Collect ~50k ViZDoom frames for tokenizer training

**Gate:** inference loop running at 30fps on random noise before any model is trained.

---

### Week 2 — Tokenizer
- [ ] Implement VQGAN: CNN encoder → VQ bottleneck (1024 codes, 256-dim, EMA) → CNN decoder
- [ ] Loss: perceptual (VGG-16) + PatchGAN + commitment (β=0.25)
- [ ] Train on ViZDoom frames (~100k steps, 2×A100, ~$30)
- [ ] Monitor: FID every 5k steps, codebook utilization every 5k steps
- [ ] Eval: reconstruction FID < 50, utilization > 80%, LPIPS < 0.15

**Gate:** pass all three tokenizer metrics before touching predictor code.

---

### Week 3 — Predictor (build + first run)
- [ ] Collect ~200k frame-sequences for predictor training (with action diversity augmentation)
- [ ] Implement causal transformer (8L, 8H, 512-dim, spatially parallel masked prediction)
- [ ] Implement persistent scene state (EMA, α=0.95)
- [ ] Implement GRU continuous state (512-dim, jointly trained)
- [ ] Implement FiLM conditioning (layers 2,4,6,8 — γ(a), β(a) projections)
- [ ] Implement inverse dynamics head (MLP on z_t, z_{t+4} → 72-class action prediction)
- [ ] Implement scheduled sampling (linear ramp 0→0.5 over 100k steps)
- [ ] First training run: 24 hours on 2×A100

**Day 1 check (mandatory — run at 24h checkpoint):**
- Action sensitivity: cosine distance between 'forward' vs 'back' predictions from identical context > 0.1
- Inverse dynamics accuracy: > 40% on held-out 4-step pairs
- If both near zero/chance: stop, debug FiLM before continuing

---

### Week 4 — Predictor (full training + demo)
- [ ] Full predictor training run: ~200k steps, 2×A100, ~$180
- [ ] Monitor: prediction accuracy, rollout FID, token entropy per step
- [ ] Slot trained predictor into inference shell
- [ ] Run ViZDoom interactive demo
- [ ] Observe which failure mode appears first — that is the next problem

**Demo gate:**
- Human navigates ≥30 seconds without collapse
- Scene responds to directional keys
- No pervasive blur
- 360-degree consistency: turn around and the room looks the same

---

### Week 5 — Fix failure modes + CS:GO prep
- [ ] Debug whichever failure mode appeared in Week 4 demo
- [ ] Download CS:GO expert dust2 subset (24GB)
- [ ] Write CS:GO HDF5 data loader (frame + action alignment)
- [ ] Train new VQGAN tokenizer on CS:GO frames (different visual domain)
- [ ] Eval CS:GO tokenizer: same gates as ViZDoom

---

### Week 6 — CS:GO predictor
- [ ] Fine-tune ViZDoom predictor on CS:GO sequences (or train from scratch if domain gap is too large)
- [ ] Run CS:GO interactive demo
- [ ] Compare: which failure modes are new vs. same as ViZDoom?

**This is the headline demo.** A working CS:GO world model at small scale is the research story.

---

### Week 7–8 — Polish + v1.5
- [ ] Decoder fine-tune on predicted tokens (closes train/inference distribution gap)
- [ ] Implement counterfactual action ranking loss (InfoNCE-style)
- [ ] Retrain predictor with contrastive loss
- [ ] Measure: does action sensitivity improve? Does collapse horizon extend?
- [ ] Record demo video for CS:GO and ViZDoom
- [ ] Write up findings

---

### v2 (post-launch, ~4 weeks)
- [ ] Implement BitNet ternary predictor (train from scratch, not quantize)
- [ ] Run identical eval protocol as fp32 baseline
- [ ] Implement uncertainty heatmap overlay (token entropy → 16×16 → upsample → overlay)
- [ ] Profile: can the ternary model run on a laptop CPU at interactive speeds?
- [ ] Write paper

---

## Compute budget

| Stage | Hardware | Duration | Cost |
|---|---|---|---|
| Tokenizer (ViZDoom) | 2×A100 | ~12h | ~$30 |
| Predictor debugging runs | 2×RTX 4090 | ~3 days | ~$50 |
| Predictor full training | 2×A100 | ~3 days | ~$180 |
| CS:GO tokenizer | 2×A100 | ~6h | ~$15 |
| CS:GO predictor fine-tune | 2×A100 | ~2 days | ~$120 |
| v1.5 retraining | 2×A100 | ~1 day | ~$60 |
| **Total v1** | | | **~$455** |
| v2 ternary ablation | 1×A100 | ~1 day | ~$30 |
| **Total v1+v2** | | | **~$485** |

> Use RTX 4090 ($0.34/hr) for all debugging and iteration runs. Only switch to A100 ($1.19/hr) for full training runs. 3.5× cost difference — never debug on an A100.

---

## Evaluation protocol

### Tokenizer
| Metric | Target | Action if failing |
|---|---|---|
| Reconstruction FID | < 50 | Train longer / increase perceptual loss weight |
| Codebook utilization | > 80% | Enable dead code reset trick |
| LPIPS | < 0.15 | Fix discriminator loss / increase GAN weight |

### Predictor
| Metric | Target | Action if failing |
|---|---|---|
| Next-frame token accuracy | > 60% | Check data pipeline, increase training time |
| Action sensitivity (cosine dist) | > 0.1 | Debug FiLM conditioning, check action tokens are reaching layers |
| Inverse dynamics accuracy | > 40% | Check 4-step gap, verify gradients flow through encoder |
| 4-step rollout FID | < 80 | Increase scheduled sampling rate |
| Token entropy @ step 30 | < 0.5 nat increase | Tune EMA α, increase context window |

### Demo
| Metric | Target |
|---|---|
| Navigation duration | ≥ 30 seconds without collapse |
| Action responsiveness | Visible, correct response to directional keys |
| Visual quality | No pervasive blur or block artifacts |
| 360-degree consistency | Room looks the same after full turn |

---

## Key risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Blur from tokenizer | Medium | Perceptual + GAN loss mandatory. Gate on LPIPS < 0.15 before proceeding. |
| Collapse at 10–15s | High | EMA scene state + GRU two-tiered memory. Tune α first. |
| Model ignores actions | High | FiLM on alternating layers + inverse dynamics head. Day 1 check at 24h. |
| Train/inference gap | High | Scheduled sampling from step 0. Self-forcing in v1.5. |
| CS:GO too hard | Medium | ViZDoom scaffold validates architecture first. CS:GO may require longer training or larger codebook. |
| Codebook collapse | Medium | Monitor utilization throughout. Dead code reset trick if <50% active. |
| RunPod cost overrun | Low | Use 4090s for debugging. Hard budget cap: stop and evaluate before spending >$200 on any single run. |

---

## Open questions (empirical — not design discussions)

1. **EMA decay α** — α=0.95 is a guess. First hyperparameter to tune if collapse occurs beyond context window.
2. **4-step vs 1-step inverse dynamics** — profile token change rate on actual game data. If >50% tokens change at 1-step, 1-step is sufficient.
3. **Context window** — 8 frames vs 16 frames. Does extending to 16 (with Flash Attention) improve coherence beyond what scene state provides?
4. **Codebook size** — 1024 codes may be too small for CS:GO. 2048/4096 as alternatives.
5. **Inference temperature** — argmax=deterministic but frozen-feeling. temperature=0.9 adds variety but risks instability.
6. **GRU ablation** — train a version without GRU and compare collapse horizons. Is it load-bearing?
7. **λ_inv weight** — sweep {0.05, 0.1, 0.2} on ViZDoom before committing to CS:GO run.

---

## Literature anchors

| Paper | What it gives us |
|---|---|
| IRIS (ICLR 2022) | VQ tokenizer + AR transformer baseline |
| REM (ICML 2024) | Parallel observation prediction (our spatially parallel masked prediction) |
| Δ-IRIS (ICML 2024) | Delta token efficiency (v2 direction) |
| GameNGen (Google 2024) | ViZDoom interactive WM baseline to compare against |
| DIAMOND (NeurIPS 2024) | CS:GO diffusion WM — the fidelity ceiling we're chasing at lower compute |
| DreamerV3 (2023) | Auxiliary objectives, EMA state design |
| Matrix-Game 3.0 (2025) | Persistent memory design — we implement a lighter EMA variant |
| ICM (Pathak et al. 2017) | Inverse dynamics head — conceptual grounding |
| BitNet b1.58 | Ternary weight training (v2) |

---

## What success looks like

**6 weeks from now:** a human navigates a hallucinated ViZDoom world for 30 seconds with a keyboard. The model is running on a single A100. The world is sharp, coherent, and responsive.

**8 weeks from now:** same demo, CS:GO dust2. That's the headline.

**12 weeks from now:** ternary predictor matches fp32 on ViZDoom. The demo runs on a laptop. That's the paper.