# FRAME — Research Notebook
*Oz Labs · 2026 · Living document — update after every training run, experiment, or architectural decision*

---

## How to use this notebook

This is the project's ground truth research log. Every training run, failed experiment, architectural decision, and surprise gets recorded here in chronological order. Wandb links expire, Slack threads get buried, commit messages are terse — this notebook is where the full story lives.

**Rules:**
- Add entries chronologically (newest at bottom of each section)
- Every entry gets a date and author
- Link to wandb runs, commits, and checkpoints — don't duplicate data, point to it
- Record **surprises and failures** as much as successes — the debugging story is the research story
- Convert relative dates to absolute dates (e.g., "yesterday" → "2026-04-10")

---

## Training Runs

Record every training run that completes (or is intentionally stopped). Format:

```
### [DATE] [COMPONENT] — [SHORT DESCRIPTION]

**Run:** wandb link
**Config:** config file used
**Hardware:** GPU type × count, duration, cost
**Checkpoint:** path or HF link

**Results:**
| Metric | Value | Gate |
|---|---|---|
| ... | ... | ... |

**Observations:** what happened, what was surprising, what to try next
```

---

### 2026-04-10 Tokenizer — ViZDoom baseline (v1)

**Run:** https://wandb.ai/guy-na8/frame-vizdoom/runs/kub4wyrl
**Config:** `configs/vizdoom_tokenizer.yaml`
**Hardware:** 2×RTX 4090, 4.7 hours (~$3.20), 5.9 it/s steady-state
**Checkpoint:** `checkpoints/vizdoom/tokenizer/tokenizer_best.pt`

**Results:**
| Metric | Value | Gate | Status |
|---|---|---|---|
| Recon loss | 0.005 | low | Good |
| Codebook utilization | 89.6% | > 80% | **Pass** |
| Perceptual loss | 0.457 | low | Healthy |
| Disc loss | 0.51 | ~0.5-1.0 | Balanced |
| LPIPS | 0.0388 | < 0.15 | **Pass** (4× under target) |
| FID | *not yet implemented* | < 50 | *pending* |

**Observations:**
- Codebook utilization ramped from 3% → 58% → 97% in first 2k steps. Dead code reset working well.
- GAN training was stable throughout — disc loss stayed in 0.5-0.7 range, no mode collapse.
- Reconstruction quality visually good: corridor geometry, weapon, textures all recognizable. Mild block artifacts at texture boundaries (expected at 16×16 token resolution).
- DDP + GAN required keeping discriminator outside accelerate wrapping. Inplace ops in BatchNorm + DDP grad tracking caused version conflicts. Solution: discriminator runs plain `.backward()`, only encoder/decoder are DDP-wrapped.
- Data loading was the initial bottleneck (0.2 it/s with HDF5 lazy loading). Fixed by preloading full dataset into RAM (~2.3GB). Jumped to 4-6 it/s.
- VQ EMA updates needed `@torch.no_grad()` + detached inputs to avoid inplace modification conflicts with DDP autograd graph.

---

## Architectural Decisions

Record decisions that deviate from the original spec, or resolve open questions.

```
### [DATE] [TOPIC] — [DECISION]

**Context:** what prompted the decision
**Options considered:** what alternatives existed
**Decision:** what was chosen and why
**Impact:** what this changes going forward
```

---

### 2026-04-10 Param count — ~30M not ~95M

**Context:** Architecture doc specified ~95M params for the predictor (8L/8H/512d/2048 FFN). Actual implementation is 30.4M.
**Options considered:** (1) increase d_model to 768 (~65M), (2) increase layers to 12 (~45M), (3) proceed with 30M and scale if capacity is insufficient.
**Decision:** Proceed with 30M. The spec dimensions (8L/8H/512d/2048) are correct; the ~95M estimate was wrong. Scale up only if predictor accuracy plateaus below 60%.
**Impact:** Faster training, lower memory, cheaper runs. First knob to turn if underfitting: d_model=768.

### 2026-04-10 FiLM initialization — small random, not zeros

**Context:** Zero-initialized FiLM projection weights made gamma/beta constant regardless of action input, producing zero gradient to action tensor. Caught by verification test.
**Decision:** Initialize FiLM projection weights with `normal_(std=0.02)`, bias gamma=1, bias beta=0. Near-identity at init but gradient flows from step 0.
**Impact:** Critical for action conditioning. Without this fix, the model would train for 24h before the Day 1 check catches that actions have no effect.

### 2026-04-10 Tokenizer encoder/decoder — ResBlocks, not plain convs

**Context:** Plain 4-layer conv encoder/decoder had 1.2M + 2.3M = 3.5M params. Too small for good reconstruction quality.
**Decision:** Added ResBlocks (2 per stage) with GroupNorm + SiLU. Encoder and decoder now 7.2M each.
**Impact:** Better reconstruction quality. Total tokenizer is 17.2M trainable + 7.6M frozen VGG + 2.8M discriminator.

---

## Failed Experiments & Dead Ends

Record things that didn't work — these are as valuable as successes.

```
### [DATE] [WHAT FAILED] — [WHY]

**What was tried:** description
**What happened:** result
**Root cause:** diagnosis
**Lesson:** what to remember
```

---

### 2026-04-10 DDP + GAN inplace ops — 3 failed attempts

**What was tried:** Wrapping discriminator with accelerate DDP alongside encoder/decoder.
**What happened:** `RuntimeError: inplace operation modified variable at version N`. Persisted through 3 fix attempts: (1) `@torch.no_grad()` on VQ EMA, (2) removing all `inplace=True` from activations, (3) `requires_grad_(False)` toggle on discriminator during gen step.
**Root cause:** DDP tracks BatchNorm running mean/var as inplace modifications. When discriminator runs twice per step (once for gen loss, once for disc loss), the version counter increments conflict. The `requires_grad_(False)` trick doesn't prevent BN stat tracking.
**Lesson:** Don't DDP-wrap the discriminator in a GAN. Keep it as a plain model on device. At 2.8M params the parallelization overhead is negligible anyway.

---

## Open Questions (Empirical)

Track questions that need experiments to answer. Move to "Architectural Decisions" once resolved.

### 2026-04-11 Predictor — ViZDoom baseline (v1)

**Run:** https://wandb.ai/guy-na8/frame-vizdoom/runs/shusone3
**Config:** `configs/vizdoom_predictor.yaml`
**Hardware:** 2×RTX 4090, 13.5 hours (~$9.20), 4.1 it/s steady-state
**Checkpoint:** `checkpoints/vizdoom/predictor/predictor_best.pt` (best at step 60k, pred_loss=1.97)
**HF:** `guychuk/frame-vizdoom-predictor`

**Results:**
| Metric | Value | Gate | Status |
|---|---|---|---|
| Prediction loss (best) | 1.97 | low | Good |
| Token accuracy (peak) | ~46% | > 60% | Below target |
| Action sensitivity | 0.2079 | > 0.1 | **Pass** (2× target) |
| Inverse dynamics acc | 51.9% | > 40% | **Pass** |
| Final pred loss | 3.17 | — | Variance from batch size 4 |
| Final token accuracy | 22.9% | — | Variance |

**Observations:**
- All architectural gates pass. FiLM conditioning is working (action sensitivity 2× target).
- Token accuracy peaked ~46% but oscillates widely due to tiny batch size (4 per GPU). The log values at any single step are noisy — wandb curves show the true trend.
- Scheduled sampling reached maximum 0.5 at step 100k and held steady.
- Inverse dynamics accuracy consistently 50%+ at checkpoints.
- **Key finding: demo shows static/near-static scene despite passing all gates.** Root cause: ViZDoom `basic` scenario data has minimal ego-motion (stationary agent in a single room). The model correctly learned "brown walls don't change" because that's the training distribution.
- Architecture is validated. Data is the bottleneck, not the model.

**Next steps:** Skip to CS:GO dataset (rich ego-motion, diverse viewpoints) rather than recollecting ViZDoom data on a different scenario.

---

### 2026-04-12 Decision — Skip to CS:GO

**Context:** ViZDoom predictor passes all architectural gates but demo shows minimal ego-motion response. Root cause: `basic` scenario training data has almost no movement.
**Options considered:** (1) Recollect ViZDoom on `deadly_corridor`/`my_way_home` scenarios, (2) Skip to CS:GO, (3) Debug architecture further.
**Decision:** Skip to CS:GO. Action sensitivity 0.21 confirms the architecture works. CS:GO `TeaPearce/CounterStrike_Deathmatch` dataset has expert gameplay with rich ego-motion — exactly what a world model needs. ViZDoom was always scaffolding.
**Impact:** Skip Week 5 ViZDoom debugging. Go directly to CS:GO tokenizer + predictor. Tokenizer architecture transfers as-is (retrain on CS:GO frames). Predictor trains from scratch on CS:GO sequences.

---

## Open Questions (Empirical)

Track questions that need experiments to answer. Move to "Architectural Decisions" once resolved.

- ~~**Action sensitivity check**~~ — RESOLVED: 0.21, architecture works
- **EMA decay α=0.95** — first hyperparameter to tune if collapse occurs beyond context window
- **Context window 8 vs 16 frames** — does extending to 16 with Flash Attention improve coherence?
- **Codebook size 1024** — may be too small for CS:GO. Try 2048/4096 if CS:GO tokenizer underperforms.
- **Inference temperature** — argmax vs temperature=0.9. Needs interactive testing.
- **GRU ablation** — is the GRU load-bearing or redundant with scene state + long context?
- **Inverse dynamics λ** — sweep {0.05, 0.1, 0.2} on ViZDoom before committing to CS:GO run
- **4-step vs 1-step inverse dynamics** — profile per-frame token change rate on actual data

---

## Compute Ledger

Track all spend against the $455 v1 budget.

| Date | Run | Hardware | Duration | Cost | Running Total |
|---|---|---|---|---|---|
| 2026-04-10 | ViZDoom tokenizer baseline | 2×RTX 4090 | 4.7h | ~$3.20 | $3.20 |
| 2026-04-11 | ViZDoom predictor baseline | 2×RTX 4090 | 13.5h | ~$9.20 | $12.40 |
| | | | | **Budget remaining** | **~$443** |

---

*Update this notebook after every training run, architectural decision, or surprise. A research project without a notebook is just a collection of checkpoints.*
