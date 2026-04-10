**FRAME**

Fast Recurrent Action-Masked Egocentric World Model

Architecture Specification - v1.1 (updated post-consortium review)

Oz Labs / Guy Leitersdorf

April 2026 - Working Draft

**What changed in v1.1:**

- **Added:** Inverse dynamics auxiliary head (Section 4.6) - unanimous consortium addition after ChatGPT/literature cross-check
- **Added:** Action diversity augmentation to training protocol (Section 5.2)
- **Added:** Counterfactual action ranking loss to v1.5 scope (Section 2)
- **Added:** Section 13 - Future directions and research roadmap (v1.5, v2, v3)
- **Confirmed:** Delta tokens deferred to v2; residual sharpness head rejected; separate codebooks rejected

# **1\. Goal**

FRAME is a **small-scale interactive game world model** - a neural network trained on egocentric game footage that, at inference time, lets a user navigate a hallucinated environment in real-time via keyboard. The model generates the next frame conditioned on the user's action, continuously, at ≥15fps on a single A100 GPU.

**Primary constraint:** the demo must feel interactive. A user hands a keyboard to someone, they fly through a hallucinated world, it responds to them.

**Three explicit failure modes** that define success/failure:

- **Blur:** output frames must be visually sharp. Perceptual quality, not just low MSE.
- **Collapse:** the hallucinated world must remain coherent for ≥30 seconds. Temporal drift is the enemy.
- **Action dropout:** the model must visibly and correctly respond to every keyboard input.

# **2\. Scope - v1 / v1.5 / v2**

| **Component**                                      | Status      |
| -------------------------------------------------- | ----------- |
| VQGAN tokenizer                                    | v1          |
| ---                                                | ---         |
| Persistent scene state (slow EMA memory)           | v1          |
| ---                                                | ---         |
| GRU continuous state (fast frame-to-frame memory)  | v1          |
| ---                                                | ---         |
| Causal transformer predictor (~100M params, fp32)  | v1          |
| ---                                                | ---         |
| FiLM action conditioning on alternating layers     | v1          |
| ---                                                | ---         |
| Action dropout training (p=0.15)                   | v1          |
| ---                                                | ---         |
| Action diversity augmentation (15% random actions) | v1 ← new    |
| ---                                                | ---         |
| Inverse dynamics auxiliary head                    | v1 ← new    |
| ---                                                | ---         |
| Scheduled sampling (train/inference gap fix)       | v1          |
| ---                                                | ---         |
| VQGAN decoder (reused from tokenizer)              | v1          |
| ---                                                | ---         |
| Interactive inference loop (keyboard → frame)      | v1          |
| ---                                                | ---         |
| Counterfactual action ranking loss (InfoNCE-style) | v1.5        |
| ---                                                | ---         |
| Decoder fine-tune on predicted tokens              | v1.5        |
| ---                                                | ---         |
| Self-forcing multi-step rollout training           | v1.5        |
| ---                                                | ---         |
| Delta token context-aware tokenizer (Δ-IRIS style) | v2          |
| ---                                                | ---         |
| Ternary predictor weights (BitNet variant)         | v2 ablation |
| ---                                                | ---         |
| Depth auxiliary head                               | v2          |
| ---                                                | ---         |
| Uncertainty heatmap overlay (token entropy)        | v2          |
| ---                                                | ---         |
| CS:GO domain transfer                              | v2          |
| ---                                                | ---         |
| Multi-future branching demo                        | v3          |
| ---                                                | ---         |
| Flow-matching decoder (faster than diffusion)      | v3          |
| ---                                                | ---         |
| Latent action interface (Genie-style)              | v3          |
| ---                                                | ---         |

# **3\. Data**

## **3.1 Training domain**

**Phase 1 (scaffold):** ViZDoom. First-person 3D shooter, discrete action space, deterministic engine, existing WM literature (GameNGen). Clean ground truth, fast iteration.

**Phase 2 (target):** CS:GO deathmatch. TeaPearce/CounterStrike_Deathmatch on HuggingFace - 190 HDF5 files, 24GB expert dust2 data, pre-aligned action labels. Action format: \[keys_pressed_onehot, Lclick, Rclick, mouse_x_onehot, mouse_y_onehot\].

| **WHY** | ViZDoom first is not a consolation prize. The tokenizer, predictor, FiLM conditioning, inference loop, and eval suite all transfer to CS:GO. You find architectural failure modes on a forgiving domain, fix them, then apply to the hard domain with a validated codebase. |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

## **3.2 Action space**

- Keyboard: {forward, back, left, right, turn-left, turn-right, shoot, noop} - 8 keys, one-hot
- Mouse: discretized into 8 horizontal × 8 vertical bins = 64 mouse tokens, one-hot
- **Total action vocab:** 72 tokens (8 keyboard + 64 mouse bins)

| **CAVEAT** | Mouse discretization loses fine-grained aiming precision. Acceptable for v1 - the goal is navigation coherence, not competitive aiming. v2 revisits continuous mouse via cross-attention. |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

## **3.3 Input resolution**

128×128 RGB. 64×64 looks like Atari (bad for demo). 256×256 drops rollout horizon to ~5s (too short). 128×128 is the right tradeoff.

## **3.4 Action diversity augmentation ← new**

Expert gameplay data is heavily biased toward competent navigation in a small region of the action space. Users will immediately walk into walls, spin in circles, and generate action sequences no expert ever produced. The model needs to have seen those transitions.

During data collection and/or training: randomly inject random actions into 15% of trajectory sequences and record the resulting (correct) transitions. This forces the model to learn what happens under unusual action sequences - critical for interactive use.

| **NOTE** | This is distinct from action dropout (which zeros the conditioning signal). Action diversity augmentation changes the actual actions in the training data to cover the full action space, not just the expert distribution. |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

# **4\. Architecture**

FRAME has six components trained in two stages. All components are domain-agnostic.

## **4.1 Stage 1 - VQGAN Tokenizer**

| **Parameter**  | Value                                                              |
| -------------- | ------------------------------------------------------------------ |
| Input          | 128×128 RGB frame                                                  |
| ---            | ---                                                                |
| Encoder        | CNN: 4 strided conv layers (64→128→256→256 channels), stride 2     |
| ---            | ---                                                                |
| Bottleneck     | VQ-VAE: codebook 1024 codes × 256-dim, EMA updates (decay=0.99)    |
| ---            | ---                                                                |
| Output tokens  | 16×16 = 256 discrete integer indices per frame                     |
| ---            | ---                                                                |
| Decoder        | CNN: 4 transposed conv layers, mirrors encoder                     |
| ---            | ---                                                                |
| Loss           | Perceptual (VGG-16) + PatchGAN discriminator + commitment (β=0.25) |
| ---            | ---                                                                |
| Precision      | fp32                                                               |
| ---            | ---                                                                |
| Approx params  | ~40M                                                               |
| ---            | ---                                                                |
| Encode latency | ~5ms / frame on A100                                               |
| ---            | ---                                                                |
| Decode latency | ~10ms / frame on A100                                              |
| ---            | ---                                                                |

| **NOTE** | VQ tokens turn next-frame prediction into classification - 256 independent softmax heads over a 1024-class vocab. Strictly easier to train than regression. Natural uncertainty quantification via softmax entropy (used in v2 heatmap). |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

| **NOTE** | VQ-VAE with L2 loss produces blur - failure mode #1. The GAN discriminator + perceptual loss force sharp, realistic textures. Non-negotiable for visual quality. Measure LPIPS after training: target &lt; 0.15. If LPIPS &gt; 0.15, fix the tokenizer before touching the predictor. |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

| **QUESTION** | Codebook utilization collapse: if <50% of codes are used after 10k steps, switch from EMA to random re-initialization of dead codes (codebook reset trick). Monitor throughout both stages - collapse is silent. |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

**Stage 1 eval gates (must pass before Stage 2):**

- Reconstruction FID on held-out frames < 50
- Codebook utilization > 80% (of 1024 codes active)
- Visual inspection: corridor geometry recognizable at 128×128

## **4.2 Stage 2 - Persistent Scene State (slow memory)**

| **Parameter**  | Value                                                                       |
| -------------- | --------------------------------------------------------------------------- |
| Dimensionality | 512-dim continuous vector                                                   |
| ---            | ---                                                                         |
| Initialization | Encoded from first frame's CNN features                                     |
| ---            | ---                                                                         |
| Update rule    | s*t = α·s*{t−1} + (1−α)·h_t, where h_t = mean-pooled predictor hidden state |
| ---            | ---                                                                         |
| EMA decay (α)  | 0.95 - slow, preserves long-horizon scene structure                         |
| ---            | ---                                                                         |
| Injection      | Prepended as global conditioning token to predictor input at every step     |
| ---            | ---                                                                         |
| Gradient       | Stopped at EMA - no gradient flows through s_t into predictor               |
| ---            | ---                                                                         |

| **NOTE** | A learned state update adds training instability via a recurrent path across the full frame sequence. The EMA is a stable hyperparameter. α=0.95 is the first thing to tune if collapse occurs beyond the context window. |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

## **4.3 Stage 2 - GRU Continuous State (fast memory)**

| **Parameter** | Value                                                                   |
| ------------- | ----------------------------------------------------------------------- |
| Architecture  | Single GRU cell: input = transformer output (512-dim), hidden = 512-dim |
| ---           | ---                                                                     |
| Parameters    | ~2.4M                                                                   |
| ---           | ---                                                                     |
| Position      | Applied per-frame step, after transformer processes full token sequence |
| ---           | ---                                                                     |
| Output        | 512-dim hidden vector concatenated to next frame's input embedding      |
| ---           | ---                                                                     |
| Timescale     | Fast - one update per frame, captures immediate motion/state change     |
| ---           | ---                                                                     |
| Gradient      | Full gradients flow - trained jointly with predictor                    |
| ---           | ---                                                                     |

**Two-tiered temporal memory:**

- **Slow (scene state, EMA):** Where have I been? Global map structure, lighting, scene identity.
- **Fast (GRU):** What just happened? Immediate motion, action effects, short-horizon continuity.

## **4.4 Stage 2 - Causal Transformer Predictor**

| **Parameter**     | Value                                                               |
| ----------------- | ------------------------------------------------------------------- |
| Architecture      | Causal transformer (GPT-style)                                      |
| ---               | ---                                                                 |
| Layers            | 8                                                                   |
| ---               | ---                                                                 |
| Attention heads   | 8                                                                   |
| ---               | ---                                                                 |
| Model dimension   | 512-dim                                                             |
| ---               | ---                                                                 |
| FFN dimension     | 2048-dim (4× model dim)                                             |
| ---               | ---                                                                 |
| Context window    | 8 frames × 256 tokens = 2048 tokens + scene state token + GRU token |
| ---               | ---                                                                 |
| Attention mask    | Causal over context; scene/GRU tokens attend to all                 |
| ---               | ---                                                                 |
| Total params      | ~95M (fp32)                                                         |
| ---               | ---                                                                 |
| KV-cache          | Enabled at inference - only new tokens computed per step            |
| ---               | ---                                                                 |
| Inference latency | ~25ms / frame on A100 with KV-cache                                 |
| ---               | ---                                                                 |

| **NOTE** | 16 frames × 256 = 4096 token context → 4096² = 16.7M element attention matrix per layer, borderline at batch size > 1. 8 frames = 2048 → 4.2M, comfortable at batch size 4. Persistent scene state compensates. Extend to 16 with Flash Attention in v2. |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

### **4.4.1 Spatially parallel masked prediction**

All 256 tokens of the next frame are predicted **simultaneously** via a single forward pass with a BERT-style mask on the target frame. Input sequence: \[context tokens\] + \[MASK × 256\]. The transformer attends causally over context and bidirectionally over the masked target. 15× faster than raster-order token-by-token generation (REM, ICML 2024).

## **4.5 Stage 2 - FiLM Action Conditioning**

| **Parameter**         | Value                                                                      |
| --------------------- | -------------------------------------------------------------------------- |
| Action embedding dim  | 128-dim (learned, from 72-dim one-hot)                                     |
| ---                   | ---                                                                        |
| FiLM layers           | Transformer layers 2, 4, 6, 8 (alternating - 4 of 8)                       |
| ---                   | ---                                                                        |
| FiLM operation        | output = γ(a) ⊙ LayerNorm(x) + β(a), where γ,β are linear projections of a |
| ---                   | ---                                                                        |
| Params per FiLM layer | 2 × (128→512) = 131k × 4 layers = 524k total                               |
| ---                   | ---                                                                        |
| Action dropout        | p=0.15 during training - OOD robustness                                    |
| ---                   | ---                                                                        |

| **NOTE** | A prepended action token can be routed around by attention. FiLM modulates every neuron in conditioned layers directly - there is no routing to exploit. This is the correct inductive bias for strong, unavoidable action conditioning. |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

## **4.6 Stage 2 - Inverse Dynamics Auxiliary Head ← new**

The inverse dynamics head is the primary architectural fix for **action dropout** (failure mode #3). It forces the model's latent representations to encode what changed due to the action, not just what the scene looks like.

| **Parameter** | Value                                                                         |
| ------------- | ----------------------------------------------------------------------------- |
| Input         | (z*t, z*{t+4}) - encoded latents of frames t and t+4 (4-step gap, not 1-step) |
| ---           | ---                                                                           |
| Architecture  | 2-layer MLP: \[512+512 → 256 → 72\], ReLU activations                         |
| ---           | ---                                                                           |
| Output        | Logits over 72-dim action vocabulary                                          |
| ---           | ---                                                                           |
| Loss          | Cross-entropy: predict the action taken between frame t and frame t+4         |
| ---           | ---                                                                           |
| Loss weight   | λ_inv = 0.1 (scale relative to main prediction loss)                          |
| ---           | ---                                                                           |
| Gradient      | Flows back through z*t and z*{t+4} into the encoder/predictor                 |
| ---           | ---                                                                           |
| Params        | ~200k - negligible                                                            |
| ---           | ---                                                                           |

| **NOTE** | In FPS games, single-frame motion under many actions (strafing, slow turns) is below the tokenizer's spatial resolution - consecutive latents may be near-identical even under non-NOOP actions. A 4-step gap allows accumulated motion to become visible in the latent. If action prediction accuracy on 1-step pairs is near chance, this is the reason. |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

| **NOTE** | The inverse dynamics gradient explicitly penalizes the encoder for producing latents that are indistinguishable under different actions. If the model ignores the action, consecutive latents will look identical regardless of what was pressed - the MLP will fail to predict the action, and the cross-entropy loss will push the encoder to encode action-relevant information. This is a direct fix, not a heuristic. |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

**Validation check:** after 10k training steps, inverse dynamics head should achieve >40% accuracy on held-out action prediction from 4-step latent pairs. If accuracy is near chance (1/72 ≈ 1.4%), the encoder is ignoring actions - stop and debug FiLM conditioning before continuing.

# **5\. Training Protocol**

## **5.1 Stage 1 - Tokenizer**

| **Setting**      | Value                                                                  |
| ---------------- | ---------------------------------------------------------------------- |
| Dataset          | All frames from ViZDoom gameplay, shuffled (temporal order not needed) |
| ---              | ---                                                                    |
| Batch size       | 32 frames                                                              |
| ---              | ---                                                                    |
| Optimizer        | Adam, lr=2e-4, cosine decay to 1e-5                                    |
| ---              | ---                                                                    |
| Steps            | ~100k (stop when FID plateaus)                                         |
| ---              | ---                                                                    |
| Compute estimate | ~8-12 hours on 2×A100                                                  |
| ---              | ---                                                                    |
| Eval frequency   | FID + codebook utilization every 5k steps                              |
| ---              | ---                                                                    |

## **5.2 Stage 2 - Predictor**

| **Setting**          | Value                                                                    |
| -------------------- | ------------------------------------------------------------------------ |
| Dataset              | Tokenized (frame tokens, action) sequences, temporal order preserved     |
| ---                  | ---                                                                      |
| Action diversity     | 15% of sequences have randomly injected actions replacing expert actions |
| ---                  | ---                                                                      |
| Sequence length      | 8 frames context + 1 target frame                                        |
| ---                  | ---                                                                      |
| Batch size           | 4 sequences                                                              |
| ---                  | ---                                                                      |
| Optimizer            | AdamW, lr=1e-4, warmup 2k steps, cosine decay                            |
| ---                  | ---                                                                      |
| Steps                | ~200k                                                                    |
| ---                  | ---                                                                      |
| Compute estimate     | ~2-3 days on 2×A100                                                      |
| ---                  | ---                                                                      |
| Loss (main)          | Cross-entropy over 256 × 1024 logits                                     |
| ---                  | ---                                                                      |
| Loss (inv. dynamics) | Cross-entropy over 72-dim action pred, weight λ=0.1                      |
| ---                  | ---                                                                      |
| Scheduled sampling   | p(use own prediction) linearly ramps 0.0→0.5 over first 100k steps       |
| ---                  | ---                                                                      |
| Gradient clipping    | max norm 1.0                                                             |
| ---                  | ---                                                                      |

| **NOTE** | At each training step, with probability p_ss replace ground-truth input tokens with the model's own argmax predictions from the previous step. Ramp p_ss from 0→0.5 over 100k steps. Do not ramp faster - instability before the model is competent causes divergence. |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

| **RISK** | The inverse dynamics loss weight λ=0.1 is a starting point. If the main prediction loss degrades while inverse dynamics accuracy rises, reduce λ. The auxiliary objective should not dominate. Monitor both losses separately throughout training. |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

## **5.3 Action sensitivity check - Day 1**

Run this check after 24 hours of predictor training on a partially trained checkpoint. Do not wait for full convergence.

- **Action sensitivity:** hold visual context fixed, vary action between 'forward' and 'back', compute cosine distance between predicted next-frame token distributions. Target > 0.1. If near zero, FiLM is not working - stop and debug.
- **Inverse dynamics accuracy:** target > 40% on held-out 4-step pairs. If near 1.4% (chance), encoder is ignoring actions.
- **If both fail:** verify action tokens are reaching the FiLM layers. Add a gradient hook to confirm non-zero gradients flow from the FiLM layers. Common bug: action dropout rate accidentally set to 1.0.

# **6\. Inference Loop**

Build this on random noise before training anything. Half of demo failures have nothing to do with the model.

| **Step**               | Operation                                                                                       |
| ---------------------- | ----------------------------------------------------------------------------------------------- |
| 1\. Keyboard capture   | Poll keyboard at 60Hz → 72-dim one-hot action vector                                            |
| ---                    | ---                                                                                             |
| 2\. Tokenizer encode   | Frame t → 256 VQ indices (~5ms, frozen)                                                         |
| ---                    | ---                                                                                             |
| 3\. Scene state update | s*t = 0.95·s*{t−1} + 0.05·h\_{t−1} (EMA)                                                        |
| ---                    | ---                                                                                             |
| 4\. Predictor forward  | \[s_t\] + \[GRU\] + \[2048 context tokens\] + \[256 MASK\] → 256×1024 logits (~25ms, KV-cached) |
| ---                    | ---                                                                                             |
| 5\. Token sampling     | argmax or temperature=0.9 → 256 predicted indices                                               |
| ---                    | ---                                                                                             |
| 6\. GRU update         | h*t = GRU(h*{t-1}, transformer_hidden_mean)                                                     |
| ---                    | ---                                                                                             |
| 7\. Tokenizer decode   | 256 indices → 128×128 RGB (~10ms, frozen)                                                       |
| ---                    | ---                                                                                             |
| 8\. Display            | Render to screen. Total: ~40ms → ~25fps ceiling                                                 |
| ---                    | ---                                                                                             |

| **NOTE** | Target ≥15fps = ≤67ms per frame. Encode (5) + predict (25) + decode (10) = 40ms. 27ms slack for keyboard polling and display. If predict time > 45ms, reduce context from 8 to 4 frames. |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

# **7\. Evaluation Protocol**

## **7.1 Tokenizer gate**

- Reconstruction FID < 50 on held-out frames
- Codebook utilization > 80%
- LPIPS < 0.15 on held-out frames - if above this, fix decoder before proceeding
- Visual inspection: corridor geometry recognizable

## **7.2 Predictor gate**

- Next-frame token prediction accuracy > 60% on held-out sequences
- **Action sensitivity > 0.1:** cosine distance between 'forward' vs 'back' predictions from identical context
- **Inverse dynamics accuracy > 40%:** on held-out 4-step pairs. If near chance, encoder is ignoring actions
- 4-step rollout FID < 80
- Token entropy stable (< 0.5 nats increase) for first 30 rollout steps

## **7.3 Demo gate**

- Human can navigate for ≥30 seconds without visual collapse
- Scene responds visibly to directional key presses
- Frames are sharp - no pervasive blur or block artifacts
- **360-degree consistency:** turn around and return to starting direction - does the room look the same? Informal but critical spatial memory test.

| **QUESTION** | Three collapse signatures to distinguish: (1) scene converges to average color - mean-field collapse; (2) oscillates between 2-3 frames - attractor loop; (3) pure noise - entropy explosion. Each has a different fix. Log per-step token entropy during rollout to distinguish them. |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

# **8\. Novelty Assessment**

| **Component**                 | **Prior work**           | **FRAME angle**            | **Novelty**       |
| ----------------------------- | ------------------------ | -------------------------- | ----------------- |
| VQ tokenizer                  | IRIS 2022                | Standard                   | None              |
| ---                           | ---                      | ---                        | ---               |
| Parallel frame prediction     | REM ICML 2024            | Adopted directly           | None              |
| ---                           | ---                      | ---                        | ---               |
| Persistent scene state        | Matrix-Game 3.0 2025     | Simpler EMA variant        | Incremental       |
| ---                           | ---                      | ---                        | ---               |
| FiLM action conditioning      | Image synthesis lit.     | Applied to WM predictor    | Incremental       |
| ---                           | ---                      | ---                        | ---               |
| GRU fast memory               | DreamerV3 2023           | Alongside transformer      | Incremental       |
| ---                           | ---                      | ---                        | ---               |
| Inverse dynamics head         | ICM 2017, RL lit.        | Applied to interactive WM  | Incremental       |
| ---                           | ---                      | ---                        | ---               |
| Action diversity augmentation | RL data collection lit.  | Applied to WM training     | Incremental       |
| ---                           | ---                      | ---                        | ---               |
| Ternary predictor (v2)        | Nobody in WM lit.        | Direct novel contribution  | Novel             |
| ---                           | ---                      | ---                        | ---               |
| Uncertainty heatmap (v2)      | Nobody in interactive WM | Novel demo artifact        | Novel             |
| ---                           | ---                      | ---                        | ---               |
| Delta tokenizer (v2)          | Δ-IRIS ICML 2024         | Applied to interactive FPS | Incremental       |
| ---                           | ---                      | ---                        | ---               |
| FPS WM at small scale         | Matrix-Game needs 8 GPUs | Single A100 target         | Novel positioning |
| ---                           | ---                      | ---                        | ---               |

**v1 claim:** minimal sufficient architecture for real-time interactive FPS world modeling at small-team scale. Identification of which components are load-bearing for the three failure modes.

**v2 claim:** first ternary-weight interactive world model. Does {−1,0,1} prediction match fp32 on this task? Either finding is publishable.

# **9\. Open Questions**

- **Q1: EMA decay α.** α=0.95 is a guess. Depends on map size and navigation speed. First hyperparameter to tune.
- **Q2: Context window.** 8 frames chosen for memory budget. Does extending to 16 with Flash Attention improve coherence beyond what scene state provides?
- **Q3: Codebook size.** 1024 codes may be too small for CS:GO visual complexity. 2048/4096 as alternatives.
- **Q4: Inference temperature.** argmax = deterministic but frozen-feeling. temperature=0.9 adds variety but risks instability.
- **Q5: Mouse discretization.** 8×8=64 bins may be too coarse for navigation requiring precise turning. 16×16=256 bins is an option.
- **Q6: Action dropout rate.** p=0.15 is a starting point. Too high → model ignores actions at test time. Too low → brittle to OOD inputs.
- **Q7: GRU ablation.** Is the GRU load-bearing or does scene state + long context make it redundant? Cheap ablation - train without GRU, compare collapse horizons.
- **Q8: Inverse dynamics λ weight.** λ=0.1 is a starting point. Sweep {0.05, 0.1, 0.2} on ViZDoom before committing.
- **Q9: 4-step vs 1-step inverse dynamics.** 4-step recommended but not confirmed. Profile token change rate per frame - if >50% change at 1-step, 1-step may be sufficient.

# **10\. Caveats and Risks**

- **Train/inference gap is the hardest problem.** Scheduled sampling helps, self-forcing (v1.5) is the stronger fix. If quality degrades after step 20, this is the cause.
- **CS:GO is genuinely hard.** Fast camera, dynamic lighting, other players, HUD. ViZDoom scaffold exists because CS:GO may require changes we can't anticipate without a working baseline.
- **Codebook collapse is silent.** Monitor utilization throughout both stages.
- **Expert data bias.** Action diversity augmentation mitigates this but doesn't fully solve it. Users will generate action sequences the model has never seen.
- **Inverse dynamics may produce weak signal at 1-step.** Use 4-step transitions. Profile single-frame token change rate on your actual game data first.
- **FiLM conditioning can be gamed.** If training action distribution is too homogeneous, the model learns to produce plausible frames regardless of action. Action diversity augmentation is the mitigation.
- **~100M params is a soft estimate.** Profile before committing to RunPod budget.

# **11\. Compute Plan (RunPod)**

| **Stage**                               | Estimate                        |
| --------------------------------------- | ------------------------------- |
| Stage 1: Tokenizer (2×A100, ~12h)       | ~\$30                           |
| ---                                     | ---                             |
| Stage 2: Predictor (2×A100, ~3 days)    | ~\$180                          |
| ---                                     | ---                             |
| Debugging budget (2×RTX 4090, ~3 days)  | ~\$50 ← use 4090s for iteration |
| ---                                     | ---                             |
| CS:GO domain transfer (2×A100, ~2 days) | ~\$120                          |
| ---                                     | ---                             |
| v2 ternary ablation (1×A100, ~1 day)    | ~\$30                           |
| ---                                     | ---                             |
| Total estimate                          | ~\$410-500                      |
| ---                                     | ---                             |

| **NOTE** | Use RTX 4090 (\$0.34/hr) for debugging runs - 3.5× cheaper than A100 (\$1.19/hr). Only switch to A100 for full training runs where context length or batch size requires 80GB VRAM. Profile your memory usage on a 4090 first. |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |

# **12\. Next Steps**

- **Build inference shell.** Keyboard → random tokens → VQGAN decode → display at 30fps on stubs. Profile every component.
- **Collect ViZDoom data.** Random policy + 15% random action augmentation, record (frame, action) at 15fps. Target ~50k frames.
- **Train VQGAN tokenizer.** Gate: FID &lt; 50, utilization &gt; 80%, LPIPS < 0.15.
- **Train predictor.** Day 1: run action sensitivity + inverse dynamics accuracy checks. If both near zero/chance, stop and debug FiLM before continuing.
- **Run ViZDoom demo.** Hand someone a keyboard. Note which failure mode appears first.
- **Transfer to CS:GO.** New tokenizer on CS:GO frames. Fine-tune predictor on CS:GO sequences.
- **v1.5: counterfactual loss.** Add InfoNCE-style contrastive action ranking on top of working baseline.
- **v2: ternary ablation.** Replace predictor weights with BitNet ternary. Run identical eval. Write up the finding.

# **13\. Future Directions and Research Roadmap**

This section documents ideas that are out of scope for v1 but worth tracking. Each entry includes the research question, why it was deferred, and what v1 result would justify pursuing it.

## **13.1 v1.5 - Counterfactual action ranking loss**

**The idea:** for a given (history, correct action, correct next frame), require the model to assign higher likelihood to the correct frame under the correct action than under sampled wrong actions. InfoNCE-style contrastive objective.

**Formulation:** L*contrast = -log \[p(f*{t+1} | h, a*correct) / Σ_k p(f*{t+1} | h, a_k)\] where a_k are K randomly sampled wrong actions.

**Why deferred:** requires a working inverse dynamics head first. If the model already ignores actions, the contrastive loss will not help - it will just become another signal the model learns to satisfy while remaining action-agnostic. Implement after inverse dynamics accuracy > 40%.

| **V2** | Implement in v1.5 after ViZDoom baseline is working. Expected benefit: stronger action conditioning, especially in ambiguous visual contexts where multiple actions produce similar-looking frames (e.g., standing still vs. shooting in an empty corridor). |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |

## **13.2 v2 - Delta-token context-aware tokenizer (Δ-IRIS)**

**The idea:** instead of encoding each frame independently, encode only what changed relative to the previous frame. The tokenizer is conditioned on the previous frame's tokens, allocating codebook capacity to changed regions. Directly from Δ-IRIS (ICML 2024).

**The efficiency argument:** in FPS games, large scene portions are static per-step - walls, floor, ceiling, HUD. If 60% of tokens are unchanged frame-to-frame, delta encoding reduces effective prediction load by 60%. This enables longer context windows or higher resolution at the same compute.

**Why deferred:** requires profiling actual per-frame token change rate on your game data. If >70% of tokens change every frame (fast action, lots of camera rotation), the efficiency gain is marginal. Measure first, implement only if the gain is confirmed.

| **V2** | Profile: after Stage 1 tokenizer is trained, run it on ViZDoom gameplay and measure the fraction of token positions that change frame-to-frame. If &lt; 40% change on average, delta tokenization is a strong v2 priority. If &gt; 70% change, deprioritize. |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |

## **13.3 v2 - Ternary predictor weights**

**The idea:** replace the transformer predictor's fp32 weights with BitNet-style ternary weights {-1, 0, 1}. Same architecture, same training procedure, different weight representation. The research question: does ternary prediction match fp32 on next-frame token prediction in a game environment?

**The demo angle:** a ternary predictor eliminates multiplications - inference is additions only. This enables the demo to run on a laptop CPU with no GPU. That is a fundamentally different story than 'we trained a world model.'

**The publication angle:** nobody in the world model literature has published a ternary-weight predictor. The finding - positive or negative - is publishable. 'Yes it matches fp32' → edge-deployable world models. 'No, and here's where it breaks' → characterization of what precision world models actually require.

| **V2** | Implementation: train fp32 baseline to convergence. Then initialize a BitNet variant from scratch (ternary training requires training from scratch, not quantizing a trained model). Run identical eval protocol. The ablation comparison IS the paper. |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

## **13.4 v2 - Depth auxiliary head**

**The idea:** add a lightweight depth prediction head on the predictor's hidden state, trained jointly with an auxiliary depth loss. Depth supervision comes from the game engine's depth buffer - essentially free in ViZDoom and available in CS:GO via Reshade/frame capture.

**The benefit:** depth auxiliary loss forces the latent to be spatially grounded - the model must maintain a 3D-consistent representation to predict accurate depth. This directly improves navigation coherence: if the model understands depth, it won't generate a wall where there should be an open corridor.

**Why deferred:** the depth buffer requires additional data collection infrastructure. Measure first whether 360-degree consistency (the spatial memory eval) is passing without it. If it is, the depth head may not be load-bearing.

| **V2** | Add to Stage 2 training as an additional loss term. Loss weight λ_depth = 0.05. Drop the head entirely at inference - it adds nothing to the interactive loop. |
| ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |

## **13.5 v2 - Uncertainty heatmap demo overlay**

**The idea:** at inference time, compute the token prediction entropy (softmax entropy over 1024-class logits) for each of the 256 spatial token positions. Render as a 16×16 heatmap upsampled to 128×128 and overlaid on the generated frame. High entropy = model is uncertain / hallucinating.

**The scientific value:** the heatmap reveals where and when the model is about to collapse before it happens. If entropy spikes in a region at step t, the model is about to generate incoherent content there at step t+1. This is a real-time interpretability tool, not just aesthetics.

**The demo value:** nobody has shipped real-time uncertainty visualization in an interactive world model. It is visually striking, scientifically honest, and makes the model's internals legible to a non-expert audience. Toggle on/off with a key press.

| **V2** | Implementation: entropy is already computed during the predictor forward pass. It is one line: H_i = -Σ_k p_k log p_k over the 1024-class softmax at each of the 256 positions. Reshape to 16×16, upsample, apply colormap (cool=certain, warm=uncertain), overlay at 50% opacity. |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

## **13.6 v3 - Multi-future branching**

**The idea:** at any point during navigation, press a key to fork the current state into N parallel futures sampled with different random seeds. The display splits into N panels showing divergent hallucinated outcomes from the same starting point.

**The scientific question:** how diverse are the model's hallucinations at the same state? High diversity = model has learned a rich distribution. Low diversity = model has collapsed to a near-deterministic mapping. This is a visual evaluation of the model's uncertainty.

**The demo value:** showing five divergent futures from one keypress is one of the most visually compelling demonstrations of what a generative world model is doing. It makes the probabilistic nature of the model tangible.

| **V2** | Requires temperature > 0 at inference (already planned). Fork is just duplicating the KV-cache state N times and running N independent forward passes with different random seeds. No architectural change required. |
| ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

## **13.7 v3 - Flow-matching decoder**

**The idea:** replace the VQGAN decoder with a flow-matching model (continuous normalizing flow from noise to frame, conditioned on token indices). Single forward pass, no denoising steps. Potentially faster than diffusion, higher quality than VQ decode, fully continuous.

**The gap:** nobody has used flow-matching as a decoder in a token-based world model. The diffusion decoder (DIAMOND) requires multiple steps. The VQ decoder is fast but limited by codebook resolution. Flow-matching may be the right middle ground - one step, learned continuous mapping.

**Why deferred:** requires implementing and training an entirely separate decoder architecture. Do this only if VQ decode quality is confirmed to be the bottleneck (LPIPS plateaus despite longer tokenizer training).

| **V2** | Justified only if LPIPS > 0.15 after tokenizer Stage 1 and longer training doesn't help. The flow-matching decoder is a significant engineering investment - validate the bottleneck before pursuing. |
| ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

## **13.8 v3 - Latent action interface (Genie-style)**

**The idea:** instead of mapping keyboard presses to explicit action tokens, learn a latent action space from video alone. The model discovers a compact action representation from consecutive frame pairs without action labels. Users then navigate in latent action space via keyboard, with a learned mapping.

**The appeal:** enables training on unlabeled video (YouTube FPV, game recordings without action logs). Massively expands the data available for training.

**Why far-deferred:** changes the fundamental nature of 'keyboard control.' The keyboard no longer maps to explicit game actions - it maps to directions in a learned latent space that may not correspond to intuitive movements. The interactive demo becomes significantly harder to make feel natural. Only pursue if the explicit action space approach fundamentally hits a data ceiling.

## **13.9 Long-term - Oz Labs product integration**

FRAME is a research project, but the architecture has direct relevance to Oz Labs product work:

- **UAV Mission Simulator:** the same architecture trained on drone FPV footage produces a neural simulator for UAV mission planning. Replace keyboard with waypoint actions. The world model learns realistic terrain response without a physics engine.
- **Synthetic ISR data generation:** a trained world model can generate novel viewpoints and scenarios for training downstream detection models. Infinite synthetic data from a finite real dataset.
- **Adversarial environment generation:** condition the world model on threat parameters to generate hallucinated environments with specified properties (urban density, vegetation cover, threat locations). Red-team simulation without real environments.
- **Edge deployment angle:** the v2 ternary predictor directly enables this. A world model that runs on a laptop CPU with no GPU is deployable in air-gapped, constrained-compute defense environments - the Oz Labs core use case.