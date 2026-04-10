# FRAME — Engineering Best Practices
*Oz Labs · 2026 · Living document — update when conventions change*

---

## Philosophy

This is a research project with a demo deadline. The engineering discipline exists to serve the research, not the other way around. Every convention below has a reason — if the reason doesn't apply to your situation, use judgment.

Three rules that override everything else:
1. **Working code beats elegant code.** A training loop that runs and logs is worth more than a refactored one that doesn't.
2. **Reproducibility is non-negotiable.** If you can't rerun an experiment and get the same result, the result doesn't exist.
3. **Debug on cheap hardware.** Never discover a bug on an A100 that you could have found on a 4090.

---

## Repository structure

```
frame/                          ← repo root (not a Python package)
│
├── data/                       ← data collection and loading only
│   ├── vizdoom/
│   │   ├── collect.py          ← collection script (runnable directly)
│   │   └── dataset.py          ← PyTorch Dataset class
│   └── csgo/
│       ├── loader.py           ← HDF5 loader for TeaPearce dataset
│       └── dataset.py
│
├── tokenizer/                  ← VQGAN: all tokenizer components
├── predictor/                  ← transformer + all conditioning components
├── inference/                  ← interactive loop only — no training code here
├── eval/                       ← evaluation scripts — no training code here
├── configs/                    ← YAML configs only — no code
├── scripts/                    ← bash entrypoints — thin wrappers around Python
├── notebooks/                  ← Jupyter for inspection and debugging only
├── docs/                       ← architecture spec, plan, this file
├── tests/                      ← unit tests for non-ML components
│
├── requirements.txt
├── requirements-dev.txt        ← black, ruff, pytest, ipykernel
├── Dockerfile
├── .env.example                ← WANDB_API_KEY, RUNPOD_API_KEY — never commit .env
├── .gitignore
└── README.md
```

**Hard rules on structure:**
- `inference/` contains zero training code. Zero.
- `eval/` contains zero training code. Zero.
- `scripts/` are bash only — they call Python, they don't implement logic.
- `notebooks/` are for inspection and debugging — never put experiment results only in a notebook.
- `configs/` are YAML only — if you're writing logic in a config, it belongs in code.

---

## Naming conventions

### Files

| Type | Convention | Example |
|---|---|---|
| Python modules | `snake_case.py` | `scene_state.py`, `inverse_dynamics.py` |
| Config files | `{domain}_{component}.yaml` | `vizdoom_tokenizer.yaml`, `csgo_predictor.yaml` |
| Checkpoint files | `{component}_{step:07d}.pt` | `tokenizer_0050000.pt`, `predictor_0200000.pt` |
| Best checkpoint | `{component}_best.pt` | `tokenizer_best.pt` |
| Run logs | `{date}_{component}_{note}.log` | `20260412_tokenizer_baseline.log` |
| Notebooks | `{nn}_{purpose}.ipynb` | `01_tokenizer_inspection.ipynb`, `02_codebook_utilization.ipynb` |

### Python

```python
# Classes: PascalCase
class VQBottleneck:
class PersistentSceneState:
class InverseDynamicsHead:

# Functions and methods: snake_case
def encode_frame(x: torch.Tensor) -> torch.Tensor:
def compute_film_modulation(action_emb: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

# Constants: UPPER_SNAKE_CASE
CODEBOOK_SIZE = 1024
CODEBOOK_DIM = 256
EMA_DECAY = 0.95

# Config keys: snake_case strings (matches YAML)
config = {
    "codebook_size": 1024,
    "ema_decay": 0.95,
}

# Tensor dimension comments: always annotate shapes
# x: (B, T, C) — batch, time, channels
# tokens: (B, 256) — batch, spatial tokens (flattened 16×16)
# action: (B, 72) — batch, one-hot action vector
```

### Experiments and runs

Every training run gets a unique name: `{domain}_{component}_{date}_{note}`

```
vizdoom_tokenizer_20260412_baseline
vizdoom_tokenizer_20260413_larger_codebook
vizdoom_predictor_20260415_baseline
vizdoom_predictor_20260416_no_gru_ablation
csgo_predictor_20260422_finetune
```

This name goes in:
- wandb run name
- checkpoint directory name
- tmux session name on RunPod

---

## Git workflow

### Branch model

```
main                    ← always runnable. no broken code ever merges here.
│
├── dev                 ← integration branch. all feature branches merge here first.
│   ├── feat/tokenizer  ← feature branches
│   ├── feat/predictor
│   ├── feat/inference-shell
│   └── fix/codebook-collapse
│
└── exp/{name}          ← experiment branches — never merge back, just archive
    ├── exp/vizdoom-baseline-20260415
    └── exp/csgo-finetune-20260422
```

**Rules:**
- `main` is always runnable. If it's broken it's an emergency.
- Feature branches branch from `dev`, merge back to `dev` via PR.
- `dev` merges to `main` only when the milestone is complete and tested.
- Experiment branches (`exp/`) are throw-away. Run an experiment, commit the results, never merge back.
- Never commit directly to `main`.

### Commit messages

Format: `{type}: {short description}`

```
feat: add FiLM conditioning to transformer layers 2,4,6,8
fix: correct action dropout rate — was 1.0, should be 0.15
refactor: extract VQ bottleneck into separate module
train: vizdoom tokenizer baseline — FID 47.3, util 84%
exp: ablation — no GRU, collapse at step 18 vs 31 with GRU
docs: update architecture spec with inverse dynamics head
chore: add wandb logging for codebook utilization
```

Types: `feat`, `fix`, `refactor`, `train`, `exp`, `docs`, `chore`, `test`

**What goes in a commit message body (when needed):**
```
train: vizdoom predictor 200k steps — action sensitivity 0.23

Training run: vizdoom_predictor_20260415_baseline
Steps: 200,000
Hardware: 2×A100, RunPod
Duration: 71 hours
Cost: ~$169

Metrics:
  - Next-frame token accuracy: 67.3%
  - Action sensitivity (cosine): 0.23  ✓ (target >0.1)
  - Inverse dynamics accuracy: 51.2%  ✓ (target >40%)
  - 4-step rollout FID: 71.4  ✓ (target <80)
  - Collapse horizon: stable to step 34

Checkpoint: checkpoints/vizdoom/predictor_best.pt
Notes: GRU ablation scheduled next.
```

Training results belong in commit messages, not just in wandb. wandb can be deleted; git is forever.

### Pull requests

- PRs require at minimum a self-review before merging to `dev`
- PR title follows the same `{type}: {description}` format
- PR description answers: what does this do, how was it tested, any known issues
- Squash merge into `dev` — keep history clean
- Delete feature branch after merge

### What never goes in git

```gitignore
# Data
data/vizdoom/raw/
data/csgo/raw/
*.hdf5
*.h5

# Checkpoints
checkpoints/
*.pt
*.pth
*.ckpt

# Secrets
.env
*.key

# Outputs
outputs/
wandb/
runs/
__pycache__/
*.pyc
.ipynb_checkpoints/

# Large files — use DVC or RunPod volumes
*.mp4
*.avi
```

**Checkpoints go on RunPod persistent volumes, not in git.** If a checkpoint matters, note the wandb run ID and step number in a commit message — that's the pointer.

---

## Experiment tracking (wandb)

### Project structure

One wandb project per phase:
- `frame-vizdoom` — all ViZDoom runs (tokenizer + predictor)
- `frame-csgo` — all CS:GO runs
- `frame-ablations` — all ablation experiments (no-GRU, ternary, etc.)

### What to log — mandatory

**Tokenizer runs:**
```python
wandb.log({
    "train/total_loss": total_loss,
    "train/recon_loss": recon_loss,
    "train/perceptual_loss": perceptual_loss,
    "train/gan_loss": gan_loss,
    "train/commitment_loss": commitment_loss,
    "eval/fid": fid,                        # every 5k steps
    "eval/lpips": lpips,                    # every 5k steps
    "eval/codebook_utilization": util,      # every 5k steps
    "eval/codebook_perplexity": perplexity, # every 5k steps
}, step=global_step)
```

**Predictor runs:**
```python
wandb.log({
    "train/total_loss": total_loss,
    "train/prediction_loss": pred_loss,
    "train/inverse_dynamics_loss": inv_loss,
    "train/scheduled_sampling_p": ss_p,
    "eval/token_accuracy": token_acc,           # every 5k steps
    "eval/action_sensitivity": action_sens,     # every 5k steps
    "eval/inverse_dynamics_acc": inv_acc,       # every 5k steps
    "eval/rollout_fid_4step": rollout_fid,      # every 10k steps
    "eval/collapse_horizon": collapse_step,     # every 10k steps
    "eval/mean_token_entropy": mean_entropy,    # every 10k steps
}, step=global_step)
```

**Log a sample grid every 10k steps:**
```python
# Tokenizer: original vs reconstruction
# Predictor: ground truth vs predicted rollout (4 steps)
wandb.log({"samples/reconstructions": wandb.Image(grid)}, step=global_step)
```

### Run naming

Set in code, not the wandb UI:
```python
wandb.init(
    project="frame-vizdoom",
    name=f"{domain}_{component}_{date}_{note}",
    config=config,
    tags=[domain, component, "v1"],
)
```

### After a run

Write a brief run note in the wandb run description:
```
Baseline tokenizer run. FID 47.3 at 95k steps, plateu visible.
Codebook util 84% - healthy. LPIPS 0.12 - passes gate.
Proceed to predictor training.
Checkpoint: /workspace/checkpoints/vizdoom/tokenizer_best.pt (RunPod vol: frame-vol-1)
```

---

## RunPod conventions

### Pod naming

`frame-{component}-{date}` — e.g., `frame-tokenizer-0412`, `frame-predictor-0415`

### Storage

```
/workspace/                     ← persistent network volume (frame-vol-1)
├── data/
│   ├── vizdoom/raw/            ← collected frames, never deleted
│   └── csgo/raw/               ← TeaPearce dataset, never deleted
├── checkpoints/
│   ├── vizdoom/
│   │   ├── tokenizer/
│   │   └── predictor/
│   └── csgo/
└── wandb/                      ← wandb offline cache
```

**Rule: raw data and checkpoints live on the persistent volume, never on container disk.** Container disk is ephemeral — a pod restart loses it.

### Session management

Always use tmux on RunPod:
```bash
tmux new -s train          # start session named 'train'
tmux attach -t train       # reattach after disconnect
```

Training command structure:
```bash
# Always log to file AND stdout
python tokenizer/train.py --config configs/vizdoom_tokenizer.yaml \
  2>&1 | tee /workspace/logs/vizdoom_tokenizer_20260412.log
```

### Before killing a pod

Checklist:
- [ ] Latest checkpoint saved to `/workspace/checkpoints/`
- [ ] wandb run finished (or marked crashed with notes)
- [ ] Run results noted in a git commit message
- [ ] Log file saved to `/workspace/logs/`

---

## Config management

All hyperparameters in YAML. Nothing hardcoded in Python except defaults.

```yaml
# configs/vizdoom_tokenizer.yaml
domain: vizdoom
component: tokenizer

model:
  codebook_size: 1024
  codebook_dim: 256
  ema_decay: 0.99
  commitment_beta: 0.25
  encoder_channels: [64, 128, 256, 256]

training:
  batch_size: 32
  lr: 2.0e-4
  lr_min: 1.0e-5
  total_steps: 100000
  warmup_steps: 1000
  grad_clip: 1.0
  log_every: 100
  eval_every: 5000
  save_every: 5000

data:
  resolution: 128
  fps: 15
  train_split: 0.95

wandb:
  project: frame-vizdoom
  tags: [vizdoom, tokenizer, v1]
```

**Rules:**
- Every experiment that changes a hyperparameter gets its own config file
- Config files are committed to git — they are the experiment record
- Never modify a config file that was used for a completed run — make a new one
- Config filename encodes the experiment: `vizdoom_tokenizer_larger_codebook.yaml`

---

## Python code standards

### Formatting

```bash
# Format: black (non-negotiable)
black .

# Lint: ruff (fast, catches real issues)
ruff check .

# Both run in pre-commit hooks
```

### Type hints

All public functions have type hints. Internal helpers can be unannotated if the types are obvious.

```python
# Good
def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode a batch of frames to discrete token indices.

    Args:
        x: Input frames. Shape: (B, 3, H, W), float32, range [-1, 1]

    Returns:
        tokens: Discrete token indices. Shape: (B, 256), int64
        z_q: Quantized embeddings. Shape: (B, 256, codebook_dim), float32
    """
    ...

# Always annotate tensor shapes in docstrings — this is non-negotiable
```

### Error handling

```python
# Fail loudly at config load time, not at training step 50k
def __init__(self, config: dict):
    assert config["codebook_size"] > 0, "codebook_size must be positive"
    assert config["ema_decay"] < 1.0, "ema_decay must be < 1.0"
    assert config["lr"] < 1.0, f"lr={config['lr']} looks wrong — did you mean 2e-4?"
```

### Device handling

```python
# Always explicit, never assume CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pass device through, don't rely on global state
model = VQBottleneck(config).to(device)
x = x.to(device)
```

### Checkpointing

```python
# Save everything needed to resume — not just the model
def save_checkpoint(path: str, model, optimizer, scheduler, step: int, config: dict):
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config,
    }, path)

# Load and verify config matches
def load_checkpoint(path: str, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    # log the step and config from checkpoint before returning
    return ckpt["step"]
```

---

## Evaluation discipline

### The eval/train separation rule

Eval code lives in `eval/`. Training code lives in `tokenizer/train.py` or `predictor/train.py`. They do not mix. An eval script should be runnable independently on any checkpoint:

```bash
python eval/action_sensitivity.py \
  --checkpoint checkpoints/vizdoom/predictor_0024000.pt \
  --config configs/vizdoom_predictor.yaml \
  --split val
```

### The day 1 check is mandatory

At the 24-hour checkpoint of predictor training, run:

```bash
python eval/action_sensitivity.py --checkpoint checkpoints/vizdoom/predictor_best.pt
python eval/inverse_acc.py --checkpoint checkpoints/vizdoom/predictor_best.pt
```

If action sensitivity < 0.1 or inverse dynamics accuracy < 40%: **stop training**. Debug FiLM conditioning. Do not continue to step 200k hoping it improves. It won't.

### Gate results go in git

When a training gate passes, commit the result:

```bash
git commit -m "train: vizdoom tokenizer passes all gates

FID: 47.3 (target <50) ✓
LPIPS: 0.12 (target <0.15) ✓
Codebook utilization: 84% (target >80%) ✓

Proceeding to predictor training.
Run: vizdoom_tokenizer_20260412_baseline
Checkpoint: RunPod frame-vol-1 /workspace/checkpoints/vizdoom/tokenizer_best.pt"
```

---

## Notebooks

Notebooks are for inspection and debugging. Not for training. Not for results.

### Naming
```
notebooks/
├── 01_tokenizer_reconstruction.ipynb   # visual inspection of reconstructions
├── 02_codebook_utilization.ipynb       # codebook usage distribution
├── 03_rollout_inspection.ipynb         # step-by-step rollout visualization
├── 04_action_sensitivity_debug.ipynb   # debug FiLM conditioning
└── 05_collapse_analysis.ipynb          # token entropy over rollout steps
```

### Rules
- Clear all outputs before committing — notebooks with outputs bloat the repo
- If you find something important in a notebook, write it up as a commit message or doc update — don't leave it buried in a notebook cell
- `jupyter nbconvert --clear-output --inplace notebooks/*.ipynb` before every commit

---

## Documentation

### What gets documented

| Thing | Where |
|---|---|
| Architecture decisions | `docs/FRAME_architecture_v1.1.docx` |
| Project plan and timeline | `docs/FRAME_plan.md` |
| Engineering conventions | `docs/BEST_PRACTICES.md` (this file) |
| Training run results | Git commit messages |
| Experiment hypotheses | Wandb run descriptions |
| Module-level docs | Docstrings in Python |
| Quick reference | `README.md` |

### What doesn't get documented separately

- Every hyperparameter choice (that's what configs are for)
- Intermediate debugging steps (that's what notebooks are for)
- Failed runs (log in wandb, note in commit, move on)

---

## The rules that matter most

In order of importance:

1. **Build the inference shell before training anything.** `python inference/loop.py --stub` at 30fps before touching model code.

2. **Never skip training gates.** FID < 50, LPIPS < 0.15, utilization > 80% before predictor. Action sensitivity > 0.1, inverse dynamics > 40% at 24h before continuing.

3. **Debug on RTX 4090, train on A100.** 3.5× cost difference. Hard cap: evaluate before any run >$200.

4. **Commit training results.** Wandb can be deleted. Git is the ground truth record.

5. **Raw data and checkpoints on persistent volume.** Container disk is ephemeral.

6. **One config file per experiment.** Never modify a config used in a completed run.

7. **Clear notebook outputs before committing.** Always.

8. **Fail loudly at init time.** Assert configs at construction, not at training step 50k.

---

*Update this document when conventions change. A convention nobody follows is noise.*