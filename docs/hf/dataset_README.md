---
license: mit
tags:
  - world-model
  - vizdoom
  - frame
  - game-footage
  - fps
task_categories:
  - video-classification
size_categories:
  - 10K<n<100K
---

# FRAME ViZDoom Dataset

Training data for the [FRAME](https://github.com/GuyNachshon/FRAME) project — a real-time interactive neural world model for FPS games.

## Dataset Description

Frame-action pairs collected from ViZDoom's `basic` scenario. Used for training the VQGAN tokenizer (Stage 1) and transformer predictor (Stage 2) of the FRAME architecture.

| Property | Value |
|---|---|
| Frames | 50,000 |
| Episodes | ~26 |
| Resolution | 128×128 RGB |
| FPS | 15 |
| Action space | 8 discrete (forward, back, left, right, turn_left, turn_right, shoot, noop) |
| Action diversity | 15% random actions injected into scripted policy |
| Format | HDF5 (gzip compressed) |
| Size | ~835 MB |

## HDF5 Structure

```
vizdoom_data.hdf5
├── frames      (50000, 128, 128, 3)  uint8    — RGB frames
├── actions     (50000, 8)            float32  — one-hot action vectors
└── episode_ids (50000,)              int32    — episode boundary markers
```

## Usage

```python
from data.vizdoom.dataset import ViZDoomFrameDataset, ViZDoomSequenceDataset

# Single frames (for tokenizer training)
frames = ViZDoomFrameDataset("vizdoom_data.hdf5")
frame = frames[0]  # (3, 128, 128) float32 in [0, 1]

# Temporal sequences (for predictor training)
sequences = ViZDoomSequenceDataset("vizdoom_data.hdf5", seq_len=9)
frames, actions = sequences[0]  # (9, 3, 128, 128), (9, 8)
```

## Collection

```bash
uv run python -m data.vizdoom.collect \
    --frames 50000 --output data/vizdoom/raw/ \
    --resolution 128 --fps 15 --random_action_prob 0.15
```

Policy: 85% scripted (forward-biased with periodic turns) + 15% random actions for action diversity augmentation.

## Citation

Part of the FRAME project by [Oz Labs](https://github.com/GuyNachshon/FRAME).
