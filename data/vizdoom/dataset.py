"""PyTorch datasets for ViZDoom frame-action data.

Two datasets:
  - ViZDoomFrameDataset: single frames for tokenizer training
  - ViZDoomSequenceDataset: temporal sequences for predictor training

Both preload the full dataset into RAM at init to avoid HDF5 I/O
bottlenecks during training. 50k frames at 128x128x3 = ~2.3GB RAM.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ViZDoomFrameDataset(Dataset):
    """Single-frame dataset for tokenizer (Stage 1) training.

    Preloads all frames into RAM as a contiguous float32 tensor.
    Returns individual frames as (3, 128, 128) float32 in [0, 1].

    Args:
        hdf5_path: Path to collected HDF5 file
    """

    def __init__(self, hdf5_path: str) -> None:
        with h5py.File(hdf5_path, "r") as f:
            # Load all frames into RAM: (N, H, W, 3) uint8 -> (N, 3, H, W) float32
            raw = f["frames"][:]  # (N, H, W, 3) uint8
        self.frames = torch.from_numpy(raw).permute(0, 3, 1, 2).float().div_(255.0)
        # (N, 3, H, W) float32 in [0, 1], contiguous in RAM

    def __len__(self) -> int:
        return self.frames.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Returns frame: (3, 128, 128) float32 in [0, 1]."""
        return self.frames[idx]


class ViZDoomSequenceDataset(Dataset):
    """Temporal sequence dataset for predictor (Stage 2) training.

    Preloads all frames and actions into RAM. Returns (frames, actions)
    sequences that never cross episode boundaries.

    Args:
        hdf5_path: Path to collected HDF5 file
        seq_len: Sequence length (default 9 = 8 context + 1 target)
    """

    def __init__(self, hdf5_path: str, seq_len: int = 9) -> None:
        self.seq_len = seq_len

        with h5py.File(hdf5_path, "r") as f:
            raw_frames = f["frames"][:]     # (N, H, W, 3) uint8
            raw_actions = f["actions"][:]   # (N, 8) float32
            episode_ids = f["episode_ids"][:]

        self.frames = torch.from_numpy(raw_frames).permute(0, 3, 1, 2).float().div_(255.0)
        self.actions = torch.from_numpy(raw_actions).float()

        # Precompute valid start indices: sequences that stay within one episode
        self.valid_starts: np.ndarray = np.array([
            i for i in range(len(episode_ids) - seq_len + 1)
            if episode_ids[i] == episode_ids[i + seq_len - 1]
        ], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns:
            frames: (seq_len, 3, H, W) float32 in [0, 1]
            actions: (seq_len, 8) float32 one-hot
        """
        start = int(self.valid_starts[idx])
        end = start + self.seq_len
        return self.frames[start:end], self.actions[start:end]
