"""PyTorch datasets for ViZDoom frame-action data.

Two datasets:
  - ViZDoomFrameDataset: single frames for tokenizer training (preloads to RAM)
  - ViZDoomSequenceDataset: temporal sequences for predictor training (numpy mmap)

ViZDoomFrameDataset preloads fully — fine for 50k frames (~2.3GB).
ViZDoomSequenceDataset uses numpy memmap for zero-copy access — handles
200k+ frames without RAM issues.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class ViZDoomFrameDataset(Dataset):
    """Single-frame dataset for tokenizer (Stage 1) training.

    Preloads all frames into RAM as a contiguous float32 tensor.
    Returns individual frames as (3, 128, 128) float32 in [0, 1].

    Args:
        hdf5_path: Path to collected HDF5 file
    """

    def __init__(self, hdf5_path: str) -> None:
        with h5py.File(hdf5_path, "r") as f:
            raw = f["frames"][:]  # (N, H, W, 3) uint8
        self.frames = torch.from_numpy(raw).permute(0, 3, 1, 2).float().div_(255.0)

    def __len__(self) -> int:
        return self.frames.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Returns frame: (3, 128, 128) float32 in [0, 1]."""
        return self.frames[idx]


def _ensure_npy_cache(hdf5_path: str) -> Path:
    """Convert HDF5 to .npy files for memmap access. Cached alongside the HDF5.

    Creates:
      - frames.npy: (N, H, W, 3) uint8
      - actions.npy: (N, 8) float32
      - episode_ids.npy: (N,) int32

    Returns:
        Path to the cache directory
    """
    cache_dir = Path(hdf5_path).parent / ".npy_cache"
    frames_path = cache_dir / "frames.npy"
    actions_path = cache_dir / "actions.npy"
    episodes_path = cache_dir / "episode_ids.npy"

    if frames_path.exists() and actions_path.exists() and episodes_path.exists():
        return cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Converting HDF5 to npy cache at {cache_dir}...")

    with h5py.File(hdf5_path, "r") as f:
        np.save(frames_path, f["frames"][:])
        np.save(actions_path, f["actions"][:])
        np.save(episodes_path, f["episode_ids"][:])

    print(f"  Cache ready.")
    return cache_dir


class ViZDoomSequenceDataset(Dataset):
    """Temporal sequence dataset for predictor (Stage 2) training.

    Uses numpy memmap for zero-copy access — no RAM preload needed.
    Handles 200k+ frames without OOM.

    Args:
        hdf5_path: Path to collected HDF5 file
        seq_len: Sequence length (default 9 = 8 context + 1 target)
    """

    def __init__(self, hdf5_path: str, seq_len: int = 9) -> None:
        self.seq_len = seq_len

        # Convert to npy for memmap (one-time, cached)
        cache_dir = _ensure_npy_cache(hdf5_path)

        self.frames = np.load(cache_dir / "frames.npy", mmap_mode="r")
        self.actions = np.load(cache_dir / "actions.npy", mmap_mode="r")
        episode_ids = np.load(cache_dir / "episode_ids.npy")

        # Precompute valid start indices: sequences within one episode
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

        # memmap read -> copy to tensor (fast, no full dataset in RAM)
        frames = np.array(self.frames[start:end])  # force read from disk
        actions = np.array(self.actions[start:end])

        frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255.0)
        actions_t = torch.from_numpy(actions).float()
        return frames_t, actions_t
