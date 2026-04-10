"""PyTorch datasets for ViZDoom frame-action data.

Two datasets:
  - ViZDoomFrameDataset: single frames for tokenizer training
  - ViZDoomSequenceDataset: temporal sequences for predictor training
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ViZDoomFrameDataset(Dataset):
    """Single-frame dataset for tokenizer (Stage 1) training.

    Returns individual frames as (3, 128, 128) float32 tensors in [0, 1].

    Args:
        hdf5_path: Path to collected HDF5 file
    """

    def __init__(self, hdf5_path: str) -> None:
        self.hdf5_path = hdf5_path
        with h5py.File(hdf5_path, "r") as f:
            self.n_frames = f["frames"].shape[0]
        # Lazy open for DataLoader worker compatibility
        self._file: h5py.File | None = None

    def _open(self) -> None:
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")

    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Returns frame: (3, 128, 128) float32 in [0, 1]."""
        self._open()
        # frames stored as (N, H, W, 3) uint8
        frame = self._file["frames"][idx]
        return torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0


class ViZDoomSequenceDataset(Dataset):
    """Temporal sequence dataset for predictor (Stage 2) training.

    Returns (frames, actions) sequences that never cross episode boundaries.

    Args:
        hdf5_path: Path to collected HDF5 file
        seq_len: Sequence length (default 9 = 8 context + 1 target)
    """

    def __init__(self, hdf5_path: str, seq_len: int = 9) -> None:
        self.hdf5_path = hdf5_path
        self.seq_len = seq_len
        self._file: h5py.File | None = None

        # Precompute valid start indices: sequences that stay within one episode
        with h5py.File(hdf5_path, "r") as f:
            episode_ids = f["episode_ids"][:]

        self.valid_starts: np.ndarray = np.array([
            i for i in range(len(episode_ids) - seq_len + 1)
            if episode_ids[i] == episode_ids[i + seq_len - 1]
        ], dtype=np.int64)

    def _open(self) -> None:
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns:
            frames: (seq_len, 3, H, W) float32 in [0, 1]
            actions: (seq_len, 8) float32 one-hot
        """
        self._open()
        start = int(self.valid_starts[idx])
        end = start + self.seq_len

        frames = self._file["frames"][start:end]   # (T, H, W, 3) uint8
        actions = self._file["actions"][start:end]  # (T, 8) float32

        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        actions = torch.from_numpy(actions).float()
        return frames, actions
