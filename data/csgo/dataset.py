"""PyTorch dataset for CS:GO frame-action sequences.

Loads pre-processed HDF5 data from the TeaPearce dataset
and returns temporal sequences for predictor training.
"""

from collections.abc import Callable

import torch
from torch.utils.data import Dataset


class CSGODataset(Dataset):
    """CS:GO frame-action sequence dataset.

    Args:
        data_dir: Directory containing processed HDF5 files
        seq_len: Sequence length (default 9 = 8 context + 1 target)
        transform: Optional transform for frames
    """

    def __init__(self, data_dir: str, seq_len: int = 9,
                 transform: Callable | None = None) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns:
            frames: (seq_len, 3, 128, 128) float32 in [0, 1]
            actions: (seq_len, 72) float32 one-hot
        """
        raise NotImplementedError
