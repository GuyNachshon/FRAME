"""HDF5 loader for TeaPearce CS:GO dataset.

Dataset: TeaPearce/CounterStrike_Deathmatch (dataset_dm_expert_dust2)
Format per file:
  frame_i_x          -> RGB screenshot
  frame_i_y          -> [keys_pressed_onehot, Lclick, Rclick, mouse_x_onehot, mouse_y_onehot]
  frame_i_helperarr  -> [kill_flag, death_flag]
"""

import h5py
import numpy as np


def load_csgo_hdf5(hdf5_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a single TeaPearce CS:GO HDF5 file.

    Extracts frames and actions, aligns them, and converts
    actions to FRAME's 72-dim one-hot format (8 keyboard + 64 mouse).

    Args:
        hdf5_path: Path to a single HDF5 file from the dataset

    Returns:
        frames: (N, 128, 128, 3) uint8 RGB
        actions: (N, 72) float32 one-hot (8 keyboard + 64 mouse bins)
    """
    raise NotImplementedError
