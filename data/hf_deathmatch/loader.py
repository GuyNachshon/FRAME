"""Loader for P-H-B-D-a16z/ViZDoom-Deathmatch-PPO-XLrg HuggingFace dataset.

Converts HF parquet format to FRAME's standard HDF5 schema:
  - frames: (N, 128, 128, 3) uint8
  - actions: (N, 8) float32 one-hot
  - episode_ids: (N,) int32

Each HF row contains 10 frames and 10 actions packed together.
Actions are integers 0-17 mapped to our 8-key one-hot format.

Usage:
    # Sample check (Task 2b — do this first)
    uv run python -m data.hf_deathmatch.loader --check

    # Full download + conversion (Task 2c)
    uv run python -m data.hf_deathmatch.loader --convert --output data/hf_deathmatch/vizdoom_deathmatch.hdf5
"""

import argparse
import base64
import io
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


# ViZDoom deathmatch action space (18 actions) mapped to our 8-key format.
# Priority: shoot > movement > turn > noop
# These mappings are approximate — inspect actual action distribution first.
# Index in our format: 0=forward, 1=back, 2=left, 3=right,
#                      4=turn_left, 5=turn_right, 6=shoot, 7=noop
#
# Will be finalized after inspecting the dataset in --check mode.
# For now, use a direct mapping based on common ViZDoom deathmatch bindings.

def _decode_frame(b64_str: str, resolution: int = 128) -> np.ndarray:
    """Decode a base64 PNG string to a uint8 numpy array.

    Args:
        b64_str: Base64-encoded PNG image
        resolution: Target resolution (square)

    Returns:
        (resolution, resolution, 3) uint8 RGB array
    """
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    if img.size != (resolution, resolution):
        img = img.resize((resolution, resolution), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def _map_action(action_int: int, n_actions: int = 18) -> np.ndarray:
    """Map a ViZDoom deathmatch integer action to our 8-dim one-hot.

    This mapping is dataset-specific. The PPO agent uses a discrete
    action space where each integer encodes a combination of buttons.
    We map to the dominant movement direction.

    Args:
        action_int: Integer action from the dataset (0 to n_actions-1)

    Returns:
        (8,) float32 one-hot action vector
    """
    # Default mapping — will be refined after --check inspection
    # ViZDoom deathmatch common mappings:
    #   0: noop, 1: attack, 2: move_right, 3: move_left,
    #   4: move_forward, 5: turn_right, 6: turn_left,
    #   7: attack+move_right, 8: attack+move_left,
    #   9: attack+move_forward, 10: attack+turn_right,
    #   11: attack+turn_left, 12: move_right+move_forward,
    #   13: move_left+move_forward, 14: turn_right+move_forward,
    #   15: turn_left+move_forward, 16: attack+turn_right+move_forward,
    #   17: attack+turn_left+move_forward
    MAPPING = {
        0: 7,   # noop
        1: 6,   # attack -> shoot
        2: 3,   # move_right -> right
        3: 2,   # move_left -> left
        4: 0,   # move_forward -> forward
        5: 5,   # turn_right
        6: 4,   # turn_left
        7: 6,   # attack+move_right -> shoot (primary: attack)
        8: 6,   # attack+move_left -> shoot
        9: 6,   # attack+move_forward -> shoot
        10: 6,  # attack+turn_right -> shoot
        11: 6,  # attack+turn_left -> shoot
        12: 0,  # move_right+forward -> forward (primary: forward)
        13: 0,  # move_left+forward -> forward
        14: 0,  # turn_right+forward -> forward
        15: 0,  # turn_left+forward -> forward
        16: 6,  # attack+turn_right+forward -> shoot
        17: 6,  # attack+turn_left+forward -> shoot
    }
    our_idx = MAPPING.get(action_int, 7)  # default noop
    one_hot = np.zeros(8, dtype=np.float32)
    one_hot[our_idx] = 1.0
    return one_hot


def sample_check(n_rows: int = 100, resolution: int = 128) -> None:
    """Task 2b: Load a small sample, inspect actions, compute frame diff.

    Gate: mean frame diff > 15 before proceeding with full download.
    """
    from datasets import load_dataset

    print("Loading sample from P-H-B-D-a16z/ViZDoom-Deathmatch-PPO-XLrg...")
    # Use streaming to avoid downloading full 49.5GB for a sample check
    ds_stream = load_dataset(
        "P-H-B-D-a16z/ViZDoom-Deathmatch-PPO-XLrg",
        split="train",
        streaming=True,
    )
    ds = list(zip(range(n_rows), ds_stream))
    ds = [row for _, row in ds]
    print(f"  Loaded {len(ds)} rows")

    # Inspect columns
    print(f"  Columns: {ds.column_names}")
    row0 = ds[0]
    print(f"  Sample row keys: {list(row0.keys())}")

    # Inspect actions
    all_actions = []
    for row in ds:
        actions = row["actions"]
        if isinstance(actions, list):
            all_actions.extend(actions)
        else:
            all_actions.append(actions)

    all_actions = np.array(all_actions)
    unique, counts = np.unique(all_actions, return_counts=True)
    print(f"\n  Action distribution ({len(all_actions)} total):")
    for u, c in zip(unique, counts):
        pct = c / len(all_actions) * 100
        print(f"    action {u:>2d}: {c:>6d} ({pct:.1f}%)")

    # Decode frames and compute frame diff
    print(f"\n  Decoding frames...")
    frames = []
    for row in ds:
        images = row["images"]
        if isinstance(images, list):
            for img_str in images:
                frames.append(_decode_frame(img_str, resolution))
        else:
            frames.append(_decode_frame(images, resolution))

    frames = np.stack(frames)
    print(f"  Frames: {frames.shape}, dtype={frames.dtype}")
    print(f"  Pixel range: [{frames.min()}, {frames.max()}]")

    # Frame diff
    diffs = np.abs(frames[1:].astype(float) - frames[:-1].astype(float)).mean(axis=(1, 2, 3))
    print(f"\n  Frame diff stats:")
    print(f"    mean: {diffs.mean():.2f}")
    print(f"    median: {np.median(diffs):.2f}")
    print(f"    max: {diffs.max():.2f}")
    print(f"    min: {diffs.min():.2f}")
    print(f"    stuck (<1.0): {(diffs < 1.0).mean():.1%}")
    print(f"    medium (1-10): {((diffs >= 1.0) & (diffs <= 10.0)).mean():.1%}")
    print(f"    high (>10): {(diffs > 10.0).mean():.1%}")

    # Gate check
    gate = "PASS" if diffs.mean() > 15 else "FAIL"
    print(f"\n  GATE: mean frame diff = {diffs.mean():.2f} (target > 15) [{gate}]")
    if gate == "FAIL":
        print("  WARNING: Frame diff below threshold. Do NOT download full dataset.")
        print("  Investigate whether this dataset has sufficient ego-motion.")

    # Save sample grid
    try:
        grid_frames = frames[:16]
        rows_list = []
        for i in range(0, 16, 4):
            row_imgs = np.concatenate(grid_frames[i:i+4], axis=1)
            rows_list.append(row_imgs)
        grid = np.concatenate(rows_list, axis=0)
        grid_img = Image.fromarray(grid)
        grid_path = "/tmp/deathmatch_check.png"
        grid_img.save(grid_path)
        print(f"\n  Sample grid saved to {grid_path}")
    except Exception as e:
        print(f"\n  Could not save grid: {e}")


def convert_full(output_path: str, resolution: int = 128,
                 max_frames: int | None = None) -> None:
    """Task 2c: Download full dataset and convert to HDF5.

    Streams rows and writes to HDF5 incrementally to avoid OOM.
    763k frames at 128x128x3 = ~37GB — cannot fit in RAM.

    Args:
        output_path: Path for output HDF5 file
        resolution: Target frame resolution
        max_frames: Optional cap on total frames (None = all)
    """
    from datasets import load_dataset

    print("Loading dataset P-H-B-D-a16z/ViZDoom-Deathmatch-PPO-XLrg...")
    ds = load_dataset(
        "P-H-B-D-a16z/ViZDoom-Deathmatch-PPO-XLrg",
        split="train",
    )
    print(f"  Loaded {len(ds)} rows")

    # Estimate total frames (10 per row)
    est_frames = min(len(ds) * 10, max_frames or len(ds) * 10)

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Writing incrementally to {output_path}...")

    with h5py.File(output_path, "w") as f:
        # Create resizable datasets
        frames_ds = f.create_dataset(
            "frames",
            shape=(0, resolution, resolution, 3),
            maxshape=(None, resolution, resolution, 3),
            dtype=np.uint8,
            chunks=(64, resolution, resolution, 3),
            compression="gzip",
            compression_opts=4,
        )
        actions_ds = f.create_dataset(
            "actions",
            shape=(0, 8),
            maxshape=(None, 8),
            dtype=np.float32,
            chunks=(1024, 8),
        )
        episodes_ds = f.create_dataset(
            "episode_ids",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=(1024,),
        )

        n_frames = 0
        # Buffer to reduce resize frequency
        buf_frames: list[np.ndarray] = []
        buf_actions: list[np.ndarray] = []
        buf_episodes: list[int] = []
        FLUSH_SIZE = 1000  # flush every 1000 frames

        for row_idx in range(len(ds)):
            row = ds[row_idx]
            episode_id = row.get("episode_id", row_idx)
            images = row["images"]
            actions = row["actions"]

            if not isinstance(images, list):
                images = [images]
            if not isinstance(actions, list):
                actions = [actions]

            for img_str, act_int in zip(images, actions):
                if max_frames and n_frames >= max_frames:
                    break

                buf_frames.append(_decode_frame(img_str, resolution))
                buf_actions.append(_map_action(act_int))
                buf_episodes.append(int(episode_id))
                n_frames += 1

            # Flush buffer to HDF5
            if len(buf_frames) >= FLUSH_SIZE:
                n_new = len(buf_frames)
                old_size = frames_ds.shape[0]
                frames_ds.resize(old_size + n_new, axis=0)
                actions_ds.resize(old_size + n_new, axis=0)
                episodes_ds.resize(old_size + n_new, axis=0)

                frames_ds[old_size:] = np.stack(buf_frames)
                actions_ds[old_size:] = np.stack(buf_actions)
                episodes_ds[old_size:] = np.array(buf_episodes, dtype=np.int32)

                buf_frames.clear()
                buf_actions.clear()
                buf_episodes.clear()

            if max_frames and n_frames >= max_frames:
                break

            if (row_idx + 1) % 2000 == 0:
                print(f"  {row_idx + 1}/{len(ds)} rows, {n_frames:,} frames")

        # Flush remaining
        if buf_frames:
            n_new = len(buf_frames)
            old_size = frames_ds.shape[0]
            frames_ds.resize(old_size + n_new, axis=0)
            actions_ds.resize(old_size + n_new, axis=0)
            episodes_ds.resize(old_size + n_new, axis=0)

            frames_ds[old_size:] = np.stack(buf_frames)
            actions_ds[old_size:] = np.stack(buf_actions)
            episodes_ds[old_size:] = np.array(buf_episodes, dtype=np.int32)

        f.attrs["n_frames"] = n_frames
        f.attrs["resolution"] = resolution
        f.attrs["source"] = "P-H-B-D-a16z/ViZDoom-Deathmatch-PPO-XLrg"

    # Quality checks (read back from file, not from RAM)
    print(f"\n  Total frames: {n_frames:,}")
    print(f"\n  Quality checks:")
    with h5py.File(output_path, "r") as f:
        actions_arr = f["actions"][:]
        act_counts = actions_arr.sum(axis=0).astype(int)
        names = ["forward", "back", "left", "right",
                 "turn_L", "turn_R", "shoot", "noop"]
        print(f"    Action distribution:")
        for name, count in zip(names, act_counts):
            print(f"      {name:>8s}: {count:>7d} ({count / n_frames * 100:.1f}%)")

        episodes = f["episode_ids"][:]
        unique_eps = np.unique(episodes)
        ep_lens = [np.sum(episodes == e) for e in unique_eps]
        print(f"    Episodes: {len(unique_eps)}")
        print(f"    Episode length: mean={np.mean(ep_lens):.0f}, "
              f"min={np.min(ep_lens)}, max={np.max(ep_lens)}")

    print(f"\n  Done. {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ViZDoom Deathmatch PPO dataset loader"
    )
    parser.add_argument("--check", action="store_true",
                        help="Sample check only (Task 2b)")
    parser.add_argument("--convert", action="store_true",
                        help="Full download + conversion (Task 2c)")
    parser.add_argument("--output", type=str,
                        default="data/hf_deathmatch/vizdoom_deathmatch.hdf5")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Cap total frames (for testing)")
    parser.add_argument("--n_check_rows", type=int, default=100,
                        help="Number of rows for --check")
    args = parser.parse_args()

    if args.check:
        sample_check(n_rows=args.n_check_rows, resolution=args.resolution)
    elif args.convert:
        convert_full(args.output, args.resolution, args.max_frames)
    else:
        parser.error("Specify --check or --convert")


if __name__ == "__main__":
    main()
