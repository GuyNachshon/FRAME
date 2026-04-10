"""ViZDoom data collection script.

Runs ViZDoom with a mixed policy (scripted + random actions),
records (frame, action) pairs at target fps, saves to HDF5.

Usage:
    uv run python data/vizdoom/collect.py \
        --frames 50000 --output data/vizdoom/raw/ \
        --resolution 128 --fps 15 --random_action_prob 0.15
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

try:
    import vizdoom as vzd
except ImportError:
    vzd = None

# 8 discrete actions matching architecture spec
# Each is a list of button states for ViZDoom's available buttons
ACTIONS = [
    [1, 0, 0, 0, 0, 0, 0],  # 0: MOVE_FORWARD
    [0, 1, 0, 0, 0, 0, 0],  # 1: MOVE_BACKWARD
    [0, 0, 1, 0, 0, 0, 0],  # 2: MOVE_LEFT
    [0, 0, 0, 1, 0, 0, 0],  # 3: MOVE_RIGHT
    [0, 0, 0, 0, 1, 0, 0],  # 4: TURN_LEFT
    [0, 0, 0, 0, 0, 1, 0],  # 5: TURN_RIGHT
    [0, 0, 0, 0, 0, 0, 1],  # 6: ATTACK
    [0, 0, 0, 0, 0, 0, 0],  # 7: NOOP
]

ACTION_NAMES = [
    "forward", "back", "left", "right",
    "turn_left", "turn_right", "shoot", "noop",
]


def create_game(scenario: str = "basic") -> "vzd.DoomGame":
    """Initialize ViZDoom with standard config.

    Args:
        scenario: ViZDoom scenario name (basic, deadly_corridor, etc.)

    Returns:
        Initialized DoomGame instance
    """
    assert vzd is not None, (
        "vizdoom not installed. Install with: uv pip install vizdoom"
    )

    game = vzd.DoomGame()

    scenario_path = f"{vzd.scenarios_path}/{scenario}.wad"
    game.set_doom_scenario_path(scenario_path)
    game.set_doom_map("map01")

    # Screen config — use closest resolution, we'll resize later
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_screen_format(vzd.ScreenFormat.RGB24)

    # Rendering
    game.set_render_hud(False)
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(True)
    game.set_render_particles(True)

    # Available buttons
    game.add_available_button(vzd.Button.MOVE_FORWARD)
    game.add_available_button(vzd.Button.MOVE_BACKWARD)
    game.add_available_button(vzd.Button.MOVE_LEFT)
    game.add_available_button(vzd.Button.MOVE_RIGHT)
    game.add_available_button(vzd.Button.TURN_LEFT)
    game.add_available_button(vzd.Button.TURN_RIGHT)
    game.add_available_button(vzd.Button.ATTACK)

    game.set_episode_timeout(2000)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.init()
    return game


def resize_frame(screen: np.ndarray, resolution: int) -> np.ndarray:
    """Resize ViZDoom screen buffer to target resolution.

    Args:
        screen: Raw screen buffer from ViZDoom, may be (H,W,3) or (3,H,W)
        resolution: Target resolution (square)

    Returns:
        (resolution, resolution, 3) uint8 array
    """
    # ViZDoom RGB24 format: (H, W, 3)
    if screen.ndim == 3 and screen.shape[0] == 3:
        screen = np.transpose(screen, (1, 2, 0))  # CHW -> HWC

    img = Image.fromarray(screen)
    img = img.resize((resolution, resolution), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def scripted_policy(step_in_episode: int) -> int:
    """Simple navigation policy: mostly forward with periodic turns.

    Forward 60%, turn_left 15%, turn_right 15%, other 10%.
    Creates diverse trajectories that cover the map.

    Args:
        step_in_episode: Current step within the episode

    Returns:
        Action index (0-7)
    """
    r = np.random.random()
    if r < 0.60:
        return 0   # forward
    elif r < 0.75:
        return 4   # turn_left
    elif r < 0.90:
        return 5   # turn_right
    elif r < 0.95:
        return 2   # strafe left
    else:
        return 3   # strafe right


def collect_frames(
    n_frames: int,
    output_path: str,
    resolution: int = 128,
    fps: int = 15,
    random_action_prob: float = 0.15,
    scenario: str = "basic",
) -> None:
    """Collect (frame, action) pairs from ViZDoom.

    Saves HDF5 with datasets:
      - frames: (N, resolution, resolution, 3) uint8
      - actions: (N, 8) float32 one-hot
      - episode_ids: (N,) int32

    Args:
        n_frames: Total frames to collect
        output_path: Directory for output HDF5
        resolution: Frame resolution (square)
        fps: Target frames per second (controls ViZDoom ticrate)
        random_action_prob: Fraction of actions replaced with random
        scenario: ViZDoom scenario name
    """
    game = create_game(scenario)

    frames_list: list[np.ndarray] = []
    actions_list: list[np.ndarray] = []
    episode_ids_list: list[int] = []
    episode_id = 0
    collected = 0

    print(f"Collecting {n_frames} frames from ViZDoom ({scenario})...")
    print(f"  Resolution: {resolution}x{resolution}")
    print(f"  Random action prob: {random_action_prob}")

    while collected < n_frames:
        game.new_episode()
        step_in_ep = 0

        while not game.is_episode_finished() and collected < n_frames:
            state = game.get_state()
            if state is None:
                break

            screen = state.screen_buffer
            frame = resize_frame(screen, resolution)

            # Pick action: random_action_prob chance of random, else scripted
            if np.random.random() < random_action_prob:
                action_idx = np.random.randint(len(ACTIONS))
            else:
                action_idx = scripted_policy(step_in_ep)

            action_onehot = np.zeros(8, dtype=np.float32)
            action_onehot[action_idx] = 1.0

            game.make_action(ACTIONS[action_idx])

            frames_list.append(frame)
            actions_list.append(action_onehot)
            episode_ids_list.append(episode_id)
            collected += 1
            step_in_ep += 1

            if collected % 1000 == 0:
                print(f"  {collected}/{n_frames} frames "
                      f"({episode_id + 1} episodes)")

        episode_id += 1

    game.close()

    # Save to HDF5
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / "vizdoom_data.hdf5"

    print(f"Saving {collected} frames to {h5_path}...")
    with h5py.File(h5_path, "w") as f:
        f.create_dataset(
            "frames", data=np.stack(frames_list),
            compression="gzip", compression_opts=4,
        )
        f.create_dataset("actions", data=np.stack(actions_list))
        f.create_dataset(
            "episode_ids",
            data=np.array(episode_ids_list, dtype=np.int32),
        )
        f.attrs["resolution"] = resolution
        f.attrs["n_frames"] = collected
        f.attrs["n_episodes"] = episode_id
        f.attrs["random_action_prob"] = random_action_prob
        f.attrs["scenario"] = scenario

    print(f"Done. {collected} frames, {episode_id} episodes.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect ViZDoom gameplay data"
    )
    parser.add_argument("--frames", type=int, default=50000,
                        help="Number of frames to collect")
    parser.add_argument("--output", type=str, default="data/vizdoom/raw/",
                        help="Output directory for HDF5")
    parser.add_argument("--resolution", type=int, default=128,
                        help="Frame resolution (square)")
    parser.add_argument("--fps", type=int, default=15,
                        help="Target FPS")
    parser.add_argument("--random_action_prob", type=float, default=0.15,
                        help="Fraction of random actions (action diversity)")
    parser.add_argument("--scenario", type=str, default="basic",
                        help="ViZDoom scenario name")
    args = parser.parse_args()

    collect_frames(
        n_frames=args.frames,
        output_path=args.output,
        resolution=args.resolution,
        fps=args.fps,
        random_action_prob=args.random_action_prob,
        scenario=args.scenario,
    )


if __name__ == "__main__":
    main()
