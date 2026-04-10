"""Keyboard and mouse capture for FRAME inference loop.

Converts pygame key state and mouse motion to a 72-dim one-hot action vector:
  - Indices 0-7: keyboard actions (exactly one set)
  - Indices 8-71: mouse bins, 8x8 grid (exactly one set)
"""

import torch
import pygame

# Keyboard action mapping: index -> (name, pygame keys)
ACTION_NAMES = [
    "forward",     # 0
    "back",        # 1
    "left",        # 2
    "right",       # 3
    "turn_left",   # 4
    "turn_right",  # 5
    "shoot",       # 6
    "noop",        # 7
]

# Priority-ordered: first match wins
KEY_MAP: list[tuple[int, list[int]]] = [
    (0, [pygame.K_w, pygame.K_UP]),       # forward
    (1, [pygame.K_s, pygame.K_DOWN]),     # back
    (2, [pygame.K_a]),                     # left
    (3, [pygame.K_d]),                     # right
    (4, [pygame.K_LEFT]),                  # turn_left
    (5, [pygame.K_RIGHT]),                 # turn_right
    (6, [pygame.K_SPACE]),                 # shoot
]

N_KEYBOARD_ACTIONS = 8
N_MOUSE_BINS = 8  # per axis
ACTION_DIM = N_KEYBOARD_ACTIONS + N_MOUSE_BINS * N_MOUSE_BINS  # 72


def _bin_mouse(rel: tuple[int, int], n_bins: int = N_MOUSE_BINS,
               sensitivity: float = 10.0) -> tuple[int, int]:
    """Bin mouse relative motion into discrete grid.

    Args:
        rel: (dx, dy) relative mouse motion from pygame
        n_bins: number of bins per axis
        sensitivity: pixels of motion that maps to the edge bin

    Returns:
        (bin_x, bin_y) each in [0, n_bins)
    """
    def _to_bin(val: float) -> int:
        # Map [-sensitivity, +sensitivity] -> [0, n_bins-1]
        normalized = (val / sensitivity + 1.0) / 2.0  # [0, 1]
        clamped = max(0.0, min(1.0 - 1e-6, normalized))
        return int(clamped * n_bins)

    return _to_bin(rel[0]), _to_bin(rel[1])


def capture_action() -> torch.Tensor:
    """Poll pygame state and return 72-dim one-hot action vector.

    Must be called after pygame.event.pump() or pygame.event.get().

    Returns:
        action: (72,) float32 tensor. One-hot in keyboard region [0:8]
                and one-hot in mouse region [8:72].
    """
    action = torch.zeros(ACTION_DIM, dtype=torch.float32)

    # Keyboard: priority-ordered, first match wins
    keys = pygame.key.get_pressed()
    keyboard_idx = 7  # default: noop
    for idx, key_list in KEY_MAP:
        if any(keys[k] for k in key_list):
            keyboard_idx = idx
            break
    action[keyboard_idx] = 1.0

    # Mouse: bin relative motion into 8x8 grid
    mouse_rel = pygame.mouse.get_rel()
    bin_x, bin_y = _bin_mouse(mouse_rel)
    mouse_idx = N_KEYBOARD_ACTIONS + bin_y * N_MOUSE_BINS + bin_x
    action[mouse_idx] = 1.0

    return action


def get_action_name(action: torch.Tensor) -> str:
    """Human-readable name for an action tensor (for debugging)."""
    kb_idx = action[:N_KEYBOARD_ACTIONS].argmax().item()
    mouse_flat = action[N_KEYBOARD_ACTIONS:].argmax().item()
    mouse_y, mouse_x = divmod(mouse_flat, N_MOUSE_BINS)
    return f"{ACTION_NAMES[kb_idx]} | mouse=({mouse_x},{mouse_y})"
