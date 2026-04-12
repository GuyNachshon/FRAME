"""Frame display for FRAME inference loop.

Renders 128x128 RGB frames to screen via pygame with FPS overlay.
"""

from collections import deque
import time

import numpy as np
import pygame
import torch

_FPS_HISTORY = 60  # frames for rolling FPS average


class FrameDisplay:
    """Pygame-based frame renderer with FPS overlay.

    Args:
        resolution: Native frame resolution (default 128)
        upscale: Display upscale factor (default 4 -> 512x512 window)
        title: Window title
    """

    def __init__(self, resolution: int = 128, upscale: int = 2,
                 title: str = "FRAME") -> None:
        self.resolution = resolution
        self.display_size = resolution * upscale

        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.display_size, self.display_size),
            pygame.SCALED,  # hardware-accelerated scaling
        )
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("monospace", 18)
        self.clock = pygame.time.Clock()
        self._frame_times: deque[float] = deque(maxlen=_FPS_HISTORY)

    def _to_surface(self, frame: torch.Tensor | np.ndarray) -> pygame.Surface:
        """Convert frame tensor/array to pygame Surface.

        Handles:
          - torch.Tensor or np.ndarray
          - CHW (3,H,W) or HWC (H,W,3)
          - float32 [0,1] or uint8 [0,255]
        """
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()

        # CHW -> HWC
        if frame.ndim == 3 and frame.shape[0] in (1, 3):
            frame = np.transpose(frame, (1, 2, 0))

        # float -> uint8
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)

        # Ensure contiguous for pygame
        frame = np.ascontiguousarray(frame)
        return pygame.surfarray.make_surface(frame.swapaxes(0, 1))

    def show(self, frame: torch.Tensor | np.ndarray) -> bool:
        """Render frame to screen.

        Args:
            frame: RGB frame, shape (3, H, W) or (H, W, 3)

        Returns:
            True if running, False if window was closed
        """
        # Handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                # Toggle mouse grab
                grabbed = pygame.event.get_grab()
                pygame.event.set_grab(not grabbed)
                pygame.mouse.set_visible(grabbed)

        now = time.perf_counter()
        self._frame_times.append(now)

        surface = self._to_surface(frame)
        scaled = pygame.transform.scale(
            surface, (self.display_size, self.display_size)
        )
        self.screen.blit(scaled, (0, 0))

        # FPS overlay
        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            fps = (len(self._frame_times) - 1) / elapsed if elapsed > 0 else 0
            fps_text = self.font.render(f"{fps:.0f} fps", True, (0, 255, 0))
            self.screen.blit(fps_text, (8, 8))

        pygame.display.flip()
        return True

    def close(self) -> None:
        pygame.quit()
