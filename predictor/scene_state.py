"""Persistent scene state (slow memory).

EMA-updated 512-dim vector that captures global scene structure.
Gradient stopped at the EMA boundary — stable hyperparameter, not learned.

Two-tiered memory design:
  - Slow (this): where have I been? Global map structure, lighting.
  - Fast (GRU): what just happened? Immediate motion, action effects.
"""

import torch
import torch.nn as nn


class PersistentSceneState(nn.Module):
    """EMA scene state: s_t = alpha * s_{t-1} + (1-alpha) * h_t.

    Prepended as a global conditioning token to predictor input.
    Gradient is stopped at the EMA boundary — the scene state
    is a running statistic, not a learned representation.

    Args:
        dim: State dimension (default 512)
        alpha: EMA decay (default 0.95). First hyperparameter to tune
               if collapse occurs beyond context window.
    """

    def __init__(self, dim: int = 512, alpha: float = 0.95) -> None:
        super().__init__()
        assert 0.0 < alpha < 1.0, f"alpha={alpha} must be in (0, 1)"
        self.dim = dim
        self.alpha = alpha
        # State is a buffer, not a parameter — no gradient
        self.register_buffer("state", torch.zeros(1, dim))

    def init_from_frame(self, frame_features: torch.Tensor) -> None:
        """Initialize scene state from first frame's features.

        Args:
            frame_features: (B, D) mean-pooled encoder features
        """
        # Use mean across batch, detached
        self.state = frame_features.detach().mean(dim=0, keepdim=True)

    def update(self, hidden: torch.Tensor) -> torch.Tensor:
        """EMA update and return current scene state token (detached).

        Args:
            hidden: (B, D) mean-pooled predictor hidden state

        Returns:
            scene_token: (B, D) detached scene state vector
        """
        # EMA: s_t = alpha * s_{t-1} + (1-alpha) * h_t
        h_mean = hidden.detach().mean(dim=0, keepdim=True)
        self.state = self.alpha * self.state + (1 - self.alpha) * h_mean
        # Expand to batch size and detach (no gradient through scene state)
        B = hidden.shape[0]
        return self.state.expand(B, -1).detach()

    def reset(self) -> None:
        """Reset scene state to zeros."""
        self.state.zero_()
