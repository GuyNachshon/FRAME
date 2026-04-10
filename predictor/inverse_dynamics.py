"""Inverse dynamics auxiliary head.

Predicts the action taken between frames t and t+4 from their latent
representations. Forces the encoder to encode action-relevant dynamics,
directly fixing action dropout (failure mode #3).

Architecture: 2-layer MLP [512+512 -> 256 -> 72], ReLU.
Loss: cross-entropy, weight lambda=0.1.
~200k params (negligible).
"""

import torch
import torch.nn as nn


class InverseDynamicsHead(nn.Module):
    """Auxiliary head: predict action from (z_t, z_{t+4}).

    Uses a 4-step gap because single-frame motion under many actions
    (strafing, slow turns) is below the tokenizer's spatial resolution.

    Args:
        latent_dim: Dimension of each latent vector (default 512)
        hidden_dim: MLP hidden dimension (default 256)
        action_dim: Output action dimension (default 72)
    """

    def __init__(self, latent_dim: int = 512, hidden_dim: int = 256,
                 action_dim: int = 72) -> None:
        super().__init__()
        # Input: concatenation of z_t and z_{t+4} -> 2*latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, z_t: torch.Tensor,
                z_t4: torch.Tensor) -> torch.Tensor:
        """Predict action from latent pair.

        Args:
            z_t: (B, D) latent at frame t
            z_t4: (B, D) latent at frame t+4

        Returns:
            logits: (B, 72) action prediction logits
        """
        return self.mlp(torch.cat([z_t, z_t4], dim=-1))
