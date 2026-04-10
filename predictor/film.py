"""FiLM (Feature-wise Linear Modulation) action conditioning.

Modulates transformer hidden states with action-dependent scale and shift:
  output = gamma(a) * LayerNorm(x) + beta(a)

Applied on alternating layers (2,4,6,8) so the action signal cannot be
routed around by attention — it directly modulates every neuron.
"""

import torch
import torch.nn as nn


class FiLMConditioning(nn.Module):
    """FiLM action conditioning layer.

    Pipeline: 72-dim one-hot -> 128-dim learned embedding -> (gamma, beta) projections.
    During training, action is zeroed with probability `dropout` to build
    OOD robustness (distinct from action diversity augmentation in data).

    Args:
        action_dim: Input action dimension (72 = 8 keyboard + 64 mouse)
        embed_dim: Action embedding dimension (default 128)
        model_dim: Transformer hidden dimension (default 512)
        dropout: Action dropout probability during training (default 0.15)
    """

    def __init__(self, action_dim: int = 72, embed_dim: int = 128,
                 model_dim: int = 512, dropout: float = 0.15) -> None:
        super().__init__()
        assert action_dim > 0
        assert 0.0 <= dropout < 1.0, f"Action dropout {dropout} looks wrong"

        self.dropout = dropout
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.ReLU(inplace=True),
        )
        # gamma and beta projections from action embedding
        self.gamma_proj = nn.Linear(embed_dim, model_dim)
        self.beta_proj = nn.Linear(embed_dim, model_dim)

        # Initialize gamma near 1, beta near 0 (near-identity at init).
        # Use small random weights (not zeros) so gradient flows from step 0.
        # Zero weights => gamma is constant => d(output)/d(action) = 0.
        nn.init.normal_(self.gamma_proj.weight, std=0.02)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.normal_(self.beta_proj.weight, std=0.02)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Apply FiLM conditioning.

        Args:
            x: (B, T, D) transformer hidden states (post-LayerNorm)
            action: (B, 72) one-hot action vector

        Returns:
            (B, T, D) modulated hidden states
        """
        # Action dropout during training: zero the action with probability p
        if self.training and self.dropout > 0:
            mask = torch.rand(action.shape[0], 1, device=action.device) > self.dropout
            action = action * mask.float()

        # action: (B, 72) -> (B, embed_dim)
        a_emb = self.action_embed(action)

        # (B, embed_dim) -> (B, model_dim) each
        gamma = self.gamma_proj(a_emb)  # (B, D)
        beta = self.beta_proj(a_emb)    # (B, D)

        # Broadcast over sequence length: (B, 1, D)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        # FiLM: gamma * x + beta
        return gamma * x + beta
