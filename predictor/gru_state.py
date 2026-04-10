"""GRU continuous state (fast memory).

Single GRU cell for per-frame state updates. Full gradient flow.
Captures immediate motion and action effects — complements the
slow EMA scene state.

~2.4M params.
"""

import torch
import torch.nn as nn


class GRUContinuousState(nn.Module):
    """GRU fast memory: per-frame update with full gradient flow.

    Args:
        input_dim: Input dimension (default 512, from transformer output)
        hidden_dim: Hidden state dimension (default 512)
    """

    def __init__(self, input_dim: int = 512,
                 hidden_dim: int = 512) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor,
                h: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """GRU step.

        Args:
            x: (B, D) input (mean-pooled transformer output)
            h: (B, D) previous hidden state, or None for zeros

        Returns:
            output: (B, D) GRU output (same as h_new for GRUCell)
            h_new: (B, D) updated hidden state
        """
        if h is None:
            h = self.reset(x.shape[0]).to(x.device)
        h_new = self.gru_cell(x, h)
        return h_new, h_new

    def reset(self, batch_size: int = 1) -> torch.Tensor:
        """Return zero-initialized hidden state.

        Args:
            batch_size: Batch size

        Returns:
            h0: (B, D) zero tensor
        """
        return torch.zeros(batch_size, self.hidden_dim)
