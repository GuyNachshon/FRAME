"""Vector quantization bottleneck with EMA codebook updates.

Implements VQ-VAE style quantization: continuous encoder outputs are
mapped to nearest codebook vectors. Codebook updated via EMA (no gradient).

Includes dead code reset: codes unused for reset_threshold steps
are re-initialized from random encoder outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """VQ bottleneck with EMA codebook.

    Maps continuous encoder features to discrete codebook indices.
    Codebook updated via exponential moving average (decay=0.99).

    Args:
        n_codes: Codebook size (default 1024)
        code_dim: Dimension of each code vector (default 256)
        ema_decay: EMA decay for codebook updates
        commitment_beta: Weight for commitment loss
        reset_threshold: Steps of non-use before resetting a dead code
    """

    def __init__(self, n_codes: int = 1024, code_dim: int = 256,
                 ema_decay: float = 0.99, commitment_beta: float = 0.25,
                 reset_threshold: int = 100) -> None:
        super().__init__()
        assert n_codes > 0
        assert code_dim > 0
        assert 0.0 < ema_decay < 1.0

        self.n_codes = n_codes
        self.code_dim = code_dim
        self.ema_decay = ema_decay
        self.commitment_beta = commitment_beta
        self.reset_threshold = reset_threshold

        # Codebook: (n_codes, code_dim)
        self.register_buffer("codebook", torch.randn(n_codes, code_dim))
        # EMA cluster size and sum for updates
        self.register_buffer("ema_count", torch.zeros(n_codes))
        self.register_buffer("ema_sum", self.codebook.clone())
        # Track usage for dead code detection
        self.register_buffer("usage_count", torch.zeros(n_codes))
        # Track utilization from last forward pass
        self._last_utilization: float = 0.0

    def _quantize(self, z_flat: torch.Tensor) -> tuple[torch.Tensor, torch.LongTensor]:
        """Find nearest codebook vectors.

        Args:
            z_flat: (N, D) flattened encoder features

        Returns:
            z_q: (N, D) quantized vectors
            indices: (N,) codebook indices
        """
        # distances: (N, n_codes) = ||z||^2 - 2*z@codebook^T + ||codebook||^2
        dist = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_flat @ self.codebook.t()
            + self.codebook.pow(2).sum(dim=1, keepdim=True).t()
        )
        indices = dist.argmin(dim=1)
        z_q = self.codebook[indices]
        return z_q, indices

    @torch.no_grad()
    def _ema_update(self, z_flat: torch.Tensor, indices: torch.LongTensor) -> None:
        """Update codebook via EMA on assigned encoder outputs.

        Wrapped in no_grad because EMA updates are statistics, not learned
        parameters. The inplace ops (.mul_, .add_, .copy_) would break
        autograd under DDP without this.
        """
        if not self.training:
            return

        # One-hot encoding: (N, n_codes)
        one_hot = F.one_hot(indices, self.n_codes).float()

        # Count assignments per code
        count = one_hot.sum(dim=0)  # (n_codes,)
        # Sum of encoder outputs per code
        code_sum = one_hot.t() @ z_flat  # (n_codes, code_dim)

        # EMA update
        self.ema_count.mul_(self.ema_decay).add_(count, alpha=1 - self.ema_decay)
        self.ema_sum.mul_(self.ema_decay).add_(code_sum, alpha=1 - self.ema_decay)

        # Laplace smoothing to avoid division by zero
        n = self.ema_count.sum()
        smoothed = (self.ema_count + 1e-5) / (n + self.n_codes * 1e-5) * n
        self.codebook.copy_(self.ema_sum / smoothed.unsqueeze(1))

        # Track usage
        self.usage_count.add_(count)

    @torch.no_grad()
    def _reset_dead_codes(self, z_flat: torch.Tensor) -> None:
        """Re-initialize dead codes from random encoder outputs."""
        if not self.training:
            return

        dead = self.usage_count < 1.0
        n_dead = dead.sum().item()
        if n_dead == 0:
            return

        # Sample random encoder outputs to replace dead codes
        n_dead = int(n_dead)
        rand_idx = torch.randint(0, z_flat.shape[0], (n_dead,),
                                 device=z_flat.device)
        self.codebook[dead] = z_flat[rand_idx].detach()
        self.ema_sum[dead] = z_flat[rand_idx].detach()
        self.ema_count[dead] = 1.0
        self.usage_count[dead] = 1.0

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        """Quantize encoder features.

        Args:
            z: (B, D, H, W) continuous features from encoder

        Returns:
            z_q: (B, D, H, W) quantized features (straight-through gradient)
            commitment_loss: scalar, beta * ||z - sg(z_q)||^2
            indices: (B, H, W) codebook indices
        """
        B, D, H, W = z.shape
        # (B, D, H, W) -> (B*H*W, D)
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)

        z_q, indices = self._quantize(z_flat)
        self._ema_update(z_flat, indices)

        # Dead code reset (periodic)
        if self.training:
            self._reset_dead_codes(z_flat)

        # Commitment loss: encourage encoder outputs to stay close to codes
        commitment_loss = self.commitment_beta * F.mse_loss(z_flat, z_q.detach())

        # Straight-through gradient: copy gradient from z_q to z
        z_q_st = z_flat + (z_q - z_flat).detach()

        # Track utilization
        unique_codes = indices.unique().numel()
        self._last_utilization = unique_codes / self.n_codes

        # Reshape back: (B*H*W, D) -> (B, D, H, W)
        z_q_st = z_q_st.reshape(B, H, W, D).permute(0, 3, 1, 2)
        indices = indices.reshape(B, H, W)

        return z_q_st, commitment_loss, indices

    def lookup(self, indices: torch.LongTensor) -> torch.Tensor:
        """Look up codebook vectors by index.

        Args:
            indices: (B, H, W) codebook indices

        Returns:
            z_q: (B, D, H, W) quantized vectors
        """
        B, H, W = indices.shape
        flat = indices.reshape(-1)
        z_q = self.codebook[flat]  # (B*H*W, D)
        return z_q.reshape(B, H, W, self.code_dim).permute(0, 3, 1, 2)

    def utilization(self) -> float:
        """Fraction of codebook codes used in last forward pass."""
        return self._last_utilization

    def reset_usage_tracking(self) -> None:
        """Reset usage counters (call periodically, e.g. every eval)."""
        self.usage_count.zero_()
