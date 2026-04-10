"""Stub components for profiling the inference loop without a trained model.

All stubs return random noise or zeros. No artificial latency —
measure real loop overhead.
"""

import torch


class StubEncoder:
    """Returns random token indices. Shape: (B, 16, 16), values in [0, 1024)."""

    def __init__(self, codebook_size: int = 1024, grid_size: int = 16) -> None:
        self.codebook_size = codebook_size
        self.grid_size = grid_size

    def __call__(self, frame: torch.Tensor) -> torch.LongTensor:
        B = frame.shape[0]
        return torch.randint(
            0, self.codebook_size, (B, self.grid_size, self.grid_size)
        )


class StubPredictor:
    """Returns random logits. Shape: (B, 256, 1024)."""

    def __init__(self, tokens_per_frame: int = 256,
                 codebook_size: int = 1024) -> None:
        self.tokens_per_frame = tokens_per_frame
        self.codebook_size = codebook_size

    def __call__(self, tokens: torch.LongTensor, action: torch.Tensor,
                 scene_state: torch.Tensor,
                 gru_state: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        return torch.randn(B, self.tokens_per_frame, self.codebook_size)


class StubDecoder:
    """Returns random RGB frame. Shape: (B, 3, 128, 128), values in [0, 1]."""

    def __init__(self, resolution: int = 128) -> None:
        self.resolution = resolution

    def __call__(self, token_indices: torch.LongTensor) -> torch.Tensor:
        B = token_indices.shape[0] if token_indices.ndim > 1 else 1
        return torch.rand(B, 3, self.resolution, self.resolution)


class StubSceneState:
    """No-op scene state. Returns zeros."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def update(self, hidden: torch.Tensor) -> torch.Tensor:
        B = hidden.shape[0]
        return torch.zeros(B, self.dim)

    def reset(self) -> None:
        pass


class StubGRUState:
    """No-op GRU state. Returns zeros."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def forward(self, x: torch.Tensor,
                h: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        zeros = torch.zeros(B, self.dim)
        return zeros, zeros

    def reset(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(batch_size, self.dim)
