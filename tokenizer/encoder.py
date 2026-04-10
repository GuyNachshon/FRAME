"""CNN encoder for VQGAN tokenizer.

Encodes 128x128 RGB frames to 16x16 x 256-dim feature maps.
Uses strided convolutions for downsampling with residual blocks for capacity.
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual block with GroupNorm."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Downsample(nn.Module):
    """Strided conv downsampling (halves spatial dims)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2,
                              padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CNNEncoder(nn.Module):
    """VQGAN encoder: 128x128 RGB -> 16x16 x 256-dim feature map.

    Architecture:
      - Input conv: 3 -> 64
      - 3 downsample stages: 64->128->256->256, each halves spatial
      - 2 residual blocks per stage
      - Final conv to codebook_dim

    128 -> 64 -> 32 -> 16 spatially.

    Args:
        in_channels: Input channels (3 for RGB)
        channels: Channel progression per downsample stage
        n_res_blocks: Residual blocks per stage
        codebook_dim: Output feature dimension
    """

    def __init__(self, in_channels: int = 3,
                 channels: list[int] | None = None,
                 n_res_blocks: int = 2,
                 codebook_dim: int = 256) -> None:
        super().__init__()
        if channels is None:
            channels = [64, 128, 256, 256]

        # Input conv
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, channels[0], 3, padding=1, bias=False),
        ]

        # Downsample stages (first 3 channels do downsampling)
        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            # Residual blocks at current resolution
            for _ in range(n_res_blocks):
                layers.append(ResBlock(in_ch))
            # Downsample
            layers.append(Downsample(in_ch, out_ch))

        # Final residual blocks at bottleneck resolution (16x16)
        for _ in range(n_res_blocks):
            layers.append(ResBlock(channels[-1]))

        # Project to codebook dimension
        layers.extend([
            nn.GroupNorm(32, channels[-1]),
            nn.SiLU(),
            nn.Conv2d(channels[-1], codebook_dim, 1),
        ])

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode frames to spatial feature maps.

        Args:
            x: (B, 3, 128, 128) float32 RGB in [0, 1]

        Returns:
            z: (B, codebook_dim, 16, 16) float32 feature map
        """
        return self.net(x)
