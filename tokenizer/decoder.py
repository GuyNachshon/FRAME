"""CNN decoder for VQGAN tokenizer.

Mirrors the encoder: decodes 16x16 x codebook_dim feature maps back
to 128x128 RGB frames via residual blocks and transposed convolutions.
"""

import torch
import torch.nn as nn

from tokenizer.encoder import ResBlock


class Upsample(nn.Module):
    """Transposed conv upsampling (doubles spatial dims)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4,
                                        stride=2, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CNNDecoder(nn.Module):
    """VQGAN decoder: 16x16 x codebook_dim -> 128x128 RGB.

    Architecture mirrors encoder in reverse:
      - Input conv from codebook_dim
      - Residual blocks at bottleneck
      - 3 upsample stages: 256->256->128->64
      - 2 residual blocks per stage
      - Final conv to RGB with sigmoid

    Args:
        out_channels: Output channels (3 for RGB)
        channels: Channel progression per stage (reverse of encoder)
        n_res_blocks: Residual blocks per stage
        codebook_dim: Input feature dimension from VQ
    """

    def __init__(self, out_channels: int = 3,
                 channels: list[int] | None = None,
                 n_res_blocks: int = 2,
                 codebook_dim: int = 256) -> None:
        super().__init__()
        if channels is None:
            channels = [256, 256, 128, 64]

        # Project from codebook dim
        layers: list[nn.Module] = [
            nn.Conv2d(codebook_dim, channels[0], 1),
        ]

        # Residual blocks at bottleneck resolution (16x16)
        for _ in range(n_res_blocks):
            layers.append(ResBlock(channels[0]))

        # Upsample stages
        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            # Upsample first
            layers.append(Upsample(in_ch, out_ch))
            # Then residual blocks at new resolution
            for _ in range(n_res_blocks):
                layers.append(ResBlock(out_ch))

        # Final projection to RGB
        layers.extend([
            nn.GroupNorm(32, channels[-1]),
            nn.SiLU(),
            nn.Conv2d(channels[-1], out_channels, 3, padding=1),
            nn.Sigmoid(),
        ])

        self.net = nn.Sequential(*layers)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized features to RGB frames.

        Args:
            z_q: (B, codebook_dim, 16, 16) quantized feature map

        Returns:
            x_recon: (B, 3, 128, 128) float32 RGB in [0, 1]
        """
        return self.net(z_q)
