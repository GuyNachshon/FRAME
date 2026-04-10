"""PatchGAN discriminator for VQGAN training.

Classifies 70x70 patches as real/fake. Used with hinge loss
to force sharp, realistic textures in reconstructions.
Standard PatchGAN from pix2pix (Isola et al. 2017).
"""

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator.

    Produces a spatial map of real/fake scores for overlapping patches.
    3-layer architecture: each layer doubles channels, uses stride 2.

    Args:
        in_channels: Input channels (3 for RGB)
        n_layers: Number of intermediate layers (default 3)
        base_channels: Base channel count (default 64)
    """

    def __init__(self, in_channels: int = 3, n_layers: int = 3,
                 base_channels: int = 64) -> None:
        super().__init__()

        layers: list[nn.Module] = [
            # First layer: no normalization
            nn.Conv2d(in_channels, base_channels, kernel_size=4,
                      stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        ch = base_channels
        for i in range(1, n_layers):
            prev_ch = ch
            ch = min(ch * 2, 512)
            layers.extend([
                nn.Conv2d(prev_ch, ch, kernel_size=4, stride=2,
                          padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.LeakyReLU(0.2, inplace=True),
            ])

        # Second-to-last layer: stride 1
        prev_ch = ch
        ch = min(ch * 2, 512)
        layers.extend([
            nn.Conv2d(prev_ch, ch, kernel_size=4, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
        ])

        # Final layer: 1-channel output (real/fake score per patch)
        layers.append(
            nn.Conv2d(ch, 1, kernel_size=4, stride=1, padding=1)
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify patches as real/fake.

        Args:
            x: (B, 3, 128, 128) RGB frames

        Returns:
            logits: (B, 1, H', W') patch classification logits
        """
        return self.net(x)
