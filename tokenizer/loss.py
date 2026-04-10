"""Combined loss for VQGAN tokenizer training.

Perceptual (VGG-16) + GAN (hinge) + commitment loss.
Non-negotiable: L2 alone produces blur (failure mode #1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG-16 features.

    Compares features at layers conv1_2, conv2_2, conv3_3, conv4_3.
    VGG weights are frozen (no gradient).
    """

    def __init__(self) -> None:
        super().__init__()
        vgg = torchvision.models.vgg16(
            weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1
        )
        # Extract feature layers
        # conv1_2: index 3, conv2_2: 8, conv3_3: 15, conv4_3: 22
        self.slices = nn.ModuleList([
            nn.Sequential(*list(vgg.features[:4])),   # conv1_2
            nn.Sequential(*list(vgg.features[4:9])),  # conv2_2
            nn.Sequential(*list(vgg.features[9:16])), # conv3_3
            nn.Sequential(*list(vgg.features[16:23])),  # conv4_3
        ])
        # Freeze all VGG weights
        for p in self.parameters():
            p.requires_grad = False

        # ImageNet normalization
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize from [0, 1] to ImageNet stats."""
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss.

        Args:
            x: (B, 3, H, W) original, in [0, 1]
            x_recon: (B, 3, H, W) reconstruction, in [0, 1]

        Returns:
            Scalar perceptual loss (sum of L1 distances at each VGG layer)
        """
        x = self._normalize(x)
        x_recon = self._normalize(x_recon)

        loss = torch.tensor(0.0, device=x.device)
        feat_x = x
        feat_r = x_recon
        for sl in self.slices:
            feat_x = sl(feat_x)
            feat_r = sl(feat_r)
            loss = loss + F.l1_loss(feat_r, feat_x.detach())

        return loss


class VQGANLoss(nn.Module):
    """Combined VQGAN loss: perceptual + GAN (hinge) + commitment.

    Args:
        perceptual_weight: Weight for VGG perceptual loss
        gan_weight: Weight for GAN generator loss
        commitment_beta: Weight for VQ commitment loss
    """

    def __init__(self, perceptual_weight: float = 1.0,
                 gan_weight: float = 0.1,
                 commitment_beta: float = 0.25) -> None:
        super().__init__()
        assert perceptual_weight > 0, "Perceptual loss is non-negotiable for sharp output"

        self.perceptual_weight = perceptual_weight
        self.gan_weight = gan_weight
        self.commitment_beta = commitment_beta
        self.perceptual = VGGPerceptualLoss()

    def forward(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        commitment_loss: torch.Tensor,
        disc_fake: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute generator losses.

        Args:
            x: (B, 3, H, W) original frames in [0, 1]
            x_recon: (B, 3, H, W) reconstructed frames in [0, 1]
            commitment_loss: scalar from VQ bottleneck
            disc_fake: discriminator output on reconstructions

        Returns:
            dict with keys: total, recon, perceptual, gan_gen, commitment
        """
        # Reconstruction loss (L1 — less blur-prone than L2)
        recon_loss = F.l1_loss(x_recon, x)

        # Perceptual loss
        perceptual_loss = self.perceptual(x, x_recon)

        # GAN generator loss (hinge): maximize disc score on fakes
        gan_gen_loss = -disc_fake.mean()

        total = (
            recon_loss
            + self.perceptual_weight * perceptual_loss
            + self.gan_weight * gan_gen_loss
            + commitment_loss
        )

        return {
            "total": total,
            "recon": recon_loss,
            "perceptual": perceptual_loss,
            "gan_gen": gan_gen_loss,
            "commitment": commitment_loss,
        }

    def discriminator_loss(self, disc_real: torch.Tensor,
                           disc_fake: torch.Tensor) -> torch.Tensor:
        """Hinge loss for discriminator.

        Args:
            disc_real: discriminator output on real frames
            disc_fake: discriminator output on fake frames (detached)

        Returns:
            Scalar discriminator loss
        """
        loss_real = F.relu(1.0 - disc_real).mean()
        loss_fake = F.relu(1.0 + disc_fake).mean()
        return 0.5 * (loss_real + loss_fake)
