from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# 32x32 RGB classifier
# =========================
class SmallCNN(nn.Module):
    """Simple CNN for 32x32 RGB images (e.g., CIFAR-10)."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 16x16

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 8x8

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


# =========================
# 64x64 RGB classifier
# =========================
class AssignmentCNN(nn.Module):
    """
    Input: 64x64x3
    Conv(3->16, 3x3, s=1, p=1) + ReLU
    MaxPool(2x2, s=2)                         # 64 -> 32
    Conv(16->32, 3x3, s=1, p=1) + ReLU
    MaxPool(2x2, s=2)                         # 32 -> 16
    Flatten
    FC(32*16*16 -> 100) + ReLU
    FC(100 -> num_classes)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 64 -> 32

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 32 -> 16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================
# GAN for MNIST (1x28x28)
# =========================
class MnistGenerator(nn.Module):
    """
    Generator: z (N, z_dim) -> fake image (N, 1, 28, 28) in [-1, 1] via Tanh.
    """
    def __init__(self, z_dim: int = 100):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True),

            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),   # 28x28
            nn.Tanh(),  # match dataset normalized to [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MnistDiscriminator(nn.Module):
    """
    Discriminator: image (N, 1, 28, 28) -> probability real/fake (N, 1).
    Uses LeakyReLU and BatchNorm for stability.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),    # 14x14
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1), # 7x7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),  # vanilla GAN
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GAN(nn.Module):
    """
    Wrapper holding generator and discriminator plus z_dim.
    """
    def __init__(self, z_dim: int = 100):
        super().__init__()
        self.z_dim = z_dim
        self.generator = MnistGenerator(z_dim)
        self.discriminator = MnistDiscriminator()

class EnergyModel(nn.Module):
    """
    Simple CNN-based Energy Model for 64x64 RGB CIFAR-10 images.
    Input: x in [-1,1], shape (B,3,64,64)
    Output: energy scalar per image, shape (B,)
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),  # 64 -> 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # 32 -> 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # 16 -> 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # 8 -> 4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.fc(x)
        return x.squeeze(-1)

# =========================
# Diffusion UNet building blocks
# =========================

class ResidualBlock(nn.Module):
    """
    Simple residual block with GroupNorm + Swish (x * sigmoid(x)).
    """
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.act = lambda x: x * torch.sigmoid(x)  # Swish

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.shortcut is None else self.shortcut(x)

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        return x + residual


class DownBlock(nn.Module):
    """
    Down-sampling block:
    - Several ResidualBlocks
    - Append last feature to skips
    - Average pool by 2
    """
    def __init__(self, width: int, in_channels: Optional[int] = None, block_depth: int = 2):
        super().__init__()
        self.width = width
        if in_channels is None:
            in_channels = width

        blocks = []
        current_in = in_channels
        for _ in range(block_depth):
            blocks.append(ResidualBlock(current_in, width))
            current_in = width
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, skips: list) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        # only store the last output of this level as a skip connection
        skips.append(x)
        x = F.avg_pool2d(x, kernel_size=2)
        return x


class UpBlock(nn.Module):
    """
    Up-sampling block:
    - Upsample by factor 2
    - Concatenate with last skip
    - Several ResidualBlocks to go back to `width` channels
    """
    def __init__(self, width: int, in_channels: Optional[int] = None, block_depth: int = 2):
        super().__init__()
        self.width = width
        if in_channels is None:
            in_channels = width
        # After upsample we will concat with skip (width channels),
        # so first block input = in_channels + width
        blocks = []
        current_in = in_channels + width
        for _ in range(block_depth):
            blocks.append(ResidualBlock(current_in, width))
            current_in = width
        self.blocks = nn.ModuleList(blocks)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x: torch.Tensor, skips: list) -> torch.Tensor:
        x = self.upsample(x)
        skip = skips.pop()  # last stored skip
        x = torch.cat([x, skip], dim=1)
        for block in self.blocks:
            x = block(x)
        return x


class SinusoidalEmbedding(nn.Module):
    """
    Sinusoidal embedding of a scalar (e.g. noise variance / time step).
    Output shape: (B, 1, 1, 2 * num_frequencies)
    """
    def __init__(self, num_frequencies: int = 16):
        super().__init__()
        self.num_frequencies = num_frequencies
        # Frequencies from 1 to 1000 on log scale
        frequencies = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies)
        )
        # Store angular speeds 2πf
        self.register_buffer(
            "angular_speeds",
            2.0 * math.pi * frequencies.view(1, 1, 1, -1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (B, 1, 1, 1) – scalar per sample
        return: (B, 1, 1, 2 * num_frequencies)
        """
        x = x.expand(-1, 1, 1, self.num_frequencies)
        sin_part = torch.sin(self.angular_speeds * x)
        cos_part = torch.cos(self.angular_speeds * x)
        return torch.cat([sin_part, cos_part], dim=-1)


class UNet(nn.Module):
    """
    UNet backbone used as the noise-prediction network in the diffusion model.

    Args:
        image_size: H = W of input images (e.g. 32 for CIFAR-10).
        num_channels: number of image channels (3 for RGB).
        embedding_dim: time/noise embedding dimension (default 32).
    """
    def __init__(self, image_size: int, num_channels: int, embedding_dim: int = 32):
        super().__init__()
        self.image_size = image_size
        self.num_channels = num_channels
        self.embedding_dim = embedding_dim

        # First conv on noisy image
        self.initial = nn.Conv2d(num_channels, 32, kernel_size=1)

        # Sinusoidal embedding of noise variance / time
        self.embedding = SinusoidalEmbedding(num_frequencies=embedding_dim // 2)
        # Project embedding channels to 32 channels
        self.embedding_proj = nn.Conv2d(embedding_dim, 32, kernel_size=1)

        # Encoder
        # Input after concat: 32 (image) + 32 (embedding) = 64 channels
        self.down1 = DownBlock(width=32, in_channels=64, block_depth=2)
        self.down2 = DownBlock(width=64, in_channels=32, block_depth=2)
        self.down3 = DownBlock(width=96, in_channels=64, block_depth=2)

        # Bottleneck
        self.mid1 = ResidualBlock(in_channels=96, out_channels=128)
        self.mid2 = ResidualBlock(in_channels=128, out_channels=128)

        # Decoder
        self.up1 = UpBlock(width=96, in_channels=128, block_depth=2)
        self.up2 = UpBlock(width=64, in_channels=96, block_depth=2)
        self.up3 = UpBlock(width=32, in_channels=64, block_depth=2)

        # Final 1x1 conv back to image channels
        self.final = nn.Conv2d(32, num_channels, kernel_size=1)
        nn.init.zeros_(self.final.weight)  # keep zero init like TF reference

    def forward(self, noisy_images: torch.Tensor, noise_variances: torch.Tensor) -> torch.Tensor:
        """
        noisy_images: (B, C, H, W)
        noise_variances: (B, 1, 1, 1) or (B, 1) broadcastable
        """
        skips = []

        # Initial image conv
        x = self.initial(noisy_images)

        # Time / noise embedding
        # embedding: (B, 1, 1, embedding_dim) -> (B, embedding_dim, 1, 1)
        noise_emb = self.embedding(noise_variances)
        noise_emb = noise_emb.permute(0, 3, 1, 2)
        # Upsample to spatial size and project
        noise_emb = F.interpolate(noise_emb, size=(self.image_size, self.image_size), mode="nearest")
        noise_emb = self.embedding_proj(noise_emb)

        # Concatenate along channels
        x = torch.cat([x, noise_emb], dim=1)

        # Down path
        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)

        # Middle
        x = self.mid1(x)
        x = self.mid2(x)

        # Up path
        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)

        # Final prediction (e.g. predicted noise)
        return self.final(x)


# =========================
# Factory
# =========================
def get_model(model_name: str,
              num_classes: int = 10,
              z_dim: int = 100,
              **kwargs):
    """
    Factory to build models by name.

    Args:
        model_name: one of
          - "cnn", "smallcnn"
          - "assignmentcnn", "assignment_cnn", "cnn64"
          - "gan"
          - "unet", "diffusion", "diffusion_unet"
        num_classes: for classifiers
        z_dim: for GAN
        **kwargs: for diffusion UNet, you can pass
            - image_size (default 32)
            - num_channels (default 3)
            - embedding_dim (default 32)

    Returns:
        nn.Module
    """
    name = (model_name or "").lower()

    if name in ("cnn", "smallcnn"):
        return SmallCNN(num_classes=num_classes)

    if name in ("assignmentcnn", "assignment_cnn", "cnn64"):
        return AssignmentCNN(num_classes=num_classes)

    if name == "gan":
        return GAN(z_dim=z_dim)

    if name in ("unet", "diffusion", "diffusion_unet"):
        image_size = kwargs.get("image_size", 32)
        num_channels = kwargs.get("num_channels", 3)
        embedding_dim = kwargs.get("embedding_dim", 32)
        return UNet(image_size=image_size,
                    num_channels=num_channels,
                    embedding_dim=embedding_dim)
    
    if name in ("energy", "ebm", "energy_model"):
        return EnergyModel()
    
    raise ValueError(
        f"Unknown model_name: {model_name!r}. "
        f"Valid: 'cnn'/'smallcnn', 'assignmentcnn'/'assignment_cnn'/'cnn64', "
        f"'gan', 'unet'/'diffusion'/'diffusion_unet'"
    )
