from typing import Optional
import torch
import torch.nn as nn


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
        num_classes: for classifiers
        z_dim: for GAN

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

    raise ValueError(f"Unknown model_name: {model_name!r}. "
                     f"Valid: 'cnn'/'smallcnn', 'assignmentcnn'/'assignment_cnn'/'cnn64', 'gan'")
