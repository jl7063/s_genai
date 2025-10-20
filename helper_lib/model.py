import torch.nn as nn

class SmallCNN(nn.Module):
    """适用于32x32 RGB图（如CIFAR-10）的简单CNN"""
    def __init__(self, num_classes=10):
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
            nn.Linear(128*4*4, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def get_model(model_name: str, num_classes: int = 10):
    if model_name.upper() == "CNN":
        return SmallCNN(num_classes=num_classes)
    raise ValueError(f"Unknown model_name: {model_name}")

import torch
import torch.nn as nn

class AssignmentCNN(nn.Module):
    """
    Input: 64x64x3
    Conv(3->16, 3x3, s=1, p=1) + ReLU
    MaxPool(2x2, s=2)
    Conv(16->32, 3x3, s=1, p=1) + ReLU
    MaxPool(2x2, s=2)
    Flatten
    FC(?, 100) + ReLU
    FC(100, 10)
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
        # 经过两次 2x2 池化后，空间从 64 -> 16，通道 32，所以展平维度 32*16*16=8192
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_model(name: str, num_classes: int = 10):
    name = name.lower()
    if name in ["assignmentcnn", "assignment_cnn", "cnn64"]:
        return AssignmentCNN(num_classes=num_classes)
    # 你原来已有的分支...
    # if name == "cnn": return 原先模型
    # else:
    return AssignmentCNN(num_classes=num_classes)

