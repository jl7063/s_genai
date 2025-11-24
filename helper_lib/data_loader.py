# helper_lib/data_loader.py

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def get_transforms(train: bool):
    """
    统一把 CIFAR-10 图片：
    - resize 到 64x64
    - 映射到 [-1, 1]，方便跟老师 notebook、Diffusion/EBM 代码对齐
    """
    tfm = [
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [0,1] -> [-1,1]
    ]

    if train:
        # 训练时加个随机翻转做数据增强
        tfm.insert(1, transforms.RandomHorizontalFlip())

    return transforms.Compose(tfm)


def get_data_loader(
    train: bool,
    batch_size: int = 64,
    num_workers: int = 2,
    data_root: str = "./data",
):
    """
    这是之前 Module 4/6 就一直在用的接口，保持不变：
    - train=True 返回训练集
    - train=False 返回测试集
    """
    tfm = get_transforms(train)
    ds = datasets.CIFAR10(
        root=data_root,
        train=train,
        download=True,
        transform=tfm,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_cifar10_dataloaders(batch_size=128, data_root="./data"):
    """
    这是给 Diffusion / EBM 用的一个小封装：
    只需要训练集 dataloader 时，可以直接调这个。
    """
    train_loader = get_data_loader(
        train=True,
        batch_size=batch_size,
        num_workers=2,
        data_root=data_root,
    )
    return train_loader
