# helper_lib/data_loader.py
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

def get_data_loader(train: bool, batch_size: int = 64, num_workers: int = 2):
    tfm = get_transforms(train)
    ds = datasets.CIFAR10(root="./data", train=train, download=True, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=True)
