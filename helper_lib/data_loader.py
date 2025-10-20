from torchvision import datasets, transforms
from torch.utils.data import DataLoader

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

def get_data_loader(data_root="./data", batch_size=128, train=True, num_workers=2):
    ds = datasets.CIFAR10(root=data_root, train=train, download=True, transform=get_transforms(train))
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers)
