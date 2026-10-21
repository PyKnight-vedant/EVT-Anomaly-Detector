"""Steps 3-4: Dataset selection, 80/10/10 split, preprocessing, DataLoaders."""
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

import config as cfg

NORMALIZE_STATS = {
    "mnist": ((0.1307,), (0.3081,)),
    "fashion_mnist": ((0.2860,), (0.3530,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}


def _build_transform(dataset: str) -> transforms.Compose:
    mean, std = NORMALIZE_STATS[dataset]
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def _raw_dataset(dataset: str, train: bool):
    tfm = _build_transform(dataset)
    if dataset == "mnist":
        return datasets.MNIST(cfg.DATA_DIR, train=train, download=True, transform=tfm)
    if dataset == "fashion_mnist":
        return datasets.FashionMNIST(cfg.DATA_DIR, train=train, download=True, transform=tfm)
    if dataset == "cifar10":
        return datasets.CIFAR10(cfg.DATA_DIR, train=train, download=True, transform=tfm)
    raise ValueError(f"Unknown dataset: {dataset}")


def get_dataloaders(
    dataset: str,
    batch_size: int = 128,
    num_workers: int = 2,
    seed: int = cfg.DEFAULT_SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Download dataset, merge train+test, re-split 80/10/10, return loaders."""
    train_raw = _raw_dataset(dataset, train=True)
    test_raw = _raw_dataset(dataset, train=False)
    full = torch.utils.data.ConcatDataset([train_raw, test_raw])

    n = len(full)
    n_train = int(cfg.TRAIN_VAL_TEST_SPLIT[0] * n)
    n_val = int(cfg.TRAIN_VAL_TEST_SPLIT[1] * n)
    n_test = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        full, [n_train, n_val, n_test], generator=generator
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    return train_loader, val_loader, test_loader


def get_subset_loader(dataset_obj, indices, batch_size=128, shuffle=False):
    return DataLoader(Subset(dataset_obj, indices), batch_size=batch_size, shuffle=shuffle)


def denormalize(x: torch.Tensor, dataset: str) -> torch.Tensor:
    """Undo normalization -> valid [0,1] image tensor (useful for attacks/backdoor)."""
    mean, std = NORMALIZE_STATS[dataset]
    mean_t = torch.tensor(mean, device=x.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=x.device).view(1, -1, 1, 1)
    return (x * std_t + mean_t).clamp(0, 1)


def normalize(x01: torch.Tensor, dataset: str) -> torch.Tensor:
    """Take [0,1]-range image tensor -> normalized (model input) space."""
    mean, std = NORMALIZE_STATS[dataset]
    mean_t = torch.tensor(mean, device=x01.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=x01.device).view(1, -1, 1, 1)
    return (x01 - mean_t) / std_t
