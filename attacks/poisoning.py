"""Step 10: Data poisoning via random label flipping."""
import random
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


class LabelFlippedDataset(Dataset):
    """Wraps a dataset, flipping the label of a random subset of samples to a
    random *different* class, at the requested fraction."""

    def __init__(self, base_dataset: Dataset, flip_fraction: float,
                 num_classes: int, seed: int = 0):
        self.base = base_dataset
        self.flip_fraction = flip_fraction
        self.num_classes = num_classes

        rng = random.Random(seed)
        n = len(base_dataset)
        n_flip = int(flip_fraction * n)
        self.flip_indices = set(rng.sample(range(n), n_flip))

        # Precompute flipped labels deterministically.
        self.flipped_label = {}
        for idx in self.flip_indices:
            _, true_label = base_dataset[idx]
            choices = [c for c in range(num_classes) if c != true_label]
            self.flipped_label[idx] = rng.choice(choices)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if idx in self.flip_indices:
            new_label = self.flipped_label[idx]
            y = torch.tensor(new_label, dtype=y.dtype) if torch.is_tensor(y) else new_label
        return x, y

    @property
    def poisoned_indices(self) -> List[int]:
        return sorted(self.flip_indices)
