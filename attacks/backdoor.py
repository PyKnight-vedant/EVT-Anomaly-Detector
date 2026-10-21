"""Step 11: Backdoor attack via a white-square trigger patch + label swap."""
import random
from typing import List

import torch
from torch.utils.data import Dataset


def apply_trigger(x01: torch.Tensor, trigger_size: int = 4) -> torch.Tensor:
    """x01: image tensor in [0,1] range, shape (C,H,W) or (B,C,H,W).
    Stamps a white square trigger in the bottom-right corner."""
    x = x01.clone()
    if x.dim() == 3:
        x[:, -trigger_size:, -trigger_size:] = 1.0
    else:
        x[:, :, -trigger_size:, -trigger_size:] = 1.0
    return x


class BackdoorDataset(Dataset):
    """Wraps a dataset (expected to yield normalized tensors); stamps the
    trigger + relabels to target_class on a poisoned subset. Requires
    denormalize/normalize functions to stamp in true pixel space."""

    def __init__(self, base_dataset: Dataset, denorm_fn, norm_fn,
                 poison_fraction: float, target_class: int,
                 trigger_size: int = 4, seed: int = 0):
        self.base = base_dataset
        self.denorm_fn = denorm_fn
        self.norm_fn = norm_fn
        self.target_class = target_class
        self.trigger_size = trigger_size

        rng = random.Random(seed)
        n = len(base_dataset)
        n_poison = int(poison_fraction * n)
        self.poison_indices = set(rng.sample(range(n), n_poison))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if idx in self.poison_indices:
            x01 = self.denorm_fn(x.unsqueeze(0)).squeeze(0)
            x01 = apply_trigger(x01, self.trigger_size)
            x = self.norm_fn(x01.unsqueeze(0)).squeeze(0)
            y = torch.tensor(self.target_class, dtype=y.dtype) if torch.is_tensor(y) else self.target_class
        return x, y

    @property
    def poisoned_indices(self) -> List[int]:
        return sorted(self.poison_indices)


def stamp_trigger_on_batch(x_norm: torch.Tensor, denorm_fn, norm_fn,
                            trigger_size: int = 4) -> torch.Tensor:
    """Apply trigger to a whole (normalized) batch — used at eval time to
    measure attack success rate on clean test data."""
    x01 = denorm_fn(x_norm)
    x01 = apply_trigger(x01, trigger_size)
    return norm_fn(x01)
