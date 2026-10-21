"""Step 14: Anomaly detection to filter poisoned/backdoored training samples.

Two lightweight, dependency-free detectors:
1. Activation-clustering-style detector: cluster penultimate-layer features
   per claimed class; flag points far from their class centroid (catches
   both label-flip and backdoor-trigger poisoning).
2. Loss-based detector: flag points with unusually high training loss
   relative to the rest of their class (label-flip poisoning tends to look
   like noisy/hard examples).
"""
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader


@torch.no_grad()
def _penultimate_features(model: nn.Module, loader: DataLoader, device):
    """Assumes model exposes `.forward` producing logits; we hook the input
    to the final linear layer as a generic 'penultimate features' proxy."""
    feats, labels = [], []
    activation = {}

    def hook(_module, inp, _out):
        activation["feat"] = inp[0].detach()

    # Find the last nn.Linear layer generically.
    last_linear = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    handle = last_linear.register_forward_hook(hook)

    for x, y in loader:
        x = x.to(device)
        model(x)
        feats.append(activation["feat"].cpu().numpy())
        labels.append(y.numpy())
    handle.remove()

    return np.concatenate(feats), np.concatenate(labels)


def activation_clustering_filter(model: nn.Module, loader: DataLoader, device,
                                  contamination: float = 0.1) -> List[int]:
    """Returns indices (into the loader's underlying dataset, in iteration
    order) flagged as anomalous via per-class 2-means clustering on
    penultimate-layer activations — the smaller cluster per class is
    flagged, following the activation-clustering backdoor-detection idea
    (Chen et al., 2018)."""
    feats, labels = _penultimate_features(model, loader, device)
    flagged = []
    offset = 0
    idx_per_class = {}
    for i, lab in enumerate(labels):
        idx_per_class.setdefault(int(lab), []).append(i)

    for lab, idxs in idx_per_class.items():
        if len(idxs) < 4:
            continue
        X = feats[idxs]
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
        counts = np.bincount(km.labels_)
        minority_cluster = counts.argmin()
        if counts.min() / counts.sum() <= max(contamination, 0.5):
            minority_idxs = [idxs[j] for j in range(len(idxs)) if km.labels_[j] == minority_cluster]
            flagged.extend(minority_idxs)

    return sorted(flagged)


@torch.no_grad()
def per_sample_loss(model: nn.Module, loader: DataLoader, device) -> np.ndarray:
    model.eval()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="none")
        losses.append(loss.cpu().numpy())
    return np.concatenate(losses)


def loss_based_filter(model: nn.Module, loader: DataLoader, device,
                       z_thresh: float = 2.5) -> List[int]:
    """Flags samples whose loss is > z_thresh std above the mean loss."""
    losses = per_sample_loss(model, loader, device)
    mean, std = losses.mean(), losses.std() + 1e-8
    z = (losses - mean) / std
    return sorted(np.where(z > z_thresh)[0].tolist())


def precision_recall_of_filter(flagged_indices: List[int], true_poison_indices: List[int],
                                n_total: int) -> dict:
    flagged = set(flagged_indices)
    poisoned = set(true_poison_indices)
    tp = len(flagged & poisoned)
    fp = len(flagged - poisoned)
    fn = len(poisoned - flagged)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"precision": precision, "recall": recall, "n_flagged": len(flagged)}
