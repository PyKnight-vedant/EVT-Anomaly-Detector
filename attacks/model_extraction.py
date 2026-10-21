"""Step 13: Model extraction (functionality stealing) attack.

Queries the victim model on a pool of (unlabeled-to-the-attacker) inputs,
collects its (soft or hard) predictions, and trains a clone model to
imitate it. Reports fidelity (agreement with victim) and query efficiency
(fidelity achieved per number of queries).
"""
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from utils.metrics import extraction_fidelity


@torch.no_grad()
def query_victim(victim_model: nn.Module, loader: DataLoader, device,
                  num_queries: int, hard_labels: bool = True):
    victim_model.eval()
    xs, ys = [], []
    collected = 0
    for x, _ in loader:
        if collected >= num_queries:
            break
        x = x.to(device)
        logits = victim_model(x)
        if hard_labels:
            y = logits.argmax(1).cpu()
        else:
            y = torch.softmax(logits, dim=1).cpu()
        xs.append(x.cpu())
        ys.append(y)
        collected += x.size(0)
    xs = torch.cat(xs)[:num_queries]
    ys = torch.cat(ys)[:num_queries]
    return xs, ys


def train_clone_model(build_model_fn: Callable, xs: torch.Tensor, ys: torch.Tensor,
                       device, epochs: int = 10, batch_size: int = 128,
                       hard_labels: bool = True, lr: float = 1e-3) -> nn.Module:
    clone = build_model_fn().to(device)
    optimizer = torch.optim.Adam(clone.parameters(), lr=lr)

    dataset = TensorDataset(xs, ys)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    clone.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = clone(x)
            if hard_labels:
                loss = nn.functional.cross_entropy(logits, y)
            else:
                # distillation-style soft-label loss (KL against victim's soft probs)
                log_probs = torch.log_softmax(logits, dim=1)
                loss = -(y * log_probs).sum(1).mean()
            loss.backward()
            optimizer.step()
    return clone


@torch.no_grad()
def evaluate_extraction(victim_model, clone_model, test_loader, device):
    victim_model.eval(); clone_model.eval()
    victim_preds, clone_preds, true_labels = [], [], []
    for x, y in test_loader:
        x = x.to(device)
        victim_preds.append(victim_model(x).argmax(1).cpu().numpy())
        clone_preds.append(clone_model(x).argmax(1).cpu().numpy())
        true_labels.append(y.numpy())
    victim_preds = np.concatenate(victim_preds)
    clone_preds = np.concatenate(clone_preds)
    true_labels = np.concatenate(true_labels)

    return {
        "fidelity": extraction_fidelity(victim_preds, clone_preds),
        "clone_accuracy": float((clone_preds == true_labels).mean()),
        "victim_accuracy": float((victim_preds == true_labels).mean()),
    }


def run_model_extraction(victim_model, query_loader, test_loader, build_model_fn,
                          device, num_queries: int = 5000, clone_epochs: int = 10,
                          hard_labels: bool = True):
    xs, ys = query_victim(victim_model, query_loader, device, num_queries, hard_labels)
    clone = train_clone_model(build_model_fn, xs, ys, device, epochs=clone_epochs,
                               hard_labels=hard_labels)
    metrics = evaluate_extraction(victim_model, clone, test_loader, device)
    metrics["num_queries"] = num_queries
    metrics["query_efficiency"] = metrics["fidelity"] / max(num_queries, 1)
    return clone, metrics
