"""Step 6: Evaluation metrics — accuracy, precision, recall, F1, confusion
matrix, training/inference time, memory usage. Also attack-specific metrics
used by Steps 7-13 (attack success rate, robust accuracy, fidelity, AUC)."""
import time
import tracemalloc
from contextlib import contextmanager
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_auc_score)


@contextmanager
def timer():
    start = time.perf_counter()
    result = {}
    yield result
    result["seconds"] = time.perf_counter() - start


@contextmanager
def memory_tracker():
    tracemalloc.start()
    result = {}
    yield result
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result["current_mb"] = current / 1e6
    result["peak_mb"] = peak / 1e6


@torch.no_grad()
def evaluate_classifier(model, loader, device) -> Dict:
    model.eval()
    all_preds, all_labels = [], []

    with memory_tracker() as mem:
        with timer() as t_infer:
            for x, y in loader:
                x = x.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(y.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "inference_time_sec": t_infer["seconds"],
        "peak_memory_mb": mem["peak_mb"],
        "n_samples": int(len(y_true)),
    }


def attack_success_rate(y_true: np.ndarray, y_pred_clean: np.ndarray,
                         y_pred_adv: np.ndarray, target_class: int = None) -> float:
    """Fraction of *originally-correct* samples the attack flips.
    If target_class is given (targeted attack), success = flipped-to-target."""
    correctly_classified = y_pred_clean == y_true
    if correctly_classified.sum() == 0:
        return 0.0
    if target_class is None:
        flipped = y_pred_adv[correctly_classified] != y_true[correctly_classified]
    else:
        flipped = y_pred_adv[correctly_classified] == target_class
    return float(flipped.mean())


def robust_accuracy(y_true: np.ndarray, y_pred_adv: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred_adv))


def membership_inference_metrics(y_true_membership, scores) -> Dict:
    preds = (scores >= 0.5).astype(int)
    return {
        "precision": float(precision_score(y_true_membership, preds, zero_division=0)),
        "recall": float(recall_score(y_true_membership, preds, zero_division=0)),
        "auc": float(roc_auc_score(y_true_membership, scores)) if len(set(y_true_membership)) > 1 else float("nan"),
    }


def extraction_fidelity(y_victim: np.ndarray, y_clone: np.ndarray) -> float:
    """Agreement rate between victim and clone predictions on the same inputs."""
    return float((y_victim == y_clone).mean())
