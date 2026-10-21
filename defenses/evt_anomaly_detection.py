"""Step 14 (extension): Extreme Value Theory (EVT) anomaly detection.

Motivation: the loss-based and activation-clustering filters in
anomaly_detection.py use ad-hoc thresholds (a fixed z-score, or "flag the
minority k-means cluster"). Per-sample loss and activation-distance
distributions are heavy-tailed, not Gaussian, so a z-score cutoff is a poor
statistical fit. EVT's Peaks-Over-Threshold (POT) method instead fits a
Generalized Pareto Distribution (GPD) to the tail of the distribution above
some baseline threshold u, and derives a detection cutoff from a target
false-positive rate rather than an arbitrary multiplier. This mirrors the
approach used in OpenMax / the Extreme Value Machine (Bendale & Boult, 2016;
Scheirer et al., 2011) for open-set / outlier recognition.

Two detectors are provided, mirroring the two existing filters:
- evt_loss_filter: EVT tail model on per-sample training loss (poisoning).
- evt_activation_filter: EVT tail model on per-sample distance-to-class-
  centroid in penultimate-layer feature space (backdoor).
Plus a third, new capability not present in anomaly_detection.py:
- evt_evasion_detector: fits EVT on *clean* logit-margin statistics, then
  flags test-time inputs (e.g. adversarial examples from Steps 7-9) whose
  margin falls in the extreme low tail — a test-time defense against
  evasion attacks, complementing the training-time poisoning/backdoor use.
"""
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import DataLoader

from defenses.anomaly_detection import _penultimate_features, per_sample_loss


def fit_gpd_threshold(scores: np.ndarray, tail_fraction: float = 0.1,
                       target_fpr: float = 0.01) -> Tuple[float, dict]:
    """Peaks-Over-Threshold: fit a GPD to the excesses of `scores` above the
    (1 - tail_fraction) empirical quantile, then invert the fitted tail CDF
    to find the score value corresponding to `target_fpr` (the fraction of
    *clean* points expected to exceed the threshold by chance).

    Returns (threshold, fit_info). Falls back to the empirical
    (1 - target_fpr) quantile if the GPD fit fails (e.g. too few points).
    """
    scores = np.asarray(scores, dtype=float)
    u = np.quantile(scores, 1 - tail_fraction)
    excesses = scores[scores > u] - u

    fit_info = {"threshold_u": float(u), "n_excesses": int(len(excesses))}

    if len(excesses) < 20:
        # Not enough tail data for a stable GPD fit; fall back to empirical quantile.
        thresh = float(np.quantile(scores, 1 - target_fpr))
        fit_info["method"] = "empirical_fallback"
        return thresh, fit_info

    # Fit GPD (shape xi, loc fixed at 0 since we're modeling excesses, scale sigma).
    xi, loc, sigma = stats.genpareto.fit(excesses, floc=0)
    fit_info.update({"method": "gpd", "xi": float(xi), "sigma": float(sigma)})

    # P(X > u + y | X > u) = (1 + xi*y/sigma)^(-1/xi)  =>  invert for target_fpr.
    p_exceed_u = tail_fraction  # empirical P(X > u)
    if p_exceed_u <= target_fpr:
        # Requested FPR is looser than the tail fraction itself; just use quantile.
        thresh = float(np.quantile(scores, 1 - target_fpr))
        fit_info["method"] = "empirical_fallback_loose_fpr"
        return thresh, fit_info

    p_within_tail = target_fpr / p_exceed_u  # conditional exceedance prob we need
    if abs(xi) < 1e-6:
        y = -sigma * np.log(p_within_tail)
    else:
        y = (sigma / xi) * (p_within_tail ** (-xi) - 1)

    thresh = float(u + y)
    return thresh, fit_info


def empirical_threshold(scores: np.ndarray, target_fpr: float = 0.01) -> float:
    """Plain (1 - target_fpr) empirical quantile -- the baseline that shows
    whether fitting a GPD adds anything beyond just picking a quantile."""
    return float(np.quantile(np.asarray(scores, dtype=float), 1 - target_fpr))


def threshold_from_reference(ref_scores, tail_fraction: float, target_fpr: float,
                             method: str = "gpd") -> Tuple[float, dict]:
    """Threshold from a (clean) reference score set. method: 'gpd' | 'empirical'."""
    if method == "empirical":
        return empirical_threshold(ref_scores, target_fpr), {"method": "empirical"}
    return fit_gpd_threshold(ref_scores, tail_fraction, target_fpr)


def evt_loss_filter(model: nn.Module, loader: DataLoader, device,
                     tail_fraction: float = 0.1, target_fpr: float = 0.01,
                     ref_loader: DataLoader = None, method: str = "gpd") -> Tuple[List[int], dict]:
    """Flags samples whose loss exceeds a threshold derived from a target FPR.

    ref_loader: CLEAN reference data (e.g. the validation split). The tail model
    is fit on its losses and the resulting threshold is applied to `loader`.
    If None, the threshold is fit on `loader` itself (contaminated by the
    poisoned samples -- kept only for the ablation).
    method: 'gpd' (EVT) or 'empirical' (plain quantile baseline)."""
    losses = per_sample_loss(model, loader, device)
    ref = per_sample_loss(model, ref_loader, device) if ref_loader is not None else losses
    thresh, fit_info = threshold_from_reference(ref, tail_fraction, target_fpr, method)
    fit_info["calibrated_on"] = "clean_reference" if ref_loader is not None else "same_data"
    flagged = sorted(np.where(losses > thresh)[0].tolist())
    return flagged, fit_info


def evt_activation_filter(model: nn.Module, loader: DataLoader, device,
                           tail_fraction: float = 0.1, target_fpr: float = 0.01,
                           ref_loader: DataLoader = None, method: str = "gpd") -> Tuple[List[int], dict]:
    """Per class: distance of each sample's penultimate features to a class
    centroid; flag samples beyond a threshold derived from a target FPR.

    ref_loader: CLEAN reference data. Centroid and tail model come from its
    features; the threshold is then applied to `loader`'s distances. If None,
    everything is computed on `loader` itself (contaminated; ablation only).
    method: 'gpd' (EVT) or 'empirical' (plain quantile baseline)."""
    feats, labels = _penultimate_features(model, loader, device)
    if ref_loader is not None:
        ref_feats, ref_labels = _penultimate_features(model, ref_loader, device)
    else:
        ref_feats, ref_labels = feats, labels

    idx_per_class = {}
    for i, lab in enumerate(labels):
        idx_per_class.setdefault(int(lab), []).append(i)

    flagged = []
    fit_infos = {}
    for lab, idxs in idx_per_class.items():
        ref_idx = np.where(ref_labels == lab)[0]
        if len(idxs) < 10 or len(ref_idx) < 10:
            continue
        centroid = ref_feats[ref_idx].mean(axis=0)
        ref_d = np.linalg.norm(ref_feats[ref_idx] - centroid, axis=1)
        dists = np.linalg.norm(feats[idxs] - centroid, axis=1)
        thresh, fit_info = threshold_from_reference(ref_d, tail_fraction, target_fpr, method)
        fit_infos[lab] = fit_info
        flagged.extend([idxs[j] for j in range(len(idxs)) if dists[j] > thresh])

    return sorted(flagged), fit_infos


@torch.no_grad()
def _clean_logit_margins(model: nn.Module, loader: DataLoader, device) -> np.ndarray:
    """Margin = top-1 logit - top-2 logit, on samples the model already
    classifies correctly. Small/negative margins on clean data are rare;
    that's the tail EVT will characterize."""
    model.eval()
    margins = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(1)
        correct = preds == y
        top2 = torch.topk(logits, 2, dim=1).values
        m = (top2[:, 0] - top2[:, 1])
        margins.append(m[correct].cpu().numpy())
    return np.concatenate(margins)


def fit_evt_evasion_detector(model: nn.Module, clean_loader: DataLoader, device,
                              tail_fraction: float = 0.1, target_fpr: float = 0.01,
                              method: str = "gpd"):
    """Fit an EVT tail model on *low-margin* clean examples (i.e. we model
    the lower tail by fitting EVT to -margin), returning a margin threshold
    below which an input is flagged as anomalous/adversarial."""
    margins = _clean_logit_margins(model, clean_loader, device)
    neg_margins = -margins  # low margin -> high "anomaly score"
    thresh_neg, fit_info = threshold_from_reference(neg_margins, tail_fraction, target_fpr, method)
    margin_threshold = -thresh_neg
    return margin_threshold, fit_info


@torch.no_grad()
def evt_flag_adversarial(model: nn.Module, x: torch.Tensor, margin_threshold: float,
                          device) -> np.ndarray:
    """Apply a fitted margin_threshold (from fit_evt_evasion_detector) to a
    batch of (possibly adversarial) inputs. Returns a boolean array: True
    where the input is flagged as anomalous/likely-adversarial."""
    model.eval()
    logits = model(x.to(device))
    top2 = torch.topk(logits, 2, dim=1).values
    margin = (top2[:, 0] - top2[:, 1]).cpu().numpy()
    return margin < margin_threshold


@torch.no_grad()
def logit_margins(model: nn.Module, x: torch.Tensor, device) -> np.ndarray:
    """Top-1 minus top-2 logit for every input in the batch."""
    model.eval()
    top2 = torch.topk(model(x.to(device)), 2, dim=1).values
    return (top2[:, 0] - top2[:, 1]).cpu().numpy()
