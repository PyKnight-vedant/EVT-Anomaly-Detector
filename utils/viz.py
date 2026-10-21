"""Step 17: Visualization — confusion matrices, ROC curves, robustness plots,
attack-success graphs, runtime comparisons."""
import os
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay

import config as cfg


def plot_confusion_matrix(cm: Sequence[Sequence[int]], class_names, title, filename):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path = os.path.join(cfg.FIGURES_DIR, filename)
    os.makedirs(cfg.FIGURES_DIR, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_roc_curve(y_true, y_score, title, filename):
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    path = os.path.join(cfg.FIGURES_DIR, filename)
    os.makedirs(cfg.FIGURES_DIR, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_robustness_curve(epsilons: List[float], accuracies: Dict[str, List[float]],
                           title, filename):
    """accuracies: {attack_name: [robust_acc_at_each_epsilon]}"""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, accs in accuracies.items():
        ax.plot(epsilons, accs, marker="o", label=name)
    ax.set_xlabel("Perturbation budget (epsilon)")
    ax.set_ylabel("Robust accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(cfg.FIGURES_DIR, filename)
    os.makedirs(cfg.FIGURES_DIR, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_bar_comparison(labels: List[str], values: List[float], ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, values, color="steelblue")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    path = os.path.join(cfg.FIGURES_DIR, filename)
    os.makedirs(cfg.FIGURES_DIR, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_training_curves(epoch_history: Dict[str, List[float]], title, filename):
    """Fig 2. epoch_history: {'train_loss': [...], 'val_acc': [...]} — same
    length lists, one entry per epoch. Requires train_baseline.py to record
    per-epoch history (see note in train_baseline.py) rather than only the
    final metrics."""
    epochs = range(1, len(next(iter(epoch_history.values()))) + 1)
    fig, ax1 = plt.subplots(figsize=(6, 5))
    ax2 = ax1.twinx()
    if "train_loss" in epoch_history:
        ax1.plot(epochs, epoch_history["train_loss"], color="firebrick", marker="o", label="Train loss")
        ax1.set_ylabel("Loss", color="firebrick")
    if "val_acc" in epoch_history:
        ax2.plot(epochs, epoch_history["val_acc"], color="steelblue", marker="s", label="Val accuracy")
        ax2.set_ylabel("Accuracy", color="steelblue")
    ax1.set_xlabel("Epoch")
    ax1.set_title(title)
    fig.tight_layout()
    path = os.path.join(cfg.FIGURES_DIR, filename)
    os.makedirs(cfg.FIGURES_DIR, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_image_grid(images: List["np.ndarray"], row_labels: List[str], col_labels: List[str],
                     title, filename):
    """Figs 6 & 8. images: flat list of length rows*cols, each a HxW or
    HxWx3 array in [0,1]. Arranged row-major: row_labels index the rows
    (e.g. sample index), col_labels index the columns (e.g. 'Clean',
    'FGSM', 'PGD', 'C&W' or 'Clean', 'Triggered')."""
    n_rows, n_cols = len(row_labels), len(col_labels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            img = images[i * n_cols + j]
            cmap = "gray" if img.ndim == 2 or img.shape[-1] == 1 else None
            ax.imshow(img.squeeze(), cmap=cmap)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(col_labels[j], fontsize=10)
            if j == 0:
                ax.set_ylabel(row_labels[i], fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    path = os.path.join(cfg.FIGURES_DIR, filename)
    os.makedirs(cfg.FIGURES_DIR, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_grouped_bar(group_labels: List[str], series: Dict[str, List[float]],
                      ylabel, title, filename):
    """Figs 10 & 13. series: {series_name: [value_per_group]}, all lists the
    same length as group_labels. E.g. for Fig 10:
    group_labels=['baseline','adv_train','dp_sgd','distillation'],
    series={'Clean accuracy': [...], 'Robust accuracy (PGD)': [...]}."""
    n_groups = len(group_labels)
    n_series = len(series)
    x = np.arange(n_groups)
    width = 0.8 / n_series
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (name, values) in enumerate(series.items()):
        ax.bar(x + i * width - 0.4 + width / 2, values, width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(cfg.FIGURES_DIR, filename)
    os.makedirs(cfg.FIGURES_DIR, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_privacy_utility_tradeoff(epsilons: List[float], accuracies: List[float],
                                   title, filename):
    """Fig 11. Requires a sweep over DP-SGD noise multipliers, each producing
    an (epsilon, clean_accuracy) pair — run defend_dp_sgd multiple times
    with different noise_multiplier values in config.py's DPSGDConfig."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(epsilons, accuracies, marker="o", color="darkorange")
    ax.set_xlabel("Privacy budget (\u03b5, lower = more private)")
    ax.set_ylabel("Clean test accuracy")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(cfg.FIGURES_DIR, filename)
    os.makedirs(cfg.FIGURES_DIR, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_gpd_tail_fit(scores: "np.ndarray", threshold: float, xi: float, sigma: float,
                       u: float, title, filename):
    """Fig 14. Histogram of raw scores (loss values or activation distances)
    with the fitted GPD tail density overlaid above u, and a vertical line
    at the derived detection threshold. scores/threshold/xi/sigma/u come
    straight out of defenses/evt_anomaly_detection.py's fit_gpd_threshold
    (pass back fit_info's 'xi', 'sigma', 'threshold_u')."""
    from scipy import stats as _stats
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(scores, bins=40, density=True, alpha=0.5, color="steelblue", label="Empirical scores")
    if xi is not None and sigma is not None:
        tail_x = np.linspace(u, scores.max(), 200)
        tail_density = _stats.genpareto.pdf(tail_x - u, xi, loc=0, scale=sigma)
        # Scale the tail density to match the fraction of mass above u.
        frac_above_u = (scores > u).mean()
        ax.plot(tail_x, tail_density * frac_above_u, color="firebrick", lw=2, label="Fitted GPD tail")
    ax.axvline(threshold, color="black", linestyle="--", label=f"EVT threshold = {threshold:.3f}")
    ax.axvline(u, color="gray", linestyle=":", label=f"POT base u = {u:.3f}")
    ax.set_xlabel("Score (loss or activation distance)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(cfg.FIGURES_DIR, filename)
    os.makedirs(cfg.FIGURES_DIR, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_detection_roc(fpr_values: List[float], detection_rates: List[float], title, filename):
    """Fig 15. Requires sweeping target_fpr through
    fit_evt_evasion_detector / defend_evt_evasion_detection and recording
    (false_positive_rate_on_clean, detection_rate_on_fgsm_adv) at each."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr_values, detection_rates, marker="o", color="seagreen")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False positive rate (on clean data)")
    ax.set_ylabel("Detection rate (on adversarial data)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(cfg.FIGURES_DIR, filename)
    os.makedirs(cfg.FIGURES_DIR, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_seed_error_bars(labels: List[str], means: List[float], stds: List[float],
                          ylabel, title, filename):
    """Fig 16. Requires each defense to be repeated across the full seed
    list (see the run_defenses.py seed-loop change discussed earlier) so
    means/stds are computed over >=5 runs, not a single seed."""
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=5, color="slateblue")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    path = os.path.join(cfg.FIGURES_DIR, filename)
    os.makedirs(cfg.FIGURES_DIR, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_runtime_comparison(labels: List[str], train_times: List[float],
                             infer_times: List[float], filename):
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, train_times, width, label="Training time (s)")
    ax.bar(x + width / 2, infer_times, width, label="Inference time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Seconds")
    ax.set_title("Runtime comparison")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(cfg.FIGURES_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path