"""Does fitting a GPD beat just taking the empirical quantile?

For a trained model we take two CLEAN score sets: a validation pool (the
reference the detector is calibrated on) and the held-out test set. For
several reference sizes n_ref and target FPRs alpha we repeatedly subsample
the pool, derive a threshold with (a) the POT/GPD method and (b) the plain
empirical (1 - alpha) quantile, and measure the REALIZED false-positive rate
on the held-out clean test scores. The method whose realized FPR stays
closer to alpha (especially at small alpha / small n_ref) is better calibrated.

Two scores are studied: negative logit margin (of correct predictions) and
per-sample cross-entropy loss.
"""
import argparse
import json
import os
import warnings

import numpy as np

import config as cfg
from defenses.evt_anomaly_detection import empirical_threshold, fit_gpd_threshold

ALPHAS = [0.001, 0.005, 0.01, 0.05]
REF_SIZES = [300, 500, 1000, 2000, None]   # None = the full validation pool
STATS = ("neg_margin", "loss")


def calibration_study(pool, test, alphas=ALPHAS, ref_sizes=REF_SIZES, tail_fraction=0.1,
                      n_reps=50, seed=0):
    """Returns {n_ref: {alpha: {method: {...}}}} for one score statistic."""
    pool, test = np.asarray(pool, float), np.asarray(test, float)
    rng = np.random.default_rng(seed)
    out = {}
    for n_ref in ref_sizes:
        n = len(pool) if n_ref is None else min(n_ref, len(pool))
        reps = 1 if n == len(pool) else n_reps
        out[str(n)] = {}
        for a in alphas:
            fp = {"gpd": [], "empirical": []}
            n_fallback = 0
            for _ in range(reps):
                ref = pool[rng.choice(len(pool), n, replace=False)] if reps > 1 else pool
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    t_g, info = fit_gpd_threshold(ref, tail_fraction, a)
                t_e = empirical_threshold(ref, a)
                fp["gpd"].append(float((test > t_g).mean()))
                fp["empirical"].append(float((test > t_e).mean()))
                n_fallback += info.get("method") != "gpd"
            res = {}
            for m, v in fp.items():
                v = np.asarray(v)
                res[m] = {"mean_fpr": float(v.mean()), "std_fpr": float(v.std()),
                          "mean_abs_rel_error": float(np.abs(v - a).mean() / a)}
            res["gpd"]["fallback_fraction"] = n_fallback / reps
            out[str(n)][str(a)] = res
    return out


def compute_scores(args, device):
    import torch
    from defenses.anomaly_detection import per_sample_loss
    from defenses.evt_anomaly_detection import _clean_logit_margins
    from models.factory import build_model
    from train_baseline import train_one_model
    from utils.data import get_dataloaders

    _, val_loader, test_loader = get_dataloaders(args.dataset, args.batch_size, seed=args.seed)
    if os.path.exists(args.checkpoint):
        model = build_model(args.model, args.dataset).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint {args.checkpoint}")
    else:
        print("No checkpoint found -> training one.")
        model, _ = train_one_model(args.dataset, args.model, args.epochs, 1e-3, args.batch_size,
                                   args.seed, device=device, quiet=True)
    model.eval()
    return {
        "neg_margin": (-_clean_logit_margins(model, val_loader, device),
                       -_clean_logit_margins(model, test_loader, device)),
        "loss": (per_sample_loss(model, val_loader, device),
                 per_sample_loss(model, test_loader, device)),
    }


def plot(results, dataset):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(STATS), 2, figsize=(8, 6), sharex=True)
    for r, stat in enumerate(STATS):
        for c, a in enumerate(("0.001", "0.01")):
            ax = axes[r][c]
            sizes = sorted(int(n) for n in results[stat])
            for m, col in (("gpd", "firebrick"), ("empirical", "steelblue")):
                mu = np.array([results[stat][str(n)][a][m]["mean_fpr"] for n in sizes]) / float(a)
                sd = np.array([results[stat][str(n)][a][m]["std_fpr"] for n in sizes]) / float(a)
                ax.plot(sizes, mu, marker="o", color=col, label=m)
                ax.fill_between(sizes, mu - sd, mu + sd, color=col, alpha=0.15)
            ax.axhline(1.0, color="k", ls="--", lw=1)
            ax.set_xscale("log")
            ax.set_title(f"{stat}, target FPR = {a}", fontsize=9)
            if c == 0:
                ax.set_ylabel("realized FPR / target")
            if r == len(STATS) - 1:
                ax.set_xlabel("reference set size $n_{ref}$")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(cfg.FIGURES_DIR, f"evt_calibration_{dataset}.png")
    fig.savefig(path, dpi=150)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist", choices=list(cfg.DATASET_INFO.keys()))
    parser.add_argument("--model", default="cnn", choices=["cnn", "resnet18"])
    parser.add_argument("--checkpoint", default=None,
                        help="Baseline checkpoint (default: results/checkpoints/baseline_<ds>_<model>.pt)")
    parser.add_argument("--epochs", type=int, default=10, help="Only used if no checkpoint exists.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=cfg.DEFAULT_SEED,
                        help="Must match the seed the checkpoint was trained with (data split).")
    parser.add_argument("--reps", type=int, default=50)
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = os.path.join(cfg.CHECKPOINT_DIR, f"baseline_{args.dataset}_{args.model}.pt")

    from utils.seed import get_device, set_seed
    set_seed(args.seed)
    device = get_device()
    scores = compute_scores(args, device)

    results = {}
    for stat, (pool, test) in scores.items():
        print(f"\n[{stat}] pool={len(pool)} test={len(test)}")
        results[stat] = calibration_study(pool, test, n_reps=args.reps, seed=args.seed)
        for n, by_a in results[stat].items():
            for a, r in by_a.items():
                print(f"  n_ref={n:>6} alpha={a:<6} realized FPR  gpd={r['gpd']['mean_fpr']:.4f}"
                      f"  emp={r['empirical']['mean_fpr']:.4f}  "
                      f"(rel.err gpd={r['gpd']['mean_abs_rel_error']:.2f} "
                      f"emp={r['empirical']['mean_abs_rel_error']:.2f}, "
                      f"gpd fallback={r['gpd']['fallback_fraction']:.2f})")

    out = os.path.join(cfg.RESULTS_DIR, f"evt_calibration_{args.dataset}_{args.model}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out}\nFigure -> {plot(results, args.dataset)}")


if __name__ == "__main__":
    main()
