"""Steps 15-17: Comparative evaluation across attacks/defenses, repeated over
multiple seeds with statistical testing, plus figure generation."""
import argparse
import json
import os

import numpy as np
import torch

import config as cfg
from attacks.fgsm import fgsm_attack
from models.factory import build_model
from train_baseline import train_one_model
from utils.data import get_dataloaders
from utils.metrics import evaluate_classifier, robust_accuracy
from utils.seed import get_device, log_manifest, set_seed
from utils.stats import (format_mean_std, one_way_anova, paired_comparison,
                         paired_t_test, summarize_runs)
from utils.viz import (plot_bar_comparison, plot_confusion_matrix,
                        plot_robustness_curve, plot_runtime_comparison,
                        plot_roc_curve, plot_training_curves, plot_image_grid,
                        plot_grouped_bar, plot_privacy_utility_tradeoff,
                        plot_gpd_tail_fit, plot_detection_roc, plot_seed_error_bars)


def repeated_baseline_runs(dataset, model_name, epochs, batch_size, seeds, device):
    """Step 16: repeat baseline training across seeds, collect metrics."""
    accs, f1s, train_times, infer_times = [], [], [], []
    last_cm, class_names = None, [str(i) for i in range(cfg.DATASET_INFO[dataset]["num_classes"])]

    for seed in seeds:
        print(f"  seed={seed} ...")
        _, metrics = train_one_model(dataset, model_name, epochs, 1e-3, batch_size,
                                      seed, device=device, quiet=True)
        accs.append(metrics["accuracy"])
        f1s.append(metrics["f1_macro"])
        train_times.append(metrics["training_time_sec"])
        infer_times.append(metrics["inference_time_sec"])
        last_cm = metrics["confusion_matrix"]

    return {
        "accuracy": summarize_runs(accs),
        "f1_macro": summarize_runs(f1s),
        "training_time_sec": summarize_runs(train_times),
        "inference_time_sec": summarize_runs(infer_times),
        "last_confusion_matrix": last_cm,
        "class_names": class_names,
    }


def robustness_across_epsilons(model, test_loader, device, epsilons, max_batches=20):
    accs = []
    for eps in epsilons:
        y_true_all, y_adv_all = [], []
        for i, (x, y) in enumerate(test_loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            x_adv = fgsm_attack(model, x, y, eps)
            preds = model(x_adv).argmax(1)
            y_true_all.append(y.cpu().numpy())
            y_adv_all.append(preds.cpu().numpy())
        y_true = np.concatenate(y_true_all)
        y_adv = np.concatenate(y_adv_all)
        accs.append(robust_accuracy(y_true, y_adv))
    return accs


def _load_multiseed(args):
    path = os.path.join(cfg.RESULTS_DIR, f"defenses_multiseed_{args.dataset}_{args.model}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def make_defense_figures(args):
    """GPD tail fit, detection ROC, privacy-utility frontier and seed error bars,
    all drawn from files written by run_defenses.py."""
    ds, mdl = args.dataset, args.model
    ms = _load_multiseed(args)
    if ms is None:
        print("  (no defenses_multiseed json -> skipping defense figures; run run_defenses.py)")
        return
    seeds = ms["seeds"]
    first = ms["per_seed"][str(seeds[0])]

    npz = os.path.join(cfg.RESULTS_DIR, f"evt_scores_{ds}_{mdl}_seed{seeds[0]}.npz")
    if os.path.exists(npz):
        d = np.load(npz)
        plot_gpd_tail_fit(d["scores"], float(d["threshold"]), float(d["xi"]), float(d["sigma"]),
                          float(d["u"]), f"GPD tail fit (-logit margin) - {ds}", f"gpd_tail_{ds}.png")

    roc = first["evt_evasion_detection"]["roc"]["fgsm_eps0.1"]
    plot_detection_roc(roc["fpr"], roc["tpr"], f"EVT-margin detector ROC (FGSM eps=0.1) - {ds}",
                       f"detection_roc_{ds}.png")

    dp_path = os.path.join(cfg.RESULTS_DIR, f"dp_sweep_{ds}_{mdl}.json")
    if os.path.exists(dp_path):
        with open(dp_path) as f:
            pts = sorted(json.load(f), key=lambda p: p["epsilon"])
        plot_privacy_utility_tradeoff([p["epsilon"] for p in pts], [p["clean_accuracy"] for p in pts],
                                      f"Privacy-utility - {ds}", f"privacy_utility_{ds}.png")

    def col(fn):
        return [fn(ms["per_seed"][str(s)]) for s in seeds]
    arms = {
        "undefended": col(lambda r: r["evt_evasion_detection"]["baseline_clean_accuracy"]),
        "adv. training": col(lambda r: r["adversarial_training"]["clean_accuracy"]),
        "distillation": col(lambda r: r["distillation"]["clean_accuracy"]),
        "DP-SGD": col(lambda r: r["dp_sgd"]["clean_accuracy"]),
    }
    plot_seed_error_bars(list(arms), [float(np.mean(v)) for v in arms.values()],
                         [float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for v in arms.values()],
                         "Clean test accuracy", f"Clean accuracy over {len(seeds)} seed(s) - {ds}",
                         f"seed_error_bars_{ds}.png")


def defense_statistics(ms):
    """Per-seed summaries, paired t-tests (+ Cohen's d, 95% CI) and one-way ANOVAs."""
    seeds = ms["seeds"]
    if len(seeds) < 2:
        return {"note": "Need >= 2 seeds for t-tests/ANOVA; re-run run_defenses.py --seeds 5."}
    per = ms["per_seed"]

    def col(fn):
        return [fn(per[str(s)]) for s in seeds]
    ev = lambda r: r["evt_evasion_detection"]
    ad = lambda r: r["anomaly_detection"]

    base_c = col(lambda r: ev(r)["baseline_clean_accuracy"])
    base_r = col(lambda r: ev(r)["baseline_robust_accuracy_pgd"])
    adv_c = col(lambda r: r["adversarial_training"]["clean_accuracy"])
    adv_r = col(lambda r: r["adversarial_training"]["robust_accuracy_pgd"])
    dis_c = col(lambda r: r["distillation"]["clean_accuracy"])
    dis_r = col(lambda r: r["distillation"]["robust_accuracy_pgd_eps0.1"])
    dp_c = col(lambda r: r["dp_sgd"]["clean_accuracy"])

    out = {"seeds": seeds, "epochs": ms.get("epochs")}
    out["summary"] = {k: summarize_runs(v) for k, v in {
        "baseline_clean_acc": base_c, "baseline_robust_acc_pgd": base_r,
        "advtrain_clean_acc": adv_c, "advtrain_robust_acc_pgd": adv_r,
        "distill_clean_acc": dis_c, "distill_robust_acc_pgd": dis_r, "dpsgd_clean_acc": dp_c,
        "dpsgd_epsilon": col(lambda r: r["dp_sgd"]["epsilon"]),
    }.items()}

    out["paired_tests"] = {
        "clean_acc: baseline vs adversarial_training": paired_comparison(base_c, adv_c),
        "clean_acc: baseline vs distillation": paired_comparison(base_c, dis_c),
        "clean_acc: baseline vs dp_sgd": paired_comparison(base_c, dp_c),
        "robust_acc_pgd: baseline vs adversarial_training": paired_comparison(base_r, adv_r),
        "robust_acc_pgd: baseline vs distillation": paired_comparison(base_r, dis_r),
    }
    out["anova"] = {
        "clean_acc across baseline/advtrain/distill/dpsgd": one_way_anova(base_c, adv_c, dis_c, dp_c),
        "label_flip recall: zscore/evt/empirical": one_way_anova(
            col(lambda r: ad(r)["label_flip_detection"]["recall"]),
            col(lambda r: ad(r)["label_flip_detection_evt"]["recall"]),
            col(lambda r: ad(r)["label_flip_detection_empirical"]["recall"])),
        "backdoor recall: clustering/evt/empirical": one_way_anova(
            col(lambda r: ad(r)["backdoor_detection"]["recall"]),
            col(lambda r: ad(r)["backdoor_detection_evt"]["recall"]),
            col(lambda r: ad(r)["backdoor_detection_empirical"]["recall"])),
    }
    # Paired EVT-vs-empirical tests (same seeds, same data, same alpha=0.01).
    out["paired_tests"]["label_flip recall: evt vs empirical"] = paired_comparison(
        col(lambda r: ad(r)["label_flip_detection_evt"]["recall"]),
        col(lambda r: ad(r)["label_flip_detection_empirical"]["recall"]))
    out["paired_tests"]["backdoor recall: evt vs empirical"] = paired_comparison(
        col(lambda r: ad(r)["backdoor_detection_evt"]["recall"]),
        col(lambda r: ad(r)["backdoor_detection_empirical"]["recall"]))

    det = {}
    for k in ("label_flip_detection", "label_flip_detection_evt", "label_flip_detection_empirical",
              "label_flip_detection_evt_uncalibrated", "backdoor_detection",
              "backdoor_detection_evt", "backdoor_detection_empirical",
              "backdoor_detection_evt_uncalibrated"):
        det[k] = {"precision": summarize_runs(col(lambda r, k=k: ad(r)[k]["precision"])),
                  "recall": summarize_runs(col(lambda r, k=k: ad(r)[k]["recall"]))}
    out["anomaly_detection_summary"] = det
    ev_sum = {"false_positive_rate_on_clean": summarize_runs(col(lambda r: ev(r)["false_positive_rate_on_clean"]))}
    for atk in ev(per[str(seeds[0])])["detection_auc"]:
        ev_sum[f"auc_{atk}"] = summarize_runs(col(lambda r, a=atk: ev(r)["detection_auc"][a]))
        ev_sum[f"detection_rate_{atk}"] = summarize_runs(
            col(lambda r, a=atk: ev(r)["alpha_sweep"]["0.01"]["gpd"][f"detection_rate_{a}"]))
    out["evasion_detector_summary"] = ev_sum
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist", choices=list(cfg.DATASET_INFO.keys()))
    parser.add_argument("--model", default="cnn", choices=["cnn", "resnet18"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seeds", type=int, default=5,
                         help="Number of seeds to repeat over (uses cfg.SEEDS[:n]).")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    seeds = cfg.SEEDS[: args.seeds]

    print(f"\n[Step 16] Repeating baseline training over {len(seeds)} seeds")
    baseline_summary = repeated_baseline_runs(
        args.dataset, args.model, args.epochs, args.batch_size, seeds, device)

    for metric in ("accuracy", "f1_macro", "training_time_sec", "inference_time_sec"):
        print(f"  {metric}: {format_mean_std(baseline_summary[metric])}")

    print("\n[Step 17] Generating figures")
    plot_confusion_matrix(
        baseline_summary["last_confusion_matrix"], baseline_summary["class_names"],
        f"Confusion Matrix - {args.dataset} baseline", f"confusion_matrix_{args.dataset}.png")

    # Robustness curve on the *last* trained model as a representative example.
    set_seed(seeds[-1])
    _, val_loader, test_loader = get_dataloaders(args.dataset, args.batch_size, seed=seeds[-1])
    model, _ = train_one_model(args.dataset, args.model, args.epochs, 1e-3,
                                args.batch_size, seeds[-1], device=device, quiet=True)
    fgsm_eps = cfg.FGSMConfig().epsilons
    robust_accs = robustness_across_epsilons(model, test_loader, device, fgsm_eps)
    plot_robustness_curve(fgsm_eps, {"FGSM": robust_accs},
                           f"Robustness vs. epsilon - {args.dataset}",
                           f"robustness_curve_{args.dataset}.png")

    plot_bar_comparison(
        ["clean"] + [f"eps={e}" for e in fgsm_eps],
        [baseline_summary["accuracy"]["mean"]] + robust_accs,
        "Accuracy", f"Clean vs adversarial accuracy - {args.dataset}",
        f"attack_success_bar_{args.dataset}.png")

    plot_runtime_comparison(
        ["baseline"],
        [baseline_summary["training_time_sec"]["mean"]],
        [baseline_summary["inference_time_sec"]["mean"]],
        f"runtime_comparison_{args.dataset}.png")

    # Additional figures: exercise all plotting functions so they generate images
    # ROC curve (binary vs rest using class-0 probability)
    y_true_list, y_score_list = [], []
    max_batches = 20
    for i, (x, y) in enumerate(test_loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        y_true_list.append(y.cpu().numpy())
        y_score_list.append(probs[:, 0])
    if y_true_list:
        y_true = np.concatenate(y_true_list)
        y_score = np.concatenate(y_score_list)
        plot_roc_curve(y_true == 0, y_score,
                       f"ROC (class0 vs rest) - {args.dataset}",
                       f"roc_curve_{args.dataset}.png")

    # Image grid: first few samples clean vs FGSM
    x_sample, y_sample = None, None
    for xb, yb in test_loader:
        x_sample = xb[:4].cpu().numpy()
        y_sample = yb[:4].cpu().numpy()
        break
    if x_sample is not None:
        eps = fgsm_eps[0] if len(fgsm_eps) > 0 else 0.1
        x_adv_sample = fgsm_attack(model, torch.from_numpy(x_sample).to(device),
                                   torch.from_numpy(y_sample).to(device), eps).cpu().numpy()
        images = []
        for i in range(len(x_sample)):
            clean = x_sample[i]
            adv = x_adv_sample[i]
            # convert from CHW to HWC when needed
            if clean.ndim == 3 and clean.shape[0] in (1, 3):
                clean = np.transpose(clean, (1, 2, 0))
            if adv.ndim == 3 and adv.shape[0] in (1, 3):
                adv = np.transpose(adv, (1, 2, 0))
            images.extend([clean.squeeze(), adv.squeeze()])
        row_labels = [str(i) for i in range(len(x_sample))]
        col_labels = ["Clean", "FGSM"]
        plot_image_grid(images, row_labels, col_labels,
                        f"Image grid - {args.dataset}", f"image_grid_{args.dataset}.png")

    # Figures below use REAL outputs of run_defenses.py (nothing synthetic).
    make_defense_figures(args)

    # If attack/defense result files already exist from run_attacks.py /
    # run_defenses.py, fold them into a combined comparative summary + stats.
    combined = {"baseline": baseline_summary}
    attacks_path = os.path.join(cfg.RESULTS_DIR, f"attacks_{args.dataset}_{args.model}.json")
    defenses_path = os.path.join(cfg.RESULTS_DIR, f"defenses_{args.dataset}_{args.model}.json")
    if os.path.exists(attacks_path):
        with open(attacks_path) as f:
            combined["attacks"] = json.load(f)
    if os.path.exists(defenses_path):
        with open(defenses_path) as f:
            combined["defenses"] = json.load(f)

    # Step 16: paired t-tests / ANOVA over per-seed defense results.
    ms_path = os.path.join(cfg.RESULTS_DIR, f"defenses_multiseed_{args.dataset}_{args.model}.json")
    if os.path.exists(ms_path):
        with open(ms_path) as f:
            combined["defense_stats"] = defense_statistics(json.load(f))
    else:
        combined["stats_note"] = ("No defenses_multiseed_*.json found; run "
                                  "`python run_defenses.py --seeds 5` first.")

    out_path = os.path.join(cfg.RESULTS_DIR, f"comparative_summary_{args.dataset}_{args.model}.json")
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)

    log_manifest(cfg.RESULTS_DIR, {"stage": "run_comparative_eval", "args": vars(args)})
    print(f"\nSaved comparative summary -> {out_path}")
    print(f"Figures saved under -> {cfg.FIGURES_DIR}")


if __name__ == "__main__":
    main()
