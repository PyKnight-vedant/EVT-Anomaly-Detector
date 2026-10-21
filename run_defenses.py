"""Step 14 driver: adversarial training, DP-SGD, distillation, anomaly
detection — each evaluated against the relevant attack from run_attacks.py."""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

import config as cfg
from attacks.backdoor import BackdoorDataset, stamp_trigger_on_batch
from attacks.carlini_wagner import cw_l2_attack
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from attacks.poisoning import LabelFlippedDataset
from defenses.adversarial_training import adversarial_train
from defenses.anomaly_detection import (activation_clustering_filter,
                                         loss_based_filter,
                                         precision_recall_of_filter)
from defenses.distillation import defensive_distillation
from defenses.dp_sgd import dp_sgd_train
from defenses.evt_anomaly_detection import (_clean_logit_margins, empirical_threshold,
                                             evt_activation_filter, evt_loss_filter,
                                             fit_gpd_threshold, logit_margins)
from models.factory import build_model
from utils.data import denormalize, get_dataloaders, normalize
from utils.metrics import evaluate_classifier, robust_accuracy, timer
from utils.seed import get_device, log_manifest, set_seed


# Target false-positive rates swept for the EVT vs. empirical-quantile comparison.
ALPHAS = [0.001, 0.005, 0.01, 0.05]


def eval_robust_accuracy(model, loader, device, epsilon, alpha, iters, max_batches=20):
    import numpy as np
    y_true_all, y_adv_all = [], []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        x_adv = pgd_attack(model, x, y, epsilon, alpha, iters)
        preds = model(x_adv).argmax(1)
        y_true_all.append(y.cpu().numpy())
        y_adv_all.append(preds.cpu().numpy())
    y_true = np.concatenate(y_true_all)
    y_adv = np.concatenate(y_adv_all)
    return robust_accuracy(y_true, y_adv)


def defend_adversarial_training(dataset, model_name, epochs, batch_size, seed, device):
    at_cfg = cfg.AdversarialTrainingConfig()
    set_seed(seed)
    train_loader, val_loader, test_loader = get_dataloaders(dataset, batch_size, seed=seed)
    model = build_model(model_name, dataset).to(device)

    with timer() as t:
        model = adversarial_train(model, train_loader, device, at_cfg.epsilon,
                                   at_cfg.alpha, at_cfg.iters, epochs)

    clean_metrics = evaluate_classifier(model, test_loader, device)
    robust_acc = eval_robust_accuracy(model, test_loader, device, at_cfg.epsilon,
                                       at_cfg.alpha, at_cfg.iters)
    result = {
        "clean_accuracy": clean_metrics["accuracy"],
        "robust_accuracy_pgd": robust_acc,
        "training_time_sec": t["seconds"],
    }
    print(f"  Adversarial training: clean_acc={result['clean_accuracy']:.3f} "
          f"robust_acc={robust_acc:.3f}")
    return result


def defend_dp_sgd(dataset, model_name, epochs, batch_size, seed, device, noise_multiplier=None):
    dp_cfg = cfg.DPSGDConfig()
    nm = dp_cfg.noise_multiplier if noise_multiplier is None else noise_multiplier
    set_seed(seed)
    train_loader, val_loader, test_loader = get_dataloaders(dataset, batch_size, seed=seed)
    model = build_model(model_name, dataset).to(device)

    with timer() as t:
        model, privacy_stats = dp_sgd_train(
            model, train_loader, device, epochs,
            nm, dp_cfg.max_grad_norm, dp_cfg.target_delta)

    clean_metrics = evaluate_classifier(model, test_loader, device)
    result = {
        "clean_accuracy": clean_metrics["accuracy"],
        "noise_multiplier": nm,
        "epsilon": privacy_stats["epsilon"],
        "delta": privacy_stats["delta"],
        "training_time_sec": t["seconds"],
    }
    print(f"  DP-SGD: clean_acc={result['clean_accuracy']:.3f} "
          f"(eps={privacy_stats['epsilon']:.2f}, delta={privacy_stats['delta']})")
    return result


def defend_distillation(dataset, model_name, epochs, batch_size, seed, device):
    dist_cfg = cfg.DistillationConfig()
    set_seed(seed)
    train_loader, val_loader, test_loader = get_dataloaders(dataset, batch_size, seed=seed)
    teacher = build_model(model_name, dataset).to(device)
    student = build_model(model_name, dataset).to(device)

    with timer() as t:
        teacher, student = defensive_distillation(
            teacher, student, train_loader, device, dist_cfg.temperature, epochs)

    clean_metrics = evaluate_classifier(student, test_loader, device)
    robust_acc = eval_robust_accuracy(student, test_loader, device, 0.1, 0.01, 10)
    result = {
        "clean_accuracy": clean_metrics["accuracy"],
        "robust_accuracy_pgd_eps0.1": robust_acc,
        "training_time_sec": t["seconds"],
    }
    print(f"  Distillation: clean_acc={result['clean_accuracy']:.3f} robust_acc={robust_acc:.3f}")
    return result


def defend_anomaly_detection(dataset, model_name, epochs, batch_size, seed, device):
    """Trains a quick model on poisoned data, then measures how well the
    ad hoc filters and the EVT / empirical-quantile filters recover the true
    poisoned indices. EVT and empirical thresholds are calibrated on the CLEAN
    validation split (val_loader) and then applied to the poisoned train set."""
    info = cfg.DATASET_INFO[dataset]
    poison_cfg = cfg.PoisonConfig()
    backdoor_cfg = cfg.BackdoorConfig()
    set_seed(seed)
    train_loader, val_loader, test_loader = get_dataloaders(dataset, batch_size, seed=seed)

    results = {}
    TAIL = 0.1

    def pr_of(flagged, truth, n):
        return precision_recall_of_filter(flagged, truth, n)

    # --- Label-flip poisoning detection ---
    frac = poison_cfg.flip_fractions[1]  # 10%
    poisoned_train = LabelFlippedDataset(train_loader.dataset, frac, info["num_classes"], seed=seed)
    poisoned_loader = DataLoader(poisoned_train, batch_size=batch_size, shuffle=False)

    model = build_model(model_name, dataset).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(max(epochs // 2, 3)):
        for x, y in poisoned_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    n = len(poisoned_train)
    truth = poisoned_train.poisoned_indices
    flagged = loss_based_filter(model, poisoned_loader, device)
    results["label_flip_detection"] = pr_of(flagged, truth, n)
    print(f"  Label-flip, z-score:  precision={results['label_flip_detection']['precision']:.3f} "
          f"recall={results['label_flip_detection']['recall']:.3f}")

    sweep = {}
    for a_ in ALPHAS:
        f_g, info_g = evt_loss_filter(model, poisoned_loader, device, TAIL, a_,
                                       ref_loader=val_loader, method="gpd")
        f_e, _ = evt_loss_filter(model, poisoned_loader, device, TAIL, a_,
                                  ref_loader=val_loader, method="empirical")
        sweep[str(a_)] = {"gpd": {**pr_of(f_g, truth, n), "gpd_fit": info_g},
                          "empirical": pr_of(f_e, truth, n)}
    results["label_flip_detection_evt"] = sweep["0.01"]["gpd"]
    results["label_flip_detection_empirical"] = sweep["0.01"]["empirical"]
    results["label_flip_alpha_sweep"] = sweep
    # Ablation: the old behaviour (fit on the contaminated training scores).
    f_c, info_c = evt_loss_filter(model, poisoned_loader, device, TAIL, 0.01, ref_loader=None)
    results["label_flip_detection_evt_uncalibrated"] = {**pr_of(f_c, truth, n), "gpd_fit": info_c}
    for k in ("label_flip_detection_evt", "label_flip_detection_empirical",
              "label_flip_detection_evt_uncalibrated"):
        print(f"  Label-flip, {k.split('detection_')[1]:<22} precision={results[k]['precision']:.3f} "
              f"recall={results[k]['recall']:.3f}")

    # --- Backdoor detection ---
    denorm_fn = lambda x: denormalize(x, dataset)
    norm_fn = lambda x: normalize(x, dataset)
    backdoor_train = BackdoorDataset(train_loader.dataset, denorm_fn, norm_fn,
                                      backdoor_cfg.poison_fraction, backdoor_cfg.target_class,
                                      backdoor_cfg.trigger_size, seed=seed)
    backdoor_loader = DataLoader(backdoor_train, batch_size=batch_size, shuffle=False)

    bd_model = build_model(model_name, dataset).to(device)
    optimizer = torch.optim.Adam(bd_model.parameters(), lr=1e-3)
    bd_model.train()
    for _ in range(max(epochs // 2, 3)):
        for x, y in backdoor_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(bd_model(x), y)
            loss.backward()
            optimizer.step()

    n_bd = len(backdoor_train)
    truth_bd = backdoor_train.poisoned_indices
    flagged_bd = activation_clustering_filter(bd_model, backdoor_loader, device)
    results["backdoor_detection"] = pr_of(flagged_bd, truth_bd, n_bd)
    print(f"  Backdoor, clustering: precision={results['backdoor_detection']['precision']:.3f} "
          f"recall={results['backdoor_detection']['recall']:.3f}")

    sweep_bd = {}
    for a_ in ALPHAS:
        f_g, info_g = evt_activation_filter(bd_model, backdoor_loader, device, TAIL, a_,
                                             ref_loader=val_loader, method="gpd")
        f_e, _ = evt_activation_filter(bd_model, backdoor_loader, device, TAIL, a_,
                                        ref_loader=val_loader, method="empirical")
        sweep_bd[str(a_)] = {"gpd": {**pr_of(f_g, truth_bd, n_bd), "gpd_fit_per_class": info_g},
                             "empirical": pr_of(f_e, truth_bd, n_bd)}
    results["backdoor_detection_evt"] = sweep_bd["0.01"]["gpd"]
    results["backdoor_detection_empirical"] = sweep_bd["0.01"]["empirical"]
    results["backdoor_alpha_sweep"] = sweep_bd
    f_c, info_c = evt_activation_filter(bd_model, backdoor_loader, device, TAIL, 0.01, ref_loader=None)
    results["backdoor_detection_evt_uncalibrated"] = {**pr_of(f_c, truth_bd, n_bd),
                                                      "gpd_fit_per_class": info_c}
    for k in ("backdoor_detection_evt", "backdoor_detection_empirical",
              "backdoor_detection_evt_uncalibrated"):
        print(f"  Backdoor, {k.split('detection_')[1]:<22} precision={results[k]['precision']:.3f} "
              f"recall={results[k]['recall']:.3f}")

    return results


def defend_evt_evasion_detection(dataset, model_name, epochs, batch_size, seed, device,
                                  tail_fraction=0.1, target_fpr=0.01, cw_batches=5):
    """Trains a plain (undefended) model, fits margin thresholds on CLEAN
    validation data (GPD and empirical-quantile), and measures the realized
    FPR on clean test data and the detection rate on several evasion attacks.
    The plain model's clean / PGD-robust accuracy is also returned and serves
    as the same-seed, same-epochs baseline arm for the paired t-tests."""
    at_cfg = cfg.AdversarialTrainingConfig()
    cw_cfg = cfg.CWConfig()
    set_seed(seed)
    train_loader, val_loader, test_loader = get_dataloaders(dataset, batch_size, seed=seed)

    model = build_model(model_name, dataset).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    clean_acc = evaluate_classifier(model, test_loader, device)["accuracy"]
    robust_acc = eval_robust_accuracy(model, test_loader, device, at_cfg.epsilon,
                                      at_cfg.alpha, at_cfg.iters)

    # Margins of correctly classified CLEAN validation data (the reference set).
    ref_neg = -_clean_logit_margins(model, val_loader, device)

    attacks = {
        "fgsm_eps0.1": (lambda x, y: fgsm_attack(model, x, y, 0.1), None),
        "fgsm_eps0.2": (lambda x, y: fgsm_attack(model, x, y, 0.2), None),
        "pgd_eps0.2_it40": (lambda x, y: pgd_attack(model, x, y, 0.2, 0.01, 40), None),
        "cw_l2": (lambda x, y: cw_l2_attack(
            model, x, y, clamp_min=-3.0, clamp_max=3.0, c=cw_cfg.c, kappa=cw_cfg.kappa,
            steps=cw_cfg.steps, lr=cw_cfg.lr, binary_search_steps=cw_cfg.binary_search_steps,
            device=device), cw_batches),  # C&W is slow: only the first few batches
    }
    margins = {"clean": []}
    for name in attacks:
        margins[name] = []
    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        margins["clean"].append(logit_margins(model, x, device))
        for name, (fn, max_b) in attacks.items():
            if max_b is not None and i >= max_b:
                continue
            margins[name].append(logit_margins(model, fn(x, y), device))
    margins = {k: np.concatenate(v) for k, v in margins.items()}

    def rates(thr):
        out = {"false_positive_rate_on_clean": float((margins["clean"] < thr).mean())}
        for name in attacks:
            out[f"detection_rate_{name}"] = float((margins[name] < thr).mean())
        return out

    sweep = {}
    for a_ in ALPHAS:
        t_neg, fit_info = fit_gpd_threshold(ref_neg, tail_fraction, a_)
        thr_g = -t_neg
        thr_e = -empirical_threshold(ref_neg, a_)
        sweep[str(a_)] = {"gpd": {"margin_threshold": thr_g, "gpd_fit": fit_info, **rates(thr_g)},
                          "empirical": {"margin_threshold": thr_e, **rates(thr_e)}}

    main_pt = sweep[str(target_fpr)]["gpd"]
    result = {
        # Backwards-compatible keys (alpha = target_fpr, GPD, FGSM eps = 0.1)
        "margin_threshold": main_pt["margin_threshold"],
        "gpd_fit": main_pt["gpd_fit"],
        "false_positive_rate_on_clean": main_pt["false_positive_rate_on_clean"],
        "detection_rate_on_fgsm_adv": main_pt["detection_rate_fgsm_eps0.1"],
        "fgsm_epsilon_used": 0.1,
        "alpha_sweep": sweep,
        "baseline_clean_accuracy": clean_acc,
        "baseline_robust_accuracy_pgd": robust_acc,
        "n_ref": int(len(ref_neg)),
    }

    # Threshold-free separability (AUC) and a subsampled ROC per attack.
    result["detection_auc"], result["roc"] = {}, {}
    for name in attacks:
        y_true = np.r_[np.zeros(len(margins["clean"])), np.ones(len(margins[name]))]
        score = -np.r_[margins["clean"], margins[name]]
        result["detection_auc"][name] = float(roc_auc_score(y_true, score))
        fpr, tpr, _ = roc_curve(y_true, score)
        keep = np.unique(np.linspace(0, len(fpr) - 1, 200).astype(int))
        result["roc"][name] = {"fpr": fpr[keep].tolist(), "tpr": tpr[keep].tolist()}

    # Real scores + fit for the GPD-tail figure (see run_comparative_eval.py).
    t_neg, fit_info = fit_gpd_threshold(ref_neg, tail_fraction, target_fpr)
    np.savez(os.path.join(cfg.RESULTS_DIR, f"evt_scores_{dataset}_{model_name}_seed{seed}.npz"),
             scores=ref_neg, threshold=t_neg, u=fit_info["threshold_u"],
             xi=fit_info.get("xi", np.nan), sigma=fit_info.get("sigma", np.nan))

    print(f"  Plain model: clean_acc={clean_acc:.3f} robust_acc(PGD)={robust_acc:.3f}")
    print(f"  EVT evasion detector (alpha={target_fpr}): FPR(clean)="
          f"{result['false_positive_rate_on_clean']:.3f}")
    for name in attacks:
        print(f"    {name:<16} detection={main_pt['detection_rate_' + name]:.3f}  "
              f"AUC={result['detection_auc'][name]:.3f}")
    return result


def run_all_defenses(args, device, seed):
    r = {}
    print("\n[Step 14a] Adversarial training")
    r["adversarial_training"] = defend_adversarial_training(
        args.dataset, args.model, args.epochs, args.batch_size, seed, device)
    print("\n[Step 14b] DP-SGD (Opacus)")
    r["dp_sgd"] = defend_dp_sgd(args.dataset, args.model, args.epochs, args.batch_size, seed, device)
    print("\n[Step 14c] Defensive distillation")
    r["distillation"] = defend_distillation(
        args.dataset, args.model, args.epochs, args.batch_size, seed, device)
    print("\n[Step 14d] Anomaly detection (z-score/clustering vs. EVT vs. empirical quantile)")
    r["anomaly_detection"] = defend_anomaly_detection(
        args.dataset, args.model, args.epochs, args.batch_size, seed, device)
    print("\n[Step 14e] EVT evasion detector + undefended baseline arm")
    r["evt_evasion_detection"] = defend_evt_evasion_detection(
        args.dataset, args.model, args.epochs, args.batch_size, seed, device)
    return r


def run_dp_sweep(args, device, seed, multipliers=(0.5, 0.75, 1.0, 1.5, 2.0)):
    """Real privacy-utility frontier: one DP-SGD run per noise multiplier."""
    points = []
    for nm in multipliers:
        r = defend_dp_sgd(args.dataset, args.model, args.epochs, args.batch_size, seed, device,
                          noise_multiplier=nm)
        points.append({"noise_multiplier": nm, "epsilon": r["epsilon"],
                       "clean_accuracy": r["clean_accuracy"]})
    return points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist", choices=list(cfg.DATASET_INFO.keys()))
    parser.add_argument("--model", default="cnn", choices=["cnn", "resnet18"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=cfg.DEFAULT_SEED,
                         help="Used only when --seeds 1.")
    parser.add_argument("--seeds", type=int, default=1,
                         help="Repeat every defense over cfg.SEEDS[:n] (needed for t-tests/ANOVA).")
    parser.add_argument("--dp_sweep", action="store_true",
                         help="Also sweep DP-SGD noise multipliers for the privacy-utility figure.")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    seeds = [args.seed] if args.seeds == 1 else cfg.SEEDS[: args.seeds]

    per_seed = {}
    for s in seeds:
        print(f"\n================ seed {s} ================")
        per_seed[str(s)] = run_all_defenses(args, device, s)
        # Written after every seed so a crash does not lose finished seeds.
        ms_path = os.path.join(cfg.RESULTS_DIR, f"defenses_multiseed_{args.dataset}_{args.model}.json")
        with open(ms_path, "w") as f:
            json.dump({"seeds": seeds[: len(per_seed)], "epochs": args.epochs,
                       "per_seed": per_seed}, f, indent=2)

    out_path = os.path.join(cfg.RESULTS_DIR, f"defenses_{args.dataset}_{args.model}.json")
    with open(out_path, "w") as f:  # first-seed results, same filename/format as before
        json.dump(per_seed[str(seeds[0])], f, indent=2)

    if args.dp_sweep:
        print("\n[DP-SGD privacy-utility sweep]")
        points = run_dp_sweep(args, device, seeds[0])
        with open(os.path.join(cfg.RESULTS_DIR, f"dp_sweep_{args.dataset}_{args.model}.json"), "w") as f:
            json.dump(points, f, indent=2)

    log_manifest(cfg.RESULTS_DIR, {"stage": "run_defenses", "args": vars(args)})
    print(f"\nSaved defense results -> {out_path}")


if __name__ == "__main__":
    main()
