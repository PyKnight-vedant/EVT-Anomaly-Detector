"""Steps 7-13 driver: run all attacks against a trained baseline model."""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as cfg
from attacks.backdoor import BackdoorDataset, stamp_trigger_on_batch
from attacks.carlini_wagner import cw_l2_attack
from attacks.fgsm import fgsm_attack
from attacks.membership_inference import run_membership_inference_attack
from attacks.model_extraction import run_model_extraction
from attacks.pgd import pgd_attack
from attacks.poisoning import LabelFlippedDataset
from models.factory import build_model
from train_baseline import train_one_model
from utils.data import denormalize, get_dataloaders, normalize
from utils.metrics import attack_success_rate, evaluate_classifier, robust_accuracy
from utils.seed import get_device, log_manifest, set_seed


@torch.no_grad()
def _predict(model, x, device):
    return model(x.to(device)).argmax(1).cpu().numpy()


def eval_attack_over_loader(model, loader, device, attack_fn, clamp_min, clamp_max,
                             target_class=None, max_batches=None):
    """Runs attack_fn on every batch, returns aggregated ASR + robust accuracy."""
    y_true_all, y_clean_all, y_adv_all = [], [], []
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        y_clean = model(x).argmax(1)
        x_adv = attack_fn(x, y)
        y_adv = model(x_adv).argmax(1)

        y_true_all.append(y.cpu().numpy())
        y_clean_all.append(y_clean.cpu().numpy())
        y_adv_all.append(y_adv.cpu().numpy())

    y_true = np.concatenate(y_true_all)
    y_clean = np.concatenate(y_clean_all)
    y_adv = np.concatenate(y_adv_all)

    return {
        "attack_success_rate": attack_success_rate(y_true, y_clean, y_adv, target_class),
        "robust_accuracy": robust_accuracy(y_true, y_adv),
        "clean_accuracy_on_batch": float((y_clean == y_true).mean()),
    }


def run_fgsm_pgd_cw(model, test_loader, device, dataset_name, max_batches=20):
    results = {"fgsm": {}, "pgd": {}, "cw": {}}
    clamp_min, clamp_max = None, None  # attacking in normalized space; no valid-range clamp

    # --- FGSM (Step 7) ---
    for eps in cfg.FGSMConfig().epsilons:
        res = eval_attack_over_loader(
            model, test_loader, device,
            attack_fn=lambda x, y, e=eps: fgsm_attack(model, x, y, e, clamp_min, clamp_max),
            clamp_min=clamp_min, clamp_max=clamp_max, max_batches=max_batches,
        )
        results["fgsm"][f"eps_{eps}"] = res
        print(f"  FGSM eps={eps}: ASR={res['attack_success_rate']:.3f} "
              f"robust_acc={res['robust_accuracy']:.3f}")

    # --- PGD (Step 8) ---
    pgd_cfg = cfg.PGDConfig()
    for eps in pgd_cfg.epsilons:
        for iters in pgd_cfg.iters_list:
            res = eval_attack_over_loader(
                model, test_loader, device,
                attack_fn=lambda x, y, e=eps, it=iters: pgd_attack(
                    model, x, y, e, pgd_cfg.alpha, it, clamp_min, clamp_max),
                clamp_min=clamp_min, clamp_max=clamp_max, max_batches=max_batches,
            )
            results["pgd"][f"eps_{eps}_iters_{iters}"] = res
            print(f"  PGD eps={eps} iters={iters}: ASR={res['attack_success_rate']:.3f} "
                  f"robust_acc={res['robust_accuracy']:.3f}")

    # --- Carlini & Wagner (Step 9) — expensive, use a small subset ---
    cw_cfg = cfg.CWConfig()
    res = eval_attack_over_loader(
        model, test_loader, device,
        attack_fn=lambda x, y: cw_l2_attack(
            model, x, y, clamp_min=-3.0, clamp_max=3.0,  # loose bounds ok in normalized space
            c=cw_cfg.c, kappa=cw_cfg.kappa, steps=cw_cfg.steps, lr=cw_cfg.lr,
            binary_search_steps=cw_cfg.binary_search_steps, device=device),
        clamp_min=None, clamp_max=None, max_batches=3,  # CW is slow; fewer batches
    )
    results["cw"] = res
    print(f"  C&W: ASR={res['attack_success_rate']:.3f} robust_acc={res['robust_accuracy']:.3f}")

    return results


def run_poisoning(dataset_name, model_name, epochs, batch_size, seed, device):
    """Step 10: retrain with varying fractions of flipped labels."""
    results = {}
    info = cfg.DATASET_INFO[dataset_name]
    for frac in cfg.PoisonConfig().flip_fractions:
        set_seed(seed)
        train_loader, val_loader, test_loader = get_dataloaders(
            dataset_name, batch_size=batch_size, seed=seed)
        poisoned_train = LabelFlippedDataset(
            train_loader.dataset, frac, info["num_classes"], seed=seed)
        poisoned_loader = DataLoader(poisoned_train, batch_size=batch_size, shuffle=True)

        model = build_model(model_name, dataset_name).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        for _ in range(epochs):
            for x, y in poisoned_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

        metrics = evaluate_classifier(model, test_loader, device)
        results[f"flip_{frac}"] = {"clean_test_accuracy": metrics["accuracy"]}
        print(f"  Poisoning flip_fraction={frac}: test_acc={metrics['accuracy']:.3f}")
    return results


def run_backdoor(dataset_name, model_name, epochs, batch_size, seed, device):
    """Step 11: trigger-patch backdoor. Reports clean accuracy + ASR on trigger."""
    bd_cfg = cfg.BackdoorConfig()
    set_seed(seed)
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name, batch_size=batch_size, seed=seed)

    denorm_fn = lambda x: denormalize(x, dataset_name)
    norm_fn = lambda x: normalize(x, dataset_name)

    backdoor_train = BackdoorDataset(
        train_loader.dataset, denorm_fn, norm_fn,
        bd_cfg.poison_fraction, bd_cfg.target_class, bd_cfg.trigger_size, seed=seed)
    backdoor_loader = DataLoader(backdoor_train, batch_size=batch_size, shuffle=True)

    model = build_model(model_name, dataset_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in backdoor_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    clean_metrics = evaluate_classifier(model, test_loader, device)

    # Attack success rate: stamp trigger on clean test images, check they get
    # classified as target_class.
    model.eval()
    correct_target = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            x_trig = stamp_trigger_on_batch(x, denorm_fn, norm_fn, bd_cfg.trigger_size)
            preds = model(x_trig).argmax(1)
            correct_target += (preds == bd_cfg.target_class).sum().item()
            total += x.size(0)
    asr = correct_target / total

    result = {
        "clean_accuracy": clean_metrics["accuracy"],
        "attack_success_rate": asr,
        "poison_fraction": bd_cfg.poison_fraction,
        "target_class": bd_cfg.target_class,
    }
    print(f"  Backdoor: clean_acc={result['clean_accuracy']:.3f} ASR={asr:.3f}")
    return result


def run_membership_inference(dataset_name, model_name, victim_model,
                              train_loader, test_loader, seed, device):
    """Step 12."""
    mi_cfg = cfg.MembershipInferenceConfig()
    build_fn = lambda: build_model(model_name, dataset_name)
    shadow_dataset = test_loader.dataset  # disjoint pool for shadow training
    metrics = run_membership_inference_attack(
        victim_model, train_loader, test_loader, build_fn, shadow_dataset,
        cfg.DATASET_INFO[dataset_name]["num_classes"], device,
        num_shadow=mi_cfg.num_shadow_models, shadow_epochs=mi_cfg.shadow_epochs, seed=seed,
    )
    print(f"  Membership inference: precision={metrics['precision']:.3f} "
          f"recall={metrics['recall']:.3f} auc={metrics['auc']:.3f}")
    return metrics


def run_extraction(dataset_name, model_name, victim_model, query_loader,
                    test_loader, device):
    """Step 13."""
    me_cfg = cfg.ModelExtractionConfig()
    build_fn = lambda: build_model(model_name, dataset_name)
    _, metrics = run_model_extraction(
        victim_model, query_loader, test_loader, build_fn, device,
        num_queries=me_cfg.num_queries, clone_epochs=me_cfg.clone_epochs,
    )
    print(f"  Model extraction: fidelity={metrics['fidelity']:.3f} "
          f"clone_acc={metrics['clone_accuracy']:.3f}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist", choices=list(cfg.DATASET_INFO.keys()))
    parser.add_argument("--model", default="cnn", choices=["cnn", "resnet18"])
    parser.add_argument("--checkpoint", default=None,
                         help="Path to trained baseline checkpoint; trains one if omitted.")
    parser.add_argument("--epochs", type=int, default=10,
                         help="Epochs for any attack-side retraining (poisoning/backdoor).")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=cfg.DEFAULT_SEED)
    parser.add_argument("--skip_expensive", action="store_true",
                         help="Skip membership inference / extraction (slow).")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    set_seed(args.seed)

    train_loader, val_loader, test_loader = get_dataloaders(
        args.dataset, batch_size=args.batch_size, seed=args.seed)

    if args.checkpoint and os.path.exists(args.checkpoint):
        model = build_model(args.model, args.dataset).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print("No checkpoint given/found — training baseline first...")
        model, _ = train_one_model(args.dataset, args.model, args.epochs, 1e-3,
                                    args.batch_size, args.seed, device=device, quiet=True)

    all_results = {}

    print("\n[Steps 7-9] Evasion attacks: FGSM / PGD / C&W")
    all_results["evasion"] = run_fgsm_pgd_cw(model, test_loader, device, args.dataset)

    print("\n[Step 10] Data poisoning (label flipping)")
    all_results["poisoning"] = run_poisoning(
        args.dataset, args.model, args.epochs, args.batch_size, args.seed, device)

    print("\n[Step 11] Backdoor attack")
    all_results["backdoor"] = run_backdoor(
        args.dataset, args.model, args.epochs, args.batch_size, args.seed, device)

    if not args.skip_expensive:
        print("\n[Step 12] Membership inference")
        all_results["membership_inference"] = run_membership_inference(
            args.dataset, args.model, model, train_loader, test_loader, args.seed, device)

        print("\n[Step 13] Model extraction")
        all_results["model_extraction"] = run_extraction(
            args.dataset, args.model, model, test_loader, test_loader, device)

    out_path = os.path.join(cfg.RESULTS_DIR, f"attacks_{args.dataset}_{args.model}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log_manifest(cfg.RESULTS_DIR, {"stage": "run_attacks", "args": vars(args)})
    print(f"\nSaved attack results -> {out_path}")


if __name__ == "__main__":
    main()
