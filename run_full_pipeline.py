"""Orchestrates the full framework end-to-end: baseline -> attacks ->
defenses -> comparative evaluation, matching Steps 1-17 of the brief."""
import argparse
import os
import subprocess
import sys

import config as cfg


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist", choices=list(cfg.DATASET_INFO.keys()))
    parser.add_argument("--model", default="cnn", choices=["cnn", "resnet18"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--attack_epochs", type=int, default=10,
                         help="Epochs for attack-side retraining (poisoning/backdoor).")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--skip_expensive_attacks", action="store_true")
    args = parser.parse_args()

    py = sys.executable

    ckpt = os.path.join(cfg.CHECKPOINT_DIR, f"baseline_{args.dataset}_{args.model}.pt")

    run([py, "train_baseline.py", "--dataset", args.dataset, "--model", args.model,
         "--epochs", str(args.epochs), "--batch_size", str(args.batch_size)])

    attack_cmd = [py, "run_attacks.py", "--dataset", args.dataset, "--model", args.model,
                  "--checkpoint", ckpt, "--epochs", str(args.attack_epochs),
                  "--batch_size", str(args.batch_size)]
    if args.skip_expensive_attacks:
        attack_cmd.append("--skip_expensive")
    run(attack_cmd)

    run([py, "run_defenses.py", "--dataset", args.dataset, "--model", args.model,
         "--epochs", str(args.attack_epochs), "--batch_size", str(args.batch_size),
         "--seeds", str(args.seeds), "--dp_sweep"])

    run([py, "run_evt_calibration.py", "--dataset", args.dataset, "--model", args.model,
         "--batch_size", str(args.batch_size)])

    run([py, "run_comparative_eval.py", "--dataset", args.dataset, "--model", args.model,
         "--epochs", str(args.epochs), "--batch_size", str(args.batch_size),
         "--seeds", str(args.seeds)])

    print("\nPipeline complete. See results/ for JSON metrics and "
          "results/figures/ for plots.")


if __name__ == "__main__":
    main()
