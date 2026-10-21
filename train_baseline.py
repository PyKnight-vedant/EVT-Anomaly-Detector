"""Steps 5-6: Train baseline CNN/ResNet18 and evaluate with full metric suite."""
import argparse
import json
import os

import torch
import torch.nn as nn
from tqdm import tqdm

import config as cfg
from models.factory import build_model
from utils.data import get_dataloaders
from utils.metrics import evaluate_classifier, timer
from utils.seed import get_device, log_manifest, set_seed


def train_one_model(dataset: str, model_name: str, epochs: int, lr: float,
                     batch_size: int, seed: int, save_path: str = None,
                     device=None, quiet=False):
    set_seed(seed)
    device = device or get_device()

    train_loader, val_loader, test_loader = get_dataloaders(
        dataset, batch_size=batch_size, seed=seed
    )

    model = build_model(model_name, dataset).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    with timer() as t_train:
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            loop = train_loader if quiet else tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for x, y in loop:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * x.size(0)
            if not quiet:
                val_metrics = evaluate_classifier(model, val_loader, device)
                print(f"  train_loss={running_loss/len(train_loader.dataset):.4f} "
                      f"val_acc={val_metrics['accuracy']:.4f}")

    test_metrics = evaluate_classifier(model, test_loader, device)
    test_metrics["training_time_sec"] = t_train["seconds"]

    if save_path:
        torch.save(model.state_dict(), save_path)

    return model, test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist",
                         choices=list(cfg.DATASET_INFO.keys()))
    parser.add_argument("--model", default="cnn", choices=["cnn", "resnet18"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=cfg.DEFAULT_SEED)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, f"baseline_{args.dataset}_{args.model}.pt")
    _, metrics = train_one_model(
        args.dataset, args.model, args.epochs, args.lr, args.batch_size,
        args.seed, save_path=ckpt_path, device=device,
    )

    out_path = os.path.join(cfg.RESULTS_DIR, f"baseline_{args.dataset}_{args.model}.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    log_manifest(cfg.RESULTS_DIR, {"stage": "train_baseline", "args": vars(args)})

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved checkpoint -> {ckpt_path}")
    print(f"Saved metrics -> {out_path}")


if __name__ == "__main__":
    main()
