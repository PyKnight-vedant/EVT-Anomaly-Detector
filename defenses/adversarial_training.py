"""Step 14: Adversarial training defense — train on a mix of clean and PGD
adversarial examples generated on-the-fly."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from attacks.pgd import pgd_attack


def adversarial_train(model: nn.Module, train_loader: DataLoader, device,
                       epsilon: float, alpha: float, iters: int, epochs: int,
                       lr: float = 1e-3, mix_ratio: float = 0.5) -> nn.Module:
    """mix_ratio: fraction of each batch replaced with adversarial examples."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            split = int(x.size(0) * mix_ratio)
            x_to_attack, y_to_attack = x[:split], y[:split]
            x_clean, y_clean = x[split:], y[split:]

            if split > 0:
                x_adv = pgd_attack(model, x_to_attack, y_to_attack, epsilon, alpha, iters)
            else:
                x_adv = x_to_attack

            x_batch = torch.cat([x_adv, x_clean], dim=0)
            y_batch = torch.cat([y_to_attack, y_clean], dim=0)

            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
    return model
