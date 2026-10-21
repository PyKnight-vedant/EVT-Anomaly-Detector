"""Step 14: Differentially-private training via Opacus DP-SGD.
Primary defense target: membership inference (Step 12)."""
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from torch.utils.data import DataLoader


def dp_sgd_train(model: nn.Module, train_loader: DataLoader, device,
                  epochs: int, noise_multiplier: float = 1.0,
                  max_grad_norm: float = 1.0, target_delta: float = 1e-5,
                  lr: float = 1e-3):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    model.train()
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    epsilon = privacy_engine.get_epsilon(delta=target_delta)
    return model, {"epsilon": epsilon, "delta": target_delta}
