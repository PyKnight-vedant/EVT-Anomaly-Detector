"""Step 7: Fast Gradient Sign Method attack.

Operates in normalized model-input space directly (perturbation magnitude
epsilon is applied in that space, which is the common convention when
epsilon values like 0.05/0.1/0.2 are quoted without a data range note).
"""
import torch
import torch.nn as nn


def fgsm_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                 epsilon: float, clamp_min: float = None, clamp_max: float = None) -> torch.Tensor:
    model.eval()
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, y)
    model.zero_grad()
    loss.backward()

    x_adv = x + epsilon * x.grad.sign()
    if clamp_min is not None and clamp_max is not None:
        x_adv = x_adv.clamp(clamp_min, clamp_max)
    return x_adv.detach()
