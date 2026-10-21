"""Step 8: Projected Gradient Descent (iterative FGSM with projection)."""
import torch
import torch.nn as nn


def pgd_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                epsilon: float, alpha: float, iters: int,
                clamp_min: float = None, clamp_max: float = None,
                random_start: bool = True) -> torch.Tensor:
    model.eval()
    x_orig = x.clone().detach()

    if random_start:
        x_adv = x_orig + torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
        if clamp_min is not None:
            x_adv = x_adv.clamp(clamp_min, clamp_max)
    else:
        x_adv = x_orig.clone()

    for _ in range(iters):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        logits = model(x_adv)
        loss = nn.functional.cross_entropy(logits, y)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            perturbation = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
            x_adv = x_orig + perturbation
            if clamp_min is not None and clamp_max is not None:
                x_adv = x_adv.clamp(clamp_min, clamp_max)

    return x_adv.detach()
