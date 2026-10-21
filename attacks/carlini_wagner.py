"""Step 9: Carlini & Wagner L2 attack (untargeted, with binary search over c).

Reference: Carlini & Wagner, "Towards Evaluating the Robustness of Neural
Networks" (2017). This is a from-scratch implementation of the L2 variant
using the tanh change-of-variables trick to keep x_adv in a valid range
without hard clamping (which breaks gradients).
"""
import torch
import torch.nn as nn


def _to_tanh_space(x, clamp_min, clamp_max):
    # map [clamp_min, clamp_max] -> (-1, 1) -> atanh space
    x_scaled = (x - clamp_min) / (clamp_max - clamp_min) * 2 - 1
    x_scaled = x_scaled.clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.atanh(x_scaled)


def _from_tanh_space(w, clamp_min, clamp_max):
    return (torch.tanh(w) + 1) / 2 * (clamp_max - clamp_min) + clamp_min


def cw_l2_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                  clamp_min: float = 0.0, clamp_max: float = 1.0,
                  c: float = 1.0, kappa: float = 0.0, steps: int = 100,
                  lr: float = 0.01, binary_search_steps: int = 5,
                  device=None) -> torch.Tensor:
    model.eval()
    device = device or x.device
    batch_size = x.size(0)

    num_classes = None

    lower_bound = torch.zeros(batch_size, device=device)
    upper_bound = torch.full((batch_size,), 1e10, device=device)
    const = torch.full((batch_size,), c, device=device)

    best_adv = x.clone().detach()
    best_l2 = torch.full((batch_size,), float("inf"), device=device)

    w_init = _to_tanh_space(x, clamp_min, clamp_max)

    for _ in range(binary_search_steps):
        w = w_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([w], lr=lr)

        for _ in range(steps):
            x_adv = _from_tanh_space(w, clamp_min, clamp_max)
            logits = model(x_adv)
            if num_classes is None:
                num_classes = logits.size(1)

            one_hot = torch.zeros_like(logits).scatter_(1, y.unsqueeze(1), 1)
            correct_logit = (logits * one_hot).sum(1)
            other_logit = (logits - one_hot * 1e4).max(1).values
            # untargeted: push correct_logit below other_logit by margin kappa
            f_loss = torch.clamp(correct_logit - other_logit + kappa, min=0)

            l2_dist = ((x_adv - x) ** 2).flatten(1).sum(1)
            loss = (l2_dist + const * f_loss).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = logits.argmax(1)
                success = (preds != y) & (l2_dist < best_l2)
                if success.any():
                    best_l2[success] = l2_dist[success]
                    best_adv[success] = x_adv[success].detach()

        with torch.no_grad():
            x_adv = _from_tanh_space(w, clamp_min, clamp_max)
            logits = model(x_adv)
            preds = logits.argmax(1)
            succeeded = preds != y

            upper_bound = torch.where(succeeded, torch.min(upper_bound, const), upper_bound)
            lower_bound = torch.where(~succeeded, torch.max(lower_bound, const), lower_bound)
            has_upper = upper_bound < 1e9
            const = torch.where(has_upper, (lower_bound + upper_bound) / 2, const * 10)

    return best_adv.detach()
