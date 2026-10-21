"""Step 14: Defensive distillation (Papernot et al.).

Train a "teacher" at high softmax temperature T, then train a "student"
of the same architecture on the teacher's soft labels at the same T. The
resulting student has smoother decision boundaries, which raises the bar
for gradient-based evasion attacks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _train_at_temperature(model, loader, device, epochs, temperature, lr,
                           soft_targets_fn=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x) / temperature
            if soft_targets_fn is None:
                loss = F.cross_entropy(logits, y)
            else:
                with torch.no_grad():
                    soft_y = soft_targets_fn(x)
                log_probs = F.log_softmax(logits, dim=1)
                loss = -(soft_y * log_probs).sum(1).mean()
            loss.backward()
            optimizer.step()
    return model


def defensive_distillation(teacher: nn.Module, student: nn.Module,
                            train_loader: DataLoader, device,
                            temperature: float = 20.0, epochs: int = 20,
                            lr: float = 1e-3):
    # Step A: train teacher at temperature T on hard labels.
    teacher = _train_at_temperature(teacher, train_loader, device, epochs, temperature, lr)

    # Step B: freeze teacher, compute its soft labels, distill into student.
    teacher.eval()

    def soft_targets_fn(x):
        return torch.softmax(teacher(x) / temperature, dim=1)

    student = _train_at_temperature(
        student, train_loader, device, epochs, temperature, lr, soft_targets_fn
    )
    return teacher, student
