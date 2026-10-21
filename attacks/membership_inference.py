"""Step 12: Membership inference attack using shadow models.

Classic Shokri et al. approach:
1. Train N shadow models that mimic the victim's training procedure on
   disjoint data (with known member/non-member splits).
2. Build an "attack dataset" of (prediction-confidence-vector, in/out label)
   from the shadow models.
3. Train a binary attack classifier on that data.
4. Apply the attack classifier to the victim model's confidence vectors on
   its actual train vs. held-out samples to score membership.
"""
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Subset

from utils.metrics import membership_inference_metrics


@torch.no_grad()
def _confidence_vectors(model: nn.Module, loader: DataLoader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_conf, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        probs = torch.softmax(model(x), dim=1).cpu().numpy()
        all_conf.append(probs)
        all_labels.append(y.numpy())
    return np.concatenate(all_conf), np.concatenate(all_labels)


def train_shadow_models(build_model_fn, dataset, num_classes: int,
                         num_shadow: int, shadow_epochs: int, device,
                         batch_size: int = 128, seed: int = 0):
    """Trains `num_shadow` shadow models on random disjoint splits of
    `dataset`; returns list of (model, member_loader, nonmember_loader)."""
    rng = np.random.RandomState(seed)
    n = len(dataset)
    shadow_results = []

    for i in range(num_shadow):
        perm = rng.permutation(n)
        half = n // 2
        member_idx, nonmember_idx = perm[:half], perm[half:]

        member_loader = DataLoader(Subset(dataset, member_idx), batch_size=batch_size, shuffle=True)
        nonmember_loader = DataLoader(Subset(dataset, nonmember_idx), batch_size=batch_size, shuffle=False)

        model = build_model_fn().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for _ in range(shadow_epochs):
            for x, y in member_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

        shadow_results.append((model, member_loader, nonmember_loader))

    return shadow_results


def build_attack_dataset(shadow_results, device):
    X, y_membership = [], []
    for model, member_loader, nonmember_loader in shadow_results:
        conf_m, _ = _confidence_vectors(model, member_loader, device)
        conf_nm, _ = _confidence_vectors(model, nonmember_loader, device)
        X.append(conf_m); y_membership.append(np.ones(len(conf_m)))
        X.append(conf_nm); y_membership.append(np.zeros(len(conf_nm)))
    X = np.concatenate(X)
    # sort each row descending so the attack model is class-order agnostic
    X = np.sort(X, axis=1)[:, ::-1]
    y_membership = np.concatenate(y_membership)
    return X, y_membership


def run_membership_inference_attack(victim_model, victim_train_loader,
                                     victim_heldout_loader, build_model_fn,
                                     shadow_dataset, num_classes, device,
                                     num_shadow=3, shadow_epochs=10, seed=0):
    shadow_results = train_shadow_models(
        build_model_fn, shadow_dataset, num_classes, num_shadow,
        shadow_epochs, device, seed=seed,
    )
    X_attack, y_attack = build_attack_dataset(shadow_results, device)

    attack_clf = LogisticRegression(max_iter=1000)
    attack_clf.fit(X_attack, y_attack)

    conf_member, _ = _confidence_vectors(victim_model, victim_train_loader, device)
    conf_nonmember, _ = _confidence_vectors(victim_model, victim_heldout_loader, device)

    conf_member_sorted = np.sort(conf_member, axis=1)[:, ::-1]
    conf_nonmember_sorted = np.sort(conf_nonmember, axis=1)[:, ::-1]

    scores_member = attack_clf.predict_proba(conf_member_sorted)[:, 1]
    scores_nonmember = attack_clf.predict_proba(conf_nonmember_sorted)[:, 1]

    y_true = np.concatenate([np.ones(len(scores_member)), np.zeros(len(scores_nonmember))])
    scores = np.concatenate([scores_member, scores_nonmember])

    return membership_inference_metrics(y_true, scores)
