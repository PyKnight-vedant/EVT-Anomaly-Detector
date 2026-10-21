"""
Central configuration for the ML security framework.
Step 18 (Reproducibility): all hyperparameters, paths, and seeds live here
so every experiment is driven from one source of truth.
"""
import os
from dataclasses import dataclass, field
from typing import List

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "datasets")
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")

for d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, CHECKPOINT_DIR):
    os.makedirs(d, exist_ok=True)

DEFAULT_SEED = 42
SEEDS = list(range(42, 52))  # 10 seeds available; pass --seeds 5 or --seeds 10

DATASET_INFO = {
    "mnist": {"in_channels": 1, "img_size": 28, "num_classes": 10},
    "fashion_mnist": {"in_channels": 1, "img_size": 28, "num_classes": 10},
    "cifar10": {"in_channels": 3, "img_size": 32, "num_classes": 10},
}

TRAIN_VAL_TEST_SPLIT = (0.8, 0.1, 0.1)  # Step 3


@dataclass
class TrainConfig:
    dataset: str = "mnist"
    model: str = "cnn"          # "cnn" or "resnet18"
    batch_size: int = 128
    epochs: int = 20
    lr: float = 1e-3
    optimizer: str = "adam"
    seed: int = DEFAULT_SEED
    num_workers: int = 2


@dataclass
class FGSMConfig:
    epsilons: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2])


@dataclass
class PGDConfig:
    epsilons: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2])
    alpha: float = 0.01
    iters_list: List[int] = field(default_factory=lambda: [10, 20, 40])


@dataclass
class CWConfig:
    c: float = 1.0
    kappa: float = 0.0
    steps: int = 100
    lr: float = 0.01
    binary_search_steps: int = 5


@dataclass
class PoisonConfig:
    flip_fractions: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.20, 0.30])


@dataclass
class BackdoorConfig:
    trigger_size: int = 4          # side length of the white-square trigger
    target_class: int = 0
    poison_fraction: float = 0.10


@dataclass
class MembershipInferenceConfig:
    num_shadow_models: int = 3
    shadow_epochs: int = 10


@dataclass
class ModelExtractionConfig:
    num_queries: int = 5000
    clone_epochs: int = 10


@dataclass
class AdversarialTrainingConfig:
    epsilon: float = 0.1
    alpha: float = 0.01
    iters: int = 10
    epochs: int = 20


@dataclass
class DPSGDConfig:
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    epochs: int = 20
    target_delta: float = 1e-5


@dataclass
class DistillationConfig:
    temperature: float = 20.0
    epochs: int = 20
