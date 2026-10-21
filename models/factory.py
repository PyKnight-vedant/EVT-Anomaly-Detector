"""Build a model by name for a given dataset."""
import torch.nn as nn

import config as cfg
from models.cnn import SimpleCNN
from models.resnet import build_resnet18


def build_model(model_name: str, dataset: str) -> nn.Module:
    info = cfg.DATASET_INFO[dataset]
    if model_name == "cnn":
        return SimpleCNN(in_channels=info["in_channels"], num_classes=info["num_classes"])
    if model_name == "resnet18":
        return build_resnet18(in_channels=info["in_channels"], num_classes=info["num_classes"])
    raise ValueError(f"Unknown model: {model_name}")
