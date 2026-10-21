"""Step 5: ResNet18 baseline (adapted for small 32x32 images, e.g. CIFAR-10)."""
import torch.nn as nn
from torchvision.models import resnet18


def build_resnet18(in_channels: int = 3, num_classes: int = 10) -> nn.Module:
    model = resnet18(weights=None, num_classes=num_classes)
    # Adapt first conv + drop maxpool for small 32x32 inputs (standard CIFAR trick).
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model
