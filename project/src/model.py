"""
model.py - CIFAR-adapted ResNet-18 for predicting annotator disagreement.

Why we change the first convolution:
- CIFAR images are only 32x32, so a 7x7, stride-2 stem downsamples too
  aggressively and discards useful local detail early.
- We switch to 3x3, stride-1, padding-1 so spatial detail is preserved.

Why we use a 2-layer MLP head:
- A 512->256->10 head has more expressive capacity than a single linear layer.
- It remains small and interpretable, which is suitable for a strong baseline.

Why there is no softmax in the model:
- The model returns raw logits for numerical stability.
- Softmax or log_softmax should be applied inside the loss/evaluation code.
"""

import torch
import torch.nn as nn
import torchvision.models as models


def build_resnet18_cifar() -> nn.Module:
    """
    Build a ResNet-18 adapted for 32x32 CIFAR input and 10-class logits output.

    Returns:
        A torch.nn.Module that maps (B, 3, 32, 32) -> (B, 10) raw logits.
    """
    model = models.resnet18(weights=None)

    # CIFAR stem: smaller kernel and no downsampling at entry.
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()

    # Replace single linear head with a 2-layer MLP head.
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 10),
    )

    return model


def model_summary(model: nn.Module) -> None:
    """
    Print parameter counts for key layer groups and total parameters.

    Args:
        model: Model returned by `build_resnet18_cifar()`.
    """
    groups = {
        "stem": [model.conv1, model.bn1],
        "layer1": [model.layer1],
        "layer2": [model.layer2],
        "layer3": [model.layer3],
        "layer4": [model.layer4],
        "head": [model.fc],
    }

    print("-" * 48)
    print(f"{'Layer Group':<16}{'Parameters':>16}{'Share':>16}")
    print("-" * 48)

    total_params = sum(param.numel() for param in model.parameters())
    if total_params == 0:
        print("Model has zero parameters.")
        return

    for name, modules in groups.items():
        group_params = sum(
            param.numel()
            for module in modules
            for param in module.parameters()
        )
        share = (group_params / total_params) * 100.0
        print(f"{name:<16}{group_params:>16,}{share:>15.2f}%")

    print("-" * 48)
    print(f"{'total':<16}{total_params:>16,}{100.0:>15.2f}%")
    print("-" * 48)


if __name__ == "__main__":
    print("Running model.py smoke test...")
    net = build_resnet18_cifar()
    dummy = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        logits = net(dummy)

    assert logits.shape == (4, 10), f"Expected output shape (4, 10), got {tuple(logits.shape)}"
    print(f"Forward pass OK: logits shape = {tuple(logits.shape)}")
    model_summary(net)
