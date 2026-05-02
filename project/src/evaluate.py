"""
evaluate.py - evaluation metrics for CIFAR-10H test split.
"""

import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .config import CONFIG
    from .dataset import get_dataloaders
    from .model import build_resnet18_cifar
except ImportError:
    from config import CONFIG
    from dataset import get_dataloaders
    from model import build_resnet18_cifar


def resolve_device(config: dict) -> torch.device:
    """
    Resolve torch device from config with safe fallback.

    Args:
        config: Global configuration dictionary.

    Returns:
        torch.device object.
    """
    requested = str(config.get("device", "cpu")).lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> Dict:
    """
    Load checkpoint weights into the model.

    Args:
        model: Model to populate with weights.
        checkpoint_path: Path to the checkpoint file.
        device: Torch device for map_location.

    Returns:
        The loaded checkpoint dictionary (if available).
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    return checkpoint if isinstance(checkpoint, dict) else {}


def kl_divergence(target: torch.Tensor, prediction: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute KL(target || prediction) per sample.

    Args:
        target: Probability targets with shape (B, 10).
        prediction: Probability predictions with shape (B, 10).
        eps: Small value to avoid log(0).

    Returns:
        Tensor of shape (B,) with per-sample KL divergence.
    """
    target_safe = torch.clamp(target, eps, 1.0)
    pred_safe = torch.clamp(prediction, eps, 1.0)
    return torch.sum(target_safe * (torch.log(target_safe) - torch.log(pred_safe)), dim=1)


def js_divergence(target: torch.Tensor, prediction: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute Jensen-Shannon divergence per sample.

    Args:
        target: Probability targets with shape (B, 10).
        prediction: Probability predictions with shape (B, 10).
        eps: Small value to avoid log(0).

    Returns:
        Tensor of shape (B,) with per-sample JSD.
    """
    mix = 0.5 * (target + prediction)
    kl_t = kl_divergence(target, mix, eps=eps)
    kl_p = kl_divergence(prediction, mix, eps=eps)
    return 0.5 * (kl_t + kl_p)


def evaluate(model: torch.nn.Module, test_loader, device: torch.device) -> Tuple[float, float, float, float]:
    """
    Evaluate the model on the CIFAR-10H test loader.

    Args:
        model: Model that outputs raw logits.
        test_loader: DataLoader for the test split.
        device: Torch device to run evaluation on.

    Returns:
        Tuple of (kl_mean, jsd_mean, cosine_mean, top1_acc).
    """
    model.eval()
    kl_sum = 0.0
    jsd_sum = 0.0
    cosine_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, soft_labels in test_loader:
            images = images.to(device)
            soft_labels = soft_labels.to(device)

            logits = model(images)
            probs = F.softmax(logits, dim=1)

            kl_vals = kl_divergence(soft_labels, probs)
            jsd_vals = js_divergence(soft_labels, probs)
            cosine_vals = F.cosine_similarity(probs, soft_labels, dim=1)

            kl_sum += float(kl_vals.sum().item())
            jsd_sum += float(jsd_vals.sum().item())
            cosine_sum += float(cosine_vals.sum().item())

            pred_top1 = probs.argmax(dim=1)
            target_top1 = soft_labels.argmax(dim=1)
            correct += int((pred_top1 == target_top1).sum().item())
            total += images.size(0)

    if total == 0:
        raise ValueError("Empty test loader: no samples to evaluate.")

    kl_mean = kl_sum / total
    jsd_mean = jsd_sum / total
    cosine_mean = cosine_sum / total
    top1_acc = correct / total

    metrics = np.array([kl_mean, jsd_mean, cosine_mean, top1_acc], dtype=np.float64)
    if not np.all(np.isfinite(metrics)):
        raise ValueError("Non-finite evaluation metric detected (NaN or inf).")

    return kl_mean, jsd_mean, cosine_mean, top1_acc


def save_metrics(output_path: Path, kl: float, jsd: float, cosine: float, top1_acc: float) -> None:
    """
    Save evaluation metrics to a text file.

    Args:
        output_path: Path to the output text file.
        kl: Mean KL divergence.
        jsd: Mean Jensen-Shannon divergence.
        cosine: Mean cosine similarity.
        top1_acc: Mean top-1 accuracy.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        file.write("----------------------------------\n")
        file.write("Evaluation Results (CIFAR-10H Test)\n")
        file.write("----------------------------------\n")
        file.write(f"KL Divergence       : {kl:.4f}\n")
        file.write(f"JSD                 : {jsd:.4f}\n")
        file.write(f"Cosine Similarity   : {cosine:.4f}\n")
        file.write(f"Top-1 Accuracy      : {top1_acc * 100.0:.2f} %\n")
        file.write("----------------------------------\n")


def main() -> None:
    """Run evaluation on CIFAR-10H test split and save results."""
    device = resolve_device(CONFIG)

    model = build_resnet18_cifar().to(device)
    checkpoint_path = CONFIG["checkpoint_path"]
    load_checkpoint(model, checkpoint_path, device)

    _, _, test_loader = get_dataloaders(CONFIG)

    kl, jsd, cosine, top1_acc = evaluate(model, test_loader, device)

    print("----------------------------------")
    print("Evaluation Results (CIFAR-10H Test)")
    print("----------------------------------")
    print(f"KL Divergence       : {kl:.4f}")
    print(f"JSD                 : {jsd:.4f}")
    print(f"Cosine Similarity   : {cosine:.4f}")
    print(f"Top-1 Accuracy      : {top1_acc * 100.0:.2f} %")
    print("----------------------------------")

    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "artifacts" / "evaluation_metrics.txt"
    save_metrics(output_path, kl, jsd, cosine, top1_acc)


if __name__ == "__main__":
    main()
