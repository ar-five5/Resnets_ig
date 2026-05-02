"""
robustness_eval.py - robustness evaluation under input corruptions.
"""

import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

try:
    from .config import CONFIG
    from .dataset import (
        CIFAR10_MEAN,
        CIFAR10_STD,
        CIFAR10HDataset,
        get_default_data_dir,
        load_cifar10h,
    )
    from .model import build_resnet18_cifar
except ImportError:
    from config import CONFIG
    from dataset import (
        CIFAR10_MEAN,
        CIFAR10_STD,
        CIFAR10HDataset,
        get_default_data_dir,
        load_cifar10h,
    )
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


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    """
    Load checkpoint weights into the model.

    Args:
        model: Model to populate with weights.
        checkpoint_path: Path to the checkpoint file.
        device: Torch device for map_location.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)


def normalize_batch(batch: torch.Tensor) -> torch.Tensor:
    """
    Normalize a batch of images using CIFAR-10 mean/std.

    Args:
        batch: Tensor with shape (B, 3, 32, 32) in [0, 1].

    Returns:
        Normalized tensor.
    """
    mean = torch.tensor(CIFAR10_MEAN, device=batch.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=batch.device).view(1, 3, 1, 1)
    return (batch - mean) / std


def add_gaussian_noise(batch: torch.Tensor, std: float = 0.1) -> torch.Tensor:
    """
    Apply Gaussian noise to a batch of images.

    Args:
        batch: Tensor with shape (B, 3, 32, 32) in [0, 1].
        std: Standard deviation of the noise.

    Returns:
        Noisy batch in [0, 1].
    """
    noise = torch.randn_like(batch) * std
    return torch.clamp(batch + noise, 0.0, 1.0)


def gaussian_kernel2d(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """
    Create a 2D Gaussian kernel.

    Args:
        kernel_size: Odd kernel size.
        sigma: Gaussian sigma.
        device: Torch device.

    Returns:
        Tensor with shape (1, 1, k, k).
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    coords = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2.0
    grid = coords.unsqueeze(0) ** 2 + coords.unsqueeze(1) ** 2
    kernel = torch.exp(-grid / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)


def apply_gaussian_blur(batch: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """
    Apply Gaussian blur to a batch of images.

    Args:
        batch: Tensor with shape (B, 3, 32, 32) in [0, 1].
        kernel_size: Size of Gaussian kernel.
        sigma: Gaussian sigma.

    Returns:
        Blurred batch in [0, 1].
    """
    kernel = gaussian_kernel2d(kernel_size, sigma, batch.device)
    kernel = kernel.repeat(3, 1, 1, 1)
    padding = kernel_size // 2
    return F.conv2d(batch, kernel, padding=padding, groups=3)


def adjust_brightness(batch: torch.Tensor, low: float = 0.8, high: float = 1.2) -> torch.Tensor:
    """
    Apply random brightness change (increase or decrease).

    Args:
        batch: Tensor with shape (B, 3, 32, 32) in [0, 1].
        low: Multiplicative factor for darkening.
        high: Multiplicative factor for brightening.

    Returns:
        Brightness-adjusted batch in [0, 1].
    """
    probs = torch.rand(batch.size(0), 1, 1, 1, device=batch.device)
    factors = torch.where(probs < 0.5, torch.tensor(low, device=batch.device), torch.tensor(high, device=batch.device))
    return torch.clamp(batch * factors, 0.0, 1.0)


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


def evaluate_condition(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    corruption: str,
) -> Tuple[float, float]:
    """
    Evaluate one corruption condition.

    Args:
        model: Model that outputs raw logits.
        test_loader: DataLoader providing test images in [0, 1].
        device: Torch device.
        corruption: One of "clean", "noise", "blur", "brightness".

    Returns:
        Tuple of (kl_mean, top1_acc).
    """
    model.eval()
    kl_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, soft_labels in test_loader:
            images = images.to(device)
            soft_labels = soft_labels.to(device)

            if corruption == "noise":
                images = add_gaussian_noise(images)
            elif corruption == "blur":
                images = apply_gaussian_blur(images)
            elif corruption == "brightness":
                images = adjust_brightness(images)
            elif corruption != "clean":
                raise ValueError(f"Unknown corruption: {corruption}")

            images = normalize_batch(images)
            logits = model(images)
            probs = F.softmax(logits, dim=1)

            kl_vals = kl_divergence(soft_labels, probs)
            kl_sum += float(kl_vals.sum().item())

            pred_top1 = probs.argmax(dim=1)
            target_top1 = soft_labels.argmax(dim=1)
            correct += int((pred_top1 == target_top1).sum().item())
            total += images.size(0)

    if total == 0:
        raise ValueError("Empty test loader: no samples to evaluate.")

    kl_mean = kl_sum / total
    acc = correct / total

    metrics = np.array([kl_mean, acc], dtype=np.float64)
    if not np.all(np.isfinite(metrics)):
        raise ValueError("Non-finite evaluation metric detected (NaN or inf).")

    return kl_mean, acc


def build_test_loader(config: dict) -> DataLoader:
    """
    Build a test loader that returns images in [0, 1] without normalization.

    Args:
        config: Configuration dictionary.

    Returns:
        DataLoader for the CIFAR-10H test split.
    """
    data_dir = config.get("data_dir", get_default_data_dir())
    images, soft_labels = load_cifar10h(data_dir=data_dir)

    split_cfg = config["cifar10h_split"]
    n_train = split_cfg["train"]
    n_val = split_cfg["val"]
    n_test = split_cfg["test"]

    if (n_train + n_val + n_test) != len(images):
        raise ValueError("CIFAR-10H split sizes must sum to the dataset size (10000).")

    rng = np.random.default_rng(config["random_seed"])
    indices = np.arange(len(images))
    rng.shuffle(indices)

    test_idx = indices[n_train + n_val : n_train + n_val + n_test]

    base_transform = transforms.ToTensor()
    test_dataset = CIFAR10HDataset(images[test_idx], soft_labels[test_idx], transform=base_transform)

    loader_kwargs = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": torch.cuda.is_available(),
        "shuffle": False,
    }

    return DataLoader(test_dataset, **loader_kwargs)


def save_results(output_path: Path, rows: Dict[str, Tuple[float, float]]) -> None:
    """
    Save robustness results to a text file.

    Args:
        output_path: Output text file path.
        rows: Mapping of condition to metrics.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        file.write("----------------------------------\n")
        file.write("Robustness Evaluation\n")
        file.write("----------------------------------\n")
        file.write("Condition        KL      Accuracy\n")
        file.write("----------------------------------\n")
        for name, (kl, acc) in rows.items():
            file.write(f"{name:<14}{kl:>7.3f} {acc * 100.0:>9.2f}\n")
        file.write("----------------------------------\n")


def print_results(rows: Dict[str, Tuple[float, float]]) -> None:
    """Print robustness results to stdout."""
    print("----------------------------------")
    print("Robustness Evaluation")
    print("----------------------------------")
    print("Condition        KL      Accuracy")
    print("----------------------------------")
    for name, (kl, acc) in rows.items():
        print(f"{name:<14}{kl:>7.3f} {acc * 100.0:>9.2f}")
    print("----------------------------------")


def main() -> None:
    """Run robustness evaluation on CIFAR-10H test split."""
    device = resolve_device(CONFIG)

    model = build_resnet18_cifar().to(device)
    load_checkpoint(model, CONFIG["checkpoint_path"], device)

    test_loader = build_test_loader(CONFIG)

    conditions = [
        ("Clean", "clean"),
        ("Noise", "noise"),
        ("Blur", "blur"),
        ("Brightness", "brightness"),
    ]

    results: Dict[str, Tuple[float, float]] = {}
    for label, key in conditions:
        kl_mean, acc = evaluate_condition(model, test_loader, device, key)
        results[label] = (kl_mean, acc)

    print_results(results)

    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "artifacts" / "robustness.txt"
    save_results(output_path, results)


if __name__ == "__main__":
    main()
