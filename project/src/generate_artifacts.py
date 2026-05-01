"""Generate required data/training artifacts from existing CIFAR-10H outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .config import CONFIG
    from .dataset import load_cifar10h
    from .model import build_resnet18_cifar
except ImportError:
    from config import CONFIG
    from dataset import load_cifar10h
    from model import build_resnet18_cifar


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def compute_entropy_bits(soft_labels: np.ndarray) -> np.ndarray:
    """Compute Shannon entropy in bits for each soft-label distribution."""
    safe = np.clip(soft_labels, 1e-12, 1.0)
    return -np.sum(soft_labels * np.log2(safe), axis=1)


def ensure_artifacts_dir(project_root: Path) -> Path:
    """Create and return the artifacts directory under project root."""
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir


def load_training_log(log_path: Path) -> List[Dict[str, float | int | str]]:
    """Load epoch-level rows from training_log.csv."""
    if not log_path.is_file():
        raise FileNotFoundError(f"Missing training log: {log_path}")

    rows: List[Dict[str, float | int | str]] = []
    with log_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "phase": row["phase"],
                    "train_loss": float(row["train_loss"]),
                    "val_loss": float(row["val_loss"]),
                }
            )
    return rows


def save_entropy_histogram(entropies: np.ndarray, out_path: Path) -> None:
    """Save histogram of CIFAR-10H entropy values."""
    plt.figure(figsize=(8, 5))
    plt.hist(entropies, bins=30, color="#4C78A8", edgecolor="white")
    plt.xlabel("Entropy (bits)")
    plt.ylabel("Number of samples")
    plt.title("CIFAR-10H Entropy Histogram")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_per_class_entropy(
    entropies: np.ndarray,
    majority_class: np.ndarray,
    out_path: Path,
) -> None:
    """Save bar chart of mean entropy grouped by majority soft-label class."""
    class_means = []
    class_counts = []
    for class_id in range(len(CIFAR10_CLASSES)):
        mask = majority_class == class_id
        class_counts.append(int(mask.sum()))
        class_means.append(float(entropies[mask].mean()) if mask.any() else 0.0)

    x = np.arange(len(CIFAR10_CLASSES))
    plt.figure(figsize=(10, 5))
    bars = plt.bar(x, class_means, color="#F58518")
    plt.xticks(x, CIFAR10_CLASSES, rotation=30, ha="right")
    plt.xlabel("Majority class")
    plt.ylabel("Average entropy (bits)")
    plt.title("Per-Class Average Entropy (Grouped by Majority Class)")

    for bar, count in zip(bars, class_counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02,
            f"n={count}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_annotator_confusion_matrix(
    soft_labels: np.ndarray,
    majority_class: np.ndarray,
    out_path: Path,
) -> None:
    """Save confusion-style matrix of average soft-label mass conditioned on majority class."""
    matrix = np.zeros((len(CIFAR10_CLASSES), len(CIFAR10_CLASSES)), dtype=np.float32)

    for class_id in range(len(CIFAR10_CLASSES)):
        mask = majority_class == class_id
        if mask.any():
            matrix[class_id] = soft_labels[mask].mean(axis=0)

    plt.figure(figsize=(8, 7))
    image = plt.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(image, fraction=0.046, pad=0.04, label="Average soft-label mass")
    plt.xticks(np.arange(len(CIFAR10_CLASSES)), CIFAR10_CLASSES, rotation=45, ha="right")
    plt.yticks(np.arange(len(CIFAR10_CLASSES)), CIFAR10_CLASSES)
    plt.xlabel("Annotator label class")
    plt.ylabel("Majority class")
    plt.title("Annotator Confusion Matrix (Average Soft-Label Mass)")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            plt.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "black",
                fontsize=7,
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def top_distribution_text(probabilities: np.ndarray, top_k: int = 3) -> str:
    """Format the top-k class probabilities into compact human-readable text."""
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    parts = [f"{CIFAR10_CLASSES[idx]} {probabilities[idx]:.2f}" for idx in top_indices]
    return " | ".join(parts)


def save_low_high_entropy_examples(
    images: np.ndarray,
    soft_labels: np.ndarray,
    entropies: np.ndarray,
    out_path: Path,
    examples_per_row: int = 4,
) -> None:
    """Save a grid of lowest- and highest-entropy examples with top label distributions."""
    sorted_indices = np.argsort(entropies)
    low_indices = sorted_indices[:examples_per_row]
    high_indices = sorted_indices[-examples_per_row:]

    fig, axes = plt.subplots(2, examples_per_row, figsize=(4 * examples_per_row, 7))

    for col, sample_index in enumerate(low_indices):
        ax = axes[0, col]
        ax.imshow(images[sample_index])
        ax.axis("off")
        ax.set_title(f"Low H={entropies[sample_index]:.2f}", fontsize=10)
        ax.set_xlabel(top_distribution_text(soft_labels[sample_index]), fontsize=8)

    for col, sample_index in enumerate(high_indices):
        ax = axes[1, col]
        ax.imshow(images[sample_index])
        ax.axis("off")
        ax.set_title(f"High H={entropies[sample_index]:.2f}", fontsize=10)
        ax.set_xlabel(top_distribution_text(soft_labels[sample_index]), fontsize=8)

    axes[0, 0].set_ylabel("Lowest entropy", fontsize=11)
    axes[1, 0].set_ylabel("Highest entropy", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def save_training_loss_curve(rows: List[Dict[str, float | int | str]], out_path: Path) -> None:
    """Save train-loss curves for pretrain and finetune phases."""
    pretrain = [row for row in rows if row["phase"] == "pretrain"]
    finetune = [row for row in rows if row["phase"] == "finetune"]

    plt.figure(figsize=(8, 5))
    if pretrain:
        plt.plot(
            [row["epoch"] for row in pretrain],
            [row["train_loss"] for row in pretrain],
            marker="o",
            label="Phase 1 (pretrain)",
        )
    if finetune:
        plt.plot(
            [row["epoch"] for row in finetune],
            [row["train_loss"] for row in finetune],
            marker="o",
            label="Phase 2 (finetune)",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_validation_loss_curve(rows: List[Dict[str, float | int | str]], out_path: Path) -> None:
    """Save validation-loss curves for pretrain and finetune phases."""
    pretrain = [row for row in rows if row["phase"] == "pretrain"]
    finetune = [row for row in rows if row["phase"] == "finetune"]

    plt.figure(figsize=(8, 5))
    if pretrain:
        plt.plot(
            [row["epoch"] for row in pretrain],
            [row["val_loss"] for row in pretrain],
            marker="o",
            label="Phase 1 val loss (CE)",
        )
    if finetune:
        plt.plot(
            [row["epoch"] for row in finetune],
            [row["val_loss"] for row in finetune],
            marker="o",
            label="Phase 2 val loss (KL)",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.title("Validation Loss Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_phase2_kl_curve(
    rows: List[Dict[str, float | int | str]],
    out_path: Path,
) -> Optional[Path]:
    """Save Phase 2 KL curve when finetune rows are present."""
    finetune = [row for row in rows if row["phase"] == "finetune"]
    if not finetune:
        return None

    plt.figure(figsize=(8, 5))
    epochs = [row["epoch"] for row in finetune]
    plt.plot(epochs, [row["train_loss"] for row in finetune], marker="o", label="Train KL")
    plt.plot(epochs, [row["val_loss"] for row in finetune], marker="o", label="Validation KL")
    plt.xlabel("Epoch")
    plt.ylabel("KL divergence")
    plt.title("Phase 2 KL Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def write_model_summary(out_path: Path) -> None:
    """Write model parameter counts grouped like model.py's summary."""
    model = build_resnet18_cifar()

    groups = {
        "stem": [model.conv1, model.bn1],
        "layer1": [model.layer1],
        "layer2": [model.layer2],
        "layer3": [model.layer3],
        "layer4": [model.layer4],
        "head": [model.fc],
    }

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    lines = []
    lines.append("Model Summary (ResNet-18 CIFAR variant)")
    lines.append("=" * 50)
    lines.append(f"Total parameters: {total_params:,}")
    lines.append(f"Trainable parameters: {trainable_params:,}")
    lines.append("")
    lines.append(f"{'Layer Group':<16}{'Parameters':>16}{'Share':>16}")
    lines.append("-" * 48)

    for name, modules in groups.items():
        group_params = sum(param.numel() for module in modules for param in module.parameters())
        share = (group_params / total_params) * 100.0
        lines.append(f"{name:<16}{group_params:>16,}{share:>15.2f}%")

    lines.append("-" * 48)
    lines.append(f"{'total':<16}{total_params:>16,}{100.0:>15.2f}%")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_data_summary(
    entropies: np.ndarray,
    split_sizes: Dict[str, int],
    out_path: Path,
) -> None:
    """Write CIFAR-10H entropy statistics and configured split sizes."""
    total = int(split_sizes["train"] + split_sizes["val"] + split_sizes["test"])

    lines = []
    lines.append("Data Summary (CIFAR-10H)")
    lines.append("=" * 30)
    lines.append(f"Entropy mean (bits): {entropies.mean():.6f}")
    lines.append(f"Entropy min (bits): {entropies.min():.6f}")
    lines.append(f"Entropy max (bits): {entropies.max():.6f}")
    lines.append("")
    lines.append("Configured split sizes:")
    lines.append(f"train: {split_sizes['train']}")
    lines.append(f"val: {split_sizes['val']}")
    lines.append(f"test: {split_sizes['test']}")
    lines.append(f"total: {total}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_training_artifacts(
    artifacts_dir: Path,
    training_rows: List[Dict[str, float | int | str]],
) -> List[Path]:
    """Generate training-curve artifacts from parsed training-log rows."""
    created_files: List[Path] = []

    train_curve_path = artifacts_dir / "training_loss_curve.png"
    save_training_loss_curve(training_rows, train_curve_path)
    created_files.append(train_curve_path)

    val_curve_path = artifacts_dir / "validation_loss_curve.png"
    save_validation_loss_curve(training_rows, val_curve_path)
    created_files.append(val_curve_path)

    phase2_path = artifacts_dir / "phase2_kl_curve.png"
    phase2_result = save_phase2_kl_curve(training_rows, phase2_path)
    if phase2_result is not None:
        created_files.append(phase2_result)

    return created_files


def generate_full_artifacts(project_root: Path, artifacts_dir: Path) -> List[Path]:
    """Generate all data-stage, training, and summary artifacts."""
    data_dir = project_root / "data"
    images, soft_labels = load_cifar10h(data_dir=str(data_dir))
    entropies = compute_entropy_bits(soft_labels)
    majority_class = np.argmax(soft_labels, axis=1)

    created_files: List[Path] = []

    entropy_hist_path = artifacts_dir / "entropy_histogram.png"
    save_entropy_histogram(entropies, entropy_hist_path)
    created_files.append(entropy_hist_path)

    per_class_entropy_path = artifacts_dir / "per_class_average_entropy.png"
    save_per_class_entropy(entropies, majority_class, per_class_entropy_path)
    created_files.append(per_class_entropy_path)

    confusion_path = artifacts_dir / "annotator_confusion_matrix.png"
    save_annotator_confusion_matrix(soft_labels, majority_class, confusion_path)
    created_files.append(confusion_path)

    low_high_path = artifacts_dir / "low_high_entropy_examples.png"
    save_low_high_entropy_examples(images, soft_labels, entropies, low_high_path)
    created_files.append(low_high_path)

    log_path = Path(CONFIG["log_path"])
    training_rows = load_training_log(log_path)
    created_files.extend(generate_training_artifacts(artifacts_dir, training_rows))

    model_summary_path = artifacts_dir / "model_summary.txt"
    write_model_summary(model_summary_path)
    created_files.append(model_summary_path)

    data_summary_path = artifacts_dir / "data_summary.txt"
    write_data_summary(entropies, CONFIG["cifar10h_split"], data_summary_path)
    created_files.append(data_summary_path)

    return created_files


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for artifact generation."""
    parser = argparse.ArgumentParser(description="Generate CIFAR-10H project artifacts.")
    parser.add_argument(
        "--training-only",
        action="store_true",
        help="Generate only training curves from training_log.csv.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate artifacts into project/artifacts/."""
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    artifacts_dir = ensure_artifacts_dir(project_root)
    log_path = Path(CONFIG["log_path"])

    if args.training_only:
        training_rows = load_training_log(log_path)
        created_files = generate_training_artifacts(artifacts_dir, training_rows)
    else:
        created_files = generate_full_artifacts(project_root, artifacts_dir)

    print("Created artifact files:")
    for path in created_files:
        print(path)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
