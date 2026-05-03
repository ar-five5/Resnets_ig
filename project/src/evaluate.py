"""
evaluate.py - evaluation metrics for CIFAR-10H test split.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

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


PRECISION_K_VALUES = (100, 200, 500)
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42


def resolve_device(config: dict) -> torch.device:
    """Resolve torch device from config with safe fallback."""
    requested = str(config.get("device", "cpu")).lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> Dict:
    """Load checkpoint weights into the model."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    return checkpoint if isinstance(checkpoint, dict) else {}


def kl_divergence_np(target: np.ndarray, prediction: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute KL(target || prediction) per sample (numpy)."""
    target_safe = np.clip(target, eps, 1.0)
    pred_safe = np.clip(prediction, eps, 1.0)
    return np.sum(target_safe * (np.log(target_safe) - np.log(pred_safe)), axis=1)


def js_divergence_np(target: np.ndarray, prediction: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute Jensen-Shannon divergence per sample (numpy)."""
    mix = 0.5 * (target + prediction)
    return 0.5 * (kl_divergence_np(target, mix, eps) + kl_divergence_np(prediction, mix, eps))


def cosine_similarity_np(target: np.ndarray, prediction: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute per-sample cosine similarity between two probability matrices."""
    num = np.sum(target * prediction, axis=1)
    denom = np.linalg.norm(target, axis=1) * np.linalg.norm(prediction, axis=1) + eps
    return num / denom


def entropy_bits_np(distributions: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Shannon entropy in bits per row."""
    safe = np.clip(distributions, eps, 1.0)
    return -np.sum(distributions * np.log2(safe), axis=1)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient between two 1-D vectors."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = np.sqrt((x_centered ** 2).sum() * (y_centered ** 2).sum())
    if denom == 0.0:
        return 0.0
    return float((x_centered * y_centered).sum() / denom)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation between two 1-D vectors."""
    x_ranks = _rankdata(x)
    y_ranks = _rankdata(y)
    return pearson_corr(x_ranks, y_ranks)


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average-rank tie-handling, matching scipy.stats.rankdata default."""
    values = np.asarray(values, dtype=np.float64)
    sorter = np.argsort(values, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(values))
    sorted_vals = values[sorter]

    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[i:j] = avg_rank
        i = j
    return ranks[inv]


def precision_at_k(
    predicted_entropy: np.ndarray,
    true_entropy: np.ndarray,
    k: int,
) -> float:
    """
    Precision@K for high-disagreement retrieval.

    Both rankings sort descending by entropy. Score is the fraction of the top-K
    predicted-entropy items that fall in the top-K true-entropy set.
    """
    if k <= 0 or k > len(predicted_entropy):
        raise ValueError(f"K={k} out of range (n={len(predicted_entropy)}).")

    pred_top = np.argsort(-predicted_entropy)[:k]
    true_top = set(np.argsort(-true_entropy)[:k].tolist())
    hits = sum(1 for idx in pred_top if idx in true_top)
    return hits / k


def collect_predictions(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on the test loader and return (preds, targets) as numpy."""
    model.eval()
    pred_chunks: List[np.ndarray] = []
    target_chunks: List[np.ndarray] = []

    with torch.no_grad():
        for images, soft_labels in test_loader:
            images = images.to(device)
            soft_labels = soft_labels.to(device)

            logits = model(images)
            probs = F.softmax(logits, dim=1)

            pred_chunks.append(probs.cpu().numpy())
            target_chunks.append(soft_labels.cpu().numpy())

    predictions = np.concatenate(pred_chunks, axis=0)
    targets = np.concatenate(target_chunks, axis=0)
    return predictions, targets


def compute_per_sample_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute per-sample KL, JSD, cosine, correctness arrays."""
    kl_per = kl_divergence_np(targets, predictions)
    jsd_per = js_divergence_np(targets, predictions)
    cosine_per = cosine_similarity_np(targets, predictions)
    correct_per = (predictions.argmax(axis=1) == targets.argmax(axis=1)).astype(np.float64)
    return {"kl": kl_per, "jsd": jsd_per, "cosine": cosine_per, "correct": correct_per}


def bootstrap_mean_std(
    per_sample: Dict[str, np.ndarray],
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Tuple[float, float]]:
    """Bootstrap (mean, std) for each per-sample metric vector."""
    n = len(next(iter(per_sample.values())))
    rng = np.random.default_rng(seed)
    results: Dict[str, Tuple[float, float]] = {}
    for name, values in per_sample.items():
        boot_means = np.empty(n_resamples, dtype=np.float64)
        for i in range(n_resamples):
            idx = rng.integers(0, n, size=n)
            boot_means[i] = float(values[idx].mean())
        results[name] = (float(boot_means.mean()), float(boot_means.std(ddof=1)))
    return results


def save_metrics(
    output_path: Path,
    point_estimates: Dict[str, float],
    bootstrap_results: Dict[str, Tuple[float, float]],
    entropy_corr: Dict[str, float],
    precision_results: Dict[int, float],
) -> None:
    """Save evaluation metrics to a text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kl = point_estimates["kl"]
    jsd = point_estimates["jsd"]
    cosine = point_estimates["cosine"]
    top1 = point_estimates["correct"]

    with output_path.open("w", encoding="utf-8") as file:
        file.write("----------------------------------\n")
        file.write("Evaluation Results (CIFAR-10H Test)\n")
        file.write("----------------------------------\n")
        file.write(f"KL Divergence       : {kl:.4f}\n")
        file.write(f"JSD                 : {jsd:.4f}\n")
        file.write(f"Cosine Similarity   : {cosine:.4f}\n")
        file.write(f"Top-1 Accuracy      : {top1 * 100.0:.2f} %\n")
        file.write("----------------------------------\n")
        file.write("Bootstrap (n=1000) mean ± std\n")
        file.write("----------------------------------\n")
        for name, label in [
            ("kl", "KL Divergence"),
            ("jsd", "JSD"),
            ("cosine", "Cosine Similarity"),
            ("correct", "Top-1 Accuracy"),
        ]:
            mean, std = bootstrap_results[name]
            if name == "correct":
                file.write(f"{label:<20}: {mean * 100.0:.2f} ± {std * 100.0:.2f} %\n")
            else:
                file.write(f"{label:<20}: {mean:.4f} ± {std:.4f}\n")
        file.write("----------------------------------\n")
        file.write("Entropy correlation (predicted vs true)\n")
        file.write("----------------------------------\n")
        file.write(f"Pearson  r          : {entropy_corr['pearson']:.4f}\n")
        file.write(f"Spearman rho        : {entropy_corr['spearman']:.4f}\n")
        file.write("----------------------------------\n")
        file.write("Precision@K (top-K by predicted entropy vs true entropy)\n")
        file.write("----------------------------------\n")
        for k in PRECISION_K_VALUES:
            file.write(f"Precision@{k:<6}: {precision_results[k]:.4f}\n")
        file.write("----------------------------------\n")


def main() -> None:
    """Run evaluation on CIFAR-10H test split and save results."""
    device = resolve_device(CONFIG)

    model = build_resnet18_cifar().to(device)
    checkpoint_path = CONFIG["checkpoint_path"]
    load_checkpoint(model, checkpoint_path, device)

    _, _, test_loader = get_dataloaders(CONFIG)

    predictions, targets = collect_predictions(model, test_loader, device)

    per_sample = compute_per_sample_metrics(predictions, targets)
    point_estimates = {name: float(values.mean()) for name, values in per_sample.items()}

    pred_entropy = entropy_bits_np(predictions)
    true_entropy = entropy_bits_np(targets)
    entropy_corr = {
        "pearson": pearson_corr(pred_entropy, true_entropy),
        "spearman": spearman_corr(pred_entropy, true_entropy),
    }

    precision_results = {
        k: precision_at_k(pred_entropy, true_entropy, k) for k in PRECISION_K_VALUES
    }

    bootstrap_results = bootstrap_mean_std(per_sample)

    print("----------------------------------")
    print("Evaluation Results (CIFAR-10H Test)")
    print("----------------------------------")
    print(f"KL Divergence       : {point_estimates['kl']:.4f}")
    print(f"JSD                 : {point_estimates['jsd']:.4f}")
    print(f"Cosine Similarity   : {point_estimates['cosine']:.4f}")
    print(f"Top-1 Accuracy      : {point_estimates['correct'] * 100.0:.2f} %")
    print(f"Pearson r (entropy) : {entropy_corr['pearson']:.4f}")
    print(f"Spearman rho        : {entropy_corr['spearman']:.4f}")
    for k in PRECISION_K_VALUES:
        print(f"Precision@{k:<6}: {precision_results[k]:.4f}")
    print("----------------------------------")

    project_root = Path(__file__).resolve().parents[1]
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    save_metrics(
        artifacts_dir / "evaluation_metrics.txt",
        point_estimates,
        bootstrap_results,
        entropy_corr,
        precision_results,
    )

    np.save(artifacts_dir / "test_predictions.npy", predictions)
    np.save(artifacts_dir / "test_targets.npy", targets)


if __name__ == "__main__":
    main()
