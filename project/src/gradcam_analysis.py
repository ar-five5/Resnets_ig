"""
gradcam_analysis.py - Grad-CAM and failure case analysis for CIFAR-10H.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

try:
    from .config import CONFIG
    from .dataset import (
        CIFAR10_MEAN,
        CIFAR10_STD,
        get_default_data_dir,
        load_cifar10h,
    )
    from .model import build_resnet18_cifar
except ImportError:
    from config import CONFIG
    from dataset import (
        CIFAR10_MEAN,
        CIFAR10_STD,
        get_default_data_dir,
        load_cifar10h,
    )
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


def entropy_bits(distribution: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Shannon entropy in bits for a probability vector."""
    safe = np.clip(distribution, eps, 1.0)
    return float(-np.sum(safe * np.log2(safe)))


def normalize_tensor(image: torch.Tensor) -> torch.Tensor:
    """Normalize a tensor image using CIFAR-10 mean/std."""
    mean = torch.tensor(CIFAR10_MEAN, device=image.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=image.device).view(1, 3, 1, 1)
    return (image - mean) / std


def build_test_indices(config: dict, total: int) -> np.ndarray:
    """
    Build CIFAR-10H test indices consistent with the configured split.

    Args:
        config: Configuration dictionary.
        total: Total dataset size.

    Returns:
        Array of indices for the test split.
    """
    split_cfg = config["cifar10h_split"]
    n_train = split_cfg["train"]
    n_val = split_cfg["val"]
    n_test = split_cfg["test"]

    if (n_train + n_val + n_test) != total:
        raise ValueError("CIFAR-10H split sizes must sum to the dataset size (10000).")

    rng = np.random.default_rng(config["random_seed"])
    indices = np.arange(total)
    rng.shuffle(indices)

    return indices[n_train + n_val : n_train + n_val + n_test]


def select_entropy_extremes(
    soft_labels: np.ndarray,
    indices: np.ndarray,
    k: int = 5,
) -> Tuple[List[int], List[int]]:
    """
    Select low- and high-entropy samples from the provided indices.

    Args:
        soft_labels: Full soft labels array.
        indices: Candidate indices (test split).
        k: Number of samples per group.

    Returns:
        (low_entropy_indices, high_entropy_indices)
    """
    entropies = np.array([entropy_bits(soft_labels[idx]) for idx in indices])
    sorted_idx = np.argsort(entropies)

    low_indices = indices[sorted_idx[:k]].tolist()
    high_indices = indices[sorted_idx[-k:]].tolist()
    return low_indices, high_indices


def compute_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    target_layer: torch.nn.Module,
) -> torch.Tensor:
    """
    Compute Grad-CAM heatmap for one input.

    Args:
        model: Model returning logits.
        input_tensor: Normalized tensor with shape (1, 3, 32, 32).
        target_class: Class index to compute Grad-CAM for.
        target_layer: Layer to hook for activations.

    Returns:
        Heatmap tensor with shape (32, 32) in [0, 1].
    """
    activations: Dict[str, torch.Tensor] = {}
    gradients: Dict[str, torch.Tensor] = {}

    def forward_hook(_, __, output):
        activations["value"] = output

        def grad_hook(grad):
            gradients["value"] = grad

        output.register_hook(grad_hook)

    hook_handle = target_layer.register_forward_hook(forward_hook)

    model.zero_grad(set_to_none=True)
    logits = model(input_tensor)
    score = logits[0, target_class]
    score.backward()

    hook_handle.remove()

    grads = gradients["value"]
    acts = activations["value"]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(32, 32), mode="bilinear", align_corners=False)

    cam = cam.squeeze(0).squeeze(0)
    cam_min = cam.min()
    cam_max = cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = torch.zeros_like(cam)
    return cam


def overlay_heatmap(image: np.ndarray, heatmap: torch.Tensor, alpha: float = 0.5) -> Image.Image:
    """
    Overlay a heatmap on an image.

    Args:
        image: Original image array (H, W, 3) uint8.
        heatmap: Heatmap tensor (H, W) in [0, 1].
        alpha: Blend factor for heatmap overlay.

    Returns:
        PIL Image with heatmap overlay.
    """
    heat = (heatmap.clamp(0, 1).detach().cpu().numpy() * 255.0).astype(np.uint8)
    heat_rgb = np.zeros_like(image)
    heat_rgb[..., 0] = heat

    base = Image.fromarray(image)
    overlay = Image.fromarray(heat_rgb)
    return Image.blend(base, overlay, alpha=alpha)


def save_side_by_side(original: np.ndarray, overlay: Image.Image, output_path: Path) -> None:
    """Save original and overlay images side-by-side."""
    original_img = Image.fromarray(original)
    combined = Image.new("RGB", (original_img.width * 2, original_img.height))
    combined.paste(original_img, (0, 0))
    combined.paste(overlay, (original_img.width, 0))
    combined.save(output_path)


def analyze_sample(
    model: torch.nn.Module,
    device: torch.device,
    image: np.ndarray,
    soft_label: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, torch.Tensor]:
    """
    Run model prediction and compute Grad-CAM for one sample.

    Returns:
        (pred_probs, soft_label, entropy, heatmap)
    """
    tensor = transforms.ToTensor()(Image.fromarray(image)).unsqueeze(0).to(device)
    tensor_norm = normalize_tensor(tensor)

    model.eval()
    with torch.no_grad():
        logits = model(tensor_norm)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_class = int(np.argmax(probs))

    heatmap = compute_gradcam(model, tensor_norm, pred_class, model.layer4)
    entropy = entropy_bits(soft_label)
    return probs, soft_label, entropy, heatmap


def cam_stats(heatmap: torch.Tensor, threshold: float = 0.6) -> Dict[str, float]:
    """Compute simple concentration stats for a heatmap."""
    heat = heatmap.clamp(0, 1)
    frac_high = float((heat > threshold).float().mean().item())
    mean_val = float(heat.mean().item())
    return {"frac_high": frac_high, "mean": mean_val}


def write_analysis(
    output_path: Path,
    entries: List[Dict[str, object]],
) -> None:
    """
    Write analysis summary to a text file.

    Args:
        output_path: Output text file path.
        entries: List of per-sample analysis dicts.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    failures = [e for e in entries if not e["correct"]]
    low_entries = [e for e in entries if e["group"] == "low_entropy"]
    high_entries = [e for e in entries if e["group"] == "high_entropy"]

    with output_path.open("w", encoding="utf-8") as file:
        file.write("Grad-CAM Failure Case Analysis\n")
        file.write("----------------------------------\n")
        file.write("Per-sample details:\n")
        for entry in entries:
            file.write("\n")
            file.write(f"Sample: {entry['name']}\n")
            file.write(f"Group: {entry['group']}\n")
            file.write(f"Entropy (bits): {entry['entropy']:.4f}\n")
            file.write(f"Predicted class: {entry['pred_class']}\n")
            file.write(f"Target class: {entry['target_class']}\n")
            file.write(f"Correct: {entry['correct']}\n")
            file.write(f"Grad-CAM mean activation: {entry['cam_mean']:.4f}\n")
            file.write(f"Grad-CAM frac > 0.6: {entry['cam_frac_high']:.4f}\n")
            file.write("Predicted distribution:\n")
            file.write("  " + " ".join([f"{p:.4f}" for p in entry["pred_probs"]]) + "\n")
            file.write("Target distribution:\n")
            file.write("  " + " ".join([f"{p:.4f}" for p in entry["target_probs"]]) + "\n")

        file.write("\n----------------------------------\n")
        file.write("Summary\n")
        file.write("----------------------------------\n")
        file.write(f"Low-entropy samples: {len(low_entries)}\n")
        file.write(f"High-entropy samples: {len(high_entries)}\n")
        file.write(f"Failures (top-1 mismatch): {len(failures)}\n")

        if failures:
            file.write("Failure cases:\n")
            for entry in failures:
                file.write(f"  - {entry['name']} (entropy={entry['entropy']:.4f})\n")
        else:
            file.write("Failure cases: none\n")

        file.write("\nInterpretation notes:\n")
        file.write("- Grad-CAM overlays are saved for visual inspection of focus regions.\n")
        file.write("- Higher entropy cases indicate softer human consensus and tend to be harder.\n")
        file.write("- Disagreement likely arises when images contain ambiguous or mixed cues.\n")


def main() -> None:
    """Run Grad-CAM analysis and save overlays plus a short report."""
    device = resolve_device(CONFIG)
    data_dir = CONFIG.get("data_dir", get_default_data_dir())
    images, soft_labels = load_cifar10h(data_dir=data_dir)

    test_indices = build_test_indices(CONFIG, len(images))
    low_indices, high_indices = select_entropy_extremes(soft_labels, test_indices, k=5)

    model = build_resnet18_cifar().to(device)
    load_checkpoint(model, CONFIG["checkpoint_path"], device)

    output_dir = Path(__file__).resolve().parents[1] / "artifacts" / "gradcam"
    output_dir.mkdir(parents=True, exist_ok=True)

    entries: List[Dict[str, object]] = []

    def process_group(indices: List[int], group: str) -> None:
        for idx in indices:
            image = images[idx]
            soft_label = soft_labels[idx]

            pred_probs, target_probs, entropy, heatmap = analyze_sample(
                model, device, image, soft_label
            )

            pred_class = int(np.argmax(pred_probs))
            target_class = int(np.argmax(target_probs))
            correct = pred_class == target_class

            overlay = overlay_heatmap(image, heatmap)
            name = f"{group}_idx{idx}.png"
            save_side_by_side(image, overlay, output_dir / name)

            stats = cam_stats(heatmap)

            entries.append(
                {
                    "name": name,
                    "group": group,
                    "entropy": entropy,
                    "pred_class": f"{pred_class} ({CIFAR10_CLASSES[pred_class]})",
                    "target_class": f"{target_class} ({CIFAR10_CLASSES[target_class]})",
                    "correct": correct,
                    "cam_mean": stats["mean"],
                    "cam_frac_high": stats["frac_high"],
                    "pred_probs": pred_probs.tolist(),
                    "target_probs": target_probs.tolist(),
                }
            )

    process_group(low_indices, "low_entropy")
    process_group(high_indices, "high_entropy")

    analysis_path = Path(__file__).resolve().parents[1] / "artifacts" / "failure_analysis.txt"
    write_analysis(analysis_path, entries)


if __name__ == "__main__":
    main()
