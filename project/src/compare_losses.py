"""
compare_losses.py - train and evaluate CIFAR-10H models with different losses.
"""

import os
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from .config import CONFIG
    from .dataset import get_dataloaders
    from .model import build_resnet18_cifar
    from .train import (
        append_log_row,
        build_cifar10_pretrain_loaders,
        build_phase2_optimizer,
        build_phase2_scheduler,
        get_optimizer_lr,
        init_log_file,
        resolve_device,
        set_backbone_requires_grad,
        set_random_seed,
    )
except ImportError:
    from config import CONFIG
    from dataset import get_dataloaders
    from model import build_resnet18_cifar
    from train import (
        append_log_row,
        build_cifar10_pretrain_loaders,
        build_phase2_optimizer,
        build_phase2_scheduler,
        get_optimizer_lr,
        init_log_file,
        resolve_device,
        set_backbone_requires_grad,
        set_random_seed,
    )


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


def loss_kl(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """KL divergence loss for soft labels."""
    log_probs = F.log_softmax(logits, dim=1)
    return F.kl_div(log_probs, target, reduction="batchmean")


def loss_jsd(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Jensen-Shannon divergence loss for soft labels."""
    probs = F.softmax(logits, dim=1)
    target_safe = torch.clamp(target, eps, 1.0)
    probs_safe = torch.clamp(probs, eps, 1.0)
    mix = 0.5 * (target_safe + probs_safe)

    kl_t = torch.sum(target_safe * (torch.log(target_safe) - torch.log(mix)), dim=1)
    kl_p = torch.sum(probs_safe * (torch.log(probs_safe) - torch.log(mix)), dim=1)
    return 0.5 * (kl_t + kl_p).mean()


def loss_soft_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Soft cross-entropy loss for soft labels."""
    log_probs = F.log_softmax(logits, dim=1)
    per_sample = -torch.sum(target * log_probs, dim=1)
    return per_sample.mean()


def pretrain_phase(
    model: torch.nn.Module,
    config: dict,
    device: torch.device,
    phase1_ckpt_path: str,
) -> Dict[str, float]:
    """
    Phase 1: pretrain model on CIFAR-10 hard labels.

    Args:
        model: Model that outputs raw logits.
        config: Global configuration dictionary.
        device: Runtime torch device.
        phase1_ckpt_path: Output checkpoint path for best Phase 1 model.

    Returns:
        Dict with best phase metrics.
    """
    train_loader, val_loader = build_cifar10_pretrain_loaders(config)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["lr_pretrain"]),
        weight_decay=float(config.get("weight_decay_pretrain", 0.0)),
    )

    epochs = int(config["epochs_pretrain"])
    patience = int(config["early_stopping_patience"])
    log_path = config["log_path"]

    best_val_acc = -1.0
    best_val_loss = float("inf")
    no_improve = 0

    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        train_bar = tqdm(train_loader, desc=f"Phase 1 | Epoch {epoch}/{epochs}", leave=False)
        for images, labels in train_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)
            train_count += images.size(0)
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = train_loss_sum / train_count

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        val_correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)

                val_loss_sum += loss.item() * images.size(0)
                val_count += images.size(0)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()

        val_loss = val_loss_sum / val_count
        val_acc = val_correct / val_count

        append_log_row(log_path, epoch, "pretrain", train_loss, val_loss)
        print(
            f"[Phase 1] Epoch {epoch:>3}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "phase": "pretrain",
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_acc": best_val_acc,
                    "val_loss": val_loss,
                },
                phase1_ckpt_path,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[Phase 1] Early stopping at epoch {epoch}.")
                break

    return {"best_val_loss": best_val_loss, "best_val_acc": best_val_acc}


def finetune_phase(
    model: torch.nn.Module,
    config: dict,
    device: torch.device,
    phase1_ckpt_path: str,
    final_ckpt_path: str,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    loss_name: str,
) -> Dict[str, float]:
    """
    Phase 2: fine-tune on CIFAR-10H soft labels with a custom loss.

    Args:
        model: Model that outputs raw logits.
        config: Global configuration dictionary.
        device: Runtime torch device.
        phase1_ckpt_path: Input checkpoint path from Phase 1.
        final_ckpt_path: Output checkpoint path for best Phase 2 model.
        loss_fn: Callable loss function.
        loss_name: String label for logging.

    Returns:
        Dict with best phase metrics.
    """
    if not os.path.isfile(phase1_ckpt_path):
        raise FileNotFoundError(f"Missing Phase 1 checkpoint: {phase1_ckpt_path}")

    phase1_ckpt = torch.load(phase1_ckpt_path, map_location=device)
    model.load_state_dict(phase1_ckpt["model_state"])
    model.to(device)

    train_loader, val_loader, _ = get_dataloaders(config)

    freeze_backbone_epochs = max(0, int(config.get("freeze_backbone_epochs", 0)))
    backbone_frozen = freeze_backbone_epochs > 0
    if backbone_frozen:
        set_backbone_requires_grad(model, False)
        for param in model.fc.parameters():
            param.requires_grad = True
        print(f"[Phase 2] Freezing backbone for first {freeze_backbone_epochs} epochs.")

    optimizer = build_phase2_optimizer(model, config)
    scheduler = build_phase2_scheduler(optimizer, config)

    epochs = int(config["epochs_finetune"])
    patience = int(config["early_stopping_patience"])
    log_path = config["log_path"]

    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        if backbone_frozen and epoch == freeze_backbone_epochs + 1:
            set_backbone_requires_grad(model, True)
            optimizer = build_phase2_optimizer(model, config)
            scheduler = build_phase2_scheduler(optimizer, config)
            if scheduler is not None:
                scheduler.best = best_val_loss
            backbone_frozen = False
            print(f"[Phase 2] Unfreezing full model at epoch {epoch}.")

        model.train()
        train_loss_sum = 0.0
        train_count = 0

        train_bar = tqdm(train_loader, desc=f"Phase 2 ({loss_name}) | Epoch {epoch}/{epochs}", leave=False)
        for images, soft_labels in train_bar:
            images = images.to(device)
            soft_labels = soft_labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, soft_labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)
            train_count += images.size(0)
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = train_loss_sum / train_count

        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for images, soft_labels in val_loader:
                images = images.to(device)
                soft_labels = soft_labels.to(device)

                logits = model(images)
                loss = loss_fn(logits, soft_labels)

                val_loss_sum += loss.item() * images.size(0)
                val_count += images.size(0)

        val_loss = val_loss_sum / val_count
        if scheduler is not None:
            scheduler.step(val_loss)
        current_lr = get_optimizer_lr(optimizer)

        append_log_row(log_path, epoch, "finetune", train_loss, val_loss)
        print(
            f"[Phase 2 - {loss_name}] Epoch {epoch:>3}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={current_lr:.6g}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(
                {
                    "phase": "finetune",
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_loss": best_val_loss,
                    "loss_name": loss_name,
                },
                final_ckpt_path,
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[Phase 2 - {loss_name}] Early stopping at epoch {epoch}.")
                break

    return {"best_val_loss": best_val_loss}


def evaluate_model(model: torch.nn.Module, test_loader, device: torch.device) -> Tuple[float, float, float, float]:
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


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    """
    Load checkpoint weights into the model.

    Args:
        model: Model to populate with weights.
        checkpoint_path: Path to the checkpoint file.
        device: Torch device for map_location.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)


def train_and_evaluate(
    loss_name: str,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    base_config: dict,
    output_dir: Path,
) -> Tuple[float, float, float, float]:
    """
    Train a model with the specified loss and evaluate on the test split.

    Args:
        loss_name: Label for the loss.
        loss_fn: Callable loss function.
        base_config: Base configuration dictionary.
        output_dir: Directory to store checkpoints and logs.

    Returns:
        Tuple of (kl_mean, jsd_mean, cosine_mean, top1_acc).
    """
    config = dict(base_config)
    config["checkpoint_path"] = str(output_dir / f"best_model_{loss_name.lower()}.pt")
    config["log_path"] = str(output_dir / f"training_log_{loss_name.lower()}.csv")

    phase1_ckpt_path = str(output_dir / f"best_model_{loss_name.lower()}_phase1.pt")

    init_log_file(config["log_path"])

    device = resolve_device(config)
    model = build_resnet18_cifar().to(device)

    pretrain_phase(model, config, device, phase1_ckpt_path)

    finetune_phase(
        model,
        config,
        device,
        phase1_ckpt_path,
        config["checkpoint_path"],
        loss_fn,
        loss_name,
    )

    load_checkpoint(model, config["checkpoint_path"], device)

    _, _, test_loader = get_dataloaders(config)
    return evaluate_model(model, test_loader, device)


def save_comparison(output_path: Path, rows: Dict[str, Tuple[float, float, float, float]]) -> None:
    """
    Save the loss comparison table to a text file.

    Args:
        output_path: Output text file path.
        rows: Mapping of loss name to metric tuple.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        file.write("----------------------------------\n")
        file.write("Loss Function Comparison\n")
        file.write("----------------------------------\n")
        file.write("Loss        KL     JSD    Cosine    Acc\n")
        file.write("----------------------------------\n")
        for loss_name, (kl, jsd, cosine, acc) in rows.items():
            file.write(
                f"{loss_name:<10}"
                f"{kl:>6.3f} "
                f"{jsd:>6.3f} "
                f"{cosine:>8.3f} "
                f"{acc * 100.0:>7.2f}\n"
            )
        file.write("----------------------------------\n")


def print_comparison(rows: Dict[str, Tuple[float, float, float, float]]) -> None:
    """Print the loss comparison table to stdout."""
    print("----------------------------------")
    print("Loss Function Comparison")
    print("----------------------------------")
    print("Loss        KL     JSD    Cosine    Acc")
    print("----------------------------------")
    for loss_name, (kl, jsd, cosine, acc) in rows.items():
        print(f"{loss_name:<10}{kl:>6.3f} {jsd:>6.3f} {cosine:>8.3f} {acc * 100.0:>7.2f}")
    print("----------------------------------")


def main() -> None:
    """Train and evaluate KL, JSD, and Soft CE models."""
    base_config = dict(CONFIG)
    base_config["epochs_pretrain"] = 10
    base_config["epochs_finetune"] = 15

    set_random_seed(int(base_config["random_seed"]))

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "artifacts" / "loss_compare"
    output_dir.mkdir(parents=True, exist_ok=True)

    losses = {
        "KL": loss_kl,
        "JSD": loss_jsd,
        "Soft CE": loss_soft_ce,
    }

    results: Dict[str, Tuple[float, float, float, float]] = {}
    for loss_name, loss_fn in losses.items():
        print("=" * 70)
        print(f"Training and evaluating: {loss_name}")
        print("=" * 70)
        metrics = train_and_evaluate(loss_name, loss_fn, base_config, output_dir)
        results[loss_name] = metrics

    print_comparison(results)

    output_path = project_root / "artifacts" / "loss_comparison.txt"
    save_comparison(output_path, results)


if __name__ == "__main__":
    main()
