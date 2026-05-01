"""
train.py - two-phase training loop for CIFAR-10H disagreement prediction.
"""

import csv
import os
import random
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms

try:
    from .config import CONFIG
    from .dataset import CIFAR10_MEAN, CIFAR10_STD, get_dataloaders, get_default_data_dir
    from .model import build_resnet18_cifar
except ImportError:
    from config import CONFIG
    from dataset import CIFAR10_MEAN, CIFAR10_STD, get_dataloaders, get_default_data_dir
    from model import build_resnet18_cifar


def set_random_seed(seed: int) -> None:
    """
    Set random seed for Python, NumPy, and PyTorch.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def init_log_file(log_path: str) -> None:
    """
    Initialize the CSV training log with the required header.

    Args:
        log_path: Output CSV path.
    """
    parent = os.path.dirname(log_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(log_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "phase", "train_loss", "val_loss"])


def append_log_row(log_path: str, epoch: int, phase: str, train_loss: float, val_loss: float) -> None:
    """
    Append one epoch entry to the CSV log.

    Args:
        log_path: Output CSV path.
        epoch: Epoch number (1-based).
        phase: "pretrain" or "finetune".
        train_loss: Mean train loss for the epoch.
        val_loss: Mean validation loss for the epoch.
    """
    with open(log_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([epoch, phase, float(train_loss), float(val_loss)])


def build_cifar10_pretrain_loaders(config: dict) -> Tuple[DataLoader, DataLoader]:
    """
    Build CIFAR-10 train/val loaders for Phase 1 pretraining.

    Uses a fixed random split with the same seed from config.
    Validation size is fixed to config["cifar10h_split"]["val"] (=2000).

    Args:
        config: Global configuration dictionary.

    Returns:
        (train_loader, val_loader)
    """
    data_dir = config.get("data_dir", get_default_data_dir())
    batch_size = int(config["batch_size"])
    num_workers = int(config["num_workers"])
    random_seed = int(config["random_seed"])
    val_size = int(config["cifar10h_split"]["val"])

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    # Same dataset split, different transforms for train vs val behavior.
    cifar_train_aug = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    cifar_train_eval = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=eval_transform,
    )

    total = len(cifar_train_aug)
    if val_size <= 0 or val_size >= total:
        raise ValueError(f"Invalid validation size: {val_size}. Must be between 1 and {total - 1}.")

    indices = np.arange(total)
    rng = np.random.default_rng(random_seed)
    rng.shuffle(indices)

    val_indices = indices[:val_size].tolist()
    train_indices = indices[val_size:].tolist()

    train_subset = Subset(cifar_train_aug, train_indices)
    val_subset = Subset(cifar_train_eval, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def set_backbone_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    """Set requires_grad for all backbone parameters while leaving `fc` untouched."""
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = requires_grad


def build_phase2_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Build a Phase 2 Adam optimizer over currently trainable parameters."""
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters available for Phase 2 optimizer.")

    return torch.optim.Adam(
        trainable_params,
        lr=float(config["lr_finetune"]),
        weight_decay=float(config.get("weight_decay_finetune", 0.0)),
    )


def build_phase2_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> Optional[torch.optim.lr_scheduler.ReduceLROnPlateau]:
    """Build optional ReduceLROnPlateau scheduler for Phase 2 validation KL."""
    if not bool(config.get("use_phase2_scheduler", False)):
        return None

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config.get("phase2_scheduler_factor", 0.5)),
        patience=int(config.get("phase2_scheduler_patience", 3)),
        min_lr=float(config.get("min_lr_finetune", 1e-6)),
    )


def get_optimizer_lr(optimizer: torch.optim.Optimizer) -> float:
    """Return the current learning rate from the first optimizer parameter group."""
    return float(optimizer.param_groups[0]["lr"])


def pretrain_phase(
    model: nn.Module,
    config: dict,
    device: torch.device,
    phase1_ckpt_path: str,
) -> Dict[str, float]:
    """
    Phase 1: pretrain model on CIFAR-10 hard labels.

    Checkpoint criterion: best validation accuracy.
    Early stopping criterion: validation loss.

    Args:
        model: Model that outputs raw logits.
        config: Global configuration dictionary.
        device: Runtime torch device.
        phase1_ckpt_path: Output checkpoint path for best Phase 1 model.

    Returns:
        Dict with best phase metrics.
    """
    # -- PHASE 1: PRETRAINING --
    train_loader, val_loader = build_cifar10_pretrain_loaders(config)
    criterion = nn.CrossEntropyLoss()
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
    model: nn.Module,
    config: dict,
    device: torch.device,
    phase1_ckpt_path: str,
    final_ckpt_path: str,
) -> Dict[str, float]:
    """
    Phase 2: fine-tune on CIFAR-10H soft labels with KL divergence.

    Checkpoint criterion: best validation KL loss.
    Early stopping criterion: validation KL loss.

    Args:
        model: Model that outputs raw logits.
        config: Global configuration dictionary.
        device: Runtime torch device.
        phase1_ckpt_path: Input checkpoint path from Phase 1.
        final_ckpt_path: Final output checkpoint path (`best_model.pt`).

    Returns:
        Dict with best phase metrics.
    """
    # -- PHASE 2: FINE-TUNING --
    if not os.path.isfile(phase1_ckpt_path):
        raise FileNotFoundError(f"Missing Phase 1 checkpoint: {phase1_ckpt_path}")

    phase1_ckpt = torch.load(phase1_ckpt_path, map_location=device)
    model.load_state_dict(phase1_ckpt["model_state"])
    model.to(device)

    train_loader, val_loader, _ = get_dataloaders(config)

    criterion = nn.KLDivLoss(reduction="batchmean")
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

    best_val_kl = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        if backbone_frozen and epoch == freeze_backbone_epochs + 1:
            set_backbone_requires_grad(model, True)
            optimizer = build_phase2_optimizer(model, config)
            scheduler = build_phase2_scheduler(optimizer, config)
            if scheduler is not None:
                scheduler.best = best_val_kl
            backbone_frozen = False
            print(f"[Phase 2] Unfreezing full model at epoch {epoch}.")

        model.train()
        train_loss_sum = 0.0
        train_count = 0

        train_bar = tqdm(train_loader, desc=f"Phase 2 | Epoch {epoch}/{epochs}", leave=False)
        for images, soft_labels in train_bar:
            images = images.to(device)
            soft_labels = soft_labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            log_probs = F.log_softmax(logits, dim=1)
            loss = criterion(log_probs, soft_labels)
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
                log_probs = F.log_softmax(logits, dim=1)
                loss = criterion(log_probs, soft_labels)

                val_loss_sum += loss.item() * images.size(0)
                val_count += images.size(0)

        val_kl = val_loss_sum / val_count
        if scheduler is not None:
            scheduler.step(val_kl)
        current_lr = get_optimizer_lr(optimizer)

        append_log_row(log_path, epoch, "finetune", train_loss, val_kl)
        print(
            f"[Phase 2] Epoch {epoch:>3}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_kl={val_kl:.4f} | lr={current_lr:.6g}"
        )

        if val_kl < best_val_kl:
            best_val_kl = val_kl
            no_improve = 0
            torch.save(
                {
                    "phase": "finetune",
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_kl": best_val_kl,
                },
                final_ckpt_path,
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[Phase 2] Early stopping at epoch {epoch}.")
                break

    return {"best_val_loss": best_val_kl}


def train_two_phase(config: dict = CONFIG) -> Dict[str, float]:
    """
    Train model with Phase 1 pretraining + Phase 2 fine-tuning.

    Args:
        config: Global configuration dictionary.

    Returns:
        Summary dictionary with best metrics and runtime.
    """
    set_random_seed(int(config["random_seed"]))
    device = resolve_device(config)

    checkpoint_path = config["checkpoint_path"]
    ckpt_dir = os.path.dirname(checkpoint_path)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    root, ext = os.path.splitext(checkpoint_path)
    phase1_ckpt_path = f"{root}_phase1{ext or '.pt'}"

    init_log_file(config["log_path"])

    print("=" * 70)
    print("Starting Two-Phase Training")
    print(f"Device: {device}")
    print("=" * 70)

    start_time = time.time()
    model = build_resnet18_cifar().to(device)

    phase1 = pretrain_phase(model, config, device, phase1_ckpt_path)

    print("\n" + "-" * 70)
    print("Phase 1 complete. Starting Phase 2.")
    print("-" * 70)

    phase2 = finetune_phase(model, config, device, phase1_ckpt_path, checkpoint_path)

    total_minutes = (time.time() - start_time) / 60.0

    print("\n" + "=" * 70)
    print("Training Summary")
    print("=" * 70)
    print(f"Best Phase 1 val loss (CE): {phase1['best_val_loss']:.6f}")
    print(f"Best Phase 1 val accuracy : {phase1['best_val_acc']:.6f}")
    print(f"Best Phase 2 val loss (KL): {phase2['best_val_loss']:.6f}")
    print(f"Total training time        : {total_minutes:.2f} minutes")
    print(f"Saved best model           : {checkpoint_path}")
    print("=" * 70)

    return {
        "phase1_best_val_loss": phase1["best_val_loss"],
        "phase1_best_val_acc": phase1["best_val_acc"],
        "phase2_best_val_loss": phase2["best_val_loss"],
        "total_time_minutes": total_minutes,
    }


if __name__ == "__main__":
    train_two_phase(CONFIG)
