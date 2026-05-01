"""
config.py - central configuration for CIFAR-10H training.
"""

from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


CONFIG = {
    # Global random seed for reproducible splits and training.
    "random_seed": 42,

    # Batch size used for all DataLoaders.
    "batch_size": 128,

    # Adam learning rate for Phase 1: CIFAR-10 hard-label pretraining.
    "lr_pretrain": 0.001,
    "weight_decay_pretrain": 1e-4,

    # Adam learning rate for Phase 2: CIFAR-10H soft-label fine-tuning.
    "lr_finetune": 0.0001,
    "weight_decay_finetune": 1e-4,
    "min_lr_finetune": 1e-6,
    "use_phase2_scheduler": True,
    "phase2_scheduler_factor": 0.5,
    "phase2_scheduler_patience": 3,
    "freeze_backbone_epochs": 3,

    # Number of epochs for CIFAR-10 pretraining.
    "epochs_pretrain": 50,

    # Number of epochs for CIFAR-10H fine-tuning.
    "epochs_finetune": 50,

    # CIFAR-10H has 10,000 soft-label examples only.
    "cifar10h_split": {"train": 6000, "val": 2000, "test": 2000},

    # Patience for early stopping during training.
    "early_stopping_patience": 10,

    # Path for saving the best model checkpoint.
    "checkpoint_path": str(PROJECT_ROOT / "best_model.pt"),

    # CSV path for epoch-level logs.
    "log_path": str(PROJECT_ROOT / "training_log.csv"),

    # Automatically choose the best available device.
    "device": (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    ),

    # Number of DataLoader worker processes.
    "num_workers": 2,
}


if __name__ == "__main__":
    print("Resolved device:", CONFIG["device"])
    print("Batch size:", CONFIG["batch_size"])
    print("Pretrain/Fine-tune epochs:", CONFIG["epochs_pretrain"], CONFIG["epochs_finetune"])
