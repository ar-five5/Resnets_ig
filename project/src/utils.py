"""Shared utilities: device resolution, checkpoint loading, entropy."""

import os

import numpy as np
import torch


def resolve_device(config: dict) -> torch.device:
    """Resolve torch device from config with safe fallback."""
    requested = str(config.get("device", "cpu")).lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> dict:
    """Load checkpoint weights into the model, return checkpoint metadata dict."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    return checkpoint if isinstance(checkpoint, dict) else {}


def entropy_bits_np(distributions: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Shannon entropy in bits per row of a probability matrix."""
    safe = np.clip(distributions, eps, 1.0)
    return -np.sum(distributions * np.log2(safe), axis=1)
