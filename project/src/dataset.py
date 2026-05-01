"""
dataset.py - data loading and dataloader setup for CIFAR-10 and CIFAR-10H.
"""

import os
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
ENTROPY_MAX = np.log2(10.0)


def get_default_data_dir() -> str:
    """Return the default data directory path (project/data)."""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(src_dir)
    return os.path.join(project_dir, "data")


def _entropy_bits(distributions: np.ndarray) -> np.ndarray:
    """Compute Shannon entropy in bits for each row of a probability matrix."""
    safe = np.clip(distributions, 1e-12, 1.0)
    return -np.sum(distributions * np.log2(safe), axis=1)


def load_cifar10h(data_dir: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CIFAR-10 test images and CIFAR-10H soft labels with sanity checks.

    Args:
        data_dir: Directory containing CIFAR cache and `cifar10h-probs.npy`.
                  If None, defaults to `project/data`.

    Returns:
        A tuple `(images, soft_labels)` where:
            - images: numpy array of shape (10000, 32, 32, 3), dtype uint8
            - soft_labels: numpy array of shape (10000, 10), dtype float32
    """
    if data_dir is None:
        data_dir = get_default_data_dir()

    # `download=True` is cache-aware and will not re-download if files exist.
    cifar10_test = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
    )
    images = cifar10_test.data

    probs_path = os.path.join(data_dir, "cifar10h-probs.npy")
    if not os.path.isfile(probs_path):
        raise FileNotFoundError(
            f"Missing file: {probs_path}. Download cifar10h-probs.npy and place it in the data directory."
        )

    soft_labels = np.load(probs_path).astype(np.float32)
    if soft_labels.shape != (10000, 10):
        raise AssertionError(f"Expected soft label shape (10000, 10), got {soft_labels.shape}.")

    row_sums = soft_labels.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        max_deviation = float(np.max(np.abs(row_sums - 1.0)))
        raise AssertionError(f"Soft-label rows must sum to 1.0 (max deviation: {max_deviation:.6f}).")

    if len(images) != len(soft_labels) or len(images) != 10000:
        raise AssertionError(
            f"Expected 10000 images and 10000 labels, got {len(images)} images and {len(soft_labels)} labels."
        )

    entropies = _entropy_bits(soft_labels)
    # Entropy bounds for a 10-class distribution are [0, log2(10)].
    if entropies.min() < -1e-6 or entropies.max() > ENTROPY_MAX + 1e-6:
        raise AssertionError(
            f"Entropy out of expected range [0, log2(10)]: min={entropies.min():.6f}, max={entropies.max():.6f}"
        )

    print("CIFAR-10H entropy stats (bits):")
    print(f"  mean={entropies.mean():.4f}, min={entropies.min():.4f}, max={entropies.max():.4f}")

    return images, soft_labels


class CIFAR10HDataset(Dataset):
    """
    Dataset for CIFAR-10H image-softlabel pairs.

    Args:
        images: numpy array of CIFAR images with shape (N, 32, 32, 3).
        soft_labels: numpy array of soft labels with shape (N, 10).
        transform: torchvision transform pipeline for images.
    """

    def __init__(self, images: np.ndarray, soft_labels: np.ndarray, transform=None):
        if len(images) != len(soft_labels):
            raise ValueError("Images and soft_labels must have the same length.")
        self.images = images
        self.soft_labels = soft_labels.astype(np.float32)
        self.transform = transform
        self._to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.images)

    def __getitem__(self, index: int):
        """
        Return one sample as `(image_tensor, soft_label_tensor)`.

        Returns:
            image_tensor: Tensor with shape (3, 32, 32)
            soft_label_tensor: Float32 tensor with shape (10,)
        """
        image = Image.fromarray(self.images[index])
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = self._to_tensor(image)

        soft_label_tensor = torch.tensor(self.soft_labels[index], dtype=torch.float32)
        return image_tensor, soft_label_tensor


def get_dataloaders(config: dict):
    """
    Build train/val/test dataloaders for CIFAR-10H soft-label training.

    Args:
        config: Configuration dictionary with at least:
                random_seed, batch_size, num_workers, cifar10h_split.
                Optionally `data_dir`.

    Returns:
        (train_loader, val_loader, test_loader)
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

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:n_train + n_val + n_test]

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

    # Advanced indexing creates split-specific arrays, keeping dataset logic simple.
    train_dataset = CIFAR10HDataset(images[train_idx], soft_labels[train_idx], transform=train_transform)
    val_dataset = CIFAR10HDataset(images[val_idx], soft_labels[val_idx], transform=eval_transform)
    test_dataset = CIFAR10HDataset(images[test_idx], soft_labels[test_idx], transform=eval_transform)

    loader_kwargs = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


def get_cifar10_pretrain_loader(config: dict) -> DataLoader:
    """
    Build the CIFAR-10 hard-label train dataloader for pretraining.

    Args:
        config: Configuration dictionary with at least:
                batch_size, num_workers. Optionally `data_dir`.

    Returns:
        DataLoader over the standard CIFAR-10 training split (50,000 images).
    """
    data_dir = config.get("data_dir", get_default_data_dir())

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    cifar10_train = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    pretrain_loader = DataLoader(
        cifar10_train,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    return pretrain_loader


if __name__ == "__main__":
    from config import CONFIG

    print("Running dataset.py smoke test...")
    train_loader, val_loader, test_loader = get_dataloaders(CONFIG)
    pretrain_loader = get_cifar10_pretrain_loader(CONFIG)

    x_soft, y_soft = next(iter(train_loader))
    x_hard, y_hard = next(iter(pretrain_loader))

    print(f"CIFAR-10H train/val/test sizes: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")
    print(f"Soft batch shapes: images={tuple(x_soft.shape)}, labels={tuple(y_soft.shape)}")
    print(f"Pretrain batch shapes: images={tuple(x_hard.shape)}, labels={tuple(y_hard.shape)}")
