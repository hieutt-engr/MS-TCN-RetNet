"""
PyTorch Dataset and DataLoader utilities for MS-TCN-RetNet / CAN-ML .pt files

Expected .pt file format
------------------------
Each split file is saved as:
{
    "X": torch.FloatTensor of shape (N, T, C),
    "y": torch.LongTensor of shape (N,)
}

Example files
-------------
train.pt
val.pt
test.pt

This module provides:
- CANMLWindowDataset
- build_dataloader(...)
- build_all_dataloaders(...)
- a simple collate function
- an example main block
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class CANMLWindowDataset(Dataset):
    """
    Dataset for preprocessed CAN-ML window tensors stored in .pt files.

    Parameters
    ----------
    pt_path : str
        Path to a .pt file containing {"X": ..., "y": ...}
    return_index : bool
        If True, __getitem__ returns (x, y, idx)
    to_float32 : bool
        Convert X to float32 if needed
    to_long : bool
        Convert y to long if needed
    """

    def __init__(
        self,
        pt_path: str,
        return_index: bool = False,
        to_float32: bool = True,
        to_long: bool = True,
    ):
        self.pt_path = str(pt_path)
        data = torch.load(self.pt_path, map_location="cpu", weights_only=True)

        if not isinstance(data, dict):
            raise ValueError(f"{self.pt_path} does not contain a dict.")

        if "X" not in data or "y" not in data:
            raise ValueError(f"{self.pt_path} must contain keys 'X' and 'y'.")

        self.X = data["X"]
        self.y = data["y"]

        if not torch.is_tensor(self.X):
            self.X = torch.tensor(self.X)
        if not torch.is_tensor(self.y):
            self.y = torch.tensor(self.y)

        if to_float32:
            self.X = self.X.float()
        if to_long:
            self.y = self.y.long()

        if self.X.ndim != 3:
            raise ValueError(
                f"Expected X to have shape (N, T, C), but got shape {tuple(self.X.shape)}"
            )
        if self.y.ndim != 1:
            raise ValueError(
                f"Expected y to have shape (N,), but got shape {tuple(self.y.shape)}"
            )
        if len(self.X) != len(self.y):
            raise ValueError(
                f"Mismatched number of samples: len(X)={len(self.X)} vs len(y)={len(self.y)}"
            )

        self.return_index = return_index

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        y = self.y[idx]
        if self.return_index:
            return x, y, idx
        return x, y

    @property
    def num_samples(self) -> int:
        return len(self.y)

    @property
    def seq_len(self) -> int:
        return int(self.X.shape[1])

    @property
    def feature_dim(self) -> int:
        return int(self.X.shape[2])

    @property
    def num_classes(self) -> int:
        return int(torch.unique(self.y).numel())

    def label_distribution(self) -> Dict[int, int]:
        vals, cnts = torch.unique(self.y, return_counts=True)
        return {int(v.item()): int(c.item()) for v, c in zip(vals, cnts)}


def canml_collate_fn(batch):
    """
    Standard collate function.

    Returns
    -------
    If dataset.return_index == False:
        X: (B, T, C)
        y: (B,)
    Else:
        X: (B, T, C)
        y: (B,)
        idx: (B,)
    """
    if len(batch[0]) == 2:
        xs, ys = zip(*batch)
        X = torch.stack(xs, dim=0)
        y = torch.stack(ys, dim=0)
        return X, y
    else:
        xs, ys, ids = zip(*batch)
        X = torch.stack(xs, dim=0)
        y = torch.stack(ys, dim=0)
        idx = torch.tensor(ids, dtype=torch.long)
        return X, y, idx


def build_dataloader(
    pt_path: str,
    batch_size: int = 128,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    return_index: bool = False,
    persistent_workers: bool = False,
) -> Tuple[CANMLWindowDataset, DataLoader]:
    """
    Build one dataset + dataloader pair from a .pt split file.
    """
    dataset = CANMLWindowDataset(
        pt_path=pt_path,
        return_index=return_index,
        to_float32=True,
        to_long=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=canml_collate_fn,
        persistent_workers=(persistent_workers and num_workers > 0),
    )
    return dataset, loader


def build_all_dataloaders(
    data_dir: str,
    train_file: str = "train.pt",
    val_file: str = "val.pt",
    test_file: str = "test.pt",
    batch_size: int = 128,
    num_workers: int = 0,
    pin_memory: bool = True,
    train_drop_last: bool = False,
    return_index: bool = False,
) -> Dict[str, object]:
    """
    Build train/val/test datasets and dataloaders from one directory.

    Returns
    -------
    dict with keys:
        train_dataset, val_dataset, test_dataset,
        train_loader, val_loader, test_loader,
        input_shape, num_classes
    """
    data_dir = Path(data_dir)

    train_path = data_dir / train_file
    val_path = data_dir / val_file
    test_path = data_dir / test_file

    train_dataset, train_loader = build_dataloader(
        pt_path=str(train_path),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=train_drop_last,
        return_index=return_index,
        persistent_workers=True,
    )

    val_dataset, val_loader = build_dataloader(
        pt_path=str(val_path),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        return_index=return_index,
        persistent_workers=True,
    )

    test_dataset, test_loader = build_dataloader(
        pt_path=str(test_path),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        return_index=return_index,
        persistent_workers=True,
    )

    input_shape = (train_dataset.seq_len, train_dataset.feature_dim)

    # Use max label + 1 so classes like 0..9 are represented correctly
    num_classes = int(max(train_dataset.y.max(), val_dataset.y.max(), test_dataset.y.max()).item()) + 1

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "input_shape": input_shape,
        "num_classes": num_classes,
    }


if __name__ == "__main__":
    # Example usage:
    data_dir = "path/to/output_preprocessed"

    bundle = build_all_dataloaders(
        data_dir=data_dir,
        batch_size=128,
        num_workers=0,
        pin_memory=True,
        train_drop_last=False,
        return_index=False,
    )

    train_dataset = bundle["train_dataset"]
    train_loader = bundle["train_loader"]

    print("Input shape (T, C):", bundle["input_shape"])
    print("Num classes:", bundle["num_classes"])
    print("Train samples:", len(train_dataset))
    print("Train label distribution:", train_dataset.label_distribution())

    first_batch = next(iter(train_loader))
    X, y = first_batch
    print("Batch X shape:", X.shape)  # (B, T, C)
    print("Batch y shape:", y.shape)  # (B,)
