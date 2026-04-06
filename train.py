"""
Training script for MS-TCN-RetNet on preprocessed CAN-ML .pt datasets.

Expected data directory
-----------------------
data_dir/
    train.pt
    val.pt
    test.pt
    metadata.json   (optional)

Features
--------
- Select GPU by --gpu_id
- Separate model file import
- Full logging, checkpoints, history, confusion matrices, reports
- Weighted sampler / class weights
- CE / Focal loss
- AMP support
"""

from __future__ import annotations

import argparse
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import WeightedRandomSampler

from dataloader import build_all_dataloaders
from networks.ms_tcn_retnet_torch import MSTCNRetNet


def make_grad_scaler(device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.GradScaler("cuda", enabled=True)
    return torch.amp.GradScaler("cpu", enabled=False)


def get_amp_context(device, enabled: bool):
    if enabled and device.type == "cuda":
        return lambda: torch.amp.autocast("cuda")
    return nullcontext


# ------------------------------
# Utils
# ------------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_log(path: Path, text: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def get_class_names(data_dir: Path, num_classes: int) -> List[str]:
    meta_path = data_dir / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            class_map = meta.get("class_map", {})
            names = ["normal"]
            inv = {}
            for fname, cid in class_map.items():
                inv[int(cid)] = Path(fname).stem
            for c in range(1, num_classes):
                names.append(inv.get(c, f"class_{c}"))
            return names[:num_classes]
        except Exception:
            pass
    return ["normal"] + [f"class_{i}" for i in range(1, num_classes)]


def compute_class_weights(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(y, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights


def build_weighted_sampler(y: torch.Tensor, num_classes: int) -> WeightedRandomSampler:
    class_weights = compute_class_weights(y, num_classes)
    sample_weights = class_weights[y.cpu()].double()
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ------------------------------
# Losses
# ------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


def build_loss(loss_name: str, class_weights: torch.Tensor | None = None, focal_gamma: float = 2.0) -> nn.Module:
    loss_name = loss_name.lower()
    if loss_name == "ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    if loss_name == "focal":
        return FocalLoss(alpha=class_weights, gamma=focal_gamma)
    raise ValueError(f"Unsupported loss_name={loss_name}")


# ------------------------------
# Evaluation
# ------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device, amp=False):
    model.eval()

    all_logits = []
    all_targets = []
    running_loss = 0.0
    n_samples = 0

    amp_ctx_factory = get_amp_context(device, enabled=amp)

    for batch in loader:
        X, y = batch[:2]
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with amp_ctx_factory():
            logits = model(X)
            loss = criterion(logits, y)

        bs = X.size(0)
        running_loss += loss.item() * bs
        n_samples += bs

        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    preds = all_logits.argmax(dim=1).numpy()
    targets = all_targets.numpy()

    avg_loss = running_loss / max(n_samples, 1)
    acc = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(targets, preds, average="weighted", zero_division=0)
    cm = confusion_matrix(targets, preds)

    return {
        "loss": avg_loss,
        "acc": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "preds": preds,
        "targets": targets,
        "cm": cm,
    }


# ------------------------------
# Train
# ------------------------------

def format_confusion_matrix(cm: np.ndarray) -> str:
    return np.array2string(cm, separator=", ", max_line_width=200)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_folder", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="ms_tcn_retnet")
    parser.add_argument("--seed", type=int, default=42)

    # GPU/device
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU index to use when device=cuda")

    # Data
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--use_weighted_sampler", action="store_true")

    # Train
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--print_freq", type=int, default=100, help="Print every N iterations during training; <=0 disables per-iter prints")

    # Scheduler
    parser.add_argument("--scheduler", type=str, default="cosine_restart", choices=["none", "cosine", "cosine_restart"])
    parser.add_argument("--t0", type=int, default=10)
    parser.add_argument("--t_mult", type=int, default=2)

    # Loss
    parser.add_argument("--loss_name", type=str, default="ce", choices=["ce", "focal"])
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    # Model
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--branch_dim", type=int, default=32)
    parser.add_argument("--tcn_dim", type=int, default=64)
    parser.add_argument("--tcn_depth", type=int, default=2)
    parser.add_argument("--tcn_kernel_size", type=int, default=3)
    parser.add_argument("--tcn_dropout", type=float, default=0.1)
    parser.add_argument("--retention_dim", type=int, default=64)
    parser.add_argument("--retention_depth", type=int, default=2)
    parser.add_argument("--retention_heads", type=int, default=4)
    parser.add_argument("--retention_dropout", type=float, default=0.1)
    parser.add_argument("--ff_expansion", type=int, default=4)
    parser.add_argument("--pool", type=str, default="mean", choices=["mean", "max"])

    args = parser.parse_args()

    seed_everything(args.seed)

    # Device selection
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        if args.gpu_id < 0 or args.gpu_id >= torch.cuda.device_count():
            raise ValueError(f"gpu_id={args.gpu_id} is invalid. Available GPUs: 0..{torch.cuda.device_count()-1}")
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.cuda.set_device(device)
        gpu_name = torch.cuda.get_device_name(device)
    else:
        device = torch.device("cpu")
        gpu_name = "CPU"

    save_folder = Path(args.save_folder)
    ensure_dir(save_folder)
    epoch_dir = save_folder / "epoch_save"
    ensure_dir(epoch_dir)

    log_file = save_folder / "training_log.txt"
    history_file = save_folder / "history.json"

    append_log(log_file, "=" * 100)
    append_log(log_file, f"Args: {vars(args)}")
    append_log(log_file, f"Selected device: {device} | {gpu_name}")

    bundle = build_all_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        train_drop_last=False,
        return_index=False,
    )

    train_dataset = bundle["train_dataset"]
    val_loader = bundle["val_loader"]
    test_loader = bundle["test_loader"]
    input_shape = bundle["input_shape"]  # (T, C)
    num_classes = bundle["num_classes"]

    if args.use_weighted_sampler:
        sampler = build_weighted_sampler(train_dataset.y, num_classes)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            drop_last=False,
            persistent_workers=(args.num_workers > 0),
        )
    else:
        train_loader = bundle["train_loader"]

    class_names = get_class_names(Path(args.data_dir), num_classes)

    model = MSTCNRetNet(
        input_dim=input_shape[1],
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        branch_dim=args.branch_dim,
        tcn_dim=args.tcn_dim,
        tcn_depth=args.tcn_depth,
        tcn_kernel_size=args.tcn_kernel_size,
        tcn_dropout=args.tcn_dropout,
        retention_dim=args.retention_dim,
        retention_depth=args.retention_depth,
        retention_heads=args.retention_heads,
        retention_dropout=args.retention_dropout,
        ff_expansion=args.ff_expansion,
        pool=args.pool,
    ).to(device)

    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(train_dataset.y, num_classes).to(device)

    criterion = build_loss(args.loss_name, class_weights=class_weights, focal_gamma=args.focal_gamma)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "cosine_restart":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.t0, T_mult=args.t_mult)
    else:
        scheduler = None

    scaler = make_grad_scaler(device, enabled=args.amp)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "val_weighted_f1": [],
        "test_loss": [],
        "test_acc": [],
        "test_macro_f1": [],
        "test_weighted_f1": [],
        "lr": [],
    }

    best_val_macro_f1 = -1.0
    best_val_acc = -1.0
    best_test_macro_f1 = -1.0

    best_val_state = None
    best_test_state = None
    best_val_cm = None
    best_test_cm = None
    best_test_report = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0

        amp_ctx_factory = get_amp_context(device, enabled=args.amp)

        num_train_batches = len(train_loader)
        for batch_idx, batch in enumerate(train_loader, start=1):
            X, y = batch[:2]
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with amp_ctx_factory():
                logits = model(X)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            bs = X.size(0)
            running_loss += loss.item() * bs
            n_samples += bs

            if args.print_freq > 0 and (batch_idx % args.print_freq == 0 or batch_idx == num_train_batches):
                iter_msg = (
                    f"Epoch [{epoch:03d}/{args.epochs:03d}] "
                    f"Iter [{batch_idx:04d}/{num_train_batches:04d}] | "
                    f"lr={optimizer.param_groups[0]['lr']:.6e} | "
                    f"loss={loss.item():.6f}"
                )
                print(iter_msg)
                append_log(log_file, iter_msg)

        train_loss = running_loss / max(n_samples, 1)

        if scheduler is not None:
            if args.scheduler == "cosine_restart":
                scheduler.step(epoch - 1 + 1.0)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        val_metrics = evaluate(model, val_loader, criterion, device, amp=args.amp)
        test_metrics = evaluate(model, test_loader, criterion, device, amp=args.amp)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["val_weighted_f1"].append(val_metrics["weighted_f1"])
        history["test_loss"].append(test_metrics["loss"])
        history["test_acc"].append(test_metrics["acc"])
        history["test_macro_f1"].append(test_metrics["macro_f1"])
        history["test_weighted_f1"].append(test_metrics["weighted_f1"])
        history["lr"].append(current_lr)

        msg = (
            f"Epoch [{epoch:03d}/{args.epochs:03d}] | "
            f"lr={current_lr:.6e} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_metrics['loss']:.6f}, val_acc={val_metrics['acc']:.4f}, "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}, val_weighted_f1={val_metrics['weighted_f1']:.4f} | "
            f"test_loss={test_metrics['loss']:.6f}, test_acc={test_metrics['acc']:.4f}, "
            f"test_macro_f1={test_metrics['macro_f1']:.4f}, test_weighted_f1={test_metrics['weighted_f1']:.4f}"
        )
        print(msg)
        append_log(log_file, msg)

        val_cm_str = format_confusion_matrix(val_metrics["cm"])
        test_cm_str = format_confusion_matrix(test_metrics["cm"])

        print("Val Confusion Matrix:")
        print(val_cm_str)
        print("Test Confusion Matrix:")
        print(test_cm_str)

        append_log(log_file, "Val Confusion Matrix:\n" + val_cm_str)
        append_log(log_file, "Test Confusion Matrix:\n" + test_cm_str)

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "history": history,
            "input_shape": input_shape,
            "num_classes": num_classes,
            "class_names": class_names,
        }

        torch.save(ckpt, epoch_dir / f"{args.model_name}_epoch_{epoch:03d}.pth")
        torch.save(ckpt, save_folder / f"{args.model_name}_last.pth")

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_val_state = ckpt
            best_val_cm = val_metrics["cm"]
            torch.save(ckpt, save_folder / f"{args.model_name}_best_macro_f1.pth")
            np.save(save_folder / "best_val_confusion_matrix.npy", best_val_cm)

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save(ckpt, save_folder / f"{args.model_name}_best_acc.pth")

        if test_metrics["macro_f1"] > best_test_macro_f1:
            best_test_macro_f1 = test_metrics["macro_f1"]
            best_test_state = ckpt
            best_test_cm = test_metrics["cm"]
            best_test_report = classification_report(
                test_metrics["targets"],
                test_metrics["preds"],
                target_names=class_names,
                zero_division=0,
                digits=4,
            )
            torch.save(ckpt, save_folder / f"{args.model_name}_best_test_macro_f1.pth")
            np.save(save_folder / "best_test_confusion_matrix.npy", best_test_cm)
            with open(save_folder / "best_test_classification_report.txt", "w", encoding="utf-8") as f:
                f.write(best_test_report)

        save_json(history_file, history)

    final_metrics = evaluate(model, test_loader, criterion, device, amp=args.amp)
    final_cm = final_metrics["cm"]
    final_report = classification_report(
        final_metrics["targets"],
        final_metrics["preds"],
        target_names=class_names,
        zero_division=0,
        digits=4,
    )
    np.save(save_folder / "final_confusion_matrix.npy", final_cm)
    with open(save_folder / "final_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(final_report)

    append_log(log_file, "=" * 100)
    append_log(log_file, f"Best VAL macro-F1: {best_val_macro_f1:.6f}")
    append_log(log_file, f"Best VAL acc: {best_val_acc:.6f}")
    append_log(log_file, f"Best TEST macro-F1: {best_test_macro_f1:.6f}")
    append_log(log_file, "Final TEST report:\n" + final_report)

    print("\nTraining finished.")
    print(f"Best VAL macro-F1: {best_val_macro_f1:.6f}")
    print(f"Best VAL acc: {best_val_acc:.6f}")
    print(f"Best TEST macro-F1: {best_test_macro_f1:.6f}")
    print(f"Outputs saved to: {save_folder}")


if __name__ == "__main__":
    main()
