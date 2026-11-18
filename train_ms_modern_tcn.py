"""
Training script for ModernTCN CAN bus IDS classification.

Uses existing preprocessed .pt files from preprocess_can_ml_subset.py.
Input: (B, T, C) -> ModernTCN handles permutation internally.
"""

from __future__ import annotations
import argparse, json, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from networks.moderntcn_can_old import ModernTCN_CAN


def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def ensure(p): Path(p).mkdir(parents=True, exist_ok=True)

_log_path = None
def set_log(p): global _log_path; _log_path = p
def plog(t=""):
    print(t)
    if _log_path:
        with open(_log_path, "a") as f: f.write(t + "\n")

def fmt_s(s):
    return f"{s:.1f}s" if s < 60 else f"{int(s//60)}m{s%60:.0f}s"

def fmt_cm(cm):
    return np.array2string(cm, separator=", ", max_line_width=200)

def banner(t, c="=", w=90):
    plog(f"\n{c*w}\n  {t}\n{c*w}\n")


def load_split(path):
    data = torch.load(path, weights_only=False)
    return data["X"], data["y"]


def build_loaders(data_dir, batch_size, num_workers, pin_memory, use_weighted_sampler=False):
    X_tr, y_tr = load_split(Path(data_dir) / "train.pt")
    X_val, y_val = load_split(Path(data_dir) / "val.pt")
    X_test, y_test = load_split(Path(data_dir) / "test.pt")

    kw = dict(num_workers=num_workers, pin_memory=pin_memory,
              persistent_workers=(num_workers > 0))

    if use_weighted_sampler:
        counts = np.bincount(y_tr.numpy())
        weights = 1.0 / counts[y_tr.numpy()]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size,
                                  sampler=sampler, **kw)
    else:
        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size,
                                  shuffle=True, **kw)

    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size,
                            shuffle=False, **kw)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size,
                             shuffle=False, **kw)

    num_classes = int(max(y_tr.max(), y_val.max(), y_test.max()).item()) + 1
    seq_len = X_tr.shape[1]
    nvars = X_tr.shape[2]

    return {
        "train_loader": train_loader, "val_loader": val_loader,
        "test_loader": test_loader,
        "n_train": len(X_tr), "n_val": len(X_val), "n_test": len(X_test),
        "seq_len": seq_len, "nvars": nvars, "num_classes": num_classes,
    }


def get_class_names(data_dir, num_classes):
    meta_path = Path(data_dir) / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        cmap = meta.get("class_map", {})
        names = ["normal"] + [Path(k).stem for k, v in sorted(cmap.items(), key=lambda x: int(x[1]))]
        return names[:num_classes]
    return ["normal"] + [f"cls_{i}" for i in range(1, num_classes)]


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    preds_all, tgts_all, loss_sum, n = [], [], 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss_sum += criterion(logits, y).item() * y.size(0)
        n += y.size(0)
        preds_all.append(logits.argmax(-1).cpu())
        tgts_all.append(y.cpu())
    preds = torch.cat(preds_all).numpy()
    tgts = torch.cat(tgts_all).numpy()
    return {
        "loss": loss_sum / max(n, 1),
        "acc": accuracy_score(tgts, preds),
        "macro_f1": f1_score(tgts, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(tgts, preds, average="weighted", zero_division=0),
        "preds": preds, "targets": tgts,
        "cm": confusion_matrix(tgts, preds),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--save_folder", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu_id", type=int, default=0)

    # Data
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--use_weighted_sampler", action="store_true")

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=4.0)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--print_freq", type=int, default=100)

    # Model
    p.add_argument("--patch_size", type=int, default=8)
    p.add_argument("--patch_stride", type=int, default=4)
    p.add_argument("--dims", type=int, nargs="+", default=[32, 64])
    p.add_argument("--num_blocks", type=int, nargs="+", default=[1, 1])
    p.add_argument("--large_sizes", type=int, nargs="+", default=[13, 13])
    p.add_argument("--small_size", type=int, default=5)
    p.add_argument("--ffn_ratio", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--head_dropout", type=float, default=0.0)
    p.add_argument("--class_dropout", type=float, default=0.0)

    # Multi-scale
    p.add_argument("--multi_scale", action="store_true", default=True)
    p.add_argument("--no_multi_scale", dest="multi_scale", action="store_false")
    p.add_argument("--ms_kernel_sizes", type=int, nargs="+", default=[3, 7, 13])
    p.add_argument("--ms_fusion", choices=["concat", "sum"], default="concat")
    p.add_argument("--head_type", choices=["flatten", "mean"], default="flatten")

    args = p.parse_args()
    seed_all(args.seed)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.cuda.set_device(device)
        gpu = torch.cuda.get_device_name(device)
    else:
        device = torch.device("cpu"); gpu = "CPU"

    sf = Path(args.save_folder); ensure(sf)
    set_log(sf / "training_log.txt")
    hf = sf / "history.json"

    plog("=" * 90)
    plog(f"Args: {vars(args)}")

    bundle = build_loaders(
        args.data_dir, args.batch_size, args.num_workers,
        args.pin_memory, args.use_weighted_sampler,
    )
    train_loader = bundle["train_loader"]
    val_loader = bundle["val_loader"]
    test_loader = bundle["test_loader"]
    seq_len = bundle["seq_len"]
    nvars = bundle["nvars"]
    num_classes = bundle["num_classes"]
    class_names = get_class_names(args.data_dir, num_classes)

    model = ModernTCN_CAN(
        seq_len=seq_len, nvars=nvars, num_classes=num_classes,
        patch_size=args.patch_size, patch_stride=args.patch_stride,
        dims=args.dims, num_blocks=args.num_blocks,
        large_sizes=args.large_sizes, small_size=args.small_size,
        ffn_ratio=args.ffn_ratio, dropout=args.dropout,
        head_dropout=args.head_dropout, class_dropout=args.class_dropout,
        multi_scale=args.multi_scale, ms_kernel_sizes=args.ms_kernel_sizes,
        ms_fusion=args.ms_fusion, head_type=args.head_type,
    ).to(device)

    pc = model.count_parameters()
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = OneCycleLR(
        optimizer, max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs, pct_start=0.3,
    )

    ms_tag = (f"MS-kernels={args.ms_kernel_sizes} fusion={args.ms_fusion}"
              if args.multi_scale else "single-scale")
    banner("Multi-Scale ModernTCN for CAN bus IDS" if args.multi_scale else "ModernTCN for CAN bus IDS")
    plog(f"  Device     : {device} ({gpu})")
    plog(f"  Input      : seq_len={seq_len}, nvars={nvars}, classes={num_classes}")
    plog(f"  Classes    : {class_names}")
    plog(f"  Patch      : size={args.patch_size}, stride={args.patch_stride}")
    plog(f"  Temporal   : {ms_tag}")
    plog(f"  Head       : {args.head_type}")
    plog(f"  Stages     : dims={args.dims}, blocks={args.num_blocks}, "
         f"large={args.large_sizes}, small={args.small_size}")
    plog(f"  FFN ratio  : {args.ffn_ratio}")
    plog(f"  Params     : backbone={pc['backbone']:,}  head={pc['head']:,}  total={pc['total']:,}")
    plog(f"  Train/Val/Test: {bundle['n_train']}/{bundle['n_val']}/{bundle['n_test']}")
    plog()

    history = {
        "train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [],
        "test_loss": [], "test_acc": [], "test_f1": [], "lr": [],
    }
    best_val_f1 = best_test_f1 = -1.0
    patience_counter = 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        model.train()
        loss_sum = n_samp = 0
        nb = len(train_loader)

        for bi, (X, y) in enumerate(train_loader, 1):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            bs = X.size(0); loss_sum += loss.item() * bs; n_samp += bs

            if args.print_freq > 0 and (bi % args.print_freq == 0 or bi == nb):
                acc = (logits.argmax(-1) == y).float().mean().item()
                lr = optimizer.param_groups[0]["lr"]
                plog(f"  [{epoch:03d}] {bi:04d}/{nb:04d} | loss={loss.item():.4f} "
                     f"acc={acc:.3f} | lr={lr:.2e}")

        train_loss = loss_sum / max(n_samp, 1)
        lr = optimizer.param_groups[0]["lr"]

        val_m = evaluate(model, val_loader, criterion, device)
        test_m = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_m["loss"])
        history["val_acc"].append(val_m["acc"])
        history["val_f1"].append(val_m["macro_f1"])
        history["test_loss"].append(test_m["loss"])
        history["test_acc"].append(test_m["acc"])
        history["test_f1"].append(test_m["macro_f1"])
        history["lr"].append(lr)

        banner(f"EPOCH {epoch:03d}/{args.epochs:03d}", "-")
        plog(f"  Train | loss={train_loss:.4f}")
        plog(f"  Val   | acc={val_m['acc']:.4f} f1={val_m['macro_f1']:.4f}")
        plog(f"  Test  | acc={test_m['acc']:.4f} f1={test_m['macro_f1']:.4f}")
        plog(f"  Time  | {fmt_s(time.time()-t_ep)} lr={lr:.2e}")

        plog("  Val CM:")
        for line in fmt_cm(val_m["cm"]).split("\n"):
            plog(f"    {line}")
        plog("  Test CM:")
        for line in fmt_cm(test_m["cm"]).split("\n"):
            plog(f"    {line}")

        ckpt = {
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args), "history": history,
            "num_classes": num_classes, "class_names": class_names,
        }
        torch.save(ckpt, sf / "last.pth")

        improved = ""
        if val_m["macro_f1"] > best_val_f1:
            best_val_f1 = val_m["macro_f1"]
            torch.save(ckpt, sf / "best_val.pth")
            improved += " [best val]"
            patience_counter = 0
        else:
            patience_counter += 1

        if test_m["macro_f1"] > best_test_f1:
            best_test_f1 = test_m["macro_f1"]
            torch.save(ckpt, sf / "best_test.pth")
            improved += " [best test]"
            report = classification_report(
                test_m["targets"], test_m["preds"],
                target_names=class_names, zero_division=0, digits=4,
            )
            with open(sf / "best_test_report.txt", "w") as f: f.write(report)

        if improved:
            plog(f"  >>>{improved}")
        plog(f"  Best | val_f1={best_val_f1:.4f} test_f1={best_test_f1:.4f}")

        if patience_counter >= args.patience:
            plog(f"\n  Early stopping at epoch {epoch} (patience={args.patience})")
            break

        with open(hf, "w") as f: json.dump(history, f, indent=2)

    # Final
    banner("DONE")
    plog(f"  Time: {fmt_s(time.time()-t_start)}")
    plog(f"  Best val F1 : {best_val_f1:.6f}")
    plog(f"  Best test F1: {best_test_f1:.6f}")

    # Structural reparameterization demo
    model.structural_reparam()
    test_reparam = evaluate(model, test_loader, criterion, device)
    plog(f"  After reparam: test_f1={test_reparam['macro_f1']:.6f} (should match)")


if __name__ == "__main__":
    main()