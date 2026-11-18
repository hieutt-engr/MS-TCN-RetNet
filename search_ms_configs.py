"""
Lightweight multi-scale kernel search for ModernTCN CAN bus IDS.

Strategy:
  1. Warm-start from vanilla baseline checkpoint (stem + ConvFFN + head)
  2. Only randomly initialize the new multi-scale DWConv branches
  3. Short training (15 epochs) per config with early stopping
  4. Rank configs by best val F1, pick top-3 for full training

Usage:
  python search_ms_configs.py \
      --data_dir ./data/2017-subaru-forester/preprocessed \
      --vanilla_ckpt ./save/moderntcn/forester/mtcn_bs256_dim128_low_dr_lr0.001_s42/best_val.pth \
      --save_folder ./save/moderntcn/forester/ms_search \
      --search_epochs 15 --early_stop_epoch 8 --early_stop_min_f1 0.90
"""

from __future__ import annotations
import argparse, json, itertools, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, TensorDataset

from networks.moderntcn_can import ModernTCN_CAN


def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def ensure(p): Path(p).mkdir(parents=True, exist_ok=True)


# ── Search space definition ──

SEARCH_CONFIGS = [
    # (name, kernel_sizes, description)
    ("vanilla_k13",       [13],             "single-scale baseline"),
    ("ms_3_13",           [3, 13],          "2-branch: short + long"),
    ("ms_5_13",           [5, 13],          "2-branch: mid + long"),
    ("ms_3_9",            [3, 9],           "2-branch: short + mid-long"),
    ("ms_3_7_13",         [3, 7, 13],       "3-branch: short + mid + long"),
    ("ms_3_9_13",         [3, 9, 13],       "3-branch: short + mid-long + long"),
    ("ms_5_9_13",         [5, 9, 13],       "3-branch: mid + mid-long + long"),
    ("ms_3_5_9_13",       [3, 5, 9, 13],    "4-branch: fine-grained"),
    ("ms_3_7_9_13",       [3, 7, 9, 13],    "4-branch: spread"),
    ("ms_3_5_7_9_13",     [3, 5, 7, 9, 13], "5-branch: full coverage"),
]


def load_split(path):
    data = torch.load(path, weights_only=False)
    return data["X"], data["y"]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds_all, tgts_all = [], []
    for X, y in loader:
        X = X.to(device)
        logits = model(X)
        preds_all.append(logits.argmax(-1).cpu())
        tgts_all.append(y)
    preds = torch.cat(preds_all).numpy()
    tgts = torch.cat(tgts_all).numpy()
    return {
        "acc": accuracy_score(tgts, preds),
        "macro_f1": f1_score(tgts, preds, average="macro", zero_division=0),
    }


def warm_start_model(model, vanilla_ckpt_path, device):
    """
    Load vanilla checkpoint weights into multi-scale model.
    Matches everything except DWConv (which has different structure).
    """
    ckpt = torch.load(vanilla_ckpt_path, map_location=device, weights_only=False)
    vanilla_sd = ckpt["model_state_dict"]
    model_sd = model.state_dict()

    loaded, skipped = 0, 0
    for key in model_sd:
        if key in vanilla_sd and model_sd[key].shape == vanilla_sd[key].shape:
            model_sd[key] = vanilla_sd[key]
            loaded += 1
        else:
            skipped += 1

    model.load_state_dict(model_sd)
    return loaded, skipped


def run_one_config(config_name, kernel_sizes, data_bundle, args, device):
    """Train one multi-scale config for search_epochs, return results."""
    train_loader = data_bundle["train_loader"]
    val_loader = data_bundle["val_loader"]

    # Build model
    model = ModernTCN_CAN(
        seq_len=data_bundle["seq_len"], nvars=data_bundle["nvars"],
        num_classes=data_bundle["num_classes"],
        patch_size=args.patch_size, patch_stride=args.patch_stride,
        dims=args.dims, num_blocks=args.num_blocks,
        large_sizes=args.large_sizes, small_size=args.small_size,
        ffn_ratio=args.ffn_ratio, dropout=args.dropout,
        multi_scale=(len(kernel_sizes) > 1),
        ms_kernel_sizes=kernel_sizes,
    ).to(device)

    # Warm-start from vanilla checkpoint
    if args.vanilla_ckpt and Path(args.vanilla_ckpt).exists():
        loaded, skipped = warm_start_model(model, args.vanilla_ckpt, device)
        warm_info = f"warm-start loaded={loaded} skipped={skipped}"
    else:
        warm_info = "from scratch"

    params = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer, max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.search_epochs, pct_start=0.3,
    )

    print(f"\n  [{config_name}] kernels={kernel_sizes} params={params:,} {warm_info}")

    history = {"epoch": [], "train_loss": [], "val_f1": [], "val_acc": []}
    best_val_f1 = -1.0
    t_start = time.time()

    for epoch in range(1, args.search_epochs + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * X.size(0)
            n += X.size(0)

        train_loss = loss_sum / max(n, 1)
        val_m = evaluate(model, val_loader, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_f1"].append(val_m["macro_f1"])
        history["val_acc"].append(val_m["acc"])

        best_val_f1 = max(best_val_f1, val_m["macro_f1"])

        print(f"    e{epoch:02d} loss={train_loss:.4f} val_f1={val_m['macro_f1']:.4f} "
              f"best={best_val_f1:.4f}")

        # Early stopping: if clearly bad by early_stop_epoch
        if epoch == args.early_stop_epoch and best_val_f1 < args.early_stop_min_f1:
            print(f"    [early stop] best_f1={best_val_f1:.4f} < {args.early_stop_min_f1} at epoch {epoch}")
            break

    elapsed = time.time() - t_start

    return {
        "config": config_name,
        "kernel_sizes": kernel_sizes,
        "params": params,
        "best_val_f1": best_val_f1,
        "final_val_f1": history["val_f1"][-1],
        "epochs_run": len(history["epoch"]),
        "time_sec": elapsed,
        "history": history,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--vanilla_ckpt", default=None, help="Path to vanilla best_val.pth for warm-start")
    p.add_argument("--save_folder", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)

    # Search
    p.add_argument("--search_epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--early_stop_epoch", type=int, default=8)
    p.add_argument("--early_stop_min_f1", type=float, default=0.90)

    # Model (fixed, same as best vanilla)
    p.add_argument("--patch_size", type=int, default=8)
    p.add_argument("--patch_stride", type=int, default=4)
    p.add_argument("--dims", type=int, nargs="+", default=[128, 256])
    p.add_argument("--num_blocks", type=int, nargs="+", default=[1, 1])
    p.add_argument("--large_sizes", type=int, nargs="+", default=[13, 13])
    p.add_argument("--small_size", type=int, default=5)
    p.add_argument("--ffn_ratio", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)

    args = p.parse_args()
    seed_all(args.seed)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")

    sf = Path(args.save_folder); ensure(sf)

    # Load data
    X_tr, y_tr = load_split(Path(args.data_dir) / "train.pt")
    X_val, y_val = load_split(Path(args.data_dir) / "val.pt")

    kw = dict(num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0))
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=True, **kw)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size * 4, shuffle=False, **kw)

    num_classes = int(max(y_tr.max(), y_val.max()).item()) + 1
    data_bundle = {
        "train_loader": train_loader, "val_loader": val_loader,
        "seq_len": X_tr.shape[1], "nvars": X_tr.shape[2], "num_classes": num_classes,
    }

    print("=" * 70)
    print("  Multi-Scale Kernel Search for ModernTCN")
    print("=" * 70)
    print(f"  Configs to search: {len(SEARCH_CONFIGS)}")
    print(f"  Epochs per config: {args.search_epochs} (early stop at {args.early_stop_epoch})")
    print(f"  Warm-start from: {args.vanilla_ckpt}")
    print(f"  Device: {device}")

    # Run search
    all_results = []
    t_total = time.time()

    for name, kernels, desc in SEARCH_CONFIGS:
        seed_all(args.seed)  # same init for fair comparison
        result = run_one_config(name, kernels, data_bundle, args, device)
        result["description"] = desc
        all_results.append(result)

    # Rank by best_val_f1
    all_results.sort(key=lambda r: r["best_val_f1"], reverse=True)

    print("\n" + "=" * 70)
    print("  SEARCH RESULTS (ranked by best val F1)")
    print("=" * 70)
    print(f"  {'Rank':<5} {'Config':<20} {'Kernels':<18} {'Params':<10} "
          f"{'Best F1':<10} {'Epochs':<8} {'Time':<8}")
    print("  " + "-" * 75)
    for i, r in enumerate(all_results):
        print(f"  {i+1:<5} {r['config']:<20} {str(r['kernel_sizes']):<18} "
              f"{r['params']:>9,} {r['best_val_f1']:<10.4f} "
              f"{r['epochs_run']:<8} {r['time_sec']:.0f}s")

    total_time = time.time() - t_total
    print(f"\n  Total search time: {total_time/60:.1f} minutes")
    print(f"  Top-3 for full training:")
    for i in range(min(3, len(all_results))):
        r = all_results[i]
        print(f"    {i+1}. {r['config']} (kernels={r['kernel_sizes']}, F1={r['best_val_f1']:.4f})")

    # Save results
    save_results = []
    for r in all_results:
        save_r = {k: v for k, v in r.items() if k != "history"}
        save_r["kernel_sizes"] = r["kernel_sizes"]
        save_results.append(save_r)

    with open(sf / "search_results.json", "w") as f:
        json.dump(save_results, f, indent=2)

    # Save full histories for visualization
    histories = {r["config"]: r["history"] for r in all_results}
    with open(sf / "search_histories.json", "w") as f:
        json.dump(histories, f, indent=2)

    print(f"\n  Results saved to {sf / 'search_results.json'}")
    print(f"  Histories saved to {sf / 'search_histories.json'}")


if __name__ == "__main__":
    main()
