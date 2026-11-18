"""
Training script for ModernTCN CAN bus IDS classification (speed-optimized).

Main changes vs. the original version:
  * GPU selection happens BEFORE torch is imported (CUDA_VISIBLE_DEVICES), so no
    CUDA context is created on other people's GPUs. Use --gpu auto|cpu|<index>
    and --list_gpus to inspect the server.
  * Whole dataset is moved to GPU once (--data_on_gpu auto|yes|no) and batches
    are sliced directly on GPU: no DataLoader workers, no per-sample collate,
    almost zero CPU usage.
  * Class-balanced sampling done on GPU (replaces WeightedRandomSampler).
  * Mixed precision (bf16 if supported, otherwise fp16 + GradScaler), TF32,
    cudnn.benchmark, fused AdamW, optional torch.compile.
  * No per-batch .item() sync; loss is accumulated on GPU.
  * Evaluation builds the confusion matrix on GPU (one sync per evaluation).
  * CPU threads limited with --cpu_threads.

Usage:
  python train_moderntcn.py --list_gpus
  python train_moderntcn.py --data_dir data/xxx --save_folder save/xxx --gpu auto
  python train_moderntcn.py ... --gpu 2 --batch_size 1024 --lr 2e-3 --compile
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# 1) Pre-parse GPU / CPU options BEFORE importing torch
# ─────────────────────────────────────────────────────────────────────────────
import argparse, os, subprocess, sys

# nvidia-smi and CUDA must use the same ordering (important with mixed GPUs)
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


def query_gpus():
    """Query GPUs via nvidia-smi (does not create any CUDA context)."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        idx, name, used, total, util = [s.strip() for s in line.split(",")]
        gpus.append(dict(index=int(idx), name=name, used=int(used),
                         total=int(total), free=int(total) - int(used),
                         util=int(util) if util.isdigit() else 0))
    return gpus


def print_gpus(gpus):
    if not gpus:
        print("No GPU found via nvidia-smi.")
        return
    print(f"{'idx':>3}  {'name':<28} {'used/total (MiB)':>18} {'free':>8} {'util':>5}")
    for g in gpus:
        print(f"{g['index']:>3}  {g['name']:<28} "
              f"{g['used']:>8}/{g['total']:<9} {g['free']:>8} {g['util']:>4}%")


def pick_gpu(spec: str):
    """Return the physical GPU index as string, or None for CPU."""
    if spec == "cpu":
        return None
    if spec == "auto":
        gpus = query_gpus()
        if not gpus:
            return None
        best = max(gpus, key=lambda g: (g["free"], -g["util"]))
        return str(best["index"])
    return spec


_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", default="auto")
_pre.add_argument("--cpu_threads", type=int, default=4)
_pre.add_argument("--list_gpus", action="store_true")
_pre_args, _ = _pre.parse_known_args()

if _pre_args.list_gpus:
    print_gpus(query_gpus())
    sys.exit(0)

PHYS_GPU = pick_gpu(_pre_args.gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = PHYS_GPU if PHYS_GPU is not None else ""
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = str(_pre_args.cpu_threads)

# ─────────────────────────────────────────────────────────────────────────────
# 2) Regular imports
# ─────────────────────────────────────────────────────────────────────────────
import json, math, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from networks.moderntcn_can import ModernTCN_CAN

torch.set_num_threads(_pre_args.cpu_threads)


# ── Utils ──

def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def ensure(p): Path(p).mkdir(parents=True, exist_ok=True)

_log_path = None
def set_log(p): global _log_path; _log_path = p
def plog(t=""):
    print(t, flush=True)
    if _log_path:
        with open(_log_path, "a") as f: f.write(t + "\n")

def fmt_s(s):
    return f"{s:.1f}s" if s < 60 else f"{int(s//60)}m{s%60:.0f}s"

def fmt_cm(cm):
    return np.array2string(cm, separator=", ", max_line_width=200)

def banner(t, c="=", w=90):
    plog(f"\n{c*w}\n  {t}\n{c*w}\n")


# ── Data: GPU-resident batch loader ──

class FastLoader:
    """
    Slices batches directly from tensors (on GPU or pinned CPU memory).
    Replaces DataLoader + TensorDataset, which collates sample by sample on CPU.
    """

    def __init__(self, X, y, batch_size, device, shuffle=False, drop_last=False,
                 balanced=False, num_classes=None, seed=0):
        self.X, self.y = X, y
        self.bs = batch_size
        self.device = device
        self.shuffle = shuffle
        self.n = len(X)
        self.drop_last = drop_last and self.n > batch_size
        self.on_device = X.device == torch.device(device) if device.type == "cuda" else True
        self.gen = torch.Generator(device=X.device)
        self.gen.manual_seed(seed)

        self.balanced = balanced
        if balanced:
            # sorted indices grouped by class, for O(1) class-balanced sampling
            order = torch.argsort(y)
            counts = torch.bincount(y, minlength=num_classes)
            self.cls_present = torch.nonzero(counts > 0).flatten()
            self.order = order
            self.counts = counts
            self.starts = torch.cumsum(counts, 0) - counts

    def __len__(self):
        return self.n // self.bs if self.drop_last else math.ceil(self.n / self.bs)

    def _indices(self):
        dev = self.X.device
        if self.balanced:
            # equal class probability, same distribution as inverse-frequency
            # WeightedRandomSampler(replacement=True)
            k = torch.randint(len(self.cls_present), (self.n,), device=dev, generator=self.gen)
            c = self.cls_present[k]
            r = torch.rand(self.n, device=dev, generator=self.gen)
            pos = self.starts[c] + (r * self.counts[c]).long().clamp_max(self.counts[c] - 1)
            return self.order[pos]
        if self.shuffle:
            return torch.randperm(self.n, device=dev, generator=self.gen)
        return None

    def __iter__(self):
        idx = self._indices()
        for i in range(len(self)):
            s, e = i * self.bs, min((i + 1) * self.bs, self.n)
            if idx is None:
                xb, yb = self.X[s:e], self.y[s:e]
            else:
                j = idx[s:e]
                xb, yb = self.X[j], self.y[j]
            if not self.on_device:
                xb = xb.pin_memory().to(self.device, non_blocking=True)
                yb = yb.pin_memory().to(self.device, non_blocking=True)
            yield xb, yb


def load_split(path):
    data = torch.load(path, weights_only=False, map_location="cpu")
    return data["X"].float().contiguous(), data["y"].long().contiguous()


def build_loaders(data_dir, batch_size, eval_batch_size, device, data_on_gpu,
                  balanced, seed):
    X_tr, y_tr = load_split(Path(data_dir) / "train.pt")
    X_val, y_val = load_split(Path(data_dir) / "val.pt")
    X_test, y_test = load_split(Path(data_dir) / "test.pt")

    num_classes = int(max(y_tr.max(), y_val.max(), y_test.max()).item()) + 1
    total_bytes = sum(t.numel() * t.element_size()
                      for t in (X_tr, y_tr, X_val, y_val, X_test, y_test))

    put_on_gpu = False
    if device.type == "cuda":
        if data_on_gpu == "yes":
            put_on_gpu = True
        elif data_on_gpu == "auto":
            free, _ = torch.cuda.mem_get_info(device)
            put_on_gpu = total_bytes < 0.5 * free

    tgt = device if put_on_gpu else torch.device("cpu")
    X_tr, y_tr = X_tr.to(tgt), y_tr.to(tgt)
    X_val, y_val = X_val.to(tgt), y_val.to(tgt)
    X_test, y_test = X_test.to(tgt), y_test.to(tgt)

    train_loader = FastLoader(X_tr, y_tr, batch_size, device, shuffle=True,
                              drop_last=True, balanced=balanced,
                              num_classes=num_classes, seed=seed)
    val_loader = FastLoader(X_val, y_val, eval_batch_size, device)
    test_loader = FastLoader(X_test, y_test, eval_batch_size, device)

    return {
        "train_loader": train_loader, "val_loader": val_loader,
        "test_loader": test_loader,
        "n_train": len(X_tr), "n_val": len(X_val), "n_test": len(X_test),
        "seq_len": X_tr.shape[1], "nvars": X_tr.shape[2],
        "num_classes": num_classes, "data_on_gpu": put_on_gpu,
        "data_mb": total_bytes / 2**20,
    }


def get_class_names(data_dir, num_classes):
    meta_path = Path(data_dir) / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        cmap = meta.get("class_map", {})
        names = ["normal"] + [Path(k).stem for k, v in sorted(cmap.items(), key=lambda x: int(x[1]))]
        return names[:num_classes]
    return ["normal"] + [f"cls_{i}" for i in range(1, num_classes)]


# ── Evaluation (confusion matrix on GPU) ──

def metrics_from_cm(cm):
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    support = cm.sum(1)
    pred = cm.sum(0)
    prec = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    rec = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(tp), where=denom > 0)
    present = (support + pred) > 0          # same label set as sklearn
    return {
        "acc": tp.sum() / max(cm.sum(), 1),
        "macro_f1": f1[present].mean() if present.any() else 0.0,
        "weighted_f1": (f1 * support).sum() / max(support.sum(), 1),
    }


@torch.no_grad()
def evaluate(model, loader, num_classes, device, amp_dtype):
    model.eval()
    C = num_classes
    cm = torch.zeros(C * C, dtype=torch.long, device=device)
    loss_sum = torch.zeros((), device=device)
    n = 0
    for X, y in loader:
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(X)
        loss_sum += F.cross_entropy(logits.float(), y, reduction="sum")
        cm += torch.bincount(y * C + logits.argmax(-1), minlength=C * C)
        n += y.numel()
    cm = cm.view(C, C).cpu().numpy()            # single sync
    out = metrics_from_cm(cm)
    out["loss"] = loss_sum.item() / max(n, 1)
    out["cm"] = cm
    return out


def report_from_cm(cm, class_names):
    """sklearn report reconstructed from the confusion matrix (no stored preds)."""
    C = cm.shape[0]
    t = np.repeat(np.arange(C), cm.sum(1))
    p = np.concatenate([np.repeat(np.arange(C), cm[i]) for i in range(C)])
    return classification_report(t, p, labels=list(range(C)),
                                  target_names=class_names, zero_division=0, digits=4)


# ── Main ──

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--save_folder", required=True)
    p.add_argument("--seed", type=int, default=42)

    # Hardware
    p.add_argument("--gpu", default="auto",
                   help="auto (most free memory) | cpu | physical GPU index from nvidia-smi")
    p.add_argument("--list_gpus", action="store_true")
    p.add_argument("--cpu_threads", type=int, default=4)
    p.add_argument("--data_on_gpu", choices=["auto", "yes", "no"], default="auto")
    p.add_argument("--amp", choices=["auto", "bf16", "fp16", "off"], default="auto")
    p.add_argument("--compile", action="store_true", help="torch.compile the model")
    p.add_argument("--compile_mode", default="default",
                   help="default | reduce-overhead | max-autotune")

    # Data
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--eval_batch_size", type=int, default=4096)
    p.add_argument("--use_weighted_sampler", action="store_true",
                   help="class-balanced sampling (done on GPU)")

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=4.0)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--print_freq", type=int, default=0, help="0 = no per-batch log")
    p.add_argument("--eval_test_every", type=int, default=1,
                   help="evaluate test set every k epochs (0 = only at the end)")

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

    args = p.parse_args()
    seed_all(args.seed)

    # ── Device ──
    if PHYS_GPU is not None and torch.cuda.is_available():
        device = torch.device("cuda:0")       # the only visible GPU
        gpu_name = torch.cuda.get_device_name(device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu"); gpu_name = "CPU"

    if device.type != "cuda" or args.amp == "off":
        amp_dtype = None
    elif args.amp == "bf16" or (args.amp == "auto" and torch.cuda.is_bf16_supported()):
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    sf = Path(args.save_folder); ensure(sf)
    set_log(sf / "training_log.txt")
    hf = sf / "history.json"

    plog("=" * 90)
    plog(f"Args: {vars(args)}")

    bundle = build_loaders(args.data_dir, args.batch_size, args.eval_batch_size,
                           device, args.data_on_gpu, args.use_weighted_sampler, args.seed)
    train_loader = bundle["train_loader"]
    val_loader = bundle["val_loader"]
    test_loader = bundle["test_loader"]
    num_classes = bundle["num_classes"]
    class_names = get_class_names(args.data_dir, num_classes)

    model = ModernTCN_CAN(
        seq_len=bundle["seq_len"], nvars=bundle["nvars"], num_classes=num_classes,
        patch_size=args.patch_size, patch_stride=args.patch_stride,
        dims=args.dims, num_blocks=args.num_blocks,
        large_sizes=args.large_sizes, small_size=args.small_size,
        ffn_ratio=args.ffn_ratio, dropout=args.dropout,
        head_dropout=args.head_dropout, class_dropout=args.class_dropout,
    ).to(device)
    run_model = torch.compile(model, mode=args.compile_mode) if args.compile else model

    pc = model.count_parameters()
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                      fused=device.type == "cuda")
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                           epochs=args.epochs, pct_start=0.3)

    banner("ModernTCN for CAN bus IDS")
    plog(f"  Device     : {device} ({gpu_name}), physical GPU id={PHYS_GPU}")
    plog(f"  AMP        : {amp_dtype}  compile={args.compile}  cpu_threads={args.cpu_threads}")
    plog(f"  Data       : {bundle['data_mb']:.0f} MB, on_gpu={bundle['data_on_gpu']}, "
         f"balanced={args.use_weighted_sampler}")
    plog(f"  Input      : seq_len={bundle['seq_len']}, nvars={bundle['nvars']}, classes={num_classes}")
    plog(f"  Classes    : {class_names}")
    plog(f"  Stages     : dims={args.dims}, blocks={args.num_blocks}, "
         f"large={args.large_sizes}, small={args.small_size}")
    plog(f"  Params     : backbone={pc['backbone']:,}  head={pc['head']:,}  total={pc['total']:,}")
    plog(f"  Train/Val/Test: {bundle['n_train']}/{bundle['n_val']}/{bundle['n_test']}  "
         f"batch={args.batch_size} steps/epoch={len(train_loader)}")
    plog()

    history = {"train_loss": [], "val_acc": [], "val_f1": [],
               "test_acc": [], "test_f1": [], "lr": [], "epoch_time": []}
    best_val_f1 = best_test_f1 = -1.0
    patience_counter = 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        run_model.train()
        loss_sum = torch.zeros((), device=device)
        n_samp = 0
        nb = len(train_loader)

        for bi, (X, y) in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = run_model(X)
            loss = criterion(logits.float(), y)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            bs = y.size(0)
            loss_sum += loss.detach() * bs          # stays on GPU, no sync
            n_samp += bs

            if args.print_freq > 0 and (bi % args.print_freq == 0 or bi == nb):
                acc = (logits.argmax(-1) == y).float().mean().item()
                lr = optimizer.param_groups[0]["lr"]
                plog(f"  [{epoch:03d}] {bi:04d}/{nb:04d} | loss={loss.item():.4f} "
                     f"acc={acc:.3f} | lr={lr:.2e}")

        train_loss = loss_sum.item() / max(n_samp, 1)
        lr = optimizer.param_groups[0]["lr"]
        t_train = time.time() - t_ep

        val_m = evaluate(run_model, val_loader, num_classes, device, amp_dtype)
        do_test = args.eval_test_every > 0 and epoch % args.eval_test_every == 0
        test_m = evaluate(run_model, test_loader, num_classes, device, amp_dtype) if do_test else None

        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_m["acc"])
        history["val_f1"].append(val_m["macro_f1"])
        history["test_acc"].append(test_m["acc"] if test_m else None)
        history["test_f1"].append(test_m["macro_f1"] if test_m else None)
        history["lr"].append(lr)
        history["epoch_time"].append(time.time() - t_ep)

        banner(f"EPOCH {epoch:03d}/{args.epochs:03d}", "-")
        plog(f"  Train | loss={train_loss:.4f}")
        plog(f"  Val   | acc={val_m['acc']:.4f} f1={val_m['macro_f1']:.4f}")
        if test_m:
            plog(f"  Test  | acc={test_m['acc']:.4f} f1={test_m['macro_f1']:.4f}")
        plog(f"  Time  | train {fmt_s(t_train)} total {fmt_s(time.time()-t_ep)} lr={lr:.2e}")
        plog("  Val CM:")
        for line in fmt_cm(val_m["cm"]).split("\n"):
            plog(f"    {line}")
        if test_m:
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

        if test_m and test_m["macro_f1"] > best_test_f1:
            best_test_f1 = test_m["macro_f1"]
            torch.save(ckpt, sf / "best_test.pth")
            improved += " [best test]"
            (sf / "best_test_report.txt").write_text(report_from_cm(test_m["cm"], class_names))

        if improved:
            plog(f"  >>>{improved}")
        plog(f"  Best | val_f1={best_val_f1:.4f} test_f1={best_test_f1:.4f}")

        with open(hf, "w") as f: json.dump(history, f, indent=2)

        if patience_counter >= args.patience:
            plog(f"\n  Early stopping at epoch {epoch} (patience={args.patience})")
            break

    # ── Final: test at best-val checkpoint, then reparam check ──
    banner("DONE")
    plog(f"  Time: {fmt_s(time.time()-t_start)}")
    plog(f"  Best val F1 : {best_val_f1:.6f}")
    plog(f"  Best test F1: {best_test_f1:.6f}")

    model.load_state_dict(torch.load(sf / "best_val.pth", weights_only=False)["model_state_dict"])
    final = evaluate(model, test_loader, num_classes, device, amp_dtype)
    plog(f"  Test @ best val: acc={final['acc']:.6f} f1={final['macro_f1']:.6f}")
    (sf / "test_at_best_val_report.txt").write_text(report_from_cm(final["cm"], class_names))

    model.structural_reparam()
    reparam = evaluate(model, test_loader, num_classes, device, amp_dtype)
    plog(f"  After reparam: test_f1={reparam['macro_f1']:.6f} (should match)")


if __name__ == "__main__":
    main()
