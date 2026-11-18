# """
# Train IVN-TCN (lightweight MS-ModernTCN + RealMLP-style embedding + TabM-style head).

# Loss follows TabM: cross-entropy applied to each of the k member predictions
# (mean over members), NOT to the averaged prediction. Inference averages the
# member probabilities.
# """

# from __future__ import annotations
# import os, sys


# def _early_cpu_threads():
#     # must run before numpy/torch import to cap BLAS/OpenMP thread pools
#     n = "4"
#     if "--cpu_threads" in sys.argv:
#         i = sys.argv.index("--cpu_threads")
#         if i + 1 < len(sys.argv):
#             n = sys.argv[i + 1]
#     for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
#         os.environ.setdefault(var, n)


# _early_cpu_threads()

# import argparse, json, random, time   # noqa: E402
# from pathlib import Path               # noqa: E402

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
# from torch.optim import AdamW
# from torch.optim.lr_scheduler import OneCycleLR
# from torch.utils.data import DataLoader, TensorDataset

# try:
#     from fast_data_seq import TensorSplit, GPUBatches
# except ImportError:
#     from utils.fast_data_seq import TensorSplit, GPUBatches

# try:
#     from networks.ivn_tcn import IVN_TCN
# except ImportError:
#     from ivn_tcn import IVN_TCN


# def seed_all(s=42):
#     random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# _log_path = None
# def plog(t=""):
#     print(t)
#     if _log_path:
#         with open(_log_path, "a") as f: f.write(t + "\n")

# def banner(t, c="=", w=90): plog(f"\n{c*w}\n  {t}\n{c*w}\n")
# def fmt_s(s): return f"{s:.1f}s" if s < 60 else f"{int(s//60)}m{s%60:.0f}s"
# def fmt_cm(cm): return np.array2string(cm, separator=", ", max_line_width=200)


# def load_split(p):
#     d = torch.load(p, weights_only=False)
#     return d["X"].float(), d["y"].long()


# def get_class_names(data_dir, n):
#     meta = Path(data_dir) / "metadata.json"
#     if meta.exists():
#         cmap = json.loads(meta.read_text()).get("class_map", {})
#         names = ["normal"] + [Path(k).stem for k, v in sorted(cmap.items(), key=lambda x: int(x[1]))]
#         return names[:n]
#     return ["normal"] + [f"cls_{i}" for i in range(1, n)]


# def member_loss(logits, y, label_smoothing):
#     B, K, C = logits.shape
#     return F.cross_entropy(logits.reshape(B * K, C), y.repeat_interleave(K),
#                            label_smoothing=label_smoothing)


# @torch.no_grad()
# def evaluate(model, loader, device, amp_dtype=None):
#     model.eval()
#     preds, tgts = [], []
#     nll = torch.zeros((), device=device)
#     n = 0
#     for X, y in loader:
#         X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
#         with torch.autocast(device.type, dtype=amp_dtype or torch.float32,
#                             enabled=amp_dtype is not None):
#             logits = model(X)
#         p = IVN_TCN.ensemble_proba(logits.float())
#         nll += F.nll_loss(torch.log(p.clamp_min(1e-12)), y, reduction="sum")
#         n += y.size(0)
#         preds.append(p.argmax(-1)); tgts.append(y)
#     preds, tgts = torch.cat(preds).cpu().numpy(), torch.cat(tgts).cpu().numpy()
#     return {"loss": nll.item() / max(n, 1), "acc": accuracy_score(tgts, preds),
#             "macro_f1": f1_score(tgts, preds, average="macro", zero_division=0),
#             "preds": preds, "targets": tgts, "cm": confusion_matrix(tgts, preds)}


# @torch.no_grad()
# def measure_latency(model, seq_len, nvars, device, iters=200):
#     model.eval()
#     out = {}
#     for bs in (1, 1024):
#         x = torch.randn(bs, seq_len, nvars, device=device)
#         for _ in range(20): model(x)
#         if device.type == "cuda": torch.cuda.synchronize()
#         t = time.perf_counter()
#         for _ in range(iters): model(x)
#         if device.type == "cuda": torch.cuda.synchronize()
#         out[bs] = (time.perf_counter() - t) / iters
#     return out


# def main():
#     global _log_path
#     p = argparse.ArgumentParser()
#     p.add_argument("--data_dir", required=True)
#     p.add_argument("--save_folder", required=True)
#     p.add_argument("--seed", type=int, default=42)
#     p.add_argument("--gpu_id", type=int, default=0)
#     p.add_argument("--batch_size", type=int, default=256)
#     p.add_argument("--num_workers", type=int, default=4, help="only used with --fast_data off")
#     p.add_argument("--pin_memory", action="store_true")
#     p.add_argument("--eval_batch_size", type=int, default=8192)
#     p.add_argument("--cpu_threads", type=int, default=4)
#     p.add_argument("--fast_data", default="auto", choices=["auto", "gpu", "cpu", "off"])
#     p.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
#     p.add_argument("--compile", action="store_true", help="torch.compile the model")
#     # training
#     p.add_argument("--epochs", type=int, default=100)
#     p.add_argument("--lr", type=float, default=1e-3)
#     p.add_argument("--weight_decay", type=float, default=1e-4)
#     p.add_argument("--grad_clip", type=float, default=4.0)
#     p.add_argument("--patience", type=int, default=15)
#     p.add_argument("--label_smoothing", type=float, default=0.0)
#     # embedding (RealMLP-style)
#     p.add_argument("--no_embedding", dest="use_embedding", action="store_false")
#     p.add_argument("--d_emb", type=int, default=8)
#     p.add_argument("--n_freq", type=int, default=16)
#     p.add_argument("--emb_sigma", type=float, default=1.0)
#     p.add_argument("--cat_col", type=int, default=0, help="arb_id column; -1 to treat as numeric")
#     # backbone
#     p.add_argument("--patch_size", type=int, default=8)
#     p.add_argument("--patch_stride", type=int, default=4)
#     p.add_argument("--dims", type=int, nargs="+", default=[64, 128])
#     p.add_argument("--num_blocks", type=int, nargs="+", default=[1, 1])
#     p.add_argument("--large_sizes", type=int, nargs="+", default=[13, 13])
#     p.add_argument("--small_size", type=int, default=5)
#     p.add_argument("--ffn_ratio", type=int, default=2)
#     p.add_argument("--dropout", type=float, default=0.1)
#     p.add_argument("--no_multi_scale", dest="multi_scale", action="store_false")
#     p.add_argument("--ms_kernel_sizes", type=int, nargs="+", default=[3, 5, 13])
#     p.add_argument("--ms_fusion", choices=["channel", "sum", "concat"], default="channel")
#     # head (TabM-style)
#     p.add_argument("--ens_k", type=int, default=8)
#     p.add_argument("--head_hidden", type=int, default=64)
#     p.add_argument("--head_dropout", type=float, default=0.1)
#     args = p.parse_args()

#     torch.set_num_threads(args.cpu_threads)
#     seed_all(args.seed)
#     if torch.cuda.is_available():
#         torch.cuda.set_device(args.gpu_id)
#         device = torch.device(f"cuda:{args.gpu_id}")
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True
#         torch.backends.cudnn.benchmark = True      # fixed shapes: autotune conv kernels once
#     else:
#         device = torch.device("cpu")
#     amp_dtype = None
#     if device.type == "cuda" and args.amp != "off":
#         amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
#     sf = Path(args.save_folder); sf.mkdir(parents=True, exist_ok=True)
#     _log_path = sf / "training_log.txt"
#     plog("=" * 90); plog(f"Args: {vars(args)}")

#     X_tr, y_tr = load_split(Path(args.data_dir) / "train.pt")
#     X_va, y_va = load_split(Path(args.data_dir) / "val.pt")
#     X_te, y_te = load_split(Path(args.data_dir) / "test.pt")
#     num_classes = int(max(y_tr.max(), y_va.max(), y_te.max())) + 1
#     seq_len, nvars = X_tr.shape[1], X_tr.shape[2]
#     class_names = get_class_names(args.data_dir, num_classes)

#     if args.fast_data != "off":
#         sp = {name: TensorSplit(X, y, device, placement=args.fast_data, name=name, log=plog)
#               for name, X, y in (("train", X_tr, y_tr), ("val", X_va, y_va), ("test", X_te, y_te))}
#         # drop_last keeps every train batch the same shape (cudnn.benchmark / compile)
#         train_loader = GPUBatches(sp["train"], args.batch_size, shuffle=True, drop_last=True)
#         val_loader = GPUBatches(sp["val"], args.eval_batch_size)
#         test_loader = GPUBatches(sp["test"], args.eval_batch_size)
#     else:
#         kw = dict(num_workers=args.num_workers, pin_memory=args.pin_memory,
#                   persistent_workers=args.num_workers > 0)
#         train_loader = DataLoader(TensorDataset(X_tr, y_tr), args.batch_size,
#                                   shuffle=True, drop_last=True, **kw)
#         val_loader = DataLoader(TensorDataset(X_va, y_va), args.eval_batch_size, shuffle=False, **kw)
#         test_loader = DataLoader(TensorDataset(X_te, y_te), args.eval_batch_size, shuffle=False, **kw)

#     model = IVN_TCN(
#         seq_len=seq_len, nvars=nvars, num_classes=num_classes,
#         use_embedding=args.use_embedding, d_emb=args.d_emb, n_freq=args.n_freq,
#         emb_sigma=args.emb_sigma, cat_col=None if args.cat_col < 0 else args.cat_col,
#         patch_size=args.patch_size, patch_stride=args.patch_stride,
#         dims=args.dims, num_blocks=args.num_blocks, large_sizes=args.large_sizes,
#         small_size=args.small_size, ffn_ratio=args.ffn_ratio, dropout=args.dropout,
#         multi_scale=args.multi_scale, ms_kernel_sizes=args.ms_kernel_sizes,
#         ms_fusion=args.ms_fusion,
#         ens_k=args.ens_k, head_hidden=args.head_hidden, head_dropout=args.head_dropout,
#     )
#     model.fit_input_stats(X_tr)          # robust scaler + arb_id vocab, train split only
#     model.to(device)
#     fwd = torch.compile(model) if args.compile else model
#     scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

#     pc = model.count_parameters()
#     optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#     scheduler = OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
#                            epochs=args.epochs, pct_start=0.3)

#     banner("IVN-TCN: light MS-ModernTCN + RealMLP embedding + TabM head")
#     plog(f"  Device    : {device}  amp={args.amp}  fast_data={args.fast_data}  compile={args.compile}")
#     plog(f"  Classes   : {class_names}")
#     emb_tag = (f"PBLD d={args.d_emb} freq={args.n_freq} sigma={args.emb_sigma} "
#                f"cat_col={args.cat_col}") if args.use_embedding else "none (raw scalar)"
#     plog(f"  Embedding : {emb_tag}")
#     if args.use_embedding and args.cat_col >= 0:
#         plog(f"  arb_id vocab: {int(model.embed.cat.n_vocab)} IDs (+ OOV)")
#     ms_tag = f"{args.ms_kernel_sizes} fusion={args.ms_fusion}" if args.multi_scale else "single k=13"
#     plog(f"  Backbone  : dims={args.dims} blocks={args.num_blocks} temporal={ms_tag}")
#     plog(f"  Head      : mean-pool + BatchEnsemble k={args.ens_k} hidden={args.head_hidden}")
#     plog(f"  Params    : embed={pc['embed']:,} backbone={pc['backbone']:,} "
#          f"head={pc['head']:,} total={pc['total']:,}")
#     plog(f"  Train/Val/Test: {len(X_tr)}/{len(X_va)}/{len(X_te)}")

#     history = {k: [] for k in ["train_loss", "val_loss", "val_f1", "test_loss", "test_f1", "lr"]}
#     best_val, best_test, patience_ctr, t0 = -1.0, -1.0, 0, time.time()

#     for epoch in range(1, args.epochs + 1):
#         te = time.time(); model.train()
#         loss_sum = torch.zeros((), device=device)
#         for X, y in train_loader:
#             X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
#             optimizer.zero_grad(set_to_none=True)
#             with torch.autocast(device.type, dtype=amp_dtype or torch.float32,
#                                 enabled=amp_dtype is not None):
#                 loss = member_loss(fwd(X).float(), y, args.label_smoothing)
#             scaler.scale(loss).backward()
#             scaler.unscale_(optimizer)
#             nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
#             scaler.step(optimizer); scaler.update(); scheduler.step()
#             loss_sum += loss.detach()
#         train_loss = loss_sum.item() / max(len(train_loader), 1)   # one sync per epoch
#         if not np.isfinite(train_loss):
#             plog("  [error] non-finite training loss, stopping"); break

#         vm = evaluate(fwd, val_loader, device, amp_dtype)
#         tm = evaluate(fwd, test_loader, device, amp_dtype)
#         lr = optimizer.param_groups[0]["lr"]
#         for k, v in [("train_loss", train_loss), ("val_loss", vm["loss"]), ("val_f1", vm["macro_f1"]),
#                      ("test_loss", tm["loss"]), ("test_f1", tm["macro_f1"]), ("lr", lr)]:
#             history[k].append(v)

#         banner(f"EPOCH {epoch:03d}/{args.epochs:03d}", "-")
#         plog(f"  Train | loss={history['train_loss'][-1]:.4f}")
#         plog(f"  Val   | loss={vm['loss']:.4f} acc={vm['acc']:.4f} f1={vm['macro_f1']:.4f}")
#         plog(f"  Test  | loss={tm['loss']:.4f} acc={tm['acc']:.4f} f1={tm['macro_f1']:.4f}")
#         plog(f"  Time  | {fmt_s(time.time()-te)} lr={lr:.2e}")
#         plog("  Val CM:");  [plog(f"    {l}") for l in fmt_cm(vm["cm"]).split("\n")]
#         plog("  Test CM:"); [plog(f"    {l}") for l in fmt_cm(tm["cm"]).split("\n")]

#         ckpt = {"epoch": epoch, "model_state_dict": model.state_dict(), "args": vars(args),
#                 "num_classes": num_classes, "class_names": class_names}
#         torch.save(ckpt, sf / "last.pth")
#         tag = ""
#         if vm["macro_f1"] > best_val:
#             best_val, patience_ctr = vm["macro_f1"], 0
#             torch.save(ckpt, sf / "best_val.pth"); tag += " [best val]"
#         else:
#             patience_ctr += 1
#         if tm["macro_f1"] > best_test:
#             best_test = tm["macro_f1"]; tag += " [best test]"
#         if tag: plog(f"  >>>{tag}")
#         plog(f"  Best | val_f1={best_val:.4f} test_f1={best_test:.4f}")
#         with open(sf / "history.json", "w") as f: json.dump(history, f, indent=2)
#         if patience_ctr >= args.patience:
#             plog(f"\n  Early stopping at epoch {epoch}"); break

#     # Test at best-val checkpoint (the number to report)
#     if not (sf / "best_val.pth").exists():
#         plog("  [error] no best_val.pth (training stopped before the first epoch finished)"); return
#     model.load_state_dict(torch.load(sf / "best_val.pth", weights_only=False)["model_state_dict"])
#     tm = evaluate(model, test_loader, device)          # fp32, as deployed
#     report = classification_report(tm["targets"], tm["preds"], target_names=class_names,
#                                    zero_division=0, digits=4)
#     (sf / "test_at_best_val_report.txt").write_text(report + "\n" + fmt_cm(tm["cm"]) + "\n")

#     model.structural_reparam()
#     tr = evaluate(model, test_loader, device)
#     lat = measure_latency(model, seq_len, nvars, device)

#     banner("DONE")
#     plog(f"  Time            : {fmt_s(time.time()-t0)}  ({fmt_s((time.time()-t0)/max(epoch,1))}/epoch)")
#     plog(f"  Params          : {pc['total']:,}")
#     plog(f"  Best val F1     : {best_val:.6f}")
#     plog(f"  Test @ best val : acc={tm['acc']:.6f} f1={tm['macro_f1']:.6f}")
#     plog(f"  After reparam   : f1={tr['macro_f1']:.6f} (should match)")
#     plog(f"  Latency ({device.type}): batch=1 {lat[1]*1e3:.3f} ms/window | "
#          f"batch=1024 {lat[1024]/1024*1e6:.2f} us/window")
#     plog(report)


# if __name__ == "__main__":
#     main()

# ─────────────────────────── V2 ───────────────────────────
"""
Train IVN-TCN (lightweight MS-ModernTCN + RealMLP-style embedding + TabM-style head).

Loss follows TabM: cross-entropy applied to each of the k member predictions
(mean over members), NOT to the averaged prediction. Inference averages the
member probabilities.
"""

from __future__ import annotations
import os, sys


def _early_cpu_threads():
    # must run before numpy/torch import to cap BLAS/OpenMP thread pools
    n = "4"
    if "--cpu_threads" in sys.argv:
        i = sys.argv.index("--cpu_threads")
        if i + 1 < len(sys.argv):
            n = sys.argv[i + 1]
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, n)


_early_cpu_threads()

import argparse, json, random, time   # noqa: E402
from pathlib import Path               # noqa: E402

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, TensorDataset

try:
    from fast_data_seq import TensorSplit, GPUBatches
except ImportError:
    from utils.fast_data_seq import TensorSplit, GPUBatches

try:
    from networks.ivn_tcn import IVN_TCN
except ImportError:
    from ivn_tcn import IVN_TCN


def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

_log_path = None
def plog(t=""):
    print(t)
    if _log_path:
        with open(_log_path, "a") as f: f.write(t + "\n")

def banner(t, c="=", w=90): plog(f"\n{c*w}\n  {t}\n{c*w}\n")
def fmt_s(s): return f"{s:.1f}s" if s < 60 else f"{int(s//60)}m{s%60:.0f}s"
def fmt_cm(cm): return np.array2string(cm, separator=", ", max_line_width=200)


def load_split(p):
    d = torch.load(p, weights_only=False)
    return d["X"].float(), d["y"].long()


def get_class_names(data_dir, n):
    meta = Path(data_dir) / "metadata.json"
    if meta.exists():
        cmap = json.loads(meta.read_text()).get("class_map", {})
        names = ["normal"] + [Path(k).stem for k, v in sorted(cmap.items(), key=lambda x: int(x[1]))]
        return names[:n]
    return ["normal"] + [f"cls_{i}" for i in range(1, n)]


def member_loss(logits, y, label_smoothing):
    B, K, C = logits.shape
    return F.cross_entropy(logits.reshape(B * K, C), y.repeat_interleave(K),
                           label_smoothing=label_smoothing)


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype=None):
    model.eval()
    preds, tgts = [], []
    nll = torch.zeros((), device=device)
    n = 0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype or torch.float32,
                            enabled=amp_dtype is not None):
            logits = model(X)
        p = IVN_TCN.ensemble_proba(logits.float())
        nll += F.nll_loss(torch.log(p.clamp_min(1e-12)), y, reduction="sum")
        n += y.size(0)
        preds.append(p.argmax(-1)); tgts.append(y)
    preds, tgts = torch.cat(preds).cpu().numpy(), torch.cat(tgts).cpu().numpy()
    return {"loss": nll.item() / max(n, 1), "acc": accuracy_score(tgts, preds),
            "macro_f1": f1_score(tgts, preds, average="macro", zero_division=0),
            "preds": preds, "targets": tgts, "cm": confusion_matrix(tgts, preds)}


@torch.no_grad()
def oov_report(model, X, y, cat_col, class_names, chunk=20000):
    """Share of frames whose arb_id is out of vocabulary, per class (val split)."""
    cat = model.embed.cat
    dev = cat.vocab.device
    hits = torch.zeros(len(class_names)); tot = torch.zeros(len(class_names))
    for i in range(0, len(y), chunk):
        ids = X[i:i + chunk, :, cat_col].to(dev)
        oov = (cat.lookup(ids) == cat.oov_index).float().mean(dim=1).cpu()   # per window
        yy = y[i:i + chunk]
        hits.index_add_(0, yy, oov); tot.index_add_(0, yy, torch.ones_like(oov))
    parts = [f"{n}={100 * h / max(t, 1):.2f}%" for n, h, t in zip(class_names, hits.tolist(), tot.tolist())]
    plog("  OOV frames per class (val): " + ", ".join(parts))


@torch.no_grad()
def measure_latency(model, seq_len, nvars, device, iters=200):
    model.eval()
    out = {}
    for bs in (1, 1024):
        x = torch.randn(bs, seq_len, nvars, device=device)
        for _ in range(20): model(x)
        if device.type == "cuda": torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters): model(x)
        if device.type == "cuda": torch.cuda.synchronize()
        out[bs] = (time.perf_counter() - t) / iters
    return out


def main():
    global _log_path
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--save_folder", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4, help="only used with --fast_data off")
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--eval_batch_size", type=int, default=8192)
    p.add_argument("--cpu_threads", type=int, default=4)
    p.add_argument("--fast_data", default="auto", choices=["auto", "gpu", "cpu", "off"])
    p.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    p.add_argument("--compile", action="store_true", help="torch.compile the model")
    # training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=4.0)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    # embedding (RealMLP-style)
    p.add_argument("--no_embedding", dest="use_embedding", action="store_false")
    p.add_argument("--d_emb", type=int, default=8)
    p.add_argument("--n_freq", type=int, default=16)
    p.add_argument("--emb_sigma", type=float, default=1.0)
    p.add_argument("--cat_col", type=int, default=0, help="arb_id column; -1 to treat as numeric")
    p.add_argument("--vocab_from", choices=["normal", "all"], default="normal",
                   help="build the arb_id vocabulary from normal train windows only, or from all")
    p.add_argument("--vocab_min_frac", type=float, default=1e-5,
                   help="an ID enters the vocab if it fills at least this fraction of frame positions")
    p.add_argument("--id_cond", action="store_true",
                   help="ID-conditioned (FiLM) modulation of numeric field embeddings")
    # backbone
    p.add_argument("--patch_size", type=int, default=8)
    p.add_argument("--patch_stride", type=int, default=4)
    p.add_argument("--dims", type=int, nargs="+", default=[64, 128])
    p.add_argument("--num_blocks", type=int, nargs="+", default=[1, 1])
    p.add_argument("--large_sizes", type=int, nargs="+", default=[13, 13])
    p.add_argument("--small_size", type=int, default=5)
    p.add_argument("--ffn_ratio", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no_multi_scale", dest="multi_scale", action="store_false")
    p.add_argument("--ms_kernel_sizes", type=int, nargs="+", default=[3, 5, 13])
    p.add_argument("--ms_fusion", choices=["channel", "sum", "concat"], default="channel")
    # head (TabM-style)
    p.add_argument("--ens_k", type=int, default=8)
    p.add_argument("--head_hidden", type=int, default=64)
    p.add_argument("--head_dropout", type=float, default=0.1)
    args = p.parse_args()

    torch.set_num_threads(args.cpu_threads)
    seed_all(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True      # fixed shapes: autotune conv kernels once
    else:
        device = torch.device("cpu")
    amp_dtype = None
    if device.type == "cuda" and args.amp != "off":
        amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    sf = Path(args.save_folder); sf.mkdir(parents=True, exist_ok=True)
    _log_path = sf / "training_log.txt"
    plog("=" * 90); plog(f"Args: {vars(args)}")

    X_tr, y_tr = load_split(Path(args.data_dir) / "train.pt")
    X_va, y_va = load_split(Path(args.data_dir) / "val.pt")
    X_te, y_te = load_split(Path(args.data_dir) / "test.pt")
    num_classes = int(max(y_tr.max(), y_va.max(), y_te.max())) + 1
    seq_len, nvars = X_tr.shape[1], X_tr.shape[2]
    class_names = get_class_names(args.data_dir, num_classes)

    if args.fast_data != "off":
        sp = {name: TensorSplit(X, y, device, placement=args.fast_data, name=name, log=plog)
              for name, X, y in (("train", X_tr, y_tr), ("val", X_va, y_va), ("test", X_te, y_te))}
        # drop_last keeps every train batch the same shape (cudnn.benchmark / compile)
        train_loader = GPUBatches(sp["train"], args.batch_size, shuffle=True, drop_last=True)
        val_loader = GPUBatches(sp["val"], args.eval_batch_size)
        test_loader = GPUBatches(sp["test"], args.eval_batch_size)
    else:
        kw = dict(num_workers=args.num_workers, pin_memory=args.pin_memory,
                  persistent_workers=args.num_workers > 0)
        train_loader = DataLoader(TensorDataset(X_tr, y_tr), args.batch_size,
                                  shuffle=True, drop_last=True, **kw)
        val_loader = DataLoader(TensorDataset(X_va, y_va), args.eval_batch_size, shuffle=False, **kw)
        test_loader = DataLoader(TensorDataset(X_te, y_te), args.eval_batch_size, shuffle=False, **kw)

    model = IVN_TCN(
        seq_len=seq_len, nvars=nvars, num_classes=num_classes,
        use_embedding=args.use_embedding, d_emb=args.d_emb, n_freq=args.n_freq,
        emb_sigma=args.emb_sigma, cat_col=None if args.cat_col < 0 else args.cat_col,
        id_cond=args.id_cond,
        patch_size=args.patch_size, patch_stride=args.patch_stride,
        dims=args.dims, num_blocks=args.num_blocks, large_sizes=args.large_sizes,
        small_size=args.small_size, ffn_ratio=args.ffn_ratio, dropout=args.dropout,
        multi_scale=args.multi_scale, ms_kernel_sizes=args.ms_kernel_sizes,
        ms_fusion=args.ms_fusion,
        ens_k=args.ens_k, head_hidden=args.head_hidden, head_dropout=args.head_dropout,
    )
    # robust scaler + arb_id vocab, train split only (vocab from normal windows by default)
    model.fit_input_stats(X_tr, y_tr if args.vocab_from == "normal" else None,
                          vocab_min_frac=args.vocab_min_frac)
    model.to(device)
    fwd = torch.compile(model) if args.compile else model
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    pc = model.count_parameters()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                           epochs=args.epochs, pct_start=0.3)

    banner("IVN-TCN: light MS-ModernTCN + RealMLP embedding + TabM head")
    plog(f"  Device    : {device}  amp={args.amp}  fast_data={args.fast_data}  compile={args.compile}")
    plog(f"  Classes   : {class_names}")
    emb_tag = (f"PBLD d={args.d_emb} freq={args.n_freq} sigma={args.emb_sigma} "
               f"cat_col={args.cat_col}") if args.use_embedding else "none (raw scalar)"
    plog(f"  Embedding : {emb_tag}")
    if args.use_embedding and args.cat_col >= 0:
        info = model.embed.cat.fit_info
        plog(f"  arb_id vocab: {info['n_vocab']} IDs (+ OOV) from "
             f"{'normal' if info['from_normal_only'] else 'all'} train windows, "
             f"min_count={info['min_count']}{'  [CAPPED at max_vocab]' if info['capped'] else ''}")
        plog(f"  ID-conditioned embedding: {'on' if args.id_cond else 'off'}")
        oov_report(model, X_va, y_va, args.cat_col, class_names)
    ms_tag = f"{args.ms_kernel_sizes} fusion={args.ms_fusion}" if args.multi_scale else "single k=13"
    plog(f"  Backbone  : dims={args.dims} blocks={args.num_blocks} temporal={ms_tag}")
    plog(f"  Head      : mean-pool + BatchEnsemble k={args.ens_k} hidden={args.head_hidden}")
    plog(f"  Params    : embed={pc['embed']:,} backbone={pc['backbone']:,} "
         f"head={pc['head']:,} total={pc['total']:,}")
    plog(f"  Train/Val/Test: {len(X_tr)}/{len(X_va)}/{len(X_te)}")

    history = {k: [] for k in ["train_loss", "val_loss", "val_f1", "test_loss", "test_f1", "lr"]}
    best_val, best_test, patience_ctr, t0 = -1.0, -1.0, 0, time.time()

    for epoch in range(1, args.epochs + 1):
        te = time.time(); model.train()
        loss_sum = torch.zeros((), device=device)
        for X, y in train_loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=amp_dtype or torch.float32,
                                enabled=amp_dtype is not None):
                loss = member_loss(fwd(X).float(), y, args.label_smoothing)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            loss_sum += loss.detach()
        train_loss = loss_sum.item() / max(len(train_loader), 1)   # one sync per epoch
        if not np.isfinite(train_loss):
            plog("  [error] non-finite training loss, stopping"); break

        vm = evaluate(fwd, val_loader, device, amp_dtype)
        tm = evaluate(fwd, test_loader, device, amp_dtype)
        lr = optimizer.param_groups[0]["lr"]
        for k, v in [("train_loss", train_loss), ("val_loss", vm["loss"]), ("val_f1", vm["macro_f1"]),
                     ("test_loss", tm["loss"]), ("test_f1", tm["macro_f1"]), ("lr", lr)]:
            history[k].append(v)

        banner(f"EPOCH {epoch:03d}/{args.epochs:03d}", "-")
        plog(f"  Train | loss={history['train_loss'][-1]:.4f}")
        plog(f"  Val   | loss={vm['loss']:.4f} acc={vm['acc']:.4f} f1={vm['macro_f1']:.4f}")
        plog(f"  Test  | loss={tm['loss']:.4f} acc={tm['acc']:.4f} f1={tm['macro_f1']:.4f}")
        plog(f"  Time  | {fmt_s(time.time()-te)} lr={lr:.2e}")
        plog("  Val CM:");  [plog(f"    {l}") for l in fmt_cm(vm["cm"]).split("\n")]
        plog("  Test CM:"); [plog(f"    {l}") for l in fmt_cm(tm["cm"]).split("\n")]

        ckpt = {"epoch": epoch, "model_state_dict": model.state_dict(), "args": vars(args),
                "num_classes": num_classes, "class_names": class_names}
        torch.save(ckpt, sf / "last.pth")
        tag = ""
        if vm["macro_f1"] > best_val:
            best_val, patience_ctr = vm["macro_f1"], 0
            torch.save(ckpt, sf / "best_val.pth"); tag += " [best val]"
        else:
            patience_ctr += 1
        if tm["macro_f1"] > best_test:
            best_test = tm["macro_f1"]; tag += " [best test]"
        if tag: plog(f"  >>>{tag}")
        plog(f"  Best | val_f1={best_val:.4f} test_f1={best_test:.4f}")
        with open(sf / "history.json", "w") as f: json.dump(history, f, indent=2)
        if patience_ctr >= args.patience:
            plog(f"\n  Early stopping at epoch {epoch}"); break

    # Test at best-val checkpoint (the number to report)
    if not (sf / "best_val.pth").exists():
        plog("  [error] no best_val.pth (training stopped before the first epoch finished)"); return
    model.load_state_dict(torch.load(sf / "best_val.pth", weights_only=False)["model_state_dict"])
    tm = evaluate(model, test_loader, device)          # fp32, as deployed
    report = classification_report(tm["targets"], tm["preds"], target_names=class_names,
                                   zero_division=0, digits=4)
    (sf / "test_at_best_val_report.txt").write_text(report + "\n" + fmt_cm(tm["cm"]) + "\n")

    model.structural_reparam()
    tr = evaluate(model, test_loader, device)
    lat = measure_latency(model, seq_len, nvars, device)

    banner("DONE")
    plog(f"  Time            : {fmt_s(time.time()-t0)}  ({fmt_s((time.time()-t0)/max(epoch,1))}/epoch)")
    plog(f"  Params          : {pc['total']:,}")
    plog(f"  Best val F1     : {best_val:.6f}")
    plog(f"  Test @ best val : acc={tm['acc']:.6f} f1={tm['macro_f1']:.6f}")
    plog(f"  After reparam   : f1={tr['macro_f1']:.6f} (should match)")
    plog(f"  Latency ({device.type}): batch=1 {lat[1]*1e3:.3f} ms/window | "
         f"batch=1024 {lat[1024]/1024*1e6:.2f} us/window")
    plog(report)


if __name__ == "__main__":
    main()