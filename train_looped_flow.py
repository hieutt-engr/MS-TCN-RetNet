"""
Training script for MS-TCN + Looped Flow with Progressive Feature Reveal.
Clean version: no adversarial evaluation, focused on monitoring iterative refinement.
"""

from __future__ import annotations
import argparse, json, random, time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from dataloader import build_all_dataloaders
from networks.ms_tcn_looped_flow import MSTCNLoopedFlow


def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def ensure(p): Path(p).mkdir(parents=True, exist_ok=True)

# Global log file path, set in main()
_log_path = None

def set_log_path(p):
    global _log_path
    _log_path = p

def plog(text=""):
    """Print to terminal AND append to log file."""
    print(text)
    if _log_path:
        with open(_log_path, "a") as f:
            f.write(text + "\n")

def fmt_s(s):
    return f"{s:.1f}s" if s < 60 else f"{int(s//60)}m{s%60:.0f}s"

def fmt_cm(cm):
    return np.array2string(cm, separator=", ", max_line_width=200)

def banner(t, c="=", w=90):
    plog(f"\n{c*w}\n  {t}\n{c*w}\n")

def count_params(m):
    t = sum(p.numel() for p in m.parameters())
    b = sum(p.numel() for n, p in m.named_parameters() if "denoiser" not in n)
    return {"total": t, "backbone": b, "denoiser": t - b}

def step_stats(logits_all, act_scores, labels):
    accs, confs = [], []
    for lg, ac in zip(logits_all, act_scores):
        accs.append((lg.argmax(-1) == labels).float().mean().item())
        confs.append(torch.sigmoid(ac).mean().item())
    return accs, confs

def fmt_curve(vals, n_show=8):
    n = len(vals)
    if n <= n_show:
        idx = list(range(n))
    else:
        idx = [0, 1, 2]
        mid = n_show - 6
        if mid > 0:
            s = (n - 6) / (mid + 1)
            idx += [int(3 + s * (j + 1)) for j in range(mid)]
        idx += [n - 3, n - 2, n - 1]
        idx = sorted(set(idx))
    return " -> ".join(f"t{i}:{vals[i]:.3f}" for i in idx)


@torch.no_grad()
def evaluate(model, loader, criterion, device, n_steps=32, num_traj=1):
    model.eval()
    preds_all, tgts_all, loss_sum, n = [], [], 0.0, 0
    for batch in loader:
        X, y = batch[0].to(device), batch[1].to(device)
        logits = model(X, n_steps=n_steps, num_trajectories=num_traj)
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
    p.add_argument("--model_name", default="ms_tcn_lf_reveal")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu_id", type=int, default=0)

    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--use_weighted_sampler", action="store_true")
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--print_freq", type=int, default=100)

    p.add_argument("--scheduler", default="cosine_restart", choices=["none", "cosine_restart"])
    p.add_argument("--t0", type=int, default=10)
    p.add_argument("--t_mult", type=int, default=2)

    # Backbone
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--branch_dim", type=int, default=32)
    p.add_argument("--tcn_dim", type=int, default=64)
    p.add_argument("--tcn_depth", type=int, default=2)
    p.add_argument("--tcn_kernel_size", type=int, default=3)
    p.add_argument("--tcn_dropout", type=float, default=0.1)
    p.add_argument("--pool", default="mean")

    # Denoiser
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--denoiser_hidden_dim", type=int, default=128)
    p.add_argument("--denoiser_expansion", type=int, default=4)
    p.add_argument("--denoiser_dropout", type=float, default=0.1)
    p.add_argument("--noise_scale", type=float, default=1.0)
    p.add_argument("--train_steps", type=int, default=16)
    p.add_argument("--act_weight", type=float, default=0.5)
    p.add_argument("--pseudotarget_warmup", type=int, default=5000)
    p.add_argument("--latent_noise_std", type=float, default=0.1)

    # Progressive reveal
    p.add_argument("--num_reveal_groups", type=int, default=4,
                   help="Number of feature groups to reveal progressively")

    # Inference
    p.add_argument("--inference_steps", type=int, default=32)
    p.add_argument("--inference_trajectories", type=int, default=1)

    # Logging
    p.add_argument("--log_curve_freq", type=int, default=200,
                   help="Print full denoising curve every N iterations")

    args = p.parse_args()
    seed_all(args.seed)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.cuda.set_device(device)
        gpu = torch.cuda.get_device_name(device)
    else:
        device = torch.device("cpu"); gpu = "CPU"

    sf = Path(args.save_folder); ensure(sf)
    lf = sf / "training_log.txt"; hf = sf / "history.json"
    set_log_path(lf)
    plog("=" * 100)
    plog(f"Args: {vars(args)}")

    bundle = build_all_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        train_drop_last=False, return_index=False,
    )
    train_loader = bundle["train_loader"]
    val_loader = bundle["val_loader"]
    test_loader = bundle["test_loader"]
    input_shape = bundle["input_shape"]
    num_classes = bundle["num_classes"]

    meta_path = Path(args.data_dir) / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        cmap = meta.get("class_map", {})
        class_names = ["normal"] + [Path(k).stem for k, v in sorted(cmap.items(), key=lambda x: int(x[1]))]
    else:
        class_names = ["normal"] + [f"cls_{i}" for i in range(1, num_classes)]

    model = MSTCNLoopedFlow(
        input_dim=input_shape[1], num_classes=num_classes,
        hidden_dim=args.hidden_dim, branch_dim=args.branch_dim,
        tcn_dim=args.tcn_dim, tcn_depth=args.tcn_depth,
        tcn_kernel_size=args.tcn_kernel_size, tcn_dropout=args.tcn_dropout,
        pool=args.pool, latent_dim=args.latent_dim,
        denoiser_hidden_dim=args.denoiser_hidden_dim,
        denoiser_expansion=args.denoiser_expansion,
        denoiser_dropout=args.denoiser_dropout,
        noise_scale=args.noise_scale, train_steps=args.train_steps,
        inference_steps=args.inference_steps,
        inference_trajectories=args.inference_trajectories,
        num_reveal_groups=args.num_reveal_groups,
    ).to(device)

    pc = count_params(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingWarmRestarts(optimizer, args.t0, args.t_mult) if args.scheduler != "none" else None
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    amp_ctx = (lambda: torch.amp.autocast("cuda")) if args.amp and device.type == "cuda" else nullcontext

    banner("MS-TCN + LOOPED FLOW (Progressive Feature Reveal)")
    plog(f"  Device         : {device} ({gpu})")
    plog(f"  Input          : {input_shape}, {num_classes} classes")
    plog(f"  Classes        : {class_names}")
    plog(f"  Reveal groups  : {args.num_reveal_groups}")
    plog(f"  Params         : backbone={pc['backbone']:,}  denoiser={pc['denoiser']:,}  total={pc['total']:,}")
    plog(f"  Train steps    : {args.train_steps}, Infer steps: {args.inference_steps}")
    plog()

    history = {
        "train_loss": [], "val_acc": [], "val_f1": [],
        "test_acc": [], "test_f1": [], "lr": [],
        "delta_first_last": [],
    }
    best_val_f1 = best_test_f1 = -1.0
    t_start = time.time()
    global_iter = 0

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        model.train()
        loss_sum = n_samp = 0
        ep_first = ep_last = ep_n = 0.0
        nb = len(train_loader)

        for bi, batch in enumerate(train_loader, 1):
            X, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad(set_to_none=True)

            global_iter += 1
            pt = min(1.0, global_iter / args.pseudotarget_warmup) if args.pseudotarget_warmup > 0 else 1.0

            with amp_ctx():
                out = model(X, labels=y, k=args.train_steps, act_weight=args.act_weight,
                            pseudotarget_prob=pt, latent_noise_std=args.latent_noise_std)
                loss = out["loss"]

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer); scaler.update()

            bs = X.size(0); loss_sum += loss.item() * bs; n_samp += bs

            with torch.no_grad():
                accs, _ = step_stats(out["logits_all"], out["act_scores"], y)
                ep_first += accs[0]; ep_last += accs[-1]; ep_n += 1

            if args.print_freq > 0 and (bi % args.print_freq == 0 or bi == nb):
                k = len(accs)
                d = accs[-1] - accs[0]
                plog(f"  [{epoch:03d}] {bi:04d}/{nb:04d} | loss={loss.item():.4f} | "
                     f"t0={accs[0]:.3f} t{k-1}={accs[-1]:.3f} (d={d:+.3f}) | pt={pt:.2f}")

            if args.log_curve_freq > 0 and bi % args.log_curve_freq == 0:
                plog(f"    Curve: [{fmt_curve(accs)}]")

        train_loss = loss_sum / max(n_samp, 1)
        avg_first = ep_first / max(ep_n, 1)
        avg_last = ep_last / max(ep_n, 1)
        avg_delta = avg_last - avg_first

        if sched:
            sched.step(epoch - 1 + 1.0)
        lr = optimizer.param_groups[0]["lr"]

        val_m = evaluate(model, val_loader, criterion, device, args.inference_steps, args.inference_trajectories)
        test_m = evaluate(model, test_loader, criterion, device, args.inference_steps, args.inference_trajectories)

        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_m["acc"]); history["val_f1"].append(val_m["macro_f1"])
        history["test_acc"].append(test_m["acc"]); history["test_f1"].append(test_m["macro_f1"])
        history["lr"].append(lr); history["delta_first_last"].append(avg_delta)

        banner(f"EPOCH {epoch:03d}/{args.epochs:03d}", "-")
        plog(f"  Train | loss={train_loss:.4f} | first={avg_first:.4f} last={avg_last:.4f} delta={avg_delta:+.4f}")
        plog(f"  Val   | acc={val_m['acc']:.4f} f1={val_m['macro_f1']:.4f}")
        plog(f"  Test  | acc={test_m['acc']:.4f} f1={test_m['macro_f1']:.4f}")
        plog(f"  Time  | {fmt_s(time.time()-t_ep)} lr={lr:.2e}")

        plog("  Val CM:")
        for line in fmt_cm(val_m["cm"]).split("\n"):
            plog(f"    {line}")
        plog("  Test CM:")
        for line in fmt_cm(test_m["cm"]).split("\n"):
            plog(f"    {line}")

        ckpt = {"epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args), "history": history, "num_classes": num_classes}
        torch.save(ckpt, sf / f"{args.model_name}_last.pth")

        improved = ""
        if val_m["macro_f1"] > best_val_f1:
            best_val_f1 = val_m["macro_f1"]
            torch.save(ckpt, sf / f"{args.model_name}_best_val.pth"); improved += " [best val]"
        if test_m["macro_f1"] > best_test_f1:
            best_test_f1 = test_m["macro_f1"]
            torch.save(ckpt, sf / f"{args.model_name}_best_test.pth"); improved += " [best test]"
            report = classification_report(test_m["targets"], test_m["preds"],
                                           target_names=class_names, zero_division=0, digits=4)
            with open(sf / "best_test_report.txt", "w") as f: f.write(report)
        if improved:
            plog(f"  >>>{improved}")
        plog(f"  Best | val_f1={best_val_f1:.4f} test_f1={best_test_f1:.4f}")

        with open(hf, "w") as f: json.dump(history, f, indent=2)

    banner("DONE")
    plog(f"  Time: {fmt_s(time.time()-t_start)}")
    plog(f"  Best val F1: {best_val_f1:.6f}")
    plog(f"  Best test F1: {best_test_f1:.6f}")


if __name__ == "__main__":
    main()