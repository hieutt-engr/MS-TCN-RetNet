"""
Payload-only FGSM-RS adversarial training for MS-TCN-RetNet.

Required files in the same project folder:
    dataloader.py or canml_pt_dataset.py
    ms_tcn_retnet_torch.py

Expected preprocessed data:
    data_dir/train.pt
    data_dir/val.pt
    data_dir/test.pt
    data_dir/metadata.json

Feature order from the current preprocessing:
    0  arbitration_id_norm
    1  dlc_norm
    2  payload_0_norm255
    3  payload_1_norm255
    4  payload_2_norm255
    5  payload_3_norm255
    6  payload_4_norm255
    7  payload_5_norm255
    8  payload_6_norm255
    9  payload_7_norm255
    10 log_delta_t_norm
    11 id_freq_recent_norm

Default adversarial mask: payload feature indices 2..9 only.
"""

from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from dataloader import build_all_dataloaders

from networks.ms_tcn_retnet_torch import MSTCNRetNet

def parse_float_or_fraction(value: str) -> float:
    s = str(value).strip()
    if "/" in s:
        a, b = s.split("/", 1)
        return float(a) / float(b)
    return float(s)


def parse_indices(value: str) -> List[int]:
    return [int(v.strip()) for v in str(value).split(",") if v.strip()]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def append_log(path: Path, text: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def save_json(path: Path, obj: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def format_confusion_matrix(cm: np.ndarray) -> str:
    return np.array2string(cm, separator=", ", max_line_width=220)


def safe_torch_load(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


def make_grad_scaler(device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.GradScaler("cuda", enabled=True)
    return torch.amp.GradScaler("cpu", enabled=False)


def get_amp_context(device, enabled: bool):
    if enabled and device.type == "cuda":
        return lambda: torch.amp.autocast("cuda")
    return nullcontext


def get_class_names(data_dir: Path, num_classes: int) -> List[str]:
    meta_path = data_dir / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            class_map = meta.get("class_map", {})
            names = ["normal"]
            inv = {int(cid): Path(fname).stem for fname, cid in class_map.items()}
            for c in range(1, num_classes):
                names.append(inv.get(c, f"class_{c}"))
            return names[:num_classes]
        except Exception:
            pass
    return ["normal"] + [f"class_{i}" for i in range(1, num_classes)]


def compute_class_weights(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(y.cpu(), minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    return counts.sum() / (num_classes * counts)


class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def build_loss(loss_name: str, class_weights: Optional[torch.Tensor], focal_gamma: float) -> nn.Module:
    if loss_name == "ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    if loss_name == "focal":
        return FocalLoss(alpha=class_weights, gamma=focal_gamma)
    raise ValueError(f"Unsupported loss_name={loss_name}")


def build_model_from_args(args, input_dim: int, num_classes: int, ckpt_args: Optional[Dict]) -> MSTCNRetNet:
    src = ckpt_args if (ckpt_args is not None and not args.disable_checkpoint_hparams) else vars(args)

    def get(name, default):
        if isinstance(src, dict):
            return src.get(name, default)
        return default

    return MSTCNRetNet(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=int(get("hidden_dim", args.hidden_dim)),
        branch_dim=int(get("branch_dim", args.branch_dim)),
        tcn_dim=int(get("tcn_dim", args.tcn_dim)),
        tcn_depth=int(get("tcn_depth", args.tcn_depth)),
        tcn_kernel_size=int(get("tcn_kernel_size", args.tcn_kernel_size)),
        tcn_dropout=float(get("tcn_dropout", args.tcn_dropout)),
        retention_dim=int(get("retention_dim", args.retention_dim)),
        retention_depth=int(get("retention_depth", args.retention_depth)),
        retention_heads=int(get("retention_heads", args.retention_heads)),
        retention_dropout=float(get("retention_dropout", args.retention_dropout)),
        ff_expansion=int(get("ff_expansion", args.ff_expansion)),
        pool=str(get("pool", args.pool)),
    )


def make_feature_mask(x: torch.Tensor, indices: List[int]) -> torch.Tensor:
    mask = torch.zeros_like(x)
    mask[:, :, indices] = 1.0
    return mask


def project_payload_linf(x_adv: torch.Tensor, x_orig: torch.Tensor, epsilon: float, payload_indices: List[int]) -> torch.Tensor:
    x_proj = x_orig.clone()
    adv_payload = x_adv[:, :, payload_indices]
    orig_payload = x_orig[:, :, payload_indices]
    adv_payload = torch.max(torch.min(adv_payload, orig_payload + epsilon), orig_payload - epsilon)
    adv_payload = adv_payload.clamp(0.0, 1.0)
    x_proj[:, :, payload_indices] = adv_payload
    return x_proj


def fgsm_rs_attack(model, x, y, criterion, epsilon: float, alpha: float, payload_indices: List[int]) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    mask = make_feature_mask(x, payload_indices)
    x_adv = x.detach().clone()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon) * mask
    x_adv = project_payload_linf(x_adv, x, epsilon, payload_indices)
    x_adv.requires_grad_(True)

    logits = model(x_adv)
    loss = criterion(logits, y)
    grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]

    x_adv = x_adv.detach() + alpha * grad.sign() * mask
    x_adv = project_payload_linf(x_adv, x, epsilon, payload_indices)
    return x_adv.detach()


def pgd_attack(model, x, y, criterion, epsilon: float, alpha: float, steps: int, payload_indices: List[int]) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    mask = make_feature_mask(x, payload_indices)
    x_adv = x.detach().clone()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon) * mask
    x_adv = project_payload_linf(x_adv, x, epsilon, payload_indices)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = criterion(logits, y)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        x_adv = x_adv.detach() + alpha * grad.sign() * mask
        x_adv = project_payload_linf(x_adv, x, epsilon, payload_indices)
    return x_adv.detach()


def compute_metrics(targets: np.ndarray, preds: np.ndarray, loss: float) -> Dict:
    attack_mask = targets != 0
    if attack_mask.sum() > 0:
        fool_to_benign = float(np.mean(preds[attack_mask] == 0))
        attack_only_acc = float(np.mean(preds[attack_mask] == targets[attack_mask]))
    else:
        fool_to_benign = 0.0
        attack_only_acc = 0.0

    return {
        "loss": float(loss),
        "acc": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(targets, preds, average="weighted", zero_division=0)),
        "fool_to_benign": fool_to_benign,
        "attack_only_acc": attack_only_acc,
        "preds": preds,
        "targets": targets,
        "cm": confusion_matrix(targets, preds),
    }


@torch.no_grad()
def evaluate_clean(model, loader, criterion, device) -> Dict:
    model.eval()
    all_logits, all_targets = [], []
    total_loss, total_n = 0.0, 0
    for batch in loader:
        x, y = batch[:2]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_n += bs
        all_logits.append(logits.cpu())
        all_targets.append(y.cpu())
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0).numpy()
    preds = logits.argmax(dim=1).numpy()
    return compute_metrics(targets, preds, total_loss / max(total_n, 1))


def evaluate_adversarial(model, loader, criterion, device, epsilon, alpha, steps, payload_indices, attack_name="pgd") -> Dict:
    model.eval()
    all_logits, all_targets = [], []
    total_loss, total_n = 0.0, 0

    for batch in loader:
        x, y = batch[:2]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.enable_grad():
            if attack_name == "fgsm_rs":
                x_adv = fgsm_rs_attack(model, x, y, criterion, epsilon, alpha, payload_indices)
            else:
                x_adv = pgd_attack(model, x, y, criterion, epsilon, alpha, steps, payload_indices)

        with torch.no_grad():
            logits = model(x_adv)
            loss = criterion(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_n += bs
        all_logits.append(logits.cpu())
        all_targets.append(y.cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0).numpy()
    preds = logits.argmax(dim=1).numpy()
    return compute_metrics(targets, preds, total_loss / max(total_n, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_folder", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="ms_tcn_retnet_fgsm_rs_at")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument(
        "--epoch_save_every",
        type=str,
        default=None,
        help="Save per-epoch checkpoint every N epochs. Use None/null/0/-1 to disable.",
    )

    parser.add_argument("--scheduler", type=str, default="cosine_restart", choices=["none", "cosine", "cosine_restart"])
    parser.add_argument("--t0", type=int, default=5)
    parser.add_argument("--t_mult", type=int, default=2)

    parser.add_argument("--loss_name", type=str, default="ce", choices=["ce", "focal"])
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    parser.add_argument("--epsilon", type=parse_float_or_fraction, default=parse_float_or_fraction("4/255"))
    parser.add_argument("--alpha", type=parse_float_or_fraction, default=None)
    parser.add_argument("--lambda_clean", type=float, default=1.0)
    parser.add_argument("--lambda_adv", type=float, default=1.0)
    parser.add_argument("--payload_indices", type=parse_indices, default=parse_indices("2,3,4,5,6,7,8,9"))
    parser.add_argument("--adv_on_attack_only", action="store_true")

    parser.add_argument("--adv_eval_freq", type=int, default=1)
    parser.add_argument("--attack_steps_eval", type=int, default=20)
    parser.add_argument("--attack_alpha_eval", type=parse_float_or_fraction, default=None)
    parser.add_argument("--eval_attack_name", type=str, default="pgd", choices=["pgd", "fgsm_rs"])

    parser.add_argument("--disable_checkpoint_hparams", action="store_true")
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

    # Normalize epoch_save_every.
    # Examples:
    #   --epoch_save_every 5    -> save epoch checkpoints at epochs 5, 10, 15, ...
    #   --epoch_save_every None -> disable per-epoch checkpoint saving
    #   omitted                 -> disable per-epoch checkpoint saving
    if args.epoch_save_every is not None:
        epoch_save_str = str(args.epoch_save_every).strip().lower()
        if epoch_save_str in {"none", "null", "false", "no", "0", "-1"}:
            args.epoch_save_every = None
        else:
            args.epoch_save_every = int(epoch_save_str)
            if args.epoch_save_every <= 0:
                args.epoch_save_every = None

    if args.alpha is None:
        args.alpha = args.epsilon
    if args.attack_alpha_eval is None:
        args.attack_alpha_eval = args.epsilon / 4.0 if args.attack_steps_eval > 1 else args.epsilon

    seed_everything(args.seed)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if args.gpu_id < 0 or args.gpu_id >= torch.cuda.device_count():
            raise ValueError(f"Invalid gpu_id={args.gpu_id}; available 0..{torch.cuda.device_count()-1}")
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.cuda.set_device(device)
        device_name = torch.cuda.get_device_name(device)
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    save_folder = Path(args.save_folder)
    ensure_dir(save_folder)
    ensure_dir(save_folder / "epoch_save")
    log_file = save_folder / "training_log.txt"
    history_file = save_folder / "history.json"

    append_log(log_file, "=" * 120)
    append_log(log_file, "Payload-only FGSM-RS adversarial training")
    append_log(log_file, f"Args: {vars(args)}")
    append_log(log_file, f"Device: {device} | {device_name}")

    bundle = build_all_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        train_drop_last=False,
        return_index=False,
    )
    train_loader = bundle["train_loader"]
    val_loader = bundle["val_loader"]
    test_loader = bundle["test_loader"]
    train_dataset = bundle["train_dataset"]
    input_shape = bundle["input_shape"]
    num_classes = bundle["num_classes"]
    class_names = get_class_names(Path(args.data_dir), num_classes)

    ckpt = safe_torch_load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", None) if isinstance(ckpt, dict) else None
    model = build_model_from_args(args, input_shape[1], num_classes, ckpt_args)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
    append_log(log_file, f"Loaded checkpoint: {args.checkpoint}")
    append_log(log_file, f"Missing keys: {missing}")
    append_log(log_file, f"Unexpected keys: {unexpected}")

    model = model.to(device)

    class_weights = compute_class_weights(train_dataset.y, num_classes).to(device) if args.use_class_weights else None
    if class_weights is not None:
        append_log(log_file, f"Class weights: {class_weights.detach().cpu().tolist()}")
    criterion = build_loss(args.loss_name, class_weights, args.focal_gamma)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "cosine_restart":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.t0, T_mult=args.t_mult)
    else:
        scheduler = None

    scaler = make_grad_scaler(device, args.amp)
    amp_ctx_factory = get_amp_context(device, args.amp)

    history = {
        "train_loss": [], "train_clean_loss": [], "train_adv_loss": [],
        "val_clean_acc": [], "val_clean_macro_f1": [],
        "val_adv_acc": [], "val_adv_macro_f1": [], "val_adv_fool_to_benign": [],
        "test_clean_acc": [], "test_clean_macro_f1": [],
        "test_adv_acc": [], "test_adv_macro_f1": [], "test_adv_fool_to_benign": [],
        "lr": [],
    }

    best_val_clean_macro_f1 = -1.0
    best_val_adv_macro_f1 = -1.0
    best_test_adv_macro_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = clean_sum = adv_sum = 0.0
        n_samples = 0
        num_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader, start=1):
            x, y = batch[:2]
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with amp_ctx_factory():
                logits_clean = model(x)
                clean_loss = criterion(logits_clean, y)

            if args.adv_on_attack_only:
                mask_attack = y != 0
                if mask_attack.sum() > 0:
                    x_adv_src = x[mask_attack]
                    y_adv_src = y[mask_attack]
                    with torch.enable_grad():
                        x_adv = fgsm_rs_attack(model, x_adv_src, y_adv_src, criterion, args.epsilon, args.alpha, args.payload_indices)
                    with amp_ctx_factory():
                        adv_loss = criterion(model(x_adv), y_adv_src)
                else:
                    adv_loss = torch.zeros((), device=device)
            else:
                with torch.enable_grad():
                    x_adv = fgsm_rs_attack(model, x, y, criterion, args.epsilon, args.alpha, args.payload_indices)
                with amp_ctx_factory():
                    adv_loss = criterion(model(x_adv), y)

            loss = args.lambda_clean * clean_loss + args.lambda_adv * adv_loss
            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            bs = x.size(0)
            loss_sum += loss.item() * bs
            clean_sum += clean_loss.item() * bs
            adv_sum += adv_loss.item() * bs
            n_samples += bs

            if args.print_freq > 0 and (batch_idx % args.print_freq == 0 or batch_idx == num_batches):
                msg = (f"Epoch [{epoch:03d}/{args.epochs:03d}] Iter [{batch_idx:04d}/{num_batches:04d}] | "
                       f"lr={optimizer.param_groups[0]['lr']:.6e} | "
                       f"loss={loss.item():.6f} | clean={clean_loss.item():.6f} | adv={adv_loss.item():.6f}")
                print(msg)
                append_log(log_file, msg)

        if scheduler is not None:
            if args.scheduler == "cosine_restart":
                scheduler.step(epoch - 1 + 1.0)
            else:
                scheduler.step()

        train_loss = loss_sum / max(n_samples, 1)
        train_clean_loss = clean_sum / max(n_samples, 1)
        train_adv_loss = adv_sum / max(n_samples, 1)
        lr = optimizer.param_groups[0]["lr"]

        val_clean = evaluate_clean(model, val_loader, criterion, device)
        test_clean = evaluate_clean(model, test_loader, criterion, device)

        do_adv_eval = (args.adv_eval_freq > 0 and epoch % args.adv_eval_freq == 0) or epoch == args.epochs
        if do_adv_eval:
            val_adv = evaluate_adversarial(model, val_loader, criterion, device, args.epsilon, args.attack_alpha_eval,
                                           args.attack_steps_eval, args.payload_indices, args.eval_attack_name)
            test_adv = evaluate_adversarial(model, test_loader, criterion, device, args.epsilon, args.attack_alpha_eval,
                                            args.attack_steps_eval, args.payload_indices, args.eval_attack_name)
        else:
            nan_cm = np.zeros((num_classes, num_classes), dtype=int)
            val_adv = {"acc": np.nan, "macro_f1": np.nan, "weighted_f1": np.nan, "fool_to_benign": np.nan, "cm": nan_cm}
            test_adv = {"acc": np.nan, "macro_f1": np.nan, "weighted_f1": np.nan, "fool_to_benign": np.nan, "cm": nan_cm}

        history["train_loss"].append(train_loss)
        history["train_clean_loss"].append(train_clean_loss)
        history["train_adv_loss"].append(train_adv_loss)
        history["val_clean_acc"].append(val_clean["acc"])
        history["val_clean_macro_f1"].append(val_clean["macro_f1"])
        history["val_adv_acc"].append(val_adv["acc"])
        history["val_adv_macro_f1"].append(val_adv["macro_f1"])
        history["val_adv_fool_to_benign"].append(val_adv["fool_to_benign"])
        history["test_clean_acc"].append(test_clean["acc"])
        history["test_clean_macro_f1"].append(test_clean["macro_f1"])
        history["test_adv_acc"].append(test_adv["acc"])
        history["test_adv_macro_f1"].append(test_adv["macro_f1"])
        history["test_adv_fool_to_benign"].append(test_adv["fool_to_benign"])
        history["lr"].append(lr)

        msg = (f"Epoch [{epoch:03d}/{args.epochs:03d}] | lr={lr:.6e} | "
               f"train_loss={train_loss:.6f}, clean={train_clean_loss:.6f}, adv={train_adv_loss:.6f} | "
               f"VAL clean_acc={val_clean['acc']:.4f}, clean_macroF1={val_clean['macro_f1']:.4f}, "
               f"adv_acc={val_adv['acc']:.4f}, adv_macroF1={val_adv['macro_f1']:.4f}, adv_F2B={val_adv['fool_to_benign']:.4f} | "
               f"TEST clean_acc={test_clean['acc']:.4f}, clean_macroF1={test_clean['macro_f1']:.4f}, "
               f"adv_acc={test_adv['acc']:.4f}, adv_macroF1={test_adv['macro_f1']:.4f}, adv_F2B={test_adv['fool_to_benign']:.4f}")
        print(msg)
        append_log(log_file, msg)
        append_log(log_file, "Val clean confusion matrix:\n" + format_confusion_matrix(val_clean["cm"]))
        append_log(log_file, "Val adv confusion matrix:\n" + format_confusion_matrix(val_adv["cm"]))
        append_log(log_file, "Test clean confusion matrix:\n" + format_confusion_matrix(test_clean["cm"]))
        append_log(log_file, "Test adv confusion matrix:\n" + format_confusion_matrix(test_adv["cm"]))

        ckpt_save = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "history": history,
            "input_shape": input_shape,
            "num_classes": num_classes,
            "class_names": class_names,
        }
        if args.epoch_save_every is not None and epoch % args.epoch_save_every == 0:
            epoch_ckpt_path = save_folder / "epoch_save" / f"{args.model_name}_epoch_{epoch:03d}.pth"
            torch.save(ckpt_save, epoch_ckpt_path)
            append_log(log_file, f"Saved epoch checkpoint: {epoch_ckpt_path}")

        torch.save(ckpt_save, save_folder / f"{args.model_name}_last.pth")
        save_json(history_file, history)

        if val_clean["macro_f1"] > best_val_clean_macro_f1:
            best_val_clean_macro_f1 = val_clean["macro_f1"]
            torch.save(ckpt_save, save_folder / f"{args.model_name}_best_clean_macro_f1.pth")
            np.save(save_folder / "best_val_clean_confusion_matrix.npy", val_clean["cm"])

        if do_adv_eval and val_adv["macro_f1"] > best_val_adv_macro_f1:
            best_val_adv_macro_f1 = val_adv["macro_f1"]
            torch.save(ckpt_save, save_folder / f"{args.model_name}_best_adv_macro_f1.pth")
            np.save(save_folder / "best_val_adv_confusion_matrix.npy", val_adv["cm"])

        if do_adv_eval and test_adv["macro_f1"] > best_test_adv_macro_f1:
            best_test_adv_macro_f1 = test_adv["macro_f1"]
            torch.save(ckpt_save, save_folder / f"{args.model_name}_best_test_adv_macro_f1.pth")
            np.save(save_folder / "best_test_adv_confusion_matrix.npy", test_adv["cm"])
            report = classification_report(test_adv["targets"], test_adv["preds"], target_names=class_names,
                                           zero_division=0, digits=4)
            with open(save_folder / "best_test_adv_classification_report.txt", "w", encoding="utf-8") as f:
                f.write(report)

    final_clean = evaluate_clean(model, test_loader, criterion, device)
    final_adv = evaluate_adversarial(model, test_loader, criterion, device, args.epsilon, args.attack_alpha_eval,
                                     args.attack_steps_eval, args.payload_indices, args.eval_attack_name)

    np.save(save_folder / "final_clean_confusion_matrix.npy", final_clean["cm"])
    np.save(save_folder / "final_adv_confusion_matrix.npy", final_adv["cm"])

    final_clean_report = classification_report(final_clean["targets"], final_clean["preds"], target_names=class_names,
                                               zero_division=0, digits=4)
    final_adv_report = classification_report(final_adv["targets"], final_adv["preds"], target_names=class_names,
                                             zero_division=0, digits=4)
    with open(save_folder / "final_clean_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(final_clean_report)
    with open(save_folder / "final_adv_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(final_adv_report)

    append_log(log_file, "=" * 120)
    append_log(log_file, f"Best VAL clean macro-F1: {best_val_clean_macro_f1:.6f}")
    append_log(log_file, f"Best VAL adv macro-F1: {best_val_adv_macro_f1:.6f}")
    append_log(log_file, f"Best TEST adv macro-F1: {best_test_adv_macro_f1:.6f}")
    append_log(log_file, "Final clean TEST report:\n" + final_clean_report)
    append_log(log_file, "Final adversarial TEST report:\n" + final_adv_report)

    print("\nTraining finished.")
    print(f"Best VAL clean macro-F1: {best_val_clean_macro_f1:.6f}")
    print(f"Best VAL adv macro-F1: {best_val_adv_macro_f1:.6f}")
    print(f"Best TEST adv macro-F1: {best_test_adv_macro_f1:.6f}")
    print(f"Final clean TEST macro-F1: {final_clean['macro_f1']:.6f}")
    print(f"Final adv TEST macro-F1: {final_adv['macro_f1']:.6f}")
    print(f"Final adv fool-to-benign: {final_adv['fool_to_benign']:.6f}")
    print(f"Outputs saved to: {save_folder}")


if __name__ == "__main__":
    main()
