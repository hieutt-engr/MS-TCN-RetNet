"""
Free Adversarial Training (Free-AT) for MS-TCN-RetNet on preprocessed CAN-ML .pt datasets.

This script is based on the standard MS-TCN-RetNet training script, but trains with
payload-only Free Adversarial Training and logs clean/adversarial losses and metrics.

Expected project structure
--------------------------
project/
    train_ms_tcn_retnet_free_at.py
    dataloader.py
    networks/
        ms_tcn_retnet_torch.py

Expected data directory
-----------------------
data_dir/
    train.pt
    val.pt
    test.pt
    metadata.json   (optional)

Default payload-only threat model
---------------------------------
Input feature order:
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

Only payload indices 2..9 are perturbed:
    ||delta_payload||_inf <= epsilon
    payload is clipped to [0, 1]
    non-payload features are restored exactly.

Recommended run
---------------
python train_ms_tcn_retnet_free_at.py \
  --data_dir ./data/2017-subaru-forester/preprocessed \
  --save_folder ./save/ms_tcn_retnet_free_at \
  --model_name ms_tcn_retnet_free_at \
  --device cuda --gpu_id 0 \
  --epochs 20 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --epsilon 4/255 \
  --free_replays 4 \
  --free_step_size 4/255 \
  --adv_eval_freq 1 \
  --attack_steps_eval 20 \
  --use_class_weights \
  --amp \
  --pin_memory \
  --print_freq 100 \
  --epoch_save_every 5
"""

from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import WeightedRandomSampler

from dataloader import build_all_dataloaders
from networks.ms_tcn_retnet_torch import MSTCNRetNet
from adv_ndcl_losses import AdvNDCLTotalLoss, parse_focus_classes


# ============================================================
# Parsing and utility functions
# ============================================================

def parse_float_or_fraction(value: str | float | int) -> float:
    s = str(value).strip()
    if "/" in s:
        a, b = s.split("/", 1)
        return float(a) / float(b)
    return float(s)


def parse_indices(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


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


def format_confusion_matrix(cm: np.ndarray) -> str:
    return np.array2string(cm, separator=", ", max_line_width=220)


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


def build_weighted_sampler(y: torch.Tensor, num_classes: int) -> WeightedRandomSampler:
    class_weights = compute_class_weights(y, num_classes)
    sample_weights = class_weights[y.cpu()].double()
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def forward_logits_features(model: nn.Module, x: torch.Tensor):
    """Return pooled embedding and logits for MSTCNRetNet without changing model code.

    MSTCNRetNet.forward() only returns logits, but the AdvNDCL loss needs the
    representation before the classifier. This helper mirrors MSTCNRetNet.forward().
    """
    seq = model.forward_features(x)
    if model.pool == "mean":
        feat = seq.mean(dim=1)
    elif model.pool == "max":
        feat = seq.max(dim=1).values
    else:
        raise ValueError(f"Unsupported pool={model.pool}")
    feat = model.head_norm(feat)
    feat = model.head_drop(feat)
    logits = model.classifier(feat)
    return feat, logits


# ============================================================
# Losses
# ============================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def build_loss(loss_name: str, class_weights: torch.Tensor | None = None, focal_gamma: float = 2.0) -> nn.Module:
    loss_name = loss_name.lower()
    if loss_name == "ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    if loss_name == "focal":
        return FocalLoss(alpha=class_weights, gamma=focal_gamma)
    raise ValueError(f"Unsupported loss_name={loss_name}")


# ============================================================
# Payload-only adversarial utilities
# ============================================================

def make_feature_mask(x: torch.Tensor, indices: List[int]) -> torch.Tensor:
    mask = torch.zeros_like(x)
    mask[:, :, indices] = 1.0
    return mask


def project_payload_linf(x_adv: torch.Tensor, x_orig: torch.Tensor, epsilon: float, payload_indices: List[int]) -> torch.Tensor:
    """Project X_adv into the payload-only L_inf epsilon-ball around X_orig."""
    x_proj = x_orig.clone()
    adv_payload = x_adv[:, :, payload_indices]
    orig_payload = x_orig[:, :, payload_indices]
    adv_payload = torch.max(torch.min(adv_payload, orig_payload + epsilon), orig_payload - epsilon)
    adv_payload = adv_payload.clamp(0.0, 1.0)
    x_proj[:, :, payload_indices] = adv_payload
    return x_proj


def delta_from_xadv(x_adv: torch.Tensor, x_orig: torch.Tensor, epsilon: float, payload_indices: List[int]) -> torch.Tensor:
    x_proj = project_payload_linf(x_adv, x_orig, epsilon, payload_indices)
    return (x_proj - x_orig).detach()


def init_free_delta(x: torch.Tensor, epsilon: float, payload_indices: List[int], random_start: bool = True) -> torch.Tensor:
    if random_start:
        mask = make_feature_mask(x, payload_indices)
        delta = torch.empty_like(x).uniform_(-epsilon, epsilon) * mask
        delta = delta_from_xadv(x + delta, x, epsilon, payload_indices)
    else:
        delta = torch.zeros_like(x)
    return delta.detach()


def fgsm_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, criterion: nn.Module, epsilon: float, payload_indices: List[int]) -> torch.Tensor:
    mask = make_feature_mask(x, payload_indices)
    x_adv = x.detach().clone().requires_grad_(True)
    logits = model(x_adv)
    loss = criterion(logits, y)
    grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
    x_adv = x_adv.detach() + epsilon * grad.sign() * mask
    return project_payload_linf(x_adv, x, epsilon, payload_indices).detach()


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
    epsilon: float,
    alpha: float,
    steps: int,
    payload_indices: List[int],
    random_start: bool = True,
) -> torch.Tensor:
    mask = make_feature_mask(x, payload_indices)
    if random_start:
        delta = torch.empty_like(x).uniform_(-epsilon, epsilon) * mask
        delta = delta_from_xadv(x + delta, x, epsilon, payload_indices)
    else:
        delta = torch.zeros_like(x)

    x_adv = project_payload_linf(x + delta, x, epsilon, payload_indices).detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = criterion(logits, y)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        x_adv = x_adv.detach() + alpha * grad.sign() * mask
        x_adv = project_payload_linf(x_adv, x, epsilon, payload_indices).detach()

    return x_adv.detach()


# ============================================================
# Evaluation
# ============================================================

def compute_metrics(targets: np.ndarray, preds: np.ndarray, loss: float) -> Dict:
    acc = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(targets, preds, average="weighted", zero_division=0)
    cm = confusion_matrix(targets, preds)

    attack_mask = targets != 0
    if attack_mask.sum() > 0:
        fool_to_benign = float(np.mean(preds[attack_mask] == 0))
        attack_only_acc = float(np.mean(preds[attack_mask] == targets[attack_mask]))
    else:
        fool_to_benign = 0.0
        attack_only_acc = 0.0

    return {
        "loss": float(loss),
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "fool_to_benign": fool_to_benign,
        "attack_only_acc": attack_only_acc,
        "preds": preds,
        "targets": targets,
        "cm": cm,
    }


@torch.no_grad()
def evaluate_clean(model: nn.Module, loader, criterion, device, amp: bool = False) -> Dict:
    model.eval()
    all_logits, all_targets = [], []
    running_loss, n_samples = 0.0, 0
    amp_ctx_factory = get_amp_context(device, enabled=amp)

    for batch in loader:
        x, y = batch[:2]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with amp_ctx_factory():
            logits = model(x)
            loss = criterion(logits, y)
        bs = x.size(0)
        running_loss += loss.item() * bs
        n_samples += bs
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

    logits_all = torch.cat(all_logits, dim=0)
    targets_all = torch.cat(all_targets, dim=0)
    preds = logits_all.argmax(dim=1).numpy()
    targets = targets_all.numpy()
    return compute_metrics(targets, preds, running_loss / max(n_samples, 1))


def evaluate_adversarial(model, loader, criterion, device, epsilon: float, alpha: float, steps: int, payload_indices: List[int], attack_name: str = "pgd") -> Dict:
    model.eval()
    all_logits, all_targets = [], []
    running_loss, n_samples = 0.0, 0

    for batch in loader:
        x, y = batch[:2]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.enable_grad():
            if attack_name.lower() == "fgsm":
                x_adv = fgsm_attack(model, x, y, criterion, epsilon, payload_indices)
            else:
                x_adv = pgd_attack(model, x, y, criterion, epsilon, alpha, steps, payload_indices, random_start=True)
        with torch.no_grad():
            logits = model(x_adv)
            loss = criterion(logits, y)
        bs = x.size(0)
        running_loss += loss.item() * bs
        n_samples += bs
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

    logits_all = torch.cat(all_logits, dim=0)
    targets_all = torch.cat(all_targets, dim=0)
    preds = logits_all.argmax(dim=1).numpy()
    targets = targets_all.numpy()
    return compute_metrics(targets, preds, running_loss / max(n_samples, 1))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_folder", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="ms_tcn_retnet_free_at")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--gpu_id", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--use_weighted_sampler", action="store_true")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--epoch_save_every", type=str, default=None, help="Save every N epochs. Use None/null/0/-1 to disable.")

    # Free-AT parameters
    parser.add_argument("--free_replays", type=int, default=4)
    parser.add_argument("--epsilon", type=parse_float_or_fraction, default=parse_float_or_fraction("4/255"))
    parser.add_argument("--free_step_size", type=parse_float_or_fraction, default=None)
    parser.add_argument("--payload_indices", type=parse_indices, default=parse_indices("2,3,4,5,6,7,8,9"))
    parser.add_argument("--free_random_start", action="store_true", default=True)
    parser.add_argument("--no_free_random_start", action="store_false", dest="free_random_start")
    parser.add_argument("--lambda_adv", type=float, default=1.0)
    parser.add_argument("--lambda_clean", type=float, default=0.0, help="Optional clean CE term in addition to Free-AT loss")
    parser.add_argument("--adv_on_attack_only", action="store_true")

    # Adversarial validation
    parser.add_argument("--adv_eval_freq", type=int, default=1)
    parser.add_argument("--eval_attack_name", type=str, default="pgd", choices=["pgd", "fgsm"])
    parser.add_argument("--attack_steps_eval", type=int, default=20)
    parser.add_argument("--attack_alpha_eval", type=parse_float_or_fraction, default=None)

    parser.add_argument("--scheduler", type=str, default="cosine_restart", choices=["none", "cosine", "cosine_restart"])
    parser.add_argument("--t0", type=int, default=10)
    parser.add_argument("--t_mult", type=int, default=2)

    parser.add_argument("--loss_name", type=str, default="ce", choices=["ce", "focal", "reweighted"])
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    # Adv-NDCL loss parameters. Clean CE is always included.
    parser.add_argument("--adv_ce_weight", type=float, default=1.0)
    parser.add_argument("--kl_weight", type=float, default=0.1)
    parser.add_argument("--kl_temperature", type=float, default=1.0)
    parser.add_argument("--kl_symmetric", action="store_true")
    parser.add_argument("--feature_consistency_weight", type=float, default=0.0)
    parser.add_argument("--feature_consistency_mode", type=str, default="cosine", choices=["cosine", "mse"])
    parser.add_argument("--adv_ndcl_weight", type=float, default=0.003)
    parser.add_argument("--adv_ndcl_temperature", type=float, default=0.1)
    parser.add_argument("--adv_ndcl_margin", type=float, default=0.2)
    parser.add_argument("--adv_ndcl_hard_ratio", type=float, default=0.10)
    parser.add_argument("--adv_ndcl_min_hard_negatives", type=int, default=1)
    parser.add_argument("--adv_ndcl_anchor_mode", type=str, default="attack_only", choices=["all", "attack_only", "normal_only", "focus_only"])
    parser.add_argument("--adv_ndcl_negative_mode", type=str, default="normal_for_attack", choices=["all", "normal_for_attack", "attack_for_normal", "binary_boundary"])
    parser.add_argument("--normal_class", type=int, default=0)
    parser.add_argument("--focus_classes", type=str, default=None)
    parser.add_argument("--adv_ndcl_warmup_epochs", type=int, default=5, help="Epochs using CE+advCE+KL before enabling AdvNDCL.")
    parser.add_argument("--adv_ndcl_ramp_epochs", type=int, default=10, help="Linear ramp length for AdvNDCL/consistency weights.")

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

    if args.epoch_save_every is not None:
        s = str(args.epoch_save_every).strip().lower()
        if s in {"none", "null", "false", "no", "0", "-1"}:
            args.epoch_save_every = None
        else:
            args.epoch_save_every = int(s)
            if args.epoch_save_every <= 0:
                args.epoch_save_every = None

    if args.free_step_size is None:
        args.free_step_size = args.epsilon
    if args.attack_alpha_eval is None:
        args.attack_alpha_eval = args.epsilon / 4.0 if args.attack_steps_eval > 1 else args.epsilon
    if args.free_replays <= 0:
        raise ValueError("--free_replays must be positive.")

    seed_everything(args.seed)

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

    append_log(log_file, "=" * 120)
    append_log(log_file, "MS-TCN-RetNet Free-AT with Adv-NDCL Loss")
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
    input_shape = bundle["input_shape"]
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
        append_log(log_file, f"Class weights: {class_weights.detach().cpu().tolist()}")

    # CE/focal criterion is used for adversarial evaluation and for crafting PGD/FGSM attacks.
    # Training uses AdvNDCLTotalLoss, which contains clean CE, adversarial CE,
    # clean-adv KL consistency, optional feature consistency, and AdvNDCL.
    eval_criterion = build_loss("ce" if args.loss_name == "reweighted" else args.loss_name, class_weights=class_weights, focal_gamma=args.focal_gamma)
    criterion = eval_criterion
    train_criterion = AdvNDCLTotalLoss(
        num_classes=num_classes,
        ce_type=args.loss_name,
        class_weights=class_weights,
        focal_gamma=args.focal_gamma,
        adv_ce_weight=args.adv_ce_weight,
        kl_weight=args.kl_weight,
        kl_temperature=args.kl_temperature,
        kl_symmetric=args.kl_symmetric,
        feature_consistency_weight=args.feature_consistency_weight,
        feature_consistency_mode=args.feature_consistency_mode,
        adv_ndcl_weight=args.adv_ndcl_weight,
        adv_ndcl_temperature=args.adv_ndcl_temperature,
        adv_ndcl_margin=args.adv_ndcl_margin,
        adv_ndcl_hard_ratio=args.adv_ndcl_hard_ratio,
        adv_ndcl_min_hard_negatives=args.adv_ndcl_min_hard_negatives,
        adv_ndcl_anchor_mode=args.adv_ndcl_anchor_mode,
        adv_ndcl_negative_mode=args.adv_ndcl_negative_mode,
        normal_class=args.normal_class,
        focus_classes=parse_focus_classes(args.focus_classes),
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "cosine_restart":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.t0, T_mult=args.t_mult)
    else:
        scheduler = None

    scaler = make_grad_scaler(device, enabled=args.amp)
    amp_ctx_factory = get_amp_context(device, enabled=args.amp)

    history = {
        "train_total_loss": [], "train_adv_loss": [], "train_clean_loss": [], "train_replay_loss": [],
        "train_kl_loss": [], "train_feat_cons_loss": [], "train_adv_ndcl_loss": [],
        "train_adv_ndcl_scale": [],
        "val_clean_loss": [], "val_clean_acc": [], "val_clean_macro_f1": [], "val_clean_weighted_f1": [],
        "val_adv_loss": [], "val_adv_acc": [], "val_adv_macro_f1": [], "val_adv_weighted_f1": [], "val_adv_fool_to_benign": [],
        "test_clean_loss": [], "test_clean_acc": [], "test_clean_macro_f1": [], "test_clean_weighted_f1": [],
        "test_adv_loss": [], "test_adv_acc": [], "test_adv_macro_f1": [], "test_adv_weighted_f1": [], "test_adv_fool_to_benign": [],
        "lr": [],
    }

    best_val_clean_macro_f1 = -1.0
    best_val_adv_macro_f1 = -1.0
    best_test_adv_macro_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss_sum = adv_loss_sum = clean_loss_sum = replay_loss_sum = 0.0
        kl_loss_sum = feat_cons_loss_sum = adv_ndcl_loss_sum = 0.0
        n_samples = n_replay_updates = 0
        num_train_batches = len(train_loader)

        # CE and adv CE are active from the beginning. AdvNDCL and consistency
        # terms are warmed up/ramped to avoid destabilizing early representations.
        if epoch <= args.adv_ndcl_warmup_epochs:
            aux_scale = 0.0
        elif args.adv_ndcl_ramp_epochs <= 0:
            aux_scale = 1.0
        else:
            aux_scale = min(1.0, (epoch - args.adv_ndcl_warmup_epochs) / float(args.adv_ndcl_ramp_epochs))

        for batch_idx, batch in enumerate(train_loader, start=1):
            X, y = batch[:2]
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if args.adv_on_attack_only:
                attack_mask = y != 0
                X_adv_base = X[attack_mask] if attack_mask.sum() > 0 else X
                y_adv_base = y[attack_mask] if attack_mask.sum() > 0 else y
            else:
                X_adv_base, y_adv_base = X, y

            delta = init_free_delta(
                X_adv_base,
                epsilon=args.epsilon,
                payload_indices=args.payload_indices,
                random_start=args.free_random_start,
            )

            batch_total = batch_adv = batch_clean = 0.0
            batch_kl = batch_feat_cons = batch_adv_ndcl = 0.0

            for _ in range(args.free_replays):
                optimizer.zero_grad(set_to_none=True)

                X_adv = project_payload_linf(X_adv_base + delta, X_adv_base, args.epsilon, args.payload_indices).detach()
                X_adv.requires_grad_(True)

                with amp_ctx_factory():
                    if args.adv_on_attack_only:
                        # Use all clean samples for clean CE, but adversarial terms only on attacks.
                        feat_clean_all, logits_clean_all = forward_logits_features(model, X)
                        clean_full = train_criterion.ce_loss(logits_clean_all, y)

                        feat_clean_sub, logits_clean_sub = forward_logits_features(model, X_adv_base)
                        feat_adv, logits_adv = forward_logits_features(model, X_adv)
                        sub_out = train_criterion(
                            logits_clean=logits_clean_sub,
                            feat_clean=feat_clean_sub,
                            targets=y_adv_base,
                            logits_adv=logits_adv,
                            feat_adv=feat_adv,
                            adv_scale=args.lambda_adv,
                            kl_scale=aux_scale,
                            feature_consistency_scale=aux_scale,
                            adv_ndcl_scale=aux_scale,
                        )
                        clean_loss = clean_full
                        adv_loss = sub_out.ce_adv
                        loss = clean_full + (sub_out.total - sub_out.ce_clean)
                        kl_loss = sub_out.kl
                        feat_cons_loss = sub_out.feature_consistency
                        adv_ndcl_loss = sub_out.adv_ndcl
                    else:
                        feat_clean, logits_clean = forward_logits_features(model, X)
                        feat_adv, logits_adv = forward_logits_features(model, X_adv)
                        out = train_criterion(
                            logits_clean=logits_clean,
                            feat_clean=feat_clean,
                            targets=y,
                            logits_adv=logits_adv,
                            feat_adv=feat_adv,
                            adv_scale=args.lambda_adv,
                            kl_scale=aux_scale,
                            feature_consistency_scale=aux_scale,
                            adv_ndcl_scale=aux_scale,
                        )
                        loss = out.total
                        clean_loss = out.ce_clean
                        adv_loss = out.ce_adv
                        kl_loss = out.kl
                        feat_cons_loss = out.feature_consistency
                        adv_ndcl_loss = out.adv_ndcl

                scaler.scale(loss).backward()

                if X_adv.grad is not None:
                    grad = X_adv.grad.detach()
                    mask = make_feature_mask(X_adv, args.payload_indices)
                    X_adv_next = X_adv.detach() + args.free_step_size * grad.sign() * mask
                    delta = delta_from_xadv(X_adv_next, X_adv_base, args.epsilon, args.payload_indices)
                else:
                    delta = delta.detach()

                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

                scaler.step(optimizer)
                scaler.update()

                bs = X.size(0)
                total_loss_sum += loss.item() * bs
                adv_loss_sum += adv_loss.item() * bs
                clean_loss_sum += clean_loss.item() * bs
                kl_loss_sum += kl_loss.item() * bs
                feat_cons_loss_sum += feat_cons_loss.item() * bs
                adv_ndcl_loss_sum += adv_ndcl_loss.item() * bs
                replay_loss_sum += loss.item()
                n_replay_updates += 1

                batch_total += loss.item()
                batch_adv += adv_loss.item()
                batch_clean += clean_loss.item()
                batch_kl += kl_loss.item()
                batch_feat_cons += feat_cons_loss.item()
                batch_adv_ndcl += adv_ndcl_loss.item()

            n_samples += X.size(0)

            if args.print_freq > 0 and (batch_idx % args.print_freq == 0 or batch_idx == num_train_batches):
                iter_msg = (
                    f"Epoch [{epoch:03d}/{args.epochs:03d}] Iter [{batch_idx:04d}/{num_train_batches:04d}] | "
                    f"lr={optimizer.param_groups[0]['lr']:.6e} | free_replays={args.free_replays} | "
                    f"loss={batch_total / args.free_replays:.6f} | "
                    f"cleanCE={batch_clean / args.free_replays:.6f} | advCE={batch_adv / args.free_replays:.6f} | "
                    f"KL={batch_kl / args.free_replays:.6f} | Feat={batch_feat_cons / args.free_replays:.6f} | "
                    f"AdvNDCL={batch_adv_ndcl / args.free_replays:.6f} | aux_scale={aux_scale:.3f}"
                )
                print(iter_msg)
                append_log(log_file, iter_msg)

        denom = max(n_samples * args.free_replays, 1)
        train_total_loss = total_loss_sum / denom
        train_adv_loss = adv_loss_sum / denom
        train_clean_loss = clean_loss_sum / denom
        train_kl_loss = kl_loss_sum / denom
        train_feat_cons_loss = feat_cons_loss_sum / denom
        train_adv_ndcl_loss = adv_ndcl_loss_sum / denom
        train_replay_loss = replay_loss_sum / max(n_replay_updates, 1)

        if scheduler is not None:
            if args.scheduler == "cosine_restart":
                scheduler.step(epoch - 1 + 1.0)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        val_clean = evaluate_clean(model, val_loader, criterion, device, amp=args.amp)
        test_clean = evaluate_clean(model, test_loader, criterion, device, amp=args.amp)

        do_adv_eval = (args.adv_eval_freq > 0 and epoch % args.adv_eval_freq == 0) or epoch == args.epochs
        if do_adv_eval:
            val_adv = evaluate_adversarial(model, val_loader, criterion, device, args.epsilon, args.attack_alpha_eval, args.attack_steps_eval, args.payload_indices, args.eval_attack_name)
            test_adv = evaluate_adversarial(model, test_loader, criterion, device, args.epsilon, args.attack_alpha_eval, args.attack_steps_eval, args.payload_indices, args.eval_attack_name)
        else:
            val_adv = {"loss": float("nan"), "acc": float("nan"), "macro_f1": float("nan"), "weighted_f1": float("nan"), "fool_to_benign": float("nan"), "cm": None}
            test_adv = {"loss": float("nan"), "acc": float("nan"), "macro_f1": float("nan"), "weighted_f1": float("nan"), "fool_to_benign": float("nan"), "cm": None}

        history["train_total_loss"].append(train_total_loss)
        history["train_adv_loss"].append(train_adv_loss)
        history["train_clean_loss"].append(train_clean_loss)
        history["train_kl_loss"].append(train_kl_loss)
        history["train_feat_cons_loss"].append(train_feat_cons_loss)
        history["train_adv_ndcl_loss"].append(train_adv_ndcl_loss)
        history["train_adv_ndcl_scale"].append(aux_scale)
        history["train_replay_loss"].append(train_replay_loss)
        history["val_clean_loss"].append(val_clean["loss"])
        history["val_clean_acc"].append(val_clean["acc"])
        history["val_clean_macro_f1"].append(val_clean["macro_f1"])
        history["val_clean_weighted_f1"].append(val_clean["weighted_f1"])
        history["val_adv_loss"].append(val_adv["loss"])
        history["val_adv_acc"].append(val_adv["acc"])
        history["val_adv_macro_f1"].append(val_adv["macro_f1"])
        history["val_adv_weighted_f1"].append(val_adv["weighted_f1"])
        history["val_adv_fool_to_benign"].append(val_adv["fool_to_benign"])
        history["test_clean_loss"].append(test_clean["loss"])
        history["test_clean_acc"].append(test_clean["acc"])
        history["test_clean_macro_f1"].append(test_clean["macro_f1"])
        history["test_clean_weighted_f1"].append(test_clean["weighted_f1"])
        history["test_adv_loss"].append(test_adv["loss"])
        history["test_adv_acc"].append(test_adv["acc"])
        history["test_adv_macro_f1"].append(test_adv["macro_f1"])
        history["test_adv_weighted_f1"].append(test_adv["weighted_f1"])
        history["test_adv_fool_to_benign"].append(test_adv["fool_to_benign"])
        history["lr"].append(current_lr)

        msg = (
            f"Epoch [{epoch:03d}/{args.epochs:03d}] | lr={current_lr:.6e} | "
            f"train_total={train_total_loss:.6f}, cleanCE={train_clean_loss:.6f}, advCE={train_adv_loss:.6f}, "
            f"KL={train_kl_loss:.6f}, Feat={train_feat_cons_loss:.6f}, AdvNDCL={train_adv_ndcl_loss:.6f}, "
            f"aux_scale={aux_scale:.3f}, replay_loss={train_replay_loss:.6f} | "
            f"VAL clean_loss={val_clean['loss']:.6f}, clean_acc={val_clean['acc']:.4f}, clean_macroF1={val_clean['macro_f1']:.4f}, "
            f"adv_loss={val_adv['loss']:.6f}, adv_acc={val_adv['acc']:.4f}, adv_macroF1={val_adv['macro_f1']:.4f}, adv_F2B={val_adv['fool_to_benign']:.4f} | "
            f"TEST clean_loss={test_clean['loss']:.6f}, clean_acc={test_clean['acc']:.4f}, clean_macroF1={test_clean['macro_f1']:.4f}, "
            f"adv_loss={test_adv['loss']:.6f}, adv_acc={test_adv['acc']:.4f}, adv_macroF1={test_adv['macro_f1']:.4f}, adv_F2B={test_adv['fool_to_benign']:.4f}"
        )
        print(msg)
        append_log(log_file, msg)

        val_clean_cm_str = format_confusion_matrix(val_clean["cm"])
        test_clean_cm_str = format_confusion_matrix(test_clean["cm"])
        print("Val Clean Confusion Matrix:")
        print(val_clean_cm_str)
        print("Test Clean Confusion Matrix:")
        print(test_clean_cm_str)
        append_log(log_file, "Val Clean Confusion Matrix:\n" + val_clean_cm_str)
        append_log(log_file, "Test Clean Confusion Matrix:\n" + test_clean_cm_str)

        if do_adv_eval:
            val_adv_cm_str = format_confusion_matrix(val_adv["cm"])
            test_adv_cm_str = format_confusion_matrix(test_adv["cm"])
            print("Val Adversarial Confusion Matrix:")
            print(val_adv_cm_str)
            print("Test Adversarial Confusion Matrix:")
            print(test_adv_cm_str)
            append_log(log_file, "Val Adversarial Confusion Matrix:\n" + val_adv_cm_str)
            append_log(log_file, "Test Adversarial Confusion Matrix:\n" + test_adv_cm_str)
        else:
            skip_msg = f"Adversarial evaluation skipped at epoch {epoch}; adv_eval_freq={args.adv_eval_freq}."
            print(skip_msg)
            append_log(log_file, skip_msg)

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

        if args.epoch_save_every is not None and epoch % args.epoch_save_every == 0:
            epoch_path = epoch_dir / f"{args.model_name}_epoch_{epoch:03d}.pth"
            torch.save(ckpt, epoch_path)
            append_log(log_file, f"Saved epoch checkpoint: {epoch_path}")

        torch.save(ckpt, save_folder / f"{args.model_name}_last.pth")

        if val_clean["macro_f1"] > best_val_clean_macro_f1:
            best_val_clean_macro_f1 = val_clean["macro_f1"]
            torch.save(ckpt, save_folder / f"{args.model_name}_best_clean_macro_f1.pth")
            np.save(save_folder / "best_val_clean_confusion_matrix.npy", val_clean["cm"])

        if do_adv_eval and val_adv["macro_f1"] > best_val_adv_macro_f1:
            best_val_adv_macro_f1 = val_adv["macro_f1"]
            torch.save(ckpt, save_folder / f"{args.model_name}_best_adv_macro_f1.pth")
            np.save(save_folder / "best_val_adv_confusion_matrix.npy", val_adv["cm"])

        if do_adv_eval and test_adv["macro_f1"] > best_test_adv_macro_f1:
            best_test_adv_macro_f1 = test_adv["macro_f1"]
            torch.save(ckpt, save_folder / f"{args.model_name}_best_test_adv_macro_f1.pth")
            np.save(save_folder / "best_test_adv_confusion_matrix.npy", test_adv["cm"])
            report = classification_report(test_adv["targets"], test_adv["preds"], target_names=class_names, zero_division=0, digits=4)
            with open(save_folder / "best_test_adv_classification_report.txt", "w", encoding="utf-8") as f:
                f.write(report)

        save_json(history_file, history)

    final_clean = evaluate_clean(model, test_loader, criterion, device, amp=args.amp)
    final_adv = evaluate_adversarial(model, test_loader, criterion, device, args.epsilon, args.attack_alpha_eval, args.attack_steps_eval, args.payload_indices, args.eval_attack_name)

    np.save(save_folder / "final_clean_confusion_matrix.npy", final_clean["cm"])
    np.save(save_folder / "final_adv_confusion_matrix.npy", final_adv["cm"])

    final_clean_report = classification_report(final_clean["targets"], final_clean["preds"], target_names=class_names, zero_division=0, digits=4)
    final_adv_report = classification_report(final_adv["targets"], final_adv["preds"], target_names=class_names, zero_division=0, digits=4)

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
