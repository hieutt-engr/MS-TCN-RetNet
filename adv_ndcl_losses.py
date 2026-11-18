"""
Adversarial NDCL-inspired losses for IVN/CAN intrusion detection.

This file only defines the loss side. The training script should create
payload-constrained adversarial examples (e.g., FGSM/PGD/Free-AT) and pass both
clean and adversarial forward outputs into AdvNDCLTotalLoss.

Expected tensors:
    logits_clean: (B, num_classes)
    feat_clean  : (B, D)
    logits_adv  : (B, num_classes) or None
    feat_adv    : (B, D) or None
    targets     : (B,)

Recommended first configuration:
    criterion = AdvNDCLTotalLoss(
        num_classes=10,
        class_weights=class_weights,        # optional
        ce_type="ce",
        adv_ce_weight=1.0,
        kl_weight=0.1,
        feature_consistency_weight=0.0,
        adv_ndcl_weight=0.003,
        adv_ndcl_temperature=0.1,
        adv_ndcl_margin=0.2,
        adv_ndcl_hard_ratio=0.10,
        adv_ndcl_anchor_mode="attack_only",
        adv_ndcl_negative_mode="normal_for_attack",
    )

Total objective:
    L = L_CE(clean)
        + lambda_adv * L_CE(adv)
        + gamma * KL(p_clean || p_adv)
        + eta * L_feat_consistency(z_clean, z_adv)
        + alpha * L_AdvNDCL(z_clean, z_adv, y)

The AdvNDCL term uses adversarial samples from other classes as hard negatives.
For each anchor, it selects top-k closest adversarial negatives by cosine
similarity and penalizes them if they are above a similarity margin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Set

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Basic classification losses
# -----------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        targets = targets.long()
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


class ReweightedCrossEntropyLoss(nn.Module):
    """
    Batch-wise class-balanced CE inspired by NDCL's reweighted CE.
    Within each class, harder samples receive larger softmax weights.
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        targets = targets.long()
        ce_each = F.cross_entropy(logits, targets, reduction="none")

        total = logits.new_tensor(0.0)
        valid_classes = 0
        for cls in torch.unique(targets):
            mask = targets == cls
            if mask.sum() == 0:
                continue
            cls_losses = ce_each[mask]
            cls_weights = torch.softmax(cls_losses.detach(), dim=0)
            total = total + torch.sum(cls_weights * cls_losses)
            valid_classes += 1

        if valid_classes == 0:
            return logits.new_tensor(0.0)
        return total / valid_classes


def build_ce_loss(
    ce_type: str = "ce",
    class_weights: Optional[torch.Tensor] = None,
    focal_gamma: float = 2.0,
) -> nn.Module:
    ce_type = ce_type.lower().strip()
    if ce_type == "ce":
        return nn.CrossEntropyLoss(weight=class_weights.float() if class_weights is not None else None)
    if ce_type == "focal":
        return FocalLoss(alpha=class_weights, gamma=focal_gamma)
    if ce_type == "reweighted":
        return ReweightedCrossEntropyLoss()
    raise ValueError(f"Unsupported ce_type={ce_type}. Use ce | focal | reweighted.")


# -----------------------------------------------------------------------------
# Clean-adversarial consistency losses
# -----------------------------------------------------------------------------
class CleanAdvKLLoss(nn.Module):
    """
    Prediction consistency between clean and adversarial logits.

    Default is one-way KL with clean prediction detached:
        KL(p_clean.detach() || p_adv)
    This makes adversarial predictions match clean predictions without pulling the
    clean branch toward unstable adversarial predictions.

    Set symmetric=True to also include KL(p_adv.detach() || p_clean).
    """

    def __init__(self, temperature: float = 1.0, symmetric: bool = False, detach_clean: bool = True):
        super().__init__()
        self.temperature = float(temperature)
        self.symmetric = bool(symmetric)
        self.detach_clean = bool(detach_clean)

    def forward(self, logits_clean: torch.Tensor, logits_adv: torch.Tensor) -> torch.Tensor:
        t = max(self.temperature, 1e-8)
        lc = logits_clean.float() / t
        la = logits_adv.float() / t

        p_clean = F.softmax(lc, dim=1)
        log_p_adv = F.log_softmax(la, dim=1)
        target_clean = p_clean.detach() if self.detach_clean else p_clean
        loss = F.kl_div(log_p_adv, target_clean, reduction="batchmean") * (t * t)

        if self.symmetric:
            p_adv = F.softmax(la, dim=1).detach()
            log_p_clean = F.log_softmax(lc, dim=1)
            loss = 0.5 * (loss + F.kl_div(log_p_clean, p_adv, reduction="batchmean") * (t * t))
        return loss


class FeatureConsistencyLoss(nn.Module):
    """
    Pull clean and adversarial embeddings of the same sample together.

    mode='cosine': mean(1 - cos(z_clean, z_adv))
    mode='mse'   : mean squared error between L2-normalized embeddings
    """

    def __init__(self, mode: str = "cosine"):
        super().__init__()
        self.mode = mode.lower().strip()

    def forward(self, feat_clean: torch.Tensor, feat_adv: torch.Tensor) -> torch.Tensor:
        zc = F.normalize(feat_clean.float(), p=2, dim=1)
        za = F.normalize(feat_adv.float(), p=2, dim=1)
        if self.mode == "cosine":
            return (1.0 - torch.sum(zc * za, dim=1)).mean()
        if self.mode == "mse":
            return F.mse_loss(zc, za)
        raise ValueError(f"Unsupported feature consistency mode={self.mode}")


# -----------------------------------------------------------------------------
# Adversarial Negative-Dominant Hard Negative Loss
# -----------------------------------------------------------------------------
class AdversarialNegativeDominantLoss(nn.Module):
    """
    NDCL-inspired adversarial hard-negative loss.

    For each anchor z_i, build a negative pool from adversarial embeddings
    z_adv_j with y_j != y_i. Select top-k closest negatives by cosine similarity,
    then penalize these similarities if they exceed a margin:

        L_i = mean_{j in hard_neg(i)} softplus((sim(z_i, z_adv_j) - margin) / tau)

    This is a practical margin version of negative-dominant learning: it does not
    need many positive samples and focuses optimization on the nearest wrong-class
    adversarial samples.

    anchor_mode:
        all         : all samples are anchors
        attack_only : only y != normal_class anchors
        normal_only : only y == normal_class anchors
        focus_only  : only classes in focus_classes are anchors

    negative_mode:
        all               : any different class can be a negative
        normal_for_attack : for attack anchors, use normal adversarial samples if available
        attack_for_normal : for normal anchors, use attack adversarial samples if available
        binary_boundary   : attack anchors use normal negatives; normal anchors use attack negatives
    """

    def __init__(
        self,
        temperature: float = 0.1,
        margin: float = 0.2,
        hard_ratio: float = 0.10,
        min_hard_negatives: int = 1,
        anchor_mode: str = "attack_only",
        negative_mode: str = "normal_for_attack",
        normal_class: int = 0,
        focus_classes: Optional[Sequence[int]] = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        if not (0.0 < hard_ratio <= 1.0):
            raise ValueError("hard_ratio must be in (0, 1].")
        self.temperature = float(temperature)
        self.margin = float(margin)
        self.hard_ratio = float(hard_ratio)
        self.min_hard_negatives = int(min_hard_negatives)
        self.anchor_mode = anchor_mode.lower().strip()
        self.negative_mode = negative_mode.lower().strip()
        self.normal_class = int(normal_class)
        self.focus_classes: Optional[Set[int]] = None if focus_classes is None else {int(c) for c in focus_classes}
        self.eps = float(eps)

    def _anchor_allowed(self, cls: int) -> bool:
        if self.anchor_mode == "all":
            return True
        if self.anchor_mode == "attack_only":
            return cls != self.normal_class
        if self.anchor_mode == "normal_only":
            return cls == self.normal_class
        if self.anchor_mode == "focus_only":
            return self.focus_classes is not None and cls in self.focus_classes
        raise ValueError(f"Unsupported anchor_mode={self.anchor_mode}")

    def _negative_mask(self, targets: torch.Tensor, anchor_index: int) -> torch.Tensor:
        cls_i = int(targets[anchor_index].item())
        base = targets != targets[anchor_index]

        if self.negative_mode == "all":
            return base

        normal_mask = targets == self.normal_class
        attack_mask = targets != self.normal_class

        if self.negative_mode == "normal_for_attack":
            if cls_i != self.normal_class and normal_mask.any():
                return normal_mask
            return base

        if self.negative_mode == "attack_for_normal":
            if cls_i == self.normal_class and attack_mask.any():
                return attack_mask
            return base

        if self.negative_mode == "binary_boundary":
            if cls_i == self.normal_class and attack_mask.any():
                return attack_mask
            if cls_i != self.normal_class and normal_mask.any():
                return normal_mask
            return base

        raise ValueError(f"Unsupported negative_mode={self.negative_mode}")

    def forward(self, feat_clean: torch.Tensor, feat_adv: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.long()
        B = targets.numel()
        if B <= 1:
            return feat_clean.new_tensor(0.0)

        z_anchor = F.normalize(feat_clean.float(), p=2, dim=1)
        z_adv = F.normalize(feat_adv.float(), p=2, dim=1)
        sim = torch.matmul(z_anchor, z_adv.t()).clamp(-1.0, 1.0)

        total = feat_clean.new_tensor(0.0)
        valid = 0
        tau = max(self.temperature, self.eps)

        for i in range(B):
            cls_i = int(targets[i].item())
            if not self._anchor_allowed(cls_i):
                continue

            neg_mask = self._negative_mask(targets, i)
            neg_idx = torch.where(neg_mask)[0]
            if neg_idx.numel() == 0:
                continue

            n_neg = int(neg_idx.numel())
            k = max(self.min_hard_negatives, int(round(self.hard_ratio * n_neg)))
            k = min(k, n_neg)

            hard_scores = sim[i, neg_idx]
            hard_vals = torch.topk(hard_scores, k=k, largest=True).values
            # Minimize similarity of hard adversarial negatives below margin.
            loss_i = F.softplus((hard_vals - self.margin) / tau).mean()
            total = total + loss_i
            valid += 1

        if valid == 0:
            return feat_clean.new_tensor(0.0)
        return total / valid


# -----------------------------------------------------------------------------
# Total wrapper
# -----------------------------------------------------------------------------
@dataclass
class AdvNDCLLossOutput:
    total: torch.Tensor
    ce_clean: torch.Tensor
    ce_adv: torch.Tensor
    kl: torch.Tensor
    feature_consistency: torch.Tensor
    adv_ndcl: torch.Tensor
    adv_ce_weight: float
    kl_weight: float
    feature_consistency_weight: float
    adv_ndcl_weight: float


class AdvNDCLTotalLoss(nn.Module):
    """
    Total adversarial defense loss.

    The adversarial branch is optional. If logits_adv or feat_adv is None, the
    corresponding terms are returned as zeros and the total becomes clean CE.
    This makes the class easy to use with warm-up schedules.
    """

    def __init__(
        self,
        num_classes: int,
        ce_type: str = "ce",
        class_weights: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0,
        adv_ce_weight: float = 1.0,
        kl_weight: float = 0.1,
        kl_temperature: float = 1.0,
        kl_symmetric: bool = False,
        feature_consistency_weight: float = 0.0,
        feature_consistency_mode: str = "cosine",
        adv_ndcl_weight: float = 0.003,
        adv_ndcl_temperature: float = 0.1,
        adv_ndcl_margin: float = 0.2,
        adv_ndcl_hard_ratio: float = 0.10,
        adv_ndcl_min_hard_negatives: int = 1,
        adv_ndcl_anchor_mode: str = "attack_only",
        adv_ndcl_negative_mode: str = "normal_for_attack",
        normal_class: int = 0,
        focus_classes: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.num_classes = int(num_classes)

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

        self.ce_loss = build_ce_loss(ce_type=ce_type, class_weights=self.class_weights, focal_gamma=focal_gamma)
        self.kl_loss = CleanAdvKLLoss(temperature=kl_temperature, symmetric=kl_symmetric, detach_clean=True)
        self.feature_consistency_loss = FeatureConsistencyLoss(mode=feature_consistency_mode)
        self.adv_ndcl_loss = AdversarialNegativeDominantLoss(
            temperature=adv_ndcl_temperature,
            margin=adv_ndcl_margin,
            hard_ratio=adv_ndcl_hard_ratio,
            min_hard_negatives=adv_ndcl_min_hard_negatives,
            anchor_mode=adv_ndcl_anchor_mode,
            negative_mode=adv_ndcl_negative_mode,
            normal_class=normal_class,
            focus_classes=focus_classes,
        )

        self.adv_ce_weight = float(adv_ce_weight)
        self.kl_weight = float(kl_weight)
        self.feature_consistency_weight = float(feature_consistency_weight)
        self.adv_ndcl_weight = float(adv_ndcl_weight)

    def forward(
        self,
        logits_clean: torch.Tensor,
        feat_clean: torch.Tensor,
        targets: torch.Tensor,
        logits_adv: Optional[torch.Tensor] = None,
        feat_adv: Optional[torch.Tensor] = None,
        adv_scale: float = 1.0,
        kl_scale: float = 1.0,
        feature_consistency_scale: float = 1.0,
        adv_ndcl_scale: float = 1.0,
    ) -> AdvNDCLLossOutput:
        targets = targets.long()
        ce_clean = self.ce_loss(logits_clean, targets)

        zero = ce_clean.new_tensor(0.0)
        ce_adv = zero
        kl = zero
        feat_cons = zero
        adv_ndcl = zero

        has_adv_logits = logits_adv is not None
        has_adv_features = feat_adv is not None

        if has_adv_logits:
            ce_adv = self.ce_loss(logits_adv, targets)
            kl = self.kl_loss(logits_clean, logits_adv)

        if has_adv_features:
            feat_cons = self.feature_consistency_loss(feat_clean, feat_adv)
            adv_ndcl = self.adv_ndcl_loss(feat_clean, feat_adv, targets)

        adv_w = self.adv_ce_weight * float(adv_scale)
        kl_w = self.kl_weight * float(kl_scale)
        fc_w = self.feature_consistency_weight * float(feature_consistency_scale)
        ndcl_w = self.adv_ndcl_weight * float(adv_ndcl_scale)

        total = ce_clean + adv_w * ce_adv + kl_w * kl + fc_w * feat_cons + ndcl_w * adv_ndcl

        return AdvNDCLLossOutput(
            total=total,
            ce_clean=ce_clean,
            ce_adv=ce_adv,
            kl=kl,
            feature_consistency=feat_cons,
            adv_ndcl=adv_ndcl,
            adv_ce_weight=adv_w,
            kl_weight=kl_w,
            feature_consistency_weight=fc_w,
            adv_ndcl_weight=ndcl_w,
        )


# -----------------------------------------------------------------------------
# Small parser utility for CLI compatibility
# -----------------------------------------------------------------------------
def parse_focus_classes(text: Optional[str]) -> Optional[tuple[int, ...]]:
    if text is None:
        return None
    text = str(text).strip()
    if text == "" or text.lower() in {"none", "null", "all"}:
        return None
    return tuple(int(x.strip()) for x in text.split(",") if x.strip() != "")
