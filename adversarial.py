"""
Adversarial attack module for CAN bus IDS models.

Supports FGSM and PGD attacks, plus evaluation utilities for
measuring robustness at different denoising step counts.
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_logits(model, x, **infer_kw):
    """Get logits from model, handling both dict and tensor outputs."""
    out = model(x, **infer_kw)
    if isinstance(out, dict):
        return out["logits_all"][-1]
    return out


# ──────────────────────────────────────────────────────────────
# FGSM
# ──────────────────────────────────────────────────────────────

def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    n_steps: int = 1,
    num_trajectories: int = 1,
) -> torch.Tensor:
    """
    Fast Gradient Sign Method.

    Parameters
    ----------
    model   : the IDS model (must be in eval mode)
    x       : (B, T, C) clean input
    y       : (B,) true labels
    epsilon : perturbation budget (L-inf)
    n_steps : denoising steps for the model (for looped flow)
    num_trajectories : number of trajectories (for looped flow)

    Returns
    -------
    x_adv : (B, T, C) adversarial input
    """
    x_adv = x.clone().detach().requires_grad_(True)

    logits = _get_logits(
        model, x_adv, n_steps=n_steps, num_trajectories=num_trajectories
    )
    loss = F.cross_entropy(logits, y)
    loss.backward()

    with torch.no_grad():
        perturbation = epsilon * x_adv.grad.sign()
        x_adv = (x + perturbation).clamp(0, 1)

    return x_adv.detach()


# ──────────────────────────────────────────────────────────────
# PGD
# ──────────────────────────────────────────────────────────────

def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    alpha: float = 0.01,
    num_iter: int = 10,
    n_steps: int = 1,
    num_trajectories: int = 1,
    random_start: bool = True,
) -> torch.Tensor:
    """
    Projected Gradient Descent attack.

    Parameters
    ----------
    model    : the IDS model
    x        : (B, T, C) clean input
    y        : (B,) true labels
    epsilon  : perturbation budget (L-inf)
    alpha    : step size per PGD iteration
    num_iter : number of PGD iterations
    n_steps  : denoising steps for the model
    num_trajectories : number of trajectories
    random_start : initialize with random perturbation within epsilon ball

    Returns
    -------
    x_adv : (B, T, C) adversarial input
    """
    x_adv = x.clone().detach()

    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = x_adv.clamp(0, 1)

    for _ in range(num_iter):
        x_adv.requires_grad_(True)

        logits = _get_logits(
            model, x_adv, n_steps=n_steps, num_trajectories=num_trajectories
        )
        loss = F.cross_entropy(logits, y)
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            # Project back to epsilon ball around original x
            delta = (x_adv - x).clamp(-epsilon, epsilon)
            x_adv = (x + delta).clamp(0, 1)

    return x_adv.detach()


# ──────────────────────────────────────────────────────────────
# Adversarial evaluation
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_clean(model, loader, device, n_steps=32, num_trajectories=1):
    """Evaluate model on clean data."""
    model.eval()
    correct = total = 0
    for batch in loader:
        X, y = batch[0].to(device), batch[1].to(device)
        logits = _get_logits(model, X, n_steps=n_steps, num_trajectories=num_trajectories)
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def evaluate_adversarial(
    model, loader, device,
    attack_fn,
    epsilon: float,
    n_steps_attack: int = 1,
    n_steps_defense: int = 32,
    num_trajectories: int = 1,
    **attack_kw,
):
    """
    Evaluate model under adversarial attack.

    Attack is crafted using n_steps_attack (attacker's knowledge of the model).
    Defense uses n_steps_defense (can be different from attack).

    Parameters
    ----------
    attack_fn        : fgsm_attack or pgd_attack
    epsilon          : perturbation budget
    n_steps_attack   : steps used by attacker to craft adversarial examples
    n_steps_defense  : steps used by defender at inference
    num_trajectories : trajectories for defense
    """
    model.eval()
    correct = total = 0

    for batch in loader:
        X, y = batch[0].to(device), batch[1].to(device)

        # Enable gradients for attack crafting
        model.zero_grad()
        for p in model.parameters():
            p.requires_grad_(True)

        # Craft adversarial examples (attacker uses n_steps_attack)
        X_adv = attack_fn(
            model, X, y, epsilon=epsilon,
            n_steps=n_steps_attack,
            num_trajectories=1,  # attacker uses 1 trajectory
            **attack_kw,
        )

        # Evaluate defense (defender uses n_steps_defense + multi-trajectory)
        with torch.no_grad():
            logits = _get_logits(
                model, X_adv,
                n_steps=n_steps_defense,
                num_trajectories=num_trajectories,
            )
            correct += (logits.argmax(-1) == y).sum().item()
            total += y.size(0)

    return correct / max(total, 1)


def robustness_scaling_analysis(
    model, loader, device,
    attack_fn,
    epsilon: float,
    step_counts: list[int] = [1, 2, 4, 8, 16, 32],
    trajectory_counts: list[int] = [1, 3, 5],
    n_steps_attack: int = 1,
    **attack_kw,
):
    """
    Run robustness analysis across different inference configurations.

    Returns dict of results for plotting.
    """
    results = {
        "epsilon": epsilon,
        "step_scaling": {},
        "trajectory_scaling": {},
    }

    # Scale denoising steps (1 trajectory)
    print(f"\n  Step scaling (eps={epsilon}, 1 trajectory):")
    for n in step_counts:
        acc = evaluate_adversarial(
            model, loader, device, attack_fn, epsilon,
            n_steps_attack=n_steps_attack,
            n_steps_defense=n,
            num_trajectories=1,
            **attack_kw,
        )
        results["step_scaling"][n] = acc
        print(f"    steps={n:3d} -> acc={acc:.4f}")

    # Scale trajectories (fixed steps)
    best_steps = max(step_counts)
    print(f"\n  Trajectory scaling (eps={epsilon}, {best_steps} steps):")
    for nt in trajectory_counts:
        acc = evaluate_adversarial(
            model, loader, device, attack_fn, epsilon,
            n_steps_attack=n_steps_attack,
            n_steps_defense=best_steps,
            num_trajectories=nt,
            **attack_kw,
        )
        results["trajectory_scaling"][nt] = acc
        print(f"    trajectories={nt} -> acc={acc:.4f}")

    return results
