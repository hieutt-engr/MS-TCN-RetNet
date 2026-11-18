"""
MS-TCN + Looped Denoising with Progressive Feature Reveal.

Core idea: reveal feature groups gradually across denoising steps.
Early steps see only timing features, later steps add ID, payload, etc.
This forces genuine iterative refinement: each step adds new information
that the previous step did not have access to.

Input : (B, T, C)
Output: (B, num_classes) logits
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── MS-TCN backbone ──

class GLU(nn.Module):
    def forward(self, x):
        a, b = torch.chunk(x, 2, dim=-1)
        return a * torch.sigmoid(b)

class FeedForward(nn.Module):
    def __init__(self, dim, expansion=4, dropout=0.1):
        super().__init__()
        h = dim * expansion
        self.net = nn.Sequential(
            nn.Linear(dim, h * 2), GLU(), nn.Dropout(dropout),
            nn.Linear(h, dim), nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class MultiScaleTemporalConv(nn.Module):
    def __init__(self, in_dim, branch_dim=32, dropout=0.1):
        super().__init__()
        self.b1 = nn.Conv1d(in_dim, branch_dim, 3, padding=1, dilation=1)
        self.b2 = nn.Conv1d(in_dim, branch_dim, 3, padding=2, dilation=2)
        self.b3 = nn.Conv1d(in_dim, branch_dim, 5, padding=2, dilation=1)
        self.norm = nn.LayerNorm(branch_dim * 3)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        xt = x.transpose(1, 2)
        z = torch.cat([self.b1(xt), self.b2(xt), self.b3(xt)], dim=1)
        return self.drop(F.gelu(self.norm(z.transpose(1, 2))))

class ResidualTCNBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(dim, dim, kernel_size, padding=pad, dilation=dilation)
        self.norm1 = nn.LayerNorm(dim)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size, padding=pad, dilation=dilation)
        self.norm2 = nn.LayerNorm(dim)
        self.drop2 = nn.Dropout(dropout)
        self.res = nn.Conv1d(dim, dim, 1)
        self.out_norm = nn.LayerNorm(dim)
        self.ks, self.dil = kernel_size, dilation
    def _trim(self, x):
        t = (self.ks - 1) * self.dil
        return x[:, :, :-t] if t else x
    def forward(self, x):
        xt = x.transpose(1, 2)
        z = self.drop1(F.gelu(self.norm1(self._trim(self.conv1(xt)).transpose(1, 2))))
        z = self.drop2(F.gelu(self.norm2(self._trim(self.conv2(z.transpose(1, 2))).transpose(1, 2))))
        return F.gelu(self.out_norm(z + self.res(xt).transpose(1, 2)))


# ── Progressive Feature Reveal ──

class ProgressiveReveal(nn.Module):
    """
    Reveal feature groups based on denoising timestep t.
    Group i becomes visible at t >= i/num_groups via a soft gate.
    Unrevealed dims get a learnable placeholder instead of zeros.
    """
    def __init__(self, dim: int, num_groups: int = 4):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = dim // num_groups
        assert dim % num_groups == 0

        self.gate_sharpness = nn.Parameter(torch.full((num_groups,), 10.0))
        self.placeholder = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.placeholder, std=0.02)

        thresholds = torch.tensor([i / num_groups for i in range(num_groups)])
        self.register_buffer("thresholds", thresholds)

    def forward(self, context: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """context: (B, D), t: (B,) in [0,1] -> (B, D) partially revealed"""
        if t.dim() == 2:
            t = t.squeeze(-1)
        gates = torch.sigmoid(
            (t.unsqueeze(-1) - self.thresholds.unsqueeze(0)) * self.gate_sharpness.unsqueeze(0)
        )
        gates_exp = gates.repeat_interleave(self.group_size, dim=-1)
        ph = self.placeholder.unsqueeze(0).expand_as(context)
        return gates_exp * context + (1 - gates_exp) * ph


# ── Stateful Denoiser ──

class StatefulDenoiserBlock(nn.Module):
    def __init__(self, latent_dim, context_dim, hidden_dim, expansion=4, dropout=0.1):
        super().__init__()
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn1 = FeedForward(hidden_dim, expansion, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn2 = FeedForward(hidden_dim, expansion, dropout)
        self.state_norm = nn.LayerNorm(hidden_dim)
        self.state_ffn = FeedForward(hidden_dim, expansion, dropout)
        self.state_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.out_proj = nn.Linear(hidden_dim, latent_dim)
        self.out_norm = nn.LayerNorm(latent_dim)

    def forward(self, x_noisy, context, t, z):
        h = self.latent_proj(x_noisy) + self.context_proj(context) + self.time_mlp(t) + z
        h = h + self.ffn1(self.norm1(h))
        h = h + self.ffn2(self.norm2(h))
        x_pred = self.out_norm(self.out_proj(h))
        z_c = self.state_ffn(self.state_norm(h))
        g = self.state_gate(torch.cat([z, z_c], -1))
        return x_pred, g * z_c + (1 - g) * z

class ACTHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
    def forward(self, z): return self.net(z).squeeze(-1)


# ── Looped Denoiser ──

class LoopedDenoiser(nn.Module):
    def __init__(self, num_classes, context_dim, latent_dim=64, hidden_dim=128,
                 expansion=4, noise_scale=1.0, dropout=0.1, num_reveal_groups=4):
        super().__init__()
        self.C, self.D, self.H = num_classes, latent_dim, hidden_dim
        self.sigma = noise_scale
        self.class_to_latent = nn.Linear(num_classes, latent_dim, bias=False)
        self.latent_to_class = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, num_classes))
        self.reveal = ProgressiveReveal(context_dim, num_reveal_groups)
        self.denoiser = StatefulDenoiserBlock(latent_dim, context_dim, hidden_dim, expansion, dropout)
        self.act_head = ACTHead(hidden_dim)
        self.z0 = nn.Parameter(torch.zeros(hidden_dim))

    def _noise(self, B, dev):
        return torch.randn(B, self.D, device=dev) * self.sigma

    def train_forward(self, ctx, labels, k=16, act_weight=0.5,
                      pseudotarget_prob=1.0, latent_noise_std=0.1):
        B, dev = ctx.shape[0], ctx.device
        x1 = self.class_to_latent(F.one_hot(labels, self.C).float())
        if latent_noise_std > 0 and self.training:
            x1 = x1 + torch.randn_like(x1) * latent_noise_std

        ts = torch.rand(B, k, device=dev).sort(-1).values
        z = self.z0.unsqueeze(0).expand(B, -1)
        ce_sum = act_sum = torch.tensor(0.0, device=dev)
        logits_all, act_scores, prev = [], [], None

        for i in range(k):
            t_i = ts[:, i]
            x0 = self._noise(B, dev)
            if i > 0 and prev is not None:
                m = torch.rand(B, 1, device=dev) < pseudotarget_prob
                tgt = torch.where(m, prev.detach(), x1)
            else:
                tgt = x1
            interp = (1 - t_i.unsqueeze(-1)) * x0 + t_i.unsqueeze(-1) * tgt

            ctx_r = self.reveal(ctx, t_i)
            pred, z_new = self.denoiser(interp, ctx_r, t_i.unsqueeze(-1), z.detach())
            prev = pred

            logits = self.latent_to_class(pred)
            logits_all.append(logits)
            ce_sum = ce_sum + F.cross_entropy(logits, labels)
            with torch.no_grad():
                ok = (logits.argmax(-1) == labels).float()
            act = self.act_head(z_new)
            act_scores.append(act)
            act_sum = act_sum + F.binary_cross_entropy_with_logits(act.clamp(-20, 20), ok)
            z = z_new

        return {"loss": ce_sum / k + act_weight * act_sum / k,
                "logits_all": logits_all, "act_scores": act_scores}

    @torch.no_grad()
    def inference_forward(self, ctx, n_steps=32, num_trajectories=1):
        B, dev = ctx.shape[0], ctx.device
        grid = torch.linspace(0, 1, n_steps + 1, device=dev)
        best_logits = best_act = None

        for _ in range(num_trajectories):
            x = self._noise(B, dev)
            z = self.z0.unsqueeze(0).expand(B, -1)
            for i in range(n_steps):
                tc, tn = grid[i].item(), grid[i + 1].item()
                t_vec = torch.full((B,), tc, device=dev)
                ctx_r = self.reveal(ctx, t_vec)
                pred, z = self.denoiser(x, ctx_r, torch.full((B, 1), tc, device=dev), z)
                x = x + (tn - tc) * (pred - x) / (1 - tc + 1e-6)
            logits = self.latent_to_class(pred)
            act_s = self.act_head(z)
            if best_logits is None:
                best_logits, best_act = logits, act_s
            else:
                better = act_s > best_act
                best_logits = torch.where(better.unsqueeze(-1), logits, best_logits)
                best_act = torch.where(better, act_s, best_act)
        return best_logits


# ── Full Model ──

class MSTCNLoopedFlow(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=64, branch_dim=32,
                 tcn_dim=64, tcn_depth=2, tcn_kernel_size=3, tcn_dropout=0.1,
                 pool="mean", latent_dim=64, denoiser_hidden_dim=128,
                 denoiser_expansion=4, denoiser_dropout=0.1, noise_scale=1.0,
                 train_steps=16, inference_steps=32, inference_trajectories=1,
                 num_reveal_groups=4):
        super().__init__()
        self.pool_mode = pool
        self.train_steps = train_steps
        self.inference_steps = inference_steps
        self.inference_trajectories = inference_trajectories

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.ms_conv = MultiScaleTemporalConv(hidden_dim, branch_dim, tcn_dropout)
        self.ms_proj = nn.Linear(branch_dim * 3, tcn_dim)
        self.ms_proj_norm = nn.LayerNorm(tcn_dim)
        self.tcn_blocks = nn.ModuleList([
            ResidualTCNBlock(tcn_dim, tcn_kernel_size, 2**i, tcn_dropout)
            for i in range(tcn_depth)
        ])
        self.denoiser = LoopedDenoiser(
            num_classes, tcn_dim, latent_dim, denoiser_hidden_dim,
            denoiser_expansion, noise_scale, denoiser_dropout,
            num_reveal_groups,
        )

    def extract_context(self, x):
        x = self.input_norm(self.input_proj(x))
        x = self.ms_proj_norm(self.ms_proj(self.ms_conv(x)))
        for blk in self.tcn_blocks:
            x = blk(x)
        return x.mean(1) if self.pool_mode == "mean" else x.max(1).values

    def forward(self, x, labels=None, **kw):
        ctx = self.extract_context(x)
        if self.training and labels is not None:
            return self.denoiser.train_forward(
                ctx, labels, k=kw.get("k", self.train_steps),
                act_weight=kw.get("act_weight", 0.5),
                pseudotarget_prob=kw.get("pseudotarget_prob", 1.0),
                latent_noise_std=kw.get("latent_noise_std", 0.1),
            )
        return self.denoiser.inference_forward(
            ctx, n_steps=kw.get("n_steps", self.inference_steps),
            num_trajectories=kw.get("num_trajectories", self.inference_trajectories),
        )
