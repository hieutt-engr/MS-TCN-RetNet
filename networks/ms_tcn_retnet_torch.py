"""
PyTorch implementation of MS-TCN-RetNet for CAN/IVN sequence classification.

Input
-----
x: Tensor of shape (B, T, C)

Output
------
logits: Tensor of shape (B, num_classes)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = torch.chunk(x, 2, dim=-1)
        return a * torch.sigmoid(b)


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = dim * expansion
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.glu = GLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.glu(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MultiScaleTemporalConv(nn.Module):
    def __init__(self, in_dim: int, branch_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.b1 = nn.Conv1d(in_dim, branch_dim, kernel_size=3, padding=1, dilation=1)
        self.b2 = nn.Conv1d(in_dim, branch_dim, kernel_size=3, padding=2, dilation=2)
        self.b3 = nn.Conv1d(in_dim, branch_dim, kernel_size=5, padding=2, dilation=1)
        self.norm = nn.LayerNorm(branch_dim * 3)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, C, T)
        xt = x.transpose(1, 2)
        z1 = self.b1(xt)
        z2 = self.b2(xt)
        z3 = self.b3(xt)
        z = torch.cat([z1, z2, z3], dim=1).transpose(1, 2)  # (B, T, 3*branch_dim)
        z = self.norm(z)
        z = F.gelu(z)
        z = self.drop(z)
        return z


class ResidualTCNBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=pad, dilation=dilation)
        self.norm1 = nn.LayerNorm(dim)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=pad, dilation=dilation)
        self.norm2 = nn.LayerNorm(dim)
        self.drop2 = nn.Dropout(dropout)

        self.res_proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.out_norm = nn.LayerNorm(dim)

        self.kernel_size = kernel_size
        self.dilation = dilation

    def _causal_trim(self, x: torch.Tensor) -> torch.Tensor:
        trim = (self.kernel_size - 1) * self.dilation
        return x[:, :, :-trim] if trim > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        xt = x.transpose(1, 2)

        z = self.conv1(xt)
        z = self._causal_trim(z).transpose(1, 2)
        z = self.norm1(z)
        z = F.gelu(z)
        z = self.drop1(z)

        zt = z.transpose(1, 2)
        z = self.conv2(zt)
        z = self._causal_trim(z).transpose(1, 2)
        z = self.norm2(z)
        z = F.gelu(z)
        z = self.drop2(z)

        res = self.res_proj(xt).transpose(1, 2)
        out = self.out_norm(z + res)
        out = F.gelu(out)
        return out


class MultiScaleRetention(nn.Module):
    """
    Practical causal retention-style operator.
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        decay_init = torch.linspace(0.0, 2.0, steps=num_heads)
        self.decay_logits = nn.Parameter(decay_init)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B,T,D) -> (B,H,T,Hd)
        B, T, D = x.shape
        x = x.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = self._reshape_heads(self.q_proj(x))
        k = self._reshape_heads(self.k_proj(x))
        v = self._reshape_heads(self.v_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,H,T,T)

        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        dist = torch.arange(T, device=x.device)
        dist = (dist[:, None] - dist[None, :]).clamp(min=0).float()  # (T,T)

        gammas = torch.sigmoid(self.decay_logits).clamp(1e-4, 1 - 1e-4)  # (H,)
        decay_bias = torch.log(gammas).view(1, self.num_heads, 1, 1) * dist.view(1, 1, T, T)

        scores = scores + decay_bias
        scores = scores.masked_fill(~causal.view(1, 1, T, T), -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # (B,H,T,Hd)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        out = self.out_proj(out)
        return out


class RetentiveBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, ff_expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.retention = MultiScaleRetention(dim=dim, num_heads=num_heads, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim=dim, expansion=ff_expansion, dropout=dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.retention(self.norm1(x)))
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x


class MSTCNRetNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        branch_dim: int = 32,
        tcn_dim: int = 64,
        tcn_depth: int = 2,
        tcn_kernel_size: int = 3,
        tcn_dropout: float = 0.1,
        retention_dim: int = 64,
        retention_depth: int = 2,
        retention_heads: int = 4,
        retention_dropout: float = 0.1,
        ff_expansion: int = 4,
        pool: str = "mean",
    ):
        super().__init__()
        self.pool = pool

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.ms_conv = MultiScaleTemporalConv(in_dim=hidden_dim, branch_dim=branch_dim, dropout=tcn_dropout)
        self.ms_proj = nn.Linear(branch_dim * 3, tcn_dim)
        self.ms_proj_norm = nn.LayerNorm(tcn_dim)

        self.tcn_blocks = nn.ModuleList([
            ResidualTCNBlock(
                dim=tcn_dim,
                kernel_size=tcn_kernel_size,
                dilation=2 ** i,
                dropout=tcn_dropout,
            )
            for i in range(tcn_depth)
        ])

        self.to_ret_dim = None
        self.to_ret_norm = None
        if tcn_dim != retention_dim:
            self.to_ret_dim = nn.Linear(tcn_dim, retention_dim)
            self.to_ret_norm = nn.LayerNorm(retention_dim)

        self.ret_blocks = nn.ModuleList([
            RetentiveBlock(
                dim=retention_dim,
                num_heads=retention_heads,
                ff_expansion=ff_expansion,
                dropout=retention_dropout,
            )
            for _ in range(retention_depth)
        ])

        self.head_norm = nn.LayerNorm(retention_dim)
        self.head_drop = nn.Dropout(retention_dropout)
        self.classifier = nn.Linear(retention_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.input_norm(x)

        x = self.ms_conv(x)
        x = self.ms_proj(x)
        x = self.ms_proj_norm(x)

        for blk in self.tcn_blocks:
            x = blk(x)

        if self.to_ret_dim is not None:
            x = self.to_ret_dim(x)
            x = self.to_ret_norm(x)

        for blk in self.ret_blocks:
            x = blk(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)

        if self.pool == "mean":
            z = x.mean(dim=1)
        elif self.pool == "max":
            z = x.max(dim=1).values
        else:
            raise ValueError(f"Unsupported pool={self.pool}")

        z = self.head_norm(z)
        z = self.head_drop(z)
        logits = self.classifier(z)
        return logits
