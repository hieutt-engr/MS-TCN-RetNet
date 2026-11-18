"""
MS-TCN -> Liquid Dynamic Gate -> RetNet for CAN/IVN IDS.

Input shape: (B, T, C). For the current preprocessing, C=12 and
log_delta_t_norm is at index 10.

The Liquid Dynamic Gate does not update learned parameters during inference.
It changes the effective temporal dynamics through input/state-conditioned
adaptive gates and time constants.
"""

from __future__ import annotations
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
    """Multi-scale local temporal extractor."""
    def __init__(self, in_dim: int, branch_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.b1 = nn.Conv1d(in_dim, branch_dim, kernel_size=3, padding=1, dilation=1)
        self.b2 = nn.Conv1d(in_dim, branch_dim, kernel_size=3, padding=2, dilation=2)
        self.b3 = nn.Conv1d(in_dim, branch_dim, kernel_size=5, padding=2, dilation=1)
        self.norm = nn.LayerNorm(branch_dim * 3)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x.transpose(1, 2)  # (B,C,T)
        z = torch.cat([self.b1(xt), self.b2(xt), self.b3(xt)], dim=1)
        z = z.transpose(1, 2)  # (B,T,3*branch_dim)
        z = self.norm(z)
        z = F.gelu(z)
        return self.drop(z)


class ResidualTCNBlock(nn.Module):
    """Causal residual dilated TCN block."""
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

    def _trim(self, x: torch.Tensor) -> torch.Tensor:
        trim = (self.kernel_size - 1) * self.dilation
        return x[:, :, :-trim] if trim > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x.transpose(1, 2)
        z = self._trim(self.conv1(xt)).transpose(1, 2)
        z = self.drop1(F.gelu(self.norm1(z)))
        z = self._trim(self.conv2(z.transpose(1, 2))).transpose(1, 2)
        z = self.drop2(F.gelu(self.norm2(z)))
        res = self.res_proj(xt).transpose(1, 2)
        return F.gelu(self.out_norm(z + res))


class LiquidDynamicGate(nn.Module):
    """
    Liquid-inspired adaptive temporal modulation.

    c_t     = tanh(Wx h_t + Wh s_{t-1})
    tau_t   = softplus(Wtau [h_t, s_{t-1}, dt_t])
    alpha_t = exp(-tau_t * dt_factor_t)
    s_t     = alpha_t * s_{t-1} + (1-alpha_t) * c_t

    Then the output uses a residual form:
        y_t = LayerNorm(h_t + Dropout(Wo s_t))
    """
    def __init__(
        self,
        dim: int,
        dt_dim: int = 1,
        dropout: float = 0.1,
        use_input_gate: bool = True,
        min_tau: float = 1e-4,
        max_dt_factor: float = 5.0,
    ):
        super().__init__()
        self.dim = dim
        self.use_input_gate = use_input_gate
        self.min_tau = min_tau
        self.max_dt_factor = max_dt_factor
        self.x_proj = nn.Linear(dim, dim)
        self.h_proj = nn.Linear(dim, dim, bias=False)
        self.tau_proj = nn.Linear(dim * 2 + dt_dim, dim)
        self.gate_proj = nn.Linear(dim * 2 + dt_dim, dim) if use_input_gate else None
        self.out_proj = nn.Linear(dim, dim)
        self.out_norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, dt: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        if dt is None:
            dt = torch.ones(B, T, 1, device=x.device, dtype=x.dtype)
        elif dt.dim() == 2:
            dt = dt.unsqueeze(-1)
        dt = dt.to(device=x.device, dtype=x.dtype)
        dt_factor = 1.0 + self.max_dt_factor * dt.clamp(0.0, 1.0)

        state = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            xt = x[:, t, :]
            dtt = dt[:, t, :]
            dtft = dt_factor[:, t, :]
            ctx = torch.cat([xt, state, dtt], dim=-1)
            candidate = torch.tanh(self.x_proj(xt) + self.h_proj(state))
            tau = F.softplus(self.tau_proj(ctx)) + self.min_tau
            alpha = torch.exp(-tau * dtft).clamp(0.0, 1.0)
            liquid_state = alpha * state + (1.0 - alpha) * candidate
            if self.use_input_gate:
                gate = torch.sigmoid(self.gate_proj(ctx))
                state = gate * xt + (1.0 - gate) * liquid_state
            else:
                state = liquid_state
            outputs.append(state.unsqueeze(1))
        y = torch.cat(outputs, dim=1)
        y = self.out_proj(y)
        return self.out_norm(x + self.drop(y))


class MultiScaleRetention(nn.Module):
    """Causal retention-style module with per-head exponential decay."""
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
        self.decay_logits = nn.Parameter(torch.linspace(0.0, 2.0, steps=num_heads))

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = self._heads(self.q_proj(x))
        k = self._heads(self.k_proj(x))
        v = self._heads(self.v_proj(x))
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        idx = torch.arange(T, device=x.device)
        dist = (idx[:, None] - idx[None, :]).clamp(min=0).float()
        gammas = torch.sigmoid(self.decay_logits).clamp(1e-4, 1.0 - 1e-4)
        decay_bias = torch.log(gammas).view(1, self.num_heads, 1, 1) * dist.view(1, 1, T, T)
        scores = scores + decay_bias
        scores = scores.masked_fill(~causal.view(1, 1, T, T), -1e9)
        attn = self.attn_drop(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        return self.out_proj(out)


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


class MSTCNLiquidRetNet(nn.Module):
    """MS-TCN -> Liquid Dynamic Gate -> RetNet classifier."""
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
        liquid_dim: Optional[int] = None,
        liquid_dropout: float = 0.1,
        liquid_use_input_gate: bool = True,
        liquid_max_dt_factor: float = 5.0,
        retention_dim: int = 64,
        retention_depth: int = 2,
        retention_heads: int = 4,
        retention_dropout: float = 0.1,
        ff_expansion: int = 4,
        pool: str = "mean",
        delta_t_index: Optional[int] = 10,
    ):
        super().__init__()
        self.pool = pool
        self.delta_t_index = delta_t_index
        liquid_dim = tcn_dim if liquid_dim is None else liquid_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.ms_conv = MultiScaleTemporalConv(hidden_dim, branch_dim, tcn_dropout)
        self.ms_proj = nn.Linear(branch_dim * 3, tcn_dim)
        self.ms_proj_norm = nn.LayerNorm(tcn_dim)

        self.tcn_blocks = nn.ModuleList([
            ResidualTCNBlock(tcn_dim, tcn_kernel_size, dilation=2 ** i, dropout=tcn_dropout)
            for i in range(tcn_depth)
        ])

        self.to_liquid_dim = nn.Linear(tcn_dim, liquid_dim) if tcn_dim != liquid_dim else None
        self.to_liquid_norm = nn.LayerNorm(liquid_dim) if tcn_dim != liquid_dim else None

        self.liquid_gate = LiquidDynamicGate(
            dim=liquid_dim,
            dropout=liquid_dropout,
            use_input_gate=liquid_use_input_gate,
            max_dt_factor=liquid_max_dt_factor,
        )

        self.to_ret_dim = nn.Linear(liquid_dim, retention_dim) if liquid_dim != retention_dim else None
        self.to_ret_norm = nn.LayerNorm(retention_dim) if liquid_dim != retention_dim else None

        self.ret_blocks = nn.ModuleList([
            RetentiveBlock(retention_dim, retention_heads, ff_expansion, retention_dropout)
            for _ in range(retention_depth)
        ])

        self.head_norm = nn.LayerNorm(retention_dim)
        self.head_drop = nn.Dropout(retention_dropout)
        self.classifier = nn.Linear(retention_dim, num_classes)

    def _extract_dt(self, raw_x: torch.Tensor) -> Optional[torch.Tensor]:
        if self.delta_t_index is None:
            return None
        if self.delta_t_index < 0 or self.delta_t_index >= raw_x.shape[-1]:
            return None
        return raw_x[..., self.delta_t_index:self.delta_t_index + 1]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        dt = self._extract_dt(x)
        x = self.input_norm(self.input_proj(x))
        x = self.ms_conv(x)
        x = self.ms_proj_norm(self.ms_proj(x))
        for block in self.tcn_blocks:
            x = block(x)
        if self.to_liquid_dim is not None:
            x = self.to_liquid_norm(self.to_liquid_dim(x))
        x = self.liquid_gate(x, dt=dt)
        if self.to_ret_dim is not None:
            x = self.to_ret_norm(self.to_ret_dim(x))
        for block in self.ret_blocks:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        if self.pool == "mean":
            z = x.mean(dim=1)
        elif self.pool == "max":
            z = x.max(dim=1).values
        else:
            raise ValueError(f"Unsupported pool={self.pool}")
        z = self.head_drop(self.head_norm(z))
        return self.classifier(z)


if __name__ == "__main__":
    B, T, C = 8, 64, 12
    model = MSTCNLiquidRetNet(input_dim=C, num_classes=10, delta_t_index=10)
    x = torch.randn(B, T, C)
    logits = model(x)
    print("Input:", x.shape)
    print("Logits:", logits.shape)
