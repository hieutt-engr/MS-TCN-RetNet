"""
Multi-Scale ModernTCN for CAN bus IDS multi-class classification.

Based on ModernTCN (Luo & Wang, ICLR 2024 Spotlight) with multi-scale
temporal convolution extension inspired by InceptionTime (Fawaz et al. 2020)
and MSTAR (Cao et al. 2024).

Key design:
  - Multi-scale DWConv: parallel branches with different kernel sizes
    capture attack patterns at multiple temporal granularities
    (short bursts, medium-range patterns, long-range sequences)
  - Three-stage mixing preserved: temporal (MS-DWConv) -> feature -> variable
  - Structural reparameterization for each branch independently
  - No RevIN (absolute feature values carry semantic meaning for IDS)

Input : (B, T, C) where T=seq_len (e.g. 64), C=num_variables (e.g. 12)
Output: (B, num_classes) logits
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Utilities ──

def conv_bn(in_ch, out_ch, kernel_size, stride, padding, groups, dilation=1, bias=False):
    if padding is None:
        padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride,
                  padding=padding, dilation=dilation, groups=groups, bias=bias),
        nn.BatchNorm1d(out_ch),
    )


def fuse_bn(conv, bn):
    """Fuse Conv1d + BatchNorm1d into a single Conv1d (for inference speed)."""
    kernel = conv.weight
    mu, var = bn.running_mean, bn.running_var
    gamma, beta, eps = bn.weight, bn.bias, bn.eps
    std = (var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1)
    return kernel * t, beta - mu * gamma / std


# ── Reparameterizable Large Kernel Conv ──

class ReparamLargeKernelConv(nn.Module):
    """
    Training: two parallel branches (large kernel + small kernel), each with BN.
    Inference: merge into single conv for zero-overhead deployment.
    """

    def __init__(self, channels, kernel_size, small_kernel, stride=1,
                 groups=1, small_kernel_merged=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        padding = kernel_size // 2

        if small_kernel_merged:
            self.reparam = nn.Conv1d(
                channels, channels, kernel_size, stride=stride,
                padding=padding, groups=groups, bias=True,
            )
        else:
            self.lkb = conv_bn(channels, channels, kernel_size,
                               stride, padding, groups)
            if small_kernel is not None:
                self.skb = conv_bn(channels, channels, small_kernel,
                                   stride, small_kernel // 2, groups)

    def forward(self, x):
        if hasattr(self, "reparam"):
            return self.reparam(x)
        out = self.lkb(x)
        if hasattr(self, "skb"):
            out = out + self.skb(x)
        return out

    def merge_kernel(self):
        eq_k, eq_b = fuse_bn(self.lkb[0], self.lkb[1])
        if hasattr(self, "skb"):
            sk, sb = fuse_bn(self.skb[0], self.skb[1])
            eq_b += sb
            pad = (self.kernel_size - self.small_kernel) // 2
            eq_k += F.pad(sk, [pad, pad])
        self.reparam = nn.Conv1d(
            self.lkb[0].in_channels, self.lkb[0].out_channels,
            self.kernel_size, stride=self.lkb[0].stride[0],
            padding=self.kernel_size // 2,
            groups=self.lkb[0].groups, bias=True,
        )
        self.reparam.weight.data = eq_k
        self.reparam.bias.data = eq_b
        del self.lkb
        if hasattr(self, "skb"):
            del self.skb


# ── Multi-Scale Depthwise Conv ──

class MultiScaleDWConv(nn.Module):
    """
    Parallel depthwise conv branches at different kernel sizes.

    fusion="sum"    : average of branch outputs (original version; unstable
                      in full training because all branches share one fixed
                      1/K weight and their gradients conflict)
    fusion="concat" : branch outputs are concatenated per variable and fused
                      by a learnable pointwise projection (groups=M), so each
                      variable learns how much to trust each temporal scale.
                      Inspired by the concat + projection used in MS-TCN-RetNet,
                      but the projection never mixes different variables, so the
                      ModernTCN per-variable structure is preserved.

    Input/Output: (B, M*D, N)
    """

    def __init__(self, channels, kernel_sizes, small_size=5,
                 groups=1, small_kernel_merged=False,
                 fusion="concat", nvars=None):
        super().__init__()
        self.fusion = fusion
        self.num_branches = len(kernel_sizes)
        self.branches = nn.ModuleList([
            ReparamLargeKernelConv(
                channels=channels, kernel_size=ks,
                small_kernel=small_size if ks > small_size else None,
                groups=groups, small_kernel_merged=small_kernel_merged,
            )
            for ks in kernel_sizes
        ])

        if fusion == "sum":
            self.scale = 1.0 / self.num_branches
        elif fusion == "concat":
            assert nvars is not None and channels % nvars == 0
            self.nvars = nvars
            self.dmodel = channels // nvars
            K, M, D = self.num_branches, nvars, self.dmodel
            # Per-variable fusion: (K*D) -> D inside each variable
            self.proj = nn.Conv1d(M * K * D, M * D, kernel_size=1, groups=M)
            self._init_proj_as_average()
        else:
            raise ValueError(f"Unknown fusion={fusion}")

    def _init_proj_as_average(self):
        """Start as the plain average of branches, then let training reweight."""
        K, D = self.num_branches, self.dmodel
        with torch.no_grad():
            w = torch.zeros(self.nvars * D, K * D, 1)
            eye = torch.eye(D) / K
            for m in range(self.nvars):
                for k in range(K):
                    w[m * D:(m + 1) * D, k * D:(k + 1) * D, 0] = eye
            self.proj.weight.copy_(w)
            self.proj.bias.zero_()

    def forward(self, x):
        if self.fusion == "sum":
            out = 0
            for branch in self.branches:
                out = out + branch(x)
            return out * self.scale

        # concat: (B, M*D, N) per branch -> (B, M, K, D, N) -> (B, M*K*D, N)
        B, _, N = x.shape
        M, D = self.nvars, self.dmodel
        outs = [branch(x).reshape(B, M, D, N) for branch in self.branches]
        z = torch.stack(outs, dim=2).reshape(B, M * self.num_branches * D, N)
        return self.proj(z)

    def merge_kernel(self):
        for branch in self.branches:
            if hasattr(branch, "merge_kernel"):
                branch.merge_kernel()


# ── ModernTCN Block ──

class ModernTCNBlock(nn.Module):
    """
    One ModernTCN block with multi-scale temporal mixing:
    MS-DWConv -> BN -> ConvFFN1 (feature) -> ConvFFN2 (variable) + residual.

    Input/Output: (B, M, D, N)
    M = num_variables, D = feature_dim, N = temporal_length (patches)
    """

    def __init__(self, nvars, dmodel, ffn_ratio, large_size, small_size,
                 small_kernel_merged=False, dropout=0.1,
                 multi_scale=True, ms_kernel_sizes=None, ms_fusion="concat"):
        super().__init__()
        dff = dmodel * ffn_ratio

        # Temporal mixing: multi-scale or single-scale DWConv
        if multi_scale and ms_kernel_sizes is not None and len(ms_kernel_sizes) > 1:
            self.dw = MultiScaleDWConv(
                channels=nvars * dmodel, kernel_sizes=ms_kernel_sizes,
                small_size=small_size, groups=nvars * dmodel,
                small_kernel_merged=small_kernel_merged,
                fusion=ms_fusion, nvars=nvars,
            )
        else:
            self.dw = ReparamLargeKernelConv(
                channels=nvars * dmodel, kernel_size=large_size,
                small_kernel=small_size, groups=nvars * dmodel,
                small_kernel_merged=small_kernel_merged,
            )
        self.norm = nn.BatchNorm1d(dmodel)

        # Feature mixing: ConvFFN1, groups=M (mix D features within each variable)
        self.ffn1_pw1 = nn.Conv1d(nvars * dmodel, nvars * dff, 1, groups=nvars)
        self.ffn1_act = nn.GELU()
        self.ffn1_pw2 = nn.Conv1d(nvars * dff, nvars * dmodel, 1, groups=nvars)
        self.ffn1_drop1 = nn.Dropout(dropout)
        self.ffn1_drop2 = nn.Dropout(dropout)

        # Variable mixing: ConvFFN2, groups=D (mix M variables at each feature)
        self.ffn2_pw1 = nn.Conv1d(nvars * dmodel, nvars * dff, 1, groups=dmodel)
        self.ffn2_act = nn.GELU()
        self.ffn2_pw2 = nn.Conv1d(nvars * dff, nvars * dmodel, 1, groups=dmodel)
        self.ffn2_drop1 = nn.Dropout(dropout)
        self.ffn2_drop2 = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, M, D, N)
        residual = x
        B, M, D, N = x.shape

        # 1) Temporal mixing
        x = x.reshape(B, M * D, N)
        x = self.dw(x)
        x = x.reshape(B, M, D, N)

        # BatchNorm per variable
        x = x.reshape(B * M, D, N)
        x = self.norm(x)
        x = x.reshape(B, M, D, N)

        # 2) Feature mixing (within each variable)
        x = x.reshape(B, M * D, N)
        x = self.ffn1_drop1(self.ffn1_pw1(x))
        x = self.ffn1_act(x)
        x = self.ffn1_drop2(self.ffn1_pw2(x))
        x = x.reshape(B, M, D, N)

        # 3) Variable mixing (across variables at each feature)
        x = x.permute(0, 2, 1, 3)       # (B, D, M, N)
        x = x.reshape(B, D * M, N)
        x = self.ffn2_drop1(self.ffn2_pw1(x))
        x = self.ffn2_act(x)
        x = self.ffn2_drop2(self.ffn2_pw2(x))
        x = x.reshape(B, D, M, N)
        x = x.permute(0, 2, 1, 3)       # (B, M, D, N)

        return x + residual


# ── Full Model ──

class ModernTCN_CAN(nn.Module):
    """
    Multi-Scale ModernTCN for CAN bus IDS classification.

    Args:
        seq_len         : temporal length of input window (e.g. 64)
        nvars           : number of input variables/features (e.g. 12)
        num_classes     : number of output classes (e.g. 10)
        patch_size      : size of each patch for stem embedding
        patch_stride    : stride of patching
        dims            : list of feature dimensions per stage (e.g. [128, 256])
        num_blocks      : list of block counts per stage (e.g. [1, 1])
        large_sizes     : list of large kernel sizes per stage (e.g. [13, 13])
        small_size      : small kernel size for reparameterization (default 5)
        ffn_ratio       : expansion ratio for ConvFFN (default 2)
        dropout         : backbone dropout
        head_dropout    : classification head dropout
        multi_scale     : enable multi-scale DWConv (default True)
        ms_kernel_sizes : kernel sizes for parallel branches (e.g. [3, 7, 13])
        ms_fusion       : "concat" (learnable per-variable fusion) or "sum"
        head_type       : "flatten" (original ModernTCN head) or
                          "mean" (temporal mean pooling, as in MS-TCN-RetNet)
    """

    def __init__(
        self,
        seq_len: int = 64,
        nvars: int = 12,
        num_classes: int = 10,
        patch_size: int = 8,
        patch_stride: int = 4,
        dims: list = None,
        num_blocks: list = None,
        large_sizes: list = None,
        small_size: int = 5,
        ffn_ratio: int = 2,
        dropout: float = 0.1,
        head_dropout: float = 0.0,
        class_dropout: float = 0.0,
        small_kernel_merged: bool = False,
        multi_scale: bool = True,
        ms_kernel_sizes: list = None,
        ms_fusion: str = "concat",
        head_type: str = "flatten",
    ):
        super().__init__()
        if dims is None:
            dims = [128, 256]
        if num_blocks is None:
            num_blocks = [1, 1]
        if large_sizes is None:
            large_sizes = [13, 13]
        if ms_kernel_sizes is None:
            ms_kernel_sizes = [3, 7, 13]

        self.nvars = nvars
        self.multi_scale = multi_scale
        self.ms_kernel_sizes = ms_kernel_sizes
        self.num_stages = len(num_blocks)
        assert len(dims) == self.num_stages
        assert len(large_sizes) == self.num_stages

        # Stem: per-variable patch embedding via Conv1d(1 -> dims[0])
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.stem = nn.Sequential(
            nn.Conv1d(1, dims[0], kernel_size=patch_size, stride=patch_stride),
            nn.BatchNorm1d(dims[0]),
        )

        # Downsample layers between stages
        self.downsample_layers = nn.ModuleList()
        downsample_ratio = 2
        self.downsample_ratio = downsample_ratio
        for i in range(self.num_stages - 1):
            self.downsample_layers.append(nn.Sequential(
                nn.BatchNorm1d(dims[i]),
                nn.Conv1d(dims[i], dims[i + 1], kernel_size=downsample_ratio,
                          stride=downsample_ratio),
            ))

        # Backbone stages
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            blocks = nn.ModuleList([
                ModernTCNBlock(
                    nvars=nvars, dmodel=dims[i], ffn_ratio=ffn_ratio,
                    large_size=large_sizes[i], small_size=small_size,
                    small_kernel_merged=small_kernel_merged, dropout=dropout,
                    multi_scale=multi_scale, ms_kernel_sizes=ms_kernel_sizes,
                    ms_fusion=ms_fusion,
                )
                for _ in range(num_blocks[i])
            ])
            self.stages.append(blocks)

        # Classification head
        patch_num = seq_len // patch_stride
        d_final = dims[-1]
        # After all downsampling, compute final temporal length
        final_n = patch_num
        for _ in range(self.num_stages - 1):
            final_n = (final_n + downsample_ratio - 1) // downsample_ratio

        self.head_type = head_type
        if head_type == "flatten":
            # Original: flatten (M * D * N) -> Linear
            self.head_nf = nvars * d_final * final_n
            self.head = nn.Sequential(
                nn.GELU(),
                nn.Dropout(class_dropout),
                nn.Linear(self.head_nf, num_classes),
            )
        elif head_type == "mean":
            # Mean over time N -> (M * D) -> LayerNorm -> Linear
            # Keeps per-variable info, drops per-patch memorization
            self.head_nf = nvars * d_final
            self.head = nn.Sequential(
                nn.LayerNorm(self.head_nf),
                nn.GELU(),
                nn.Dropout(class_dropout),
                nn.Linear(self.head_nf, num_classes),
            )
        else:
            raise ValueError(f"Unknown head_type={head_type}")

    def forward_backbone(self, x):
        """
        x: (B, M, L) where M=nvars, L=seq_len
        returns: (B, M, D, N)
        """
        B, M, L = x.shape

        # Stem: embed each variable independently
        # (B, M, L) -> (B*M, 1, L) -> Conv1d -> (B*M, D, N) -> (B, M, D, N)
        x = x.reshape(B * M, 1, L)
        if self.patch_size != self.patch_stride:
            pad_len = self.patch_size - self.patch_stride
            x = F.pad(x, (0, pad_len), mode="replicate")
        x = self.stem(x)
        _, D, N = x.shape
        x = x.reshape(B, M, D, N)

        # Stages
        for i in range(self.num_stages):
            if i > 0:
                # Downsample: per-variable
                B, M, D, N = x.shape
                x = x.reshape(B * M, D, N)
                if N % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                    x = F.pad(x, (0, pad_len), mode="replicate")
                x = self.downsample_layers[i - 1](x)
                _, D, N = x.shape
                x = x.reshape(B, M, D, N)

            for blk in self.stages[i]:
                x = blk(x)

        return x

    def forward(self, x):
        """
        x: (B, T, C) from dataloader  (T=seq_len, C=nvars)
        returns: (B, num_classes) logits
        """
        # Permute to (B, M=C, L=T) for ModernTCN
        x = x.permute(0, 2, 1)

        # Backbone
        x = self.forward_backbone(x)  # (B, M, D, N)

        if self.head_type == "mean":
            x = x.mean(dim=-1)               # (B, M, D)
        x = x.reshape(x.shape[0], -1)        # (B, M*D) or (B, M*D*N)
        x = self.head(x)
        return x

    def structural_reparam(self):
        """Merge large+small kernels for fast inference."""
        for m in self.modules():
            if hasattr(m, "merge_kernel"):
                m.merge_kernel()

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        backbone = sum(
            p.numel() for n, p in self.named_parameters()
            if "head" not in n
        )
        return {"total": total, "backbone": backbone, "head": total - backbone}