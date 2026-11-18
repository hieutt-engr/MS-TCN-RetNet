# """
# IVN-TCN: lightweight Multi-Scale ModernTCN for CAN bus IDS.

# Three parts, each chosen for in-vehicle constraints (single forward pass,
# small parameter count, low latency per window):

# 1) Input embedding (RealMLP-inspired, Holzmueller et al., NeurIPS 2024;
#    Gorishniy et al., NeurIPS 2022)
#    - Robust scaling (median / IQR fitted on train) + smooth clipping
#    - Learnable per-variable scaling (soft feature selection)
#    - PBLD embedding per numerical variable: cos(2*pi*w*x + b) -> Linear,
#      concatenated with the scaled scalar itself (DenseNet-style passthrough)
#    - Optional categorical embedding for arbitration ID, with an
#      out-of-vocabulary token (unseen IDs are themselves an attack signal)
#    Every variable has its own embedding parameters (never shared), which
#    matches the per-variable design of ModernTCN.

# 2) Backbone: Multi-Scale ModernTCN blocks (small dims, light fusion)

# 3) Head: temporal mean pooling + TabM-style BatchEnsemble MLP
#    (Gorishniy et al., ICLR 2025). k members share weights and differ by
#    rank-1 adapters (r, s, b). Ensembling is applied ONLY to the head, so
#    the convolutional backbone runs once per window.

# Input : (B, T, C)
# Output: (B, k, num_classes) logits, one set per ensemble member
# """

# from __future__ import annotations
# import math

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# try:
#     from networks.moderntcn_can_old import ModernTCNBlock
# except ImportError:
#     from networks.moderntcn_can_old import ModernTCNBlock


# # ─────────────────────────── 1) Input embedding ───────────────────────────

# class RobustSmoothScaler(nn.Module):
#     """(x - median) / IQR, then smooth clip x / sqrt(1 + (x/c)^2). Fitted on train."""

#     def __init__(self, nvars, clip=3.0):
#         super().__init__()
#         self.clip = clip
#         self.register_buffer("center", torch.zeros(nvars))
#         self.register_buffer("scale", torch.ones(nvars))

#     @torch.no_grad()
#     def fit(self, X, max_rows=500_000):
#         x = X.reshape(-1, X.shape[-1]).float()
#         if x.shape[0] > max_rows:
#             idx = torch.randperm(x.shape[0])[:max_rows]
#             x = x[idx]
#         q = torch.quantile(x, torch.tensor([0.25, 0.5, 0.75]), dim=0)
#         iqr = q[2] - q[0]
#         std = x.std(dim=0)
#         # Mostly-constant payload bytes have IQR = 0: fall back to std, then 1
#         scale = torch.where(iqr > 1e-6, iqr, torch.where(std > 1e-6, std, torch.ones_like(std)))
#         self.center.copy_(q[1])
#         self.scale.copy_(scale)

#     def forward(self, x):
#         x = (x - self.center) / self.scale
#         return x / torch.sqrt(1.0 + (x / self.clip) ** 2)


# class PBLDEmbedding(nn.Module):
#     """
#     Periodic-Bias-Linear-DenseNet embedding, one independent module per variable
#     (vectorized). x: (..., M) -> (..., M, d_out)
#     """

#     def __init__(self, nvars, n_freq=16, d_out=8, sigma=1.0):
#         super().__init__()
#         assert d_out >= 2
#         self.w = nn.Parameter(torch.randn(nvars, n_freq) * sigma)
#         self.b = nn.Parameter(torch.empty(nvars, n_freq).uniform_(-math.pi, math.pi))
#         self.lin_w = nn.Parameter(torch.empty(nvars, n_freq, d_out - 1))
#         self.lin_b = nn.Parameter(torch.zeros(nvars, d_out - 1))
#         nn.init.uniform_(self.lin_w, -1 / math.sqrt(n_freq), 1 / math.sqrt(n_freq))

#     def forward(self, x):
#         z = torch.cos(2 * math.pi * x.unsqueeze(-1) * self.w + self.b)       # (..., M, K)
#         z = torch.einsum("...mk,mkd->...md", z, self.lin_w) + self.lin_b      # (..., M, d-1)
#         return torch.cat([x.unsqueeze(-1), z], dim=-1)                        # (..., M, d)


# class CategoryEmbedding(nn.Module):
#     """Map a float-encoded categorical column to indices (exact match) + OOV token."""

#     def __init__(self, max_vocab=512, d_out=8, tol=1e-5):
#         super().__init__()
#         self.max_vocab, self.tol = max_vocab, tol
#         self.register_buffer("vocab", torch.zeros(max_vocab))
#         self.register_buffer("n_vocab", torch.tensor(0))
#         self.emb = nn.Embedding(max_vocab + 1, d_out)   # last index = OOV

#     @torch.no_grad()
#     def fit(self, values, min_count=20):
#         v, c = torch.unique(values.reshape(-1).float(), return_counts=True)
#         v = v[c >= min_count]
#         if v.numel() > self.max_vocab:
#             v = v[torch.argsort(c[c >= min_count], descending=True)[: self.max_vocab]]
#         v = torch.sort(v).values
#         self.vocab.zero_()
#         self.vocab[: v.numel()] = v
#         self.n_vocab.fill_(v.numel())

#     def forward(self, x):
#         n = int(self.n_vocab)
#         vocab = self.vocab[:n]
#         idx = torch.searchsorted(vocab, x.contiguous()).clamp(max=max(n - 1, 0))
#         lo = (idx - 1).clamp(min=0)
#         near = torch.where((x - vocab[lo]).abs() < (x - vocab[idx]).abs(), lo, idx)
#         hit = (x - vocab[near]).abs() < self.tol
#         idx = torch.where(hit, near, torch.full_like(near, self.max_vocab))
#         return self.emb(idx)


# class InputEmbedding(nn.Module):
#     """(B, T, M) raw features -> (B, T, M, E) per-variable embeddings."""

#     def __init__(self, nvars, d_emb=8, n_freq=16, sigma=1.0,
#                  cat_col=None, max_vocab=512):
#         super().__init__()
#         self.nvars, self.cat_col = nvars, cat_col
#         self.num_cols = [i for i in range(nvars) if i != cat_col]
#         n_num = len(self.num_cols)
#         self.scaler = RobustSmoothScaler(n_num)
#         self.feat_scale = nn.Parameter(torch.ones(n_num))        # learnable scaling
#         self.pbld = PBLDEmbedding(n_num, n_freq=n_freq, d_out=d_emb, sigma=sigma)
#         self.cat = CategoryEmbedding(max_vocab, d_emb) if cat_col is not None else None

#     @torch.no_grad()
#     def fit(self, X):
#         self.scaler.fit(X[..., self.num_cols])
#         if self.cat is not None:
#             self.cat.fit(X[..., self.cat_col])

#     def forward(self, x):
#         # Keep embedding in fp32 under AMP: cos(2*pi*w*x+b) and exact arb_id
#         # matching are not accurate enough in bf16/fp16.
#         with torch.autocast(device_type=x.device.type, enabled=False):
#             return self._forward(x.float())

#     def _forward(self, x):
#         xn = self.scaler(x[..., self.num_cols]) * self.feat_scale
#         e_num = self.pbld(xn)                                     # (B, T, M_num, E)
#         if self.cat is None:
#             return e_num
#         e_cat = self.cat(x[..., self.cat_col]).unsqueeze(2)       # (B, T, 1, E)
#         # restore original column order
#         parts, j = [], 0
#         for i in range(self.nvars):
#             if i == self.cat_col:
#                 parts.append(e_cat)
#             else:
#                 parts.append(e_num[:, :, j:j + 1]); j += 1
#         return torch.cat(parts, dim=2)


# # ─────────────────────────── 3) TabM-style head ───────────────────────────

# class BELinear(nn.Module):
#     """BatchEnsemble linear: y_i = s_i * (W (r_i * x_i)) + b_i, W shared by k members."""

#     def __init__(self, d_in, d_out, k):
#         super().__init__()
#         self.weight = nn.Parameter(torch.empty(d_in, d_out))
#         nn.init.kaiming_uniform_(self.weight.T, a=math.sqrt(5))
#         # TabM-style init: random +-1 input adapters, unit output adapters
#         self.r = nn.Parameter(torch.randint(0, 2, (k, d_in)).float() * 2 - 1)
#         self.s = nn.Parameter(torch.ones(k, d_out))
#         bound = 1 / math.sqrt(d_in)
#         self.bias = nn.Parameter(torch.empty(k, d_out).uniform_(-bound, bound))

#     def forward(self, x):                       # x: (B, k, d_in)
#         return (x * self.r) @ self.weight * self.s + self.bias


# class EnsembleHead(nn.Module):
#     def __init__(self, d_in, num_classes, k=8, hidden=64, dropout=0.1):
#         super().__init__()
#         self.k = k
#         self.norm = nn.LayerNorm(d_in)
#         if hidden > 0:
#             self.l1 = BELinear(d_in, hidden, k)
#             self.act, self.drop = nn.GELU(), nn.Dropout(dropout)
#             self.l2 = BELinear(hidden, num_classes, k)
#         else:
#             self.l1 = None
#             self.l2 = BELinear(d_in, num_classes, k)

#     def forward(self, z):                       # z: (B, d_in)
#         z = self.norm(z).unsqueeze(1).expand(-1, self.k, -1)
#         if self.l1 is not None:
#             z = self.drop(self.act(self.l1(z)))
#         return self.l2(z)                       # (B, k, C)


# # ─────────────────────────── Full model ───────────────────────────

# class IVN_TCN(nn.Module):
#     def __init__(
#         self,
#         seq_len=64, nvars=12, num_classes=10,
#         # embedding
#         use_embedding=True, d_emb=8, n_freq=16, emb_sigma=1.0, cat_col=0, max_vocab=512,
#         # backbone
#         patch_size=8, patch_stride=4, dims=(64, 128), num_blocks=(1, 1),
#         large_sizes=(13, 13), small_size=5, ffn_ratio=2, dropout=0.1,
#         multi_scale=True, ms_kernel_sizes=(3, 5, 13), ms_fusion="channel",
#         # head
#         ens_k=8, head_hidden=64, head_dropout=0.1,
#     ):
#         super().__init__()
#         dims, num_blocks, large_sizes = list(dims), list(num_blocks), list(large_sizes)
#         self.nvars, self.num_stages = nvars, len(dims)
#         self.patch_size, self.patch_stride = patch_size, patch_stride

#         if use_embedding:
#             self.embed = InputEmbedding(nvars, d_emb, n_freq, emb_sigma, cat_col, max_vocab)
#             in_ch = d_emb
#         else:
#             self.embed, in_ch = None, 1

#         # Stem: per-variable patching, shared across variables (as in ModernTCN)
#         self.stem = nn.Sequential(
#             nn.Conv1d(in_ch, dims[0], kernel_size=patch_size, stride=patch_stride),
#             nn.BatchNorm1d(dims[0]),
#         )
#         self.downsample_ratio = 2
#         self.downsample_layers = nn.ModuleList([
#             nn.Sequential(nn.BatchNorm1d(dims[i]),
#                           nn.Conv1d(dims[i], dims[i + 1], 2, stride=2))
#             for i in range(self.num_stages - 1)
#         ])
#         self.stages = nn.ModuleList([
#             nn.ModuleList([
#                 ModernTCNBlock(nvars=nvars, dmodel=dims[i], ffn_ratio=ffn_ratio,
#                                large_size=large_sizes[i], small_size=small_size,
#                                dropout=dropout, multi_scale=multi_scale,
#                                ms_kernel_sizes=list(ms_kernel_sizes), ms_fusion=ms_fusion)
#                 for _ in range(num_blocks[i])
#             ])
#             for i in range(self.num_stages)
#         ])
#         self.head = EnsembleHead(nvars * dims[-1], num_classes,
#                                  k=ens_k, hidden=head_hidden, dropout=head_dropout)

#     @torch.no_grad()
#     def fit_input_stats(self, X_train):
#         """Fit robust scaler and arbitration ID vocabulary on training windows."""
#         if self.embed is not None:
#             self.embed.fit(X_train)

#     def forward(self, x):                                   # (B, T, M)
#         B, T, M = x.shape
#         if self.embed is not None:
#             e = self.embed(x)                               # (B, T, M, E)
#             e = e.permute(0, 2, 3, 1).reshape(B * M, -1, T) # (B*M, E, T)
#         else:
#             e = x.permute(0, 2, 1).reshape(B * M, 1, T)
#         if self.patch_size != self.patch_stride:
#             e = F.pad(e, (0, self.patch_size - self.patch_stride), mode="replicate")
#         h = self.stem(e)
#         _, D, N = h.shape
#         h = h.reshape(B, M, D, N)

#         for i in range(self.num_stages):
#             if i > 0:
#                 B, M, D, N = h.shape
#                 h = h.reshape(B * M, D, N)
#                 if N % self.downsample_ratio:
#                     h = F.pad(h, (0, self.downsample_ratio - N % self.downsample_ratio),
#                               mode="replicate")
#                 h = self.downsample_layers[i - 1](h)
#                 _, D, N = h.shape
#                 h = h.reshape(B, M, D, N)
#             for blk in self.stages[i]:
#                 h = blk(h)

#         z = h.mean(dim=-1).reshape(B, -1)                   # temporal mean pool -> (B, M*D)
#         return self.head(z)                                 # (B, k, C)

#     @staticmethod
#     def ensemble_proba(logits):
#         """Average member probabilities (TabM inference rule)."""
#         return logits.softmax(dim=-1).mean(dim=1)

#     def structural_reparam(self):
#         for m in self.modules():
#             if hasattr(m, "merge_kernel"):
#                 m.merge_kernel()

#     def count_parameters(self):
#         def n(mod): return sum(p.numel() for p in mod.parameters()) if mod is not None else 0
#         emb, head = n(self.embed), n(self.head)
#         total = sum(p.numel() for p in self.parameters())
#         return {"embed": emb, "backbone": total - emb - head, "head": head, "total": total}



# ─────────────────────────── V2 ───────────────────────────
"""
IVN-TCN: lightweight Multi-Scale ModernTCN for CAN bus IDS.

Three parts, each chosen for in-vehicle constraints (single forward pass,
small parameter count, low latency per window):

1) Input embedding (RealMLP-inspired, Holzmueller et al., NeurIPS 2024;
   Gorishniy et al., NeurIPS 2022)
   - Robust scaling (median / IQR fitted on train) + smooth clipping
   - Learnable per-variable scaling (soft feature selection)
   - PBLD embedding per numerical variable: cos(2*pi*w*x + b) -> Linear,
     concatenated with the scaled scalar itself (DenseNet-style passthrough)
   - Optional categorical embedding for arbitration ID, with an
     out-of-vocabulary token (unseen IDs are themselves an attack signal)
   - Optional ID-conditioned modulation (FiLM): the arbitration ID of each
     frame rescales and shifts the embedding of every numeric field, because
     payload and timing semantics depend on which ID sent the frame
   Every variable has its own embedding parameters (never shared), which
   matches the per-variable design of ModernTCN.

2) Backbone: Multi-Scale ModernTCN blocks (small dims, light fusion)

3) Head: temporal mean pooling + TabM-style BatchEnsemble MLP
   (Gorishniy et al., ICLR 2025). k members share weights and differ by
   rank-1 adapters (r, s, b). Ensembling is applied ONLY to the head, so
   the convolutional backbone runs once per window.

Input : (B, T, C)
Output: (B, k, num_classes) logits, one set per ensemble member
"""

from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from networks.moderntcn_can import ModernTCNBlock
except ImportError:
    from moderntcn_can import ModernTCNBlock


# ─────────────────────────── 1) Input embedding ───────────────────────────

class RobustSmoothScaler(nn.Module):
    """(x - median) / IQR, then smooth clip x / sqrt(1 + (x/c)^2). Fitted on train."""

    def __init__(self, nvars, clip=3.0):
        super().__init__()
        self.clip = clip
        self.register_buffer("center", torch.zeros(nvars))
        self.register_buffer("scale", torch.ones(nvars))

    @torch.no_grad()
    def fit(self, X, max_rows=500_000):
        x = X.reshape(-1, X.shape[-1]).float()
        if x.shape[0] > max_rows:
            idx = torch.randperm(x.shape[0])[:max_rows]
            x = x[idx]
        q = torch.quantile(x, torch.tensor([0.25, 0.5, 0.75]), dim=0)
        iqr = q[2] - q[0]
        std = x.std(dim=0)
        # Mostly-constant payload bytes have IQR = 0: fall back to std, then 1
        scale = torch.where(iqr > 1e-6, iqr, torch.where(std > 1e-6, std, torch.ones_like(std)))
        self.center.copy_(q[1])
        self.scale.copy_(scale)

    def forward(self, x):
        x = (x - self.center) / self.scale
        return x / torch.sqrt(1.0 + (x / self.clip) ** 2)


class PBLDEmbedding(nn.Module):
    """
    Periodic-Bias-Linear-DenseNet embedding, one independent module per variable
    (vectorized). x: (..., M) -> (..., M, d_out)
    """

    def __init__(self, nvars, n_freq=16, d_out=8, sigma=1.0):
        super().__init__()
        assert d_out >= 2
        self.w = nn.Parameter(torch.randn(nvars, n_freq) * sigma)
        self.b = nn.Parameter(torch.empty(nvars, n_freq).uniform_(-math.pi, math.pi))
        self.lin_w = nn.Parameter(torch.empty(nvars, n_freq, d_out - 1))
        self.lin_b = nn.Parameter(torch.zeros(nvars, d_out - 1))
        nn.init.uniform_(self.lin_w, -1 / math.sqrt(n_freq), 1 / math.sqrt(n_freq))

    def forward(self, x):
        z = torch.cos(2 * math.pi * x.unsqueeze(-1) * self.w + self.b)       # (..., M, K)
        z = torch.einsum("...mk,mkd->...md", z, self.lin_w) + self.lin_b      # (..., M, d-1)
        return torch.cat([x.unsqueeze(-1), z], dim=-1)                        # (..., M, d)


class CategoryEmbedding(nn.Module):
    """
    Map a float-encoded categorical column (arbitration ID) to indices by exact
    match, plus one out-of-vocabulary (OOV) token.

    Vocabulary = IDs seen in NORMAL training traffic. IDs injected by attacks
    (e.g. random IDs from fuzzing) are therefore OOV, which is itself a signal.
    The threshold is a fraction of frame positions, so overlapping windows (which
    repeat each frame many times) do not inflate rare IDs.
    """

    def __init__(self, max_vocab=512, d_out=8, tol=1e-5):
        super().__init__()
        self.max_vocab, self.tol = max_vocab, tol
        self.register_buffer("vocab", torch.zeros(max_vocab))
        self.register_buffer("n_vocab", torch.tensor(0))
        self.emb = nn.Embedding(max_vocab + 1, d_out)   # last index = OOV
        self.fit_info = {}

    @property
    def oov_index(self):
        return self.max_vocab

    @torch.no_grad()
    def fit(self, values, labels=None, normal_label=0, min_frac=1e-5):
        """values: (N, T) ID column of the train windows; labels: (N,) window labels."""
        if labels is not None:
            keep = labels.reshape(-1) == normal_label
            if keep.any():
                values = values[keep]
        flat = values.reshape(-1).float()
        v, c = torch.unique(flat, return_counts=True)
        min_count = max(1, int(min_frac * flat.numel()))
        ok = c >= min_count
        v, c = v[ok], c[ok]
        capped = v.numel() > self.max_vocab
        if capped:
            v = v[torch.argsort(c, descending=True)[: self.max_vocab]]
        v = torch.sort(v).values
        self.vocab.zero_()
        self.vocab[: v.numel()] = v
        self.n_vocab.fill_(v.numel())
        self.fit_info = {"n_vocab": int(v.numel()), "capped": bool(capped),
                         "min_count": min_count, "from_normal_only": labels is not None}

    def lookup(self, x):
        """Float IDs -> vocab indices; unseen IDs -> oov_index."""
        n = int(self.n_vocab)
        if n == 0:
            return torch.full(x.shape, self.oov_index, dtype=torch.long, device=x.device)
        vocab = self.vocab[:n]
        idx = torch.searchsorted(vocab, x.contiguous()).clamp(max=n - 1)
        lo = (idx - 1).clamp(min=0)
        near = torch.where((x - vocab[lo]).abs() < (x - vocab[idx]).abs(), lo, idx)
        hit = (x - vocab[near]).abs() < self.tol
        return torch.where(hit, near, torch.full_like(near, self.oov_index))

    def forward(self, x):
        return self.emb(self.lookup(x))


class InputEmbedding(nn.Module):
    """
    (B, T, M) raw features -> (B, T, M, E) per-variable embeddings.

    id_cond=True adds the ID-conditioned modulation: the arbitration-ID
    embedding e_id of each frame produces per-field FiLM parameters,
        z_j = (1 + dgamma_j(e_id)) * PBLD_j(x_j) + beta_j(e_id),
    so the same payload value is represented differently under different IDs.
    The FiLM layer is zero-initialized (starts identical to the unconditioned
    model) and OOV IDs get dgamma = beta = 0 (fall back to the shared embedding).
    """

    def __init__(self, nvars, d_emb=8, n_freq=16, sigma=1.0,
                 cat_col=None, max_vocab=512, id_cond=False):
        super().__init__()
        self.nvars, self.cat_col, self.d_emb = nvars, cat_col, d_emb
        self.num_cols = [i for i in range(nvars) if i != cat_col]
        n_num = self.n_num = len(self.num_cols)
        self.scaler = RobustSmoothScaler(n_num)
        self.feat_scale = nn.Parameter(torch.ones(n_num))        # learnable scaling
        self.pbld = PBLDEmbedding(n_num, n_freq=n_freq, d_out=d_emb, sigma=sigma)
        self.cat = CategoryEmbedding(max_vocab, d_emb) if cat_col is not None else None
        if id_cond and self.cat is None:
            raise ValueError("id_cond requires a categorical ID column (cat_col)")
        self.id_cond = id_cond
        if id_cond:
            self.film = nn.Linear(d_emb, n_num * 2 * d_emb)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    @torch.no_grad()
    def fit(self, X, y=None, vocab_min_frac=1e-5):
        self.scaler.fit(X[..., self.num_cols])
        if self.cat is not None:
            self.cat.fit(X[..., self.cat_col], labels=y, min_frac=vocab_min_frac)

    def forward(self, x):
        # Keep embedding in fp32 under AMP: cos(2*pi*w*x+b) and exact arb_id
        # matching are not accurate enough in bf16/fp16.
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self._forward(x.float())

    def _forward(self, x):
        xn = self.scaler(x[..., self.num_cols]) * self.feat_scale
        e_num = self.pbld(xn)                                     # (B, T, M_num, E)
        if self.cat is None:
            return e_num

        idx = self.cat.lookup(x[..., self.cat_col])               # (B, T)
        e_id = self.cat.emb(idx)                                  # (B, T, E)

        if self.id_cond:
            B, T = idx.shape
            f = self.film(e_id).view(B, T, self.n_num, 2, self.d_emb)
            known = (idx != self.cat.oov_index).to(f.dtype)[..., None, None, None]
            f = f * known                                         # OOV -> dgamma = beta = 0
            e_num = (1.0 + f[..., 0, :]) * e_num + f[..., 1, :]

        e_cat = e_id.unsqueeze(2)                                 # (B, T, 1, E)
        parts, j = [], 0
        for i in range(self.nvars):                               # restore column order
            if i == self.cat_col:
                parts.append(e_cat)
            else:
                parts.append(e_num[:, :, j:j + 1]); j += 1
        return torch.cat(parts, dim=2)


# ─────────────────────────── 3) TabM-style head ───────────────────────────

class BELinear(nn.Module):
    """BatchEnsemble linear: y_i = s_i * (W (r_i * x_i)) + b_i, W shared by k members."""

    def __init__(self, d_in, d_out, k):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_in, d_out))
        nn.init.kaiming_uniform_(self.weight.T, a=math.sqrt(5))
        # TabM-style init: random +-1 input adapters, unit output adapters
        self.r = nn.Parameter(torch.randint(0, 2, (k, d_in)).float() * 2 - 1)
        self.s = nn.Parameter(torch.ones(k, d_out))
        bound = 1 / math.sqrt(d_in)
        self.bias = nn.Parameter(torch.empty(k, d_out).uniform_(-bound, bound))

    def forward(self, x):                       # x: (B, k, d_in)
        return (x * self.r) @ self.weight * self.s + self.bias


class EnsembleHead(nn.Module):
    def __init__(self, d_in, num_classes, k=8, hidden=64, dropout=0.1):
        super().__init__()
        self.k = k
        self.norm = nn.LayerNorm(d_in)
        if hidden > 0:
            self.l1 = BELinear(d_in, hidden, k)
            self.act, self.drop = nn.GELU(), nn.Dropout(dropout)
            self.l2 = BELinear(hidden, num_classes, k)
        else:
            self.l1 = None
            self.l2 = BELinear(d_in, num_classes, k)

    def forward(self, z):                       # z: (B, d_in)
        z = self.norm(z).unsqueeze(1).expand(-1, self.k, -1)
        if self.l1 is not None:
            z = self.drop(self.act(self.l1(z)))
        return self.l2(z)                       # (B, k, C)


# ─────────────────────────── Full model ───────────────────────────

class IVN_TCN(nn.Module):
    def __init__(
        self,
        seq_len=64, nvars=12, num_classes=10,
        # embedding
        use_embedding=True, d_emb=8, n_freq=16, emb_sigma=1.0, cat_col=0, max_vocab=512,
        id_cond=False,
        # backbone
        patch_size=8, patch_stride=4, dims=(64, 128), num_blocks=(1, 1),
        large_sizes=(13, 13), small_size=5, ffn_ratio=2, dropout=0.1,
        multi_scale=True, ms_kernel_sizes=(3, 5, 13), ms_fusion="channel",
        # head
        ens_k=8, head_hidden=64, head_dropout=0.1,
    ):
        super().__init__()
        dims, num_blocks, large_sizes = list(dims), list(num_blocks), list(large_sizes)
        self.nvars, self.num_stages = nvars, len(dims)
        self.patch_size, self.patch_stride = patch_size, patch_stride

        if use_embedding:
            self.embed = InputEmbedding(nvars, d_emb, n_freq, emb_sigma, cat_col, max_vocab,
                                        id_cond=id_cond)
            in_ch = d_emb
        else:
            self.embed, in_ch = None, 1

        # Stem: per-variable patching, shared across variables (as in ModernTCN)
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, dims[0], kernel_size=patch_size, stride=patch_stride),
            nn.BatchNorm1d(dims[0]),
        )
        self.downsample_ratio = 2
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(nn.BatchNorm1d(dims[i]),
                          nn.Conv1d(dims[i], dims[i + 1], 2, stride=2))
            for i in range(self.num_stages - 1)
        ])
        self.stages = nn.ModuleList([
            nn.ModuleList([
                ModernTCNBlock(nvars=nvars, dmodel=dims[i], ffn_ratio=ffn_ratio,
                               large_size=large_sizes[i], small_size=small_size,
                               dropout=dropout, multi_scale=multi_scale,
                               ms_kernel_sizes=list(ms_kernel_sizes), ms_fusion=ms_fusion)
                for _ in range(num_blocks[i])
            ])
            for i in range(self.num_stages)
        ])
        self.head = EnsembleHead(nvars * dims[-1], num_classes,
                                 k=ens_k, hidden=head_hidden, dropout=head_dropout)

    @torch.no_grad()
    def fit_input_stats(self, X_train, y_train=None, vocab_min_frac=1e-5):
        """Fit robust scaler and arbitration-ID vocabulary on training windows.
        With y_train, the vocabulary comes from normal windows only."""
        if self.embed is not None:
            self.embed.fit(X_train, y_train, vocab_min_frac)

    def forward(self, x):                                   # (B, T, M)
        B, T, M = x.shape
        if self.embed is not None:
            e = self.embed(x)                               # (B, T, M, E)
            e = e.permute(0, 2, 3, 1).reshape(B * M, -1, T) # (B*M, E, T)
        else:
            e = x.permute(0, 2, 1).reshape(B * M, 1, T)
        if self.patch_size != self.patch_stride:
            e = F.pad(e, (0, self.patch_size - self.patch_stride), mode="replicate")
        h = self.stem(e)
        _, D, N = h.shape
        h = h.reshape(B, M, D, N)

        for i in range(self.num_stages):
            if i > 0:
                B, M, D, N = h.shape
                h = h.reshape(B * M, D, N)
                if N % self.downsample_ratio:
                    h = F.pad(h, (0, self.downsample_ratio - N % self.downsample_ratio),
                              mode="replicate")
                h = self.downsample_layers[i - 1](h)
                _, D, N = h.shape
                h = h.reshape(B, M, D, N)
            for blk in self.stages[i]:
                h = blk(h)

        z = h.mean(dim=-1).reshape(B, -1)                   # temporal mean pool -> (B, M*D)
        return self.head(z)                                 # (B, k, C)

    @staticmethod
    def ensemble_proba(logits):
        """Average member probabilities (TabM inference rule)."""
        return logits.softmax(dim=-1).mean(dim=1)

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, "merge_kernel"):
                m.merge_kernel()

    def count_parameters(self):
        def n(mod): return sum(p.numel() for p in mod.parameters()) if mod is not None else 0
        emb, head = n(self.embed), n(self.head)
        total = sum(p.numel() for p in self.parameters())
        return {"embed": emb, "backbone": total - emb - head, "head": head, "total": total}