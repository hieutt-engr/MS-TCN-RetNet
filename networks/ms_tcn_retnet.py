"""
MS-TCN-RetNet for IVN IDS
---------------------------------
A practical TensorFlow/Keras implementation of the model design discussed:
    Input -> Projection -> Multi-scale Temporal Conv -> Residual Dilated TCN
          -> Retention Blocks -> Pooling -> Head

Notes
-----
1) This file implements a *practical retention-style block* for Keras.
   It is inspired by RetNet-style decayed causal retention, but it is not
   an official reproduction of the original RetNet paper.
2) The code supports two modes:
   - "regression": next-step / target prediction
   - "embedding": feature extraction for open-set / metric-learning pipelines
3) For anomaly detection like DCATCN, start with mode="regression" and use
   residual-based scoring outside this model.

"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models


# ============================================================
# Utility layers
# ============================================================

class GLU(layers.Layer):
    """Gated Linear Unit."""
    def call(self, x):
        a, b = tf.split(x, 2, axis=-1)
        return a * tf.nn.sigmoid(b)


class FeedForward(layers.Layer):
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1, name: Optional[str] = None):
        super().__init__(name=name)
        hidden = dim * expansion
        self.fc1 = layers.Dense(hidden * 2, use_bias=True)
        self.glu = GLU()
        self.drop1 = layers.Dropout(dropout)
        self.fc2 = layers.Dense(dim, use_bias=True)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training: bool = False):
        x = self.fc1(x)
        x = self.glu(x)
        x = self.drop1(x, training=training)
        x = self.fc2(x)
        x = self.drop2(x, training=training)
        return x


# ============================================================
# Multi-scale temporal convolution front-end
# ============================================================

class MultiScaleTemporalConv(layers.Layer):
    """
    Parallel convolution branches for short- and mid-range temporal patterns.
    """
    def __init__(
        self,
        branch_dim: int = 32,
        dropout: float = 0.1,
        activation: str = "gelu",
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self.b1 = layers.SeparableConv1D(branch_dim, kernel_size=3, dilation_rate=1, padding="same")
        self.b2 = layers.SeparableConv1D(branch_dim, kernel_size=3, dilation_rate=2, padding="same")
        self.b3 = layers.SeparableConv1D(branch_dim, kernel_size=5, dilation_rate=1, padding="same")

        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.act = layers.Activation(activation)
        self.drop = layers.Dropout(dropout)

    def call(self, x, training: bool = False):
        z1 = self.b1(x)
        z2 = self.b2(x)
        z3 = self.b3(x)
        z = tf.concat([z1, z2, z3], axis=-1)
        z = self.norm(z)
        z = self.act(z)
        z = self.drop(z, training=training)
        return z


class ResidualTCNBlock(layers.Layer):
    """
    Two-layer dilated causal TCN block with residual projection.
    """
    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        dilation_rate: int = 1,
        dropout: float = 0.1,
        activation: str = "gelu",
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.conv1 = layers.SeparableConv1D(
            dim, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="causal"
        )
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.act1 = layers.Activation(activation)
        self.drop1 = layers.Dropout(dropout)

        self.conv2 = layers.SeparableConv1D(
            dim, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="causal"
        )
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.act2 = layers.Activation(activation)
        self.drop2 = layers.Dropout(dropout)

        self.res_proj = layers.Conv1D(dim, kernel_size=1, padding="same")
        self.out_norm = layers.LayerNormalization(epsilon=1e-6)
        self.out_act = layers.Activation(activation)

    def call(self, x, training: bool = False):
        z = self.conv1(x)
        z = self.norm1(z)
        z = self.act1(z)
        z = self.drop1(z, training=training)

        z = self.conv2(z)
        z = self.norm2(z)
        z = self.act2(z)
        z = self.drop2(z, training=training)

        res = self.res_proj(x)
        out = z + res
        out = self.out_norm(out)
        out = self.out_act(out)
        return out


# ============================================================
# Retention-style module
# ============================================================

class MultiScaleRetention(layers.Layer):
    """
    A practical, retention-inspired causal temporal aggregation layer.

    It uses:
    - learned Q/K/V projections
    - causal masking
    - per-head exponential distance decay

    Complexity is O(T^2), which is acceptable for moderate window sizes.
    This is intentionally simple and robust for research prototyping.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_output_proj: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = layers.Dense(dim, use_bias=False)
        self.k_proj = layers.Dense(dim, use_bias=False)
        self.v_proj = layers.Dense(dim, use_bias=False)

        # Decay parameters are learned in unconstrained space, then squashed to (0,1).
        self.decay_logits = tf.Variable(
            initial_value=tf.linspace(0.0, 2.0, num_heads),
            trainable=True,
            name=f"{self.name or 'retention'}_decay_logits",
            dtype=tf.float32,
        )

        self.attn_drop = layers.Dropout(dropout)
        self.use_output_proj = use_output_proj
        self.out_proj = layers.Dense(dim, use_bias=False) if use_output_proj else None

    def _reshape_heads(self, x: tf.Tensor) -> tf.Tensor:
        # (B, T, D) -> (B, H, T, Hd)
        b = tf.shape(x)[0]
        t = tf.shape(x)[1]
        x = tf.reshape(x, [b, t, self.num_heads, self.head_dim])
        x = tf.transpose(x, [0, 2, 1, 3])
        return x

    def call(self, x, training: bool = False):
        # x: (B, T, D)
        q = self._reshape_heads(self.q_proj(x))
        k = self._reshape_heads(self.k_proj(x))
        v = self._reshape_heads(self.v_proj(x))

        # Dot-product scores: (B, H, T, T)
        scores = tf.matmul(q, k, transpose_b=True) * self.scale

        # Causal mask
        t = tf.shape(x)[1]
        causal = tf.linalg.band_part(tf.ones((t, t), dtype=tf.float32), -1, 0)  # lower triangular
        causal = tf.reshape(causal, [1, 1, t, t])

        # Distance decay mask
        idx = tf.range(t)
        dist = tf.cast(idx[:, None] - idx[None, :], tf.float32)  # (T, T)
        dist = tf.maximum(dist, 0.0)  # only past distances for causal side
        gammas = tf.sigmoid(self.decay_logits)  # (H,) in (0,1)
        gammas = tf.clip_by_value(gammas, 1e-4, 1.0 - 1e-4)
        log_gammas = tf.math.log(gammas)  # negative values mostly
        decay_bias = tf.reshape(log_gammas, [1, self.num_heads, 1, 1]) * tf.reshape(dist, [1, 1, t, t])

        # Apply causal mask and decay
        minus_inf = tf.constant(-1e9, dtype=scores.dtype)
        scores = scores + decay_bias
        scores = tf.where(causal > 0, scores, minus_inf)

        attn = tf.nn.softmax(scores, axis=-1)
        attn = self.attn_drop(attn, training=training)

        out = tf.matmul(attn, v)  # (B, H, T, Hd)
        out = tf.transpose(out, [0, 2, 1, 3])  # (B, T, H, Hd)
        out = tf.reshape(out, [tf.shape(x)[0], t, self.dim])

        if self.out_proj is not None:
            out = self.out_proj(out)

        return out


class RetentiveBlock(layers.Layer):
    """
    Pre-norm retention block:
        x = x + Retention(LN(x))
        x = x + FFN(LN(x))
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        ff_expansion: int = 4,
        dropout: float = 0.1,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.retention = MultiScaleRetention(dim=dim, num_heads=num_heads, dropout=dropout)
        self.drop1 = layers.Dropout(dropout)

        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = FeedForward(dim=dim, expansion=ff_expansion, dropout=dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training: bool = False):
        z = self.retention(self.norm1(x), training=training)
        x = x + self.drop1(z, training=training)

        z = self.ffn(self.norm2(x), training=training)
        x = x + self.drop2(z, training=training)
        return x


# ============================================================
# Model builder
# ============================================================

def build_ms_tcn_retnet(
    input_shape: Tuple[int, int],
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
    pool: str = "gap",
    mode: str = "regression",
    embedding_dim: int = 128,
    output_dim: int = 1,
    name: str = "MS_TCN_RetNet",
) -> tf.keras.Model:
    """
    Build the proposed MS-TCN-RetNet model.

    Parameters
    ----------
    input_shape : (T, C)
        Sequence length and input feature dimension.
    mode : {"regression", "embedding"}
        regression -> Dense(output_dim) output
        embedding  -> Dense(embedding_dim) + L2 normalize
    """
    if mode not in {"regression", "embedding"}:
        raise ValueError("mode must be either 'regression' or 'embedding'")
    if pool not in {"gap", "gmp"}:
        raise ValueError("pool must be either 'gap' or 'gmp'")

    inputs = layers.Input(shape=input_shape, name="input")

    # 1) Input projection
    x = layers.Dense(hidden_dim, name="input_projection")(inputs)
    x = layers.LayerNormalization(epsilon=1e-6, name="input_projection_norm")(x)

    # 2) Multi-scale conv front-end
    x = MultiScaleTemporalConv(branch_dim=branch_dim, dropout=tcn_dropout, name="ms_temporal_conv")(x)

    # Project concatenated branches to TCN dim
    x = layers.Conv1D(tcn_dim, kernel_size=1, padding="same", name="ms_proj")(x)
    x = layers.LayerNormalization(epsilon=1e-6, name="ms_proj_norm")(x)

    # 3) Residual dilated TCN stack
    for i in range(tcn_depth):
        x = ResidualTCNBlock(
            dim=tcn_dim,
            kernel_size=tcn_kernel_size,
            dilation_rate=2 ** i,
            dropout=tcn_dropout,
            name=f"tcn_block_{i + 1}",
        )(x)

    # Align dimension for retention stack
    if tcn_dim != retention_dim:
        x = layers.Conv1D(retention_dim, kernel_size=1, padding="same", name="to_retention_dim")(x)
        x = layers.LayerNormalization(epsilon=1e-6, name="to_retention_dim_norm")(x)

    # 4) Retentive blocks
    for i in range(retention_depth):
        x = RetentiveBlock(
            dim=retention_dim,
            num_heads=retention_heads,
            ff_expansion=ff_expansion,
            dropout=retention_dropout,
            name=f"retentive_block_{i + 1}",
        )(x)

    # 5) Pooling
    if pool == "gap":
        z = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    else:
        z = layers.GlobalMaxPooling1D(name="global_max_pool")(x)

    z = layers.LayerNormalization(epsilon=1e-6, name="pooled_norm")(z)
    z = layers.Dropout(retention_dropout, name="pooled_dropout")(z)

    # 6) Head
    if mode == "regression":
        outputs = layers.Dense(output_dim, name="regression_head")(z)
    else:
        z = layers.Dense(embedding_dim, activation="gelu", name="embedding_fc")(z)
        outputs = tf.nn.l2_normalize(z, axis=-1, name="embedding_head")

    model = models.Model(inputs=inputs, outputs=outputs, name=name)
    return model


# ============================================================
# Optional compile helper
# ============================================================

def compile_for_regression(
    model: tf.keras.Model,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    loss: str = "mse",
) -> tf.keras.Model:
    """
    Compile model for next-step / target prediction.
    """
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=["mae"])
    return model


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    # Example 1: residual-based anomaly detection / regression
    model = build_ms_tcn_retnet(
        input_shape=(64, 16),   # T=64, C=16
        hidden_dim=64,
        branch_dim=32,
        tcn_dim=64,
        tcn_depth=2,
        retention_dim=64,
        retention_depth=2,
        retention_heads=4,
        mode="regression",
        output_dim=1,
    )
    model = compile_for_regression(model)
    model.summary()

    # Example 2: embedding extractor for open-set / metric learning
    emb_model = build_ms_tcn_retnet(
        input_shape=(64, 16),
        hidden_dim=64,
        branch_dim=32,
        tcn_dim=64,
        tcn_depth=2,
        retention_dim=64,
        retention_depth=2,
        retention_heads=4,
        mode="embedding",
        embedding_dim=128,
    )
    emb_model.summary()
