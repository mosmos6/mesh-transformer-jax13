# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


# -----------------------------
# Helper: Layer configuration
# -----------------------------
@dataclass
class LayerCfg:
    d_model: int
    n_heads: int
    d_head: int
    pe: str = "rotary"
    pe_rotary_dims: int = 64


# -----------------------------
# LayerNorm（leafspec 準拠: scale/offset）
#   /.../norm/{scale,offset}
#   /proj/ReplicatedLayerNorm_0/{scale,offset}
# -----------------------------
class LayerNorm(nn.Module):
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        d = x.shape[-1]
        scale = self.param("scale", nn.initializers.ones, (d,))
        offset = self.param("offset", nn.initializers.zeros, (d,))
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        xhat = (x - mean) * jax.lax.rsqrt(var + self.epsilon)
        return xhat * scale + offset


# -----------------------------
# Dense（直下に "kernel" / "bias" を持つ）
#  ※ nn.Dense を内側に作らない（.../q/q を防ぐ）
# -----------------------------
class DenseNoBias(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        in_features = x.shape[-1]
        kernel = self.param("kernel", nn.initializers.lecun_normal(), (in_features, self.features))
        return jnp.dot(x, kernel)


class DenseBias(nn.Module):
    features: int
    compute_fp32: bool = False  # MLP で True、QKV/O/LM-Head は False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        in_features = x.shape[-1]
        kernel = self.param("kernel", nn.initializers.lecun_normal(), (in_features, self.features))
        bias = self.param("bias", nn.initializers.zeros, (self.features,))
        if self.compute_fp32:
            y = jnp.dot(x.astype(jnp.float32), kernel.astype(jnp.float32)) + bias.astype(jnp.float32)
            return y.astype(x.dtype)
        else:
            return jnp.dot(x, kernel) + bias


# -----------------------------
# Embedding: /embed/embed_layer/embedding
# -----------------------------
class _EmbedLayer(nn.Module):
    n_vocab: int
    d_model: int

    @nn.compact
    def __call__(self, ids: jnp.ndarray) -> jnp.ndarray:
        emb = self.param("embedding", nn.initializers.normal(stddev=1.0), (self.n_vocab, self.d_model))
        return jnp.take(emb, ids, axis=0)


class EmbedBlock(nn.Module):
    n_vocab: int
    d_model: int

    @nn.compact
    def __call__(self, ids: jnp.ndarray) -> jnp.ndarray:
        # サブスコープを leafspec に合わせる
        return _EmbedLayer(self.n_vocab, self.d_model, name="embed_layer")(ids)


# -----------------------------
# RoPE helpers
# -----------------------------
def _rope_angles(rotary_dims: int, seq_len: int, dtype=jnp.float32) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # out shapes: (1, seq_len, 1, rotary_dims//2)
    half = rotary_dims // 2
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, half, dtype=dtype) / half))
    t = jnp.arange(seq_len, dtype=dtype)
    angles = jnp.einsum("t,f->tf", t, inv_freq)  # (T, F)
    cos = jnp.cos(angles)[None, :, None, :]  # (1,T,1,half)
    sin = jnp.sin(angles)[None, :, None, :]  # (1,T,1,half)
    return cos, sin


def _rope_apply_pair(qBTHD: jnp.ndarray, kBTHD: jnp.ndarray, rotary_dims: int, offset) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # q,k: (B,T,H,Dh)  /  offset: 動的（tracer）OK
    if rotary_dims == 0:
        return qBTHD, kBTHD
    B, T, H, Dh = qBTHD.shape
    rot = min(rotary_dims, Dh)
    cos, sin = _rope_angles(rot, T, dtype=jnp.float32)
    # 動的 offset → 常に roll（offset=0 なら恒等）
    cos = jnp.roll(cos, shift=offset, axis=1)
    sin = jnp.roll(sin, shift=offset, axis=1)

    def _apply(xBTHD):
        x1 = xBTHD[..., :rot]
        x2 = xBTHD[..., rot:]
        x1a, x1b = jnp.split(x1, 2, axis=-1)  # (..., half)
        xr1 = x1a * cos - x1b * sin
        xr2 = x1a * sin + x1b * cos
        return jnp.concatenate([jnp.concatenate([xr1, xr2], axis=-1), x2], axis=-1)

    return _apply(qBTHD), _apply(kBTHD)


def split_heads(xBTD: jnp.ndarray, H: int) -> jnp.ndarray:
    B, T, D = xBTD.shape
    Dh = D // H
    return xBTD.reshape(B, T, H, Dh)


def merge_heads(xBTHD: jnp.ndarray) -> jnp.ndarray:
    B, T, H, Dh = xBTHD.shape
    return xBTHD.reshape(B, T, H * Dh)


# -----------------------------
# Transformer layer
# -----------------------------
class TransformerLayerShard(nn.Module):
    cfg: LayerCfg

    def setup(self):
        D = self.cfg.d_model
        # Attn projections (bias なし) — leafspec: /{q,k,v,o}/kernel
        self.q = DenseNoBias(D)            # name は属性名（q）
        self.k = DenseNoBias(D)
        self.v = DenseNoBias(D)
        self.o = DenseNoBias(D)
        # FFN (bias あり) — leafspec: /{dense_proj,dense_proj_o}/{kernel,bias}
        self.dense_proj   = DenseBias(4 * D, compute_fp32=True)
        self.dense_proj_o = DenseBias(D,     compute_fp32=True)
        # LN — leafspec: /norm/{scale,offset}
        self.norm = LayerNorm(name="norm")

    # --- attention (full) ---
    def attend_full(self, xBTD: jnp.ndarray) -> jnp.ndarray:
        B, T, D = xBTD.shape
        H = self.cfg.n_heads
        Dh = self.cfg.d_head

        qBTD = self.q(xBTD)
        kBTD = self.k(xBTD)
        vBTD = self.v(xBTD)

        qBTHD = split_heads(qBTD, H)
        kBTHD = split_heads(kBTD, H)
        vBTHD = split_heads(vBTD, H)

        # RoPE
        if self.cfg.pe == "rotary":
            qBTHD, kBTHD = _rope_apply_pair(qBTHD, kBTHD, rotary_dims=self.cfg.pe_rotary_dims, offset=0)

        # QK^T (f32) + causal mask
        qf = qBTHD.astype(jnp.float32)
        kf = kBTHD.astype(jnp.float32)
        logits = jnp.einsum("bthd,bThd->bhtT", qf, kf) * (1.0 / jnp.sqrt(float(Dh)))
        mask = jnp.tril(jnp.ones((T, T), dtype=logits.dtype))
        logits = logits + (1.0 - mask)[None, None, :, :] * jnp.array(-1e10, dtype=logits.dtype)
        probs = jax.nn.softmax(logits, axis=-1)

        yBTHD = jnp.einsum("bhtT,bThd->bthd", probs, vBTHD.astype(jnp.float32))
        yBTD = merge_heads(yBTHD).astype(xBTD.dtype)
        return self.o(yBTD)

    # --- FFN ---
    def mlp(self, xBTD: jnp.ndarray) -> jnp.ndarray:
        h = self.dense_proj(xBTD)      # f32 演算（出力は元 dtype）
        h = jax.nn.gelu(h).astype(xBTD.dtype)
        y = self.dense_proj_o(h)       # f32 演算（出力は元 dtype）
        return y

    # --- decode once (KV 使用) ---
    def decode_once(self, decode_state: dict, xB1D: jnp.ndarray, attn_bias: Optional[jnp.ndarray] = None):
        """入力1トークンで1ステップ前進
        decode_state: {"k": (T,B,H,Dh), "v": (T,B,H,Dh), "cur_index": int32[]}
        返り値: (deltaB1D, new_state)
        """
        B, one, D = xB1D.shape
        assert one == 1
        H = self.cfg.n_heads
        Dh = self.cfg.d_head

        kTBHD = decode_state["k"]
        vTBHD = decode_state["v"]
        cur = jnp.asarray(decode_state["cur_index"], dtype=jnp.int32)  # tracer OK

        qB1D = self.q(xB1D)
        kB1D = self.k(xB1D)
        vB1D = self.v(xB1D)

        qB1HD = split_heads(qB1D, H)     # (B,1,H,Dh)
        kB1HD = split_heads(kB1D, H)
        vB1HD = split_heads(vB1D, H)

        # RoPE: offset=cur（動的）
        if self.cfg.pe == "rotary":
            qB1HD, kB1HD = _rope_apply_pair(qB1HD, kB1HD, rotary_dims=self.cfg.pe_rotary_dims, offset=cur)

        # KV 更新（動的インデックス）
        kTBHD = kTBHD.at[cur].set(kB1HD[0])
        vTBHD = vTBHD.at[cur].set(vB1HD[0])

        # QK^T（0..cur）
        qf = qB1HD.astype(jnp.float32)
        kf = kTBHD.astype(jnp.float32)
        logits = jnp.einsum("bshd,tbhd->bhts", qf, kf)[:, :, :, 0]  # (B=1,H,T)
        logits = logits * (1.0 / jnp.sqrt(float(Dh)))

        T = kTBHD.shape[0]
        arangeT = jnp.arange(T, dtype=jnp.int32)
        valid = (arangeT <= cur)
        logits = logits + (1.0 - valid.astype(logits.dtype))[None, None, :] * jnp.array(-1e10, dtype=logits.dtype)

        probs = jax.nn.softmax(logits, axis=-1)  # (B=1,H,T)

        vf = vTBHD.astype(jnp.float32)
        yB1HD = jnp.einsum("bht,tbhd->bhd", probs, vf)[:, None, :, :]  # (B,1,H,Dh)
        yB1D = merge_heads(yB1HD).astype(xB1D.dtype)

        out = self.o(yB1D)
        new_state = {"k": kTBHD, "v": vTBHD, "cur_index": cur + jnp.array(1, dtype=cur.dtype)}
        return out, new_state


# -----------------------------
# Projection block (final LN + LM head)
#   /proj/ReplicatedLayerNorm_0/{scale,offset}
#   /proj/Dense_0/{kernel,bias}
# -----------------------------
class ProjectionShard(nn.Module):
    n_vocab: int
    d_model: int

    def setup(self):
        self.norm = LayerNorm(name="ReplicatedLayerNorm_0")
        # LM-Head: DenseBias を "Dense_0" という名前の submodule にする
        self.lm_head = DenseBias(self.n_vocab, compute_fp32=False, name="Dense_0")

    def __call__(self, xBTD: jnp.ndarray) -> jnp.ndarray:
        xn = self.norm(xBTD)
        logits = self.lm_head(xn)  # (B,T,V)
        return logits


__all__ = [
    "LayerCfg",
    "LayerNorm",
    "DenseNoBias",
    "DenseBias",
    "EmbedBlock",
    "TransformerLayer",
    "ProjectionShard",
    "split_heads",
    "merge_heads",
]
