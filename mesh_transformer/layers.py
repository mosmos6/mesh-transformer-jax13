# mesh_transformer/layers.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


# --------------------------
# Utility: LayerNorm (scale/offset 命名)
# --------------------------
class ReplicatedLayerNorm(nn.Module):
    """Param 名を leafspec に合わせる: {scale, offset}。eps は標準値。"""
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        d = x.shape[-1]
        scale = self.param("scale", nn.initializers.ones, (d,))
        offset = self.param("offset", nn.initializers.zeros, (d,))
        x32 = x.astype(jnp.float32)
        mean = jnp.mean(x32, axis=-1, keepdims=True)
        var = jnp.var(x32, axis=-1, keepdims=True)
        y = (x32 - mean) * jax.lax.rsqrt(var + self.eps)
        y = y * scale.astype(jnp.float32) + offset.astype(jnp.float32)
        return y.astype(x.dtype)


# --------------------------
# Rotary Positional Embedding (RoPE=64)
# --------------------------
def _rope_rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    # (..., rot/2, 2) → (..., rot)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return jnp.stack((-x2, x1), axis=-1).reshape(x.shape)

def apply_rope(q: jnp.ndarray,
               k: jnp.ndarray,
               pos: jnp.ndarray,
               rotary_dim: int,
               base: float = 10000.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    q,k: (B,T,H,Dh) または (B,1,H,Dh)
    pos: 位置インデックス int32（T,）または（1,）
    rotary_dim: 先頭 Dh のうち適用する次元（GPT‑J: 64）
    """
    Dh = q.shape[-1]
    rot = rotary_dim
    assert rot <= Dh and (rot % 2 == 0), "rotary_dim must be even and <= d_head"

    # inv_freq: (rot/2,)
    inv_freq = 1.0 / (base ** (jnp.arange(0, rot, 2, dtype=jnp.float32) / rot))  # (rot/2,)

    # pos: (..., 1) として outer
    pos = pos.astype(jnp.float32).reshape((-1, 1))
    # (..., rot/2)
    freqs = pos * inv_freq[None, :]
    # (..., rot)
    cos = jnp.repeat(jnp.cos(freqs), 2, axis=-1)
    sin = jnp.repeat(jnp.sin(freqs), 2, axis=-1)

    # ブロードキャスト: T or 1 を B,H 軸へ
    while cos.ndim < q.ndim:
        cos = cos[:, None, ...]
        sin = sin[:, None, ...]

    def _apply(x):
        x_rot = x[..., :rot]
        x_pass = x[..., rot:]
        x_rot = x_rot * cos + _rope_rotate_half(x_rot) * sin
        return jnp.concatenate([x_rot, x_pass], axis=-1)

    return _apply(q), _apply(k)


# --------------------------
# Dense (bias 有/無) ただし名前は leafspec に合わせる
# --------------------------
class DenseNoBias(nn.Module):
    features: int
    name: Optional[str] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.Dense(self.features, use_bias=False, name=self.name)(x)

class DenseBias(nn.Module):
    features: int
    name: Optional[str] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.Dense(self.features, use_bias=True, name=self.name)(x)


# --------------------------
# Embedding (/embed/embed_layer/embedding)
# --------------------------
class _EmbedLayer(nn.Module):
    n_vocab: int
    d_model: int

    @nn.compact
    def __call__(self, ids: jnp.ndarray) -> jnp.ndarray:
        # ids: (B,T) または (T,) または (1,)
        emb = self.param("embedding", nn.initializers.normal(stddev=0.02),
                         (self.n_vocab, self.d_model))
        x = jnp.take(emb, ids, axis=0)
        return x

class EmbedBlock(nn.Module):
    n_vocab: int
    d_model: int

    @nn.compact
    def __call__(self, ids: jnp.ndarray) -> jnp.ndarray:
        # 名前を leafspec に合わせるため 2 段に分ける
        embed_layer = _EmbedLayer(self.n_vocab, self.d_model, name="embed_layer")
        return embed_layer(ids)


# --------------------------
# Transformer Layer (pre‑LN, 自己注意 + MLP)
#   params path:
#     /transformer_layers_{i}/norm/{scale,offset}
#     /transformer_layers_{i}/{q,k,v,o}/kernel
#     /transformer_layers_{i}/dense_proj{,_o}/{kernel,bias}
# --------------------------
@dataclass
class LayerCfg:
    d_model: int
    n_heads: int
    d_head: int
    rotary_dim: int

class TransformerLayerShard(nn.Module):
    cfg: LayerCfg
    name: Optional[str] = None  # 外側で transformer_layers_i 指定

    def setup(self):
        D = self.cfg.d_model
        H = self.cfg.n_heads
        # pre-LN
        self.norm = ReplicatedLayerNorm(name="norm")
        # Attention (bias なし)
        self.q = DenseNoBias(D, name="q")
        self.k = DenseNoBias(D, name="k")
        self.v = DenseNoBias(D, name="v")
        self.o = DenseNoBias(D, name="o")
        # MLP (bias あり)
        self.ff1 = DenseBias(4 * D, name="dense_proj")
        self.ff2 = DenseBias(D, name="dense_proj_o")

    # ---- full-seq (context) ----
    def attend_full(self, xBTD: jnp.ndarray) -> jnp.ndarray:
        """x:(B,T,D) -> (B,T,D), 事前に pre-LN 済み想定"""
        B, T, D = xBTD.shape
        H = self.cfg.n_heads
        Dh = self.cfg.d_head
        rot = self.cfg.rotary_dim

        # Q,K,V: (B,T,D) -> (B,T,H,Dh)
        qBTD = self.q(xBTD)
        kBTD = self.k(xBTD)
        vBTD = self.v(xBTD)
        qBTHD = qBTD.reshape(B, T, H, Dh)
        kBTHD = kBTD.reshape(B, T, H, Dh)
        vBTHD = vBTD.reshape(B, T, H, Dh)

        # RoPE（位置 0..T-1）
        pos = jnp.arange(T, dtype=jnp.int32)
        qBTHD, kBTHD = apply_rope(qBTHD, kBTHD, pos, rotary_dim=rot)

        # scaled dot-prod attn
        scale = 1.0 / jnp.sqrt(jnp.float32(Dh))
        # scores: (B,H,T,T)
        scores = jnp.einsum("bthd,bshd->bhts", qBTHD, kBTHD) * scale
        # causal mask
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        mask = jnp.where(mask, 0.0, -1e9).astype(scores.dtype)
        scores = scores + mask

        w = jax.nn.softmax(scores, axis=-1)  # (B,H,T,T)
        ctx = jnp.einsum("bhts,bshd->bthd", w, vBTHD)  # (B,T,H,Dh)
        outBTD = self.o(ctx.reshape(B, T, D))          # (B,T,D)
        return outBTD

    def mlp_block(self, xBTD: jnp.ndarray) -> jnp.ndarray:
        """MLP は『計算だけ f32 → 出力だけ元 dtype(bf16)』"""
        x32 = xBTD.astype(jnp.float32)
        h = self.ff1(x32)          # (B,T,4D)
        h = jax.nn.gelu(h)
        y = self.ff2(h)            # (B,T,D)
        return y.astype(xBTD.dtype)

    @nn.compact
    def __call__(self, xBTD: jnp.ndarray) -> jnp.ndarray:
        # pre-LN
        xn = self.norm(xBTD)
        attn = self.attend_full(xn)
        x = xBTD + attn
        # FFN（pre-LN）
        xn2 = self.norm(x)
        ff = self.mlp_block(xn2)
        return x + ff

    # ---- decode_once (KV 使用) ----
    def decode_once(self,
                    decode_state: Dict[str, Any],
                    xB1D: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        decode_state: {'k':(Ttot,B,H,Dh), 'v':同, 'cur_index': scalar int32}
        xB1D: (B,1,D)  事前に pre-LN 済み想定
        返り: (deltaB1D, new_state)
        """
        B, _, D = xB1D.shape
        H = self.cfg.n_heads
        Dh = self.cfg.d_head
        rot = self.cfg.rotary_dim

        kTBHD: jnp.ndarray = decode_state["k"]
        vTBHD: jnp.ndarray = decode_state["v"]
        cur: jnp.ndarray = decode_state["cur_index"]  # int32 scalar
        Ttot = kTBHD.shape[0]

        # Q/K/V（1 ステップ）
        qB1D = self.q(xB1D)
        kB1D = self.k(xB1D)
        vB1D = self.v(xB1D)
        qBHd = qB1D.reshape(B, 1, H, Dh)
        kBHd = kB1D.reshape(B, 1, H, Dh)
        vBHd = vB1D.reshape(B, 1, H, Dh)

        # RoPE（位置=cur）
        qBHd, kBHd = apply_rope(qBHd, kBHd, pos=cur.reshape((1,)), rotary_dim=rot)

        # K/V キャッシュへ write（cur は tracer なので dynamic_update を使う）
        # kTBHD: (Ttot,B,H,Dh)
        kTBHD = jax.lax.dynamic_update_slice(kTBHD, kBHd, (cur, 0, 0, 0))
        vTBHD = jax.lax.dynamic_update_slice(vTBHD, vBHd, (cur, 0, 0, 0))

        # scores: (B,H,Ttot)  全長に対して計算し、mask で cur 以降を切る
        scale = 1.0 / jnp.sqrt(jnp.float32(Dh))
        # einsum: (B,1,H,Dh) x (T,B,H,Dh) -> (B,H,T)
        scores = jnp.einsum("bthd,tbhd->bht", qBHd, kTBHD) * scale
        # mask: pos <= cur のみ有効
        pos = jnp.arange(Ttot, dtype=jnp.int32)  # (Ttot,)
        valid = (pos[None, None, :] <= cur[None, None])  # (1,1,Ttot)
        scores = jnp.where(valid, scores, scores.dtype.type(-1e9))
        w = jax.nn.softmax(scores, axis=-1)  # (B,H,Ttot)

        ctx = jnp.einsum("bht,tbhd->bhd", w, vTBHD)  # (B,H,Dh)
        out = self.o(ctx.reshape(B, 1, D))           # (B,1,D)

        new_state = {"k": kTBHD, "v": vTBHD, "cur_index": cur + jnp.int32(1)}
        return out, new_state

    # for convenience
    def init_decode_state(self, total_len: int, batch: int) -> Dict[str, Any]:
        H = self.cfg.n_heads
        Dh = self.cfg.d_head
        k = jnp.zeros((total_len, batch, H, Dh), dtype=jnp.bfloat16)
        v = jnp.zeros((total_len, batch, H, Dh), dtype=jnp.bfloat16)
        return {"k": k, "v": v, "cur_index": jnp.int32(0)}


# --------------------------
# Projection (/proj/…)
#   /proj/ReplicatedLayerNorm_0/{scale,offset}
#   /proj/Dense_0/{kernel,bias}
# --------------------------
class ProjectionShard(nn.Module):
    d_model: int
    n_vocab: int

    def setup(self):
        # 名前は leafspec に合わせる
        self.layer_norm = ReplicatedLayerNorm(name="ReplicatedLayerNorm_0")
        self.out = DenseBias(self.n_vocab, name="Dense_0")

    @nn.compact
    def __call__(self, xB1D: jnp.ndarray) -> jnp.ndarray:
        # pre-LN → Dense  （b のゼロ中心化は外でパラメータを書き換えても可）
        xn = self.layer_norm(xB1D)
        return self.out(xn)
