# Copyright (c) 2025
# Minimal, leafspec-compatible Flax layers for GPT-J style model.
#  - Keep original class names used by transformer_shard.py:
#      * ReplicatedLayerNorm
#      * TransformerLayerShard (with .norm, .attend_full, .mlp_block, .decode_once)
#      * EmbedBlock
#      * ProjectionShard
#  - Parameter paths are aligned to the provided leafspec:
#      /embed/embed_layer/embedding
#      /proj/ReplicatedLayerNorm_0/{scale,offset}
#      /proj/Dense_0/{kernel,bias}
#      /transformer_layers_*/{q,k,v,o}/kernel
#      /transformer_layers_*/dense_proj{,_o}/{kernel,bias}
#
#  - FFN (MLP) does compute in float32 and casts output back to input dtype.
#  - Attention matmul/softmax are computed in float32; 1/sqrt(d_head) applied exactly once.
#  - RoPE rotates the first `pe_rotary_dims` dims of each head; the remaining dims pass through.
#  - KV cache layout is TBHD (T, B, H, Dh) and is updated with lax.dynamic_update_slice (no Python int()).

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import jax
import jax.numpy as jnp
from jax import lax
from flax import linen as nn


# --------------------------
# Config structure
# --------------------------
@dataclass
class LayerCfg:
    d_model: int
    n_heads: int
    d_head: int
    pe_rotary_dims: int = 64
    norm: str = "layernorm"


# --------------------------
# Utilities
# --------------------------
def _gelu(x: jnp.ndarray) -> jnp.ndarray:
    # approximate gelu like in GPT-J
    return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * (x ** 3))))


def _rope_build_cos_sin(t: jnp.ndarray, rotary_dims: int, *, base: float = 10000.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build cos/sin for RoPE.

    Args:
      t: positions, shape (T,) or scalar (); dtype int32/int64
      rotary_dims: number of dims to rotate (<= d_head)
      base: rotary base

    Returns:
      cos, sin with shape (..., rotary_dims // 2)
      For full sequence call: t: (T,) -> returns (T, P), with P = rotary_dims//2
      For single step call: t: scalar -> returns (1, P)
    """
    t = jnp.asarray(t)
    if t.ndim == 0:
        t = t[None]  # -> (1,)
    # inverse frequency
    half = rotary_dims // 2
    inv_freq = 1.0 / (base ** (jnp.arange(0, half, dtype=jnp.float32) / float(half)))  # (P,)
    freqs = t.astype(jnp.float32)[:, None] * inv_freq[None, :]  # (T, P)
    return jnp.cos(freqs), jnp.sin(freqs)


def _rope_apply(q: jnp.ndarray, k: jnp.ndarray, rotary_dims: int, pos: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply RoPE to q, k.
      Shapes:
        q: (B, H, T, Dh)
        k: (B, H, T, Dh)
      rotary_dims: int <= Dh
        RoPE applies to the first `rotary_dims` dims of Dh.
      pos:
        - For full pass: jnp.arange(T) -> (T,)
        - For single step: scalar position -> ()
    """
    if rotary_dims <= 0:
        return q, k

    # cos,sin: (T, P) or (1, P) where P=rotary_dims//2
    cos, sin = _rope_build_cos_sin(pos, rotary_dims)
    # reshape for broadcasting to (B,H,T,P)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    def _apply(x: jnp.ndarray) -> jnp.ndarray:
        # x: (B,H,T,Dh)
        x_rot = x[..., :rotary_dims]
        x_pass = x[..., rotary_dims:]
        # even/odd split
        x1 = x_rot[..., 0::2]
        x2 = x_rot[..., 1::2]
        # rotate
        x1p = x1 * cos - x2 * sin
        x2p = x2 * cos + x1 * sin
        # interleave back
        x_rot2 = jnp.stack([x1p, x2p], axis=-1).reshape(x_rot.shape)
        return jnp.concatenate([x_rot2, x_pass], axis=-1)

    return _apply(q), _apply(k)


# --------------------------
# LayerNorm (replicated)
# --------------------------
class ReplicatedLayerNorm(nn.Module):
    """LayerNorm with explicit 'scale' and 'offset' names."""
    epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        d = x.shape[-1]
        scale = self.param('scale', nn.initializers.ones, (d,), jnp.bfloat16)
        offset = self.param('offset', nn.initializers.zeros, (d,), jnp.bfloat16)

        x_f32 = x.astype(jnp.float32)
        mean = jnp.mean(x_f32, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x_f32 - mean), axis=-1, keepdims=True)
        inv = jax.lax.rsqrt(var + jnp.asarray(self.epsilon, x_f32.dtype))
        y = (x_f32 - mean) * inv

        y = y * scale.astype(jnp.float32) + offset.astype(jnp.float32)
        return y.astype(x.dtype)


# --------------------------
# Dense (no bias / with bias)
#   * define params directly to match leaf paths: /.../{kernel,bias}
# --------------------------
class DenseNoBias(nn.Module):
    features: int  # output dim

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        in_features = x.shape[-1]
        kernel = self.param('kernel', nn.initializers.lecun_normal(), (in_features, self.features), jnp.bfloat16)
        y = jnp.matmul(x.astype(jnp.float32), kernel.astype(jnp.float32))
        return y.astype(x.dtype)


class DenseBias(nn.Module):
    features: int  # output dim

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        in_features = x.shape[-1]
        kernel = self.param('kernel', nn.initializers.lecun_normal(), (in_features, self.features), jnp.bfloat16)
        bias = self.param('bias', nn.initializers.zeros, (self.features,), jnp.bfloat16)
        y = jnp.matmul(x.astype(jnp.float32), kernel.astype(jnp.float32))
        y = y + bias.astype(jnp.float32)
        return y.astype(x.dtype)


# --------------------------
# Embed / Projection
# --------------------------
class EmbedBlock(nn.Module):
    """Embedding table under /embed/embed_layer/embedding"""
    n_vocab: int
    d_model: int

    @nn.compact
    def __call__(self, tok_ids: jnp.ndarray) -> jnp.ndarray:
        # flax.linen.Embed uses param name "embedding"
        emb = nn.Embed(num_embeddings=self.n_vocab, features=self.d_model, name='embed_layer',
                       embedding_init=nn.initializers.normal(stddev=1.0))(tok_ids)
        return emb

    @nn.nowrap
    def embed(self, tok_ids: jnp.ndarray) -> jnp.ndarray:
        # alias for backward compatibility
        return self.__call__(tok_ids)


class ProjectionShard(nn.Module):
    """Final projection head with optional LN before Dense.
       Params under /proj/ReplicatedLayerNorm_0 and /proj/Dense_0
    """
    d_model: int
    n_vocab: int
    use_ln: bool = True

    @nn.compact
    def __call__(self, xBTD: jnp.ndarray) -> jnp.ndarray:
        if self.use_ln:
            xBTD = ReplicatedLayerNorm(name='ReplicatedLayerNorm_0')(xBTD)
        # Dense to vocab
        logits = DenseBias(self.n_vocab, name='Dense_0')(xBTD.astype(jnp.bfloat16))
        return logits


# --------------------------
# Transformer Layer
# --------------------------
class TransformerLayerShard(nn.Module):
    """Single Transformer layer (Pre-LN), GPT-J style."""
    cfg: LayerCfg

    @nn.compact
    def __call__(self, xBTD: jnp.ndarray) -> jnp.ndarray:
        # kept for API completeness; not used by transformer_shard.py
        x = xBTD
        x = x + self.attend_full(self.norm(x))
        x = x + self.mlp_block(self.norm(x))
        return x

    # --- Expose modules with specific names to match leafspec paths ---
    @nn.compact
    def norm(self, xBTD: jnp.ndarray) -> jnp.ndarray:
        return ReplicatedLayerNorm(name='ReplicatedLayerNorm_0')(xBTD)

    @nn.compact
    def _proj_q(self, xBTD: jnp.ndarray) -> jnp.ndarray:
        return DenseNoBias(self.cfg.n_heads * self.cfg.d_head, name='q')(xBTD)

    @nn.compact
    def _proj_k(self, xBTD: jnp.ndarray) -> jnp.ndarray:
        return DenseNoBias(self.cfg.n_heads * self.cfg.d_head, name='k')(xBTD)

    @nn.compact
    def _proj_v(self, xBTD: jnp.ndarray) -> jnp.ndarray:
        return DenseNoBias(self.cfg.n_heads * self.cfg.d_head, name='v')(xBTD)

    @nn.compact
    def _proj_o(self, xBTHD: jnp.ndarray) -> jnp.ndarray:
        # input: (B,T,H,Dh) -> (B,T,D)
        B, T, H, Dh = xBTHD.shape
        xBTD = xBTHD.reshape(B, T, H * Dh)
        return DenseNoBias(self.cfg.d_model, name='o')(xBTD)

    @nn.compact
    def _dense_proj(self, xBTD: jnp.ndarray) -> jnp.ndarray:
        # first MLP linear
        return DenseBias(self.cfg.d_model * 4, name='dense_proj')(xBTD)

    @nn.compact
    def _dense_proj_o(self, xBT4D: jnp.ndarray) -> jnp.ndarray:
        return DenseBias(self.cfg.d_model, name='dense_proj_o')(xBT4D)

    # --- Attention ---
    @nn.nowrap
    def _split_heads(self, xBTD: jnp.ndarray) -> jnp.ndarray:
        B, T, D = xBTD.shape
        H, Dh = self.cfg.n_heads, self.cfg.d_head
        xBTHD = xBTD.reshape(B, T, H, Dh)
        return jnp.transpose(xBTHD, (0, 2, 1, 3))  # (B,H,T,Dh)

    @nn.nowrap
    def _merge_heads(self, xBH1D_or_BHTD: jnp.ndarray) -> jnp.ndarray:
        if xBH1D_or_BHTD.ndim == 4:
            B, H, T_, Dh = xBH1D_or_BHTD.shape
            xBTHD = jnp.transpose(xBH1D_or_BHTD, (0, 2, 1, 3))  # (B,T,H,Dh)
            return xBTHD.reshape(B, T_, H * Dh)
        raise ValueError("expected 4D tensor for _merge_heads")

    @nn.nowrap
    def _causal_bias(self, Tq: int, Tk: int) -> jnp.ndarray:
        # (1,1,Tq,Tk) additive mask in f32: 0 for allowed, -1e10 for masked
        mask = jnp.tril(jnp.ones((Tq, Tk), dtype=jnp.int32), k=0)
        bias = (1.0 - mask.astype(jnp.float32)) * (-1e10)
        return bias[None, None, :, :]

    @nn.compact
    def attend_full(self, xBTD: jnp.ndarray, attn_bias: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Full-sequence self-attention.
          xBTD: (B,T,D)
          returns: (B,T,D)
        """
        B, T, D = xBTD.shape
        H, Dh = self.cfg.n_heads, self.cfg.d_head

        qBTD = self._proj_q(xBTD)  # (B,T,H*Dh)
        kBTD = self._proj_k(xBTD)
        vBTD = self._proj_v(xBTD)

        q = self._split_heads(qBTD)  # (B,H,T,Dh)
        k = self._split_heads(kBTD)
        v = self._split_heads(vBTD)

        # RoPE on first pe_rotary_dims dims
        pos = jnp.arange(T, dtype=jnp.int32)
        q, k = _rope_apply(q, k, self.cfg.pe_rotary_dims, pos)

        # QK^T / sqrt(dh)
        scale = 1.0 / jnp.sqrt(jnp.asarray(Dh, jnp.float32))
        attn_scores = jnp.einsum('BHTD,BHSD->BHTS', q.astype(jnp.float32), k.astype(jnp.float32))
        attn_scores = attn_scores * scale

        if attn_bias is None:
            attn_bias = self._causal_bias(T, T)
        attn_scores = attn_scores + attn_bias  # broadcast OK

        attn_probs = nn.softmax(attn_scores, axis=-1)
        ctx = jnp.einsum('BHTS,BHSD->BHTD', attn_probs.astype(jnp.float32), v.astype(jnp.float32))
        out = self._proj_o(jnp.transpose(ctx, (0, 2, 1, 3)))  # (B,T,D)
        return out

    # --- MLP ---
    @nn.compact
    def mlp_block(self, xBTD: jnp.ndarray) -> jnp.ndarray:
        # compute in f32; return original dtype
        x_f32 = xBTD.astype(jnp.float32)
        h = self._dense_proj(x_f32).astype(jnp.float32)
        h = _gelu(h)
        y = self._dense_proj_o(h)
        return y.astype(xBTD.dtype)

    # --- One-step decode with KV cache ---
    @nn.compact
    def decode_once(
        self,
        decode_state: Dict[str, jnp.ndarray],
        xB1D: jnp.ndarray,
        attn_bias: Optional[jnp.ndarray] = None,
    ):
        """
        Args:
          decode_state:
            {
              "k": (T, B, H, Dh),
              "v": (T, B, H, Dh),
              "cur_index": () int32
            }
          xB1D: (B,1,D)
        Returns:
          delta: (B,1,D), new_state
        """
        H, Dh = self.cfg.n_heads, self.cfg.d_head
        B = xB1D.shape[0]

        qB1TD = self._proj_q(xB1D)        # (B,1,H*Dh)
        kB1TD = self._proj_k(xB1D)
        vB1TD = self._proj_v(xB1D)

        # (B,1,H,Dh) -> (B,H,1,Dh)
        q = jnp.transpose(qB1TD.reshape(B, 1, H, Dh), (0,2,1,3))
        k = jnp.transpose(kB1TD.reshape(B, 1, H, Dh), (0,2,1,3))
        v = jnp.transpose(vB1TD.reshape(B, 1, H, Dh), (0,2,1,3))

        # RoPE at single position
        cur = jnp.asarray(decode_state["cur_index"]).astype(jnp.int32)  # ()
        q, k = _rope_apply(q, k, self.cfg.pe_rotary_dims, cur)

        # Update KV cache at cur (TBHD)
        k_cache: jnp.ndarray = decode_state["k"]  # (T,B,H,Dh)
        v_cache: jnp.ndarray = decode_state["v"]
        T = k_cache.shape[0]

        # prepare (1,B,H,Dh) to update
        k_upd = jnp.transpose(k, (2,0,1,3))  # (1,B,H,Dh)
        v_upd = jnp.transpose(v, (2,0,1,3))  # (1,B,H,Dh)

        # start indices must be int32 arrays; use lax to avoid Python int()
        start = jnp.stack([cur, jnp.int32(0), jnp.int32(0), jnp.int32(0)], axis=0)
        k_cache2 = lax.dynamic_update_slice(k_cache, k_upd, start)
        v_cache2 = lax.dynamic_update_slice(v_cache, v_upd, start)

        # Attention against all cached keys up to cur (causal)
        # scores: (B,H,T)
        qBH1D = q[..., 0, :]  # (B,H,Dh)
        scores = jnp.einsum('BHD,TBHD->BHT', qBH1D.astype(jnp.float32), k_cache2.astype(jnp.float32))
        scale = 1.0 / jnp.sqrt(jnp.asarray(Dh, jnp.float32))
        scores = scores * scale

        # causal mask up to cur
        ar = jnp.arange(T, dtype=jnp.int32)[None, None, :]
        mask = ar <= cur[None, None]
        scores = jnp.where(mask, scores, jnp.asarray(-1e10, scores.dtype))

        attn = nn.softmax(scores, axis=-1)  # (B,H,T)
        # context: (B,H,Dh)
        ctx = jnp.einsum('BHT,TBHD->BHD', attn.astype(jnp.float32), v_cache2.astype(jnp.float32))
        # project out
        ctxBH1D = ctx[:, :, None, :]  # (B,H,1,Dh)
        outB1D = self._proj_o(jnp.transpose(ctxBH1D, (0,2,1,3)))  # (B,1,D)

        # MLP residual path: Pre-LN + FF
        xn = self.norm(xB1D)
        ff = self.mlp_block(xn)
        out = outB1D + ff

        new_state = {
            "k": k_cache2,
            "v": v_cache2,
            "cur_index": cur + jnp.int32(1),
        }
        return out, new_state
