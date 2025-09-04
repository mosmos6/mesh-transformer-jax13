# Copyright (c) 2025
# Compatible with flax.linen API used by the existing mesh_transformer/transformer_shard.py
#  - Keep parameter names/paths consistent with model_leaves_new.json
#  - Pre-LN inside each Transformer layer is stored under: /transformer_layers_*/norm/{offset,scale}
#  - Proj (LM head) has: /proj/ReplicatedLayerNorm_0/{scale,offset} and /proj/Dense_0/{kernel,bias}
#  - Q, K, V, O kernels under each layer: /transformer_layers_*/{q,k,v,o}/kernel
#  - MLP under each layer: /transformer_layers_*/dense_proj{,_o}/{kernel,bias}
#
# Attention math is computed in float32; MLP math is computed in float32; outputs cast back to bf16.
# KV cache layout is TBHD: (T, B, H, Dh)
#
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import math
import jax
import jax.numpy as jnp
from flax import linen as nn


# --------------------------
# Small helpers
# --------------------------

def _to_f32(x: jnp.ndarray) -> jnp.ndarray:
    return x.astype(jnp.float32)

def _to_out_dtype(x_ref: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    # Cast x to x_ref's dtype (typically bf16)
    return x.astype(x_ref.dtype)

def _gelu(x: jnp.ndarray) -> jnp.ndarray:
    # Use exact GELU (float32) for stability
    x32 = _to_f32(x)
    return 0.5 * x32 * (1.0 + jax.lax.erf(x32 / jnp.sqrt(jnp.array(2.0, x32.dtype))))


# --------------------------
# Dense wrappers with fixed param names
# --------------------------

class DenseNoBias(nn.Module):
    features: int
    @nn.compact
    def __call__(self, x):
        # leafspec: "/.../<wrapper>/Dense_0/{kernel}"
        return nn.Dense(self.features, use_bias=False, name="Dense_0")(x)

class DenseBias(nn.Module):
    features: int
    @nn.compact
    def __call__(self, x):
        # leafspec: "/.../<wrapper>/Dense_0/{kernel,bias}"
        return nn.Dense(self.features, use_bias=True, name="Dense_0")(x)



# --------------------------
# Embedding (keeps path: /embed/embed_layer/embedding)
# --------------------------

class EmbedBlock(nn.Module):
    n_vocab: int
    d_model: int

    @nn.compact
    def __call__(self, ids: jnp.ndarray) -> jnp.ndarray:
        # ids: (B, T) or (T,)
        emb = nn.Embed(
            num_embeddings=self.n_vocab,
            features=self.d_model,
            dtype=jnp.bfloat16,
            name='embed_layer',
            embedding_init=nn.initializers.normal(stddev=1.0),
        )
        out = emb(ids)
        return out  # (B, T, D)


# --------------------------
# LayerNorm (Replicated at projection head)
# --------------------------

class ReplicatedLayerNorm(nn.Module):
    """LayerNorm whose params live under the module name (e.g., '/proj/ReplicatedLayerNorm_0/…').
    Used at the projection head only to match leafspec.
    """
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        d = x.shape[-1]
        scale = self.param('scale', nn.initializers.ones, (d,), jnp.bfloat16)
        offset = self.param('offset', nn.initializers.zeros, (d,), jnp.bfloat16)
        x32 = _to_f32(x)
        mu = jnp.mean(x32, axis=-1, keepdims=True)
        var = jnp.mean((x32 - mu) ** 2, axis=-1, keepdims=True)
        y = (x32 - mu) * jax.lax.rsqrt(var + jnp.array(self.epsilon, x32.dtype))
        y = y * _to_f32(scale) + _to_f32(offset)
        return _to_out_dtype(x, y)


# --------------------------
# Rotary positional embedding utilities
# --------------------------

def _rope_freqs(rotary_dims: int, dtype=jnp.float32) -> jnp.ndarray:
    # inv_freq shape: (rotary_dims//2,)
    half = rotary_dims // 2
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, half, dtype=dtype) / float(half)))
    return inv_freq  # (half,)

def _rope_angles(time_idx: jnp.ndarray, inv_freq: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # time_idx: (T,) or scalar; inv_freq: (half,)
    # angles: (T, half)
    angles = jnp.outer(time_idx, inv_freq)
    return jnp.cos(angles), jnp.sin(angles)

def _apply_rope(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray, rotary_dims: int) -> jnp.ndarray:
    """Apply RoPE on the first rotary_dims of the last dim of x.
    x: (..., Dh)
    cos/sin: broadcastable to (..., rotary_dims//2)
    """
    Dh = x.shape[-1]
    rot = rotary_dims
    assert rot <= Dh, "rotary_dims must be <= head_dim"
    half = rot // 2

    # Split pass and rotary parts
    x_pass = x[..., rot:]
    xr = x[..., :rot]  # (..., rot)
    # pair as (..., half, 2)
    xr_pair = xr.reshape(xr.shape[:-1] + (half, 2))
    x_even = xr_pair[..., 0]  # (..., half)
    x_odd = xr_pair[..., 1]   # (..., half)

    # Broadcast cos/sin to xr_pair
    # cos/sin: (T, half) -> (..., half) via broadcast
    rot_even = x_even * cos - x_odd * sin
    rot_odd  = x_even * sin + x_odd * cos
    xr_rot = jnp.stack([rot_even, rot_odd], axis=-1).reshape(xr.shape)

    return jnp.concatenate([xr_rot, x_pass], axis=-1)


# --------------------------
# Relative position bias (not used for rotary; kept for backward compat)
# --------------------------

class RelativePositionEmbs(nn.Module):
    """Kept for API compatibility. Returns zeros when called."""
    num_buckets: int = 0
    max_distance: int = 0
    n_heads: int = 0

    @nn.compact
    def __call__(self, qlen: int, klen: int) -> jnp.ndarray:
        # Return zero bias: (1, n_heads, qlen, klen)
        return jnp.zeros((1, self.n_heads, qlen, klen), dtype=jnp.bfloat16)


# --------------------------
# Transformer layer (Shard)
# --------------------------

@dataclass
class LayerCfg:
    d_model: int
    n_heads: int
    d_head: int
    pe: str = 'rotary'
    pe_rotary_dims: int = 64


class TransformerLayerShard(nn.Module):
    """One transformer block with pre-LN.
    Exposes methods used by transformer_shard.py:
      - norm(x)
      - attend_full(x)  # full prefix attention (B, T, D) -> (B, T, D)
      - mlp_block(x)    # MLP (B, T, D) -> (B, T, D)
      - decode_once(decode_state, xB1D, attn_bias) -> (deltaB1D, new_state)
    """
    cfg: LayerCfg
    

    # --- 互換用エイリアス（既存コードが self.n_heads 等を参照しても動くように） ---
    @property
    def d_model(self) -> int:
        return self.cfg.d_model

    @property
    def n_heads(self) -> int:
        return self.cfg.n_heads

    @property
    def d_head(self) -> int:
        return self.cfg.d_head

    @property
    def pe(self) -> str:
        return self.cfg.pe

    @property
    def pe_rotary_dims(self) -> int:
        return self.cfg.pe_rotary_dims

    @property
    def rotary_dim(self) -> int:
        # 互換エイリアス（古い呼び出し側が rotary_dim を期待しても動くように）
        return self.cfg.pe_rotary_dims


    # -------- LayerNorm inside layer: params under "/transformer_layers_*/norm/{offset,scale}"
    @nn.compact
    def norm(self, x: jnp.ndarray) -> jnp.ndarray:
        return ReplicatedLayerNorm(name='norm')(x)


    # -------- Full attention (prefix)
    @nn.compact
    def attend_full(self, xBTD: jnp.ndarray) -> jnp.ndarray:
        B, T, D = xBTD.shape
        H, Dh = self.n_heads, self.d_head
        assert D == H * Dh, "d_model must equal n_heads * d_head"

        q = DenseNoBias(D, name='q')(_to_f32(xBTD))
        k = DenseNoBias(D, name='k')(_to_f32(xBTD))
        v = DenseNoBias(D, name='v')(_to_f32(xBTD))

        # Reshape to (B,T,H,Dh)
        def to_BTHD(z):
            return z.reshape(B, T, H, Dh)

        qBTHD = to_BTHD(q)
        kBTHD = to_BTHD(k)
        vBTHD = to_BTHD(v)

        # Rotary (prefix)
        if self.pe == 'rotary' and self.pe_rotary_dims > 0:
            inv = _rope_freqs(self.pe_rotary_dims, dtype=jnp.float32)
            t_idx = jnp.arange(T, dtype=jnp.float32)
            cos, sin = _rope_angles(t_idx, inv)  # (T, half)
            # reshape to (1,T,1,half)
            cos = cos[None, :, None, :]
            sin = sin[None, :, None, :]
            q_rot = _apply_rope(_to_f32(qBTHD), cos, sin, self.pe_rotary_dims)
            k_rot = _apply_rope(_to_f32(kBTHD), cos, sin, self.pe_rotary_dims)
        else:
            q_rot = _to_f32(qBTHD)
            k_rot = _to_f32(kBTHD)

        # Scores: (B,H,T,T)
        # (B,T,H,Dh) -> (B,H,T,Dh)
        qBHtD = jnp.transpose(q_rot, (0, 2, 1, 3))
        kBHTD = jnp.transpose(k_rot, (0, 2, 1, 3))
        # scale
        scale = jnp.array(1.0 / math.sqrt(Dh), dtype=jnp.float32)
        scores = jnp.einsum('BHTD,BHSD->BHTS', qBHtD, kBHTD) * scale
        # causal mask
        m = jnp.tril(jnp.ones((T, T), dtype=bool))
        scores = jnp.where(m[None, None, :, :], scores, jnp.full_like(scores, -1e9, dtype=scores.dtype))
        probs = jax.nn.softmax(scores, axis=-1)

        vBHTD = jnp.transpose(_to_f32(vBTHD), (0, 2, 1, 3))
        ctxBHTD = jnp.einsum('BHTS,BHSD->BHTD', probs, vBHTD)
        # back to (B,T,D)
        ctxBTD = jnp.transpose(ctxBHTD, (0, 2, 1, 3)).reshape(B, T, D)
        out = DenseNoBias(D, name='o')(ctxBTD)
        return out

    # -------- MLP
    @nn.compact
    def mlp_block(self, xBTD: jnp.ndarray) -> jnp.ndarray:
        D = xBTD.shape[-1]
        hidden = 4 * D
        # Dense -> GELU -> Dense ; all math in f32
        h = DenseBias(hidden, name='dense_proj')(_to_f32(xBTD))
        h = _gelu(h)
        y = DenseBias(D, name='dense_proj_o')(h)
        return _to_out_dtype(xBTD, y)

    # -------- One-step decode using KV cache (state dict: {"k": TBHD, "v": TBHD, "cur_index": int32})
    @nn.compact
    def decode_once(self, decode_state: Dict[str, jnp.ndarray], xB1D: jnp.ndarray, attn_bias: Optional[jnp.ndarray] = None
                   ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        # xB1D: (B,1,D)
        B, one, D = xB1D.shape
        H, Dh = self.n_heads, self.d_head
        assert D == H * Dh

        # Project new q, k, v
        q = DenseNoBias(D, name='q')(_to_f32(xB1D))            # (B,1,D)
        k_new = DenseNoBias(D, name='k')(_to_f32(xB1D))        # (B,1,D)
        v_new = DenseNoBias(D, name='v')(_to_f32(xB1D))        # (B,1,D)

        # reshape to (B,1,H,Dh)
        qB1HD = q.reshape(B, 1, H, Dh)
        kB1HD = k_new.reshape(B, 1, H, Dh)
        vB1HD = v_new.reshape(B, 1, H, Dh)

        # Apply RoPE at the current index
        if self.pe == 'rotary' and self.pe_rotary_dims > 0:
            inv = _rope_freqs(self.pe_rotary_dims, dtype=jnp.float32)
            cur = decode_state['cur_index']  # int32[] tracer-friendly
            cur_f = cur.astype(jnp.float32)
            cos, sin = _rope_angles(cur_f[None], inv)  # (1, half)
            cos = cos[None, :, None, :]  # (1,1,1,half)
            sin = sin[None, :, None, :]  # (1,1,1,half)
            qB1HD = _apply_rope(_to_f32(qB1HD), cos, sin, self.pe_rotary_dims)
            kB1HD = _apply_rope(_to_f32(kB1HD), cos, sin, self.pe_rotary_dims)
        else:
            qB1HD = _to_f32(qB1HD)
            kB1HD = _to_f32(kB1HD)

        # Read cache (T, B, H, Dh)
        kTBHD = decode_state['k']
        vTBHD = decode_state['v']
        Tcache = kTBHD.shape[0]

        # Update cache at cur index: (B,H,Dh)
        cur = decode_state['cur_index']
        k_cur = jnp.transpose(kB1HD, (1, 0, 2, 3))[0]  # (B,H,Dh)
        v_cur = jnp.transpose(vB1HD, (1, 0, 2, 3))[0]  # (B,H,Dh)
        kTBHD = kTBHD.at[cur].set(k_cur)
        vTBHD = vTBHD.at[cur].set(v_cur)

        # Attention over all positions <= cur (mask future)
        # Shapes for dot:
        #   q: (B,H,1,Dh)
        #   k: (B,H,T,Dh)
        qBH1D = jnp.transpose(qB1HD, (0, 2, 1, 3))
        kBHTD = jnp.transpose(_to_f32(kTBHD), (1, 2, 0, 3))
        vBHTD = jnp.transpose(_to_f32(vTBHD), (1, 2, 0, 3))

        scale = jnp.array(1.0 / math.sqrt(Dh), dtype=jnp.float32)
        scores = jnp.einsum('BH1D,BHTD->BH1T', qBH1D, kBHTD) * scale  # (B,H,1,T)
        # causal mask with current index
        idx = jnp.arange(Tcache, dtype=cur.dtype)  # (T,)
        mask = idx[None, None, None, :] <= cur[None, None, None]
        scores = jnp.where(mask, scores, jnp.full_like(scores, -1e9, dtype=scores.dtype))
        probs = jax.nn.softmax(scores, axis=-1)  # (B,H,1,T)

        ctxBH1D = jnp.einsum('BH1T,BHTD->BH1D', probs, vBHTD)  # (B,H,1,Dh)
        ctxB1HD = jnp.transpose(ctxBH1D, (0, 2, 1, 3))        # (B,1,H,Dh)
        ctxB1D = ctxB1HD.reshape(B, 1, D)                     # (B,1,D)
        attn_out = DenseNoBias(D, name='o')(ctxB1D)           # (B,1,D)

        # Residual-add for MLP
        x_after_attn = xB1D + attn_out
        xn2 = self.norm(x_after_attn)
        ff = self.mlp_block(xn2)

        delta_total = attn_out + ff
        new_state = {
            'k': kTBHD,
            'v': vTBHD,
            'cur_index': cur + jnp.array(1, dtype=cur.dtype),
        }
        return delta_total, new_state


# --------------------------
# Projection (LM Head)
# --------------------------

class ProjectionShard(nn.Module):
    d_model: int
    n_vocab: int

    @nn.compact
    def __call__(self, xBTD: jnp.ndarray) -> jnp.ndarray:
        # optional LN (params under /proj/ReplicatedLayerNorm_0)
        xn = ReplicatedLayerNorm(name='ReplicatedLayerNorm_0')(xBTD)
        logits = nn.Dense(self.n_vocab, use_bias=True, name='Dense_0')(_to_f32(xn))
        # return logits in float32 for numerical stability of softmax
        return logits.astype(jnp.float32)
