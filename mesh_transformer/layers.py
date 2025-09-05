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

# ---- debug: print dtype once (jit trace 時に1回だけ) ----
DEBUG_DTYPE_ONCE = False
_DEBUG_PRINTED = set()

def _print_once(tag: str, **xs):
    # jax.debug.print は JIT 内でも出る。ここは trace 時に1回だけ呼ぶ。
    if DEBUG_DTYPE_ONCE and tag not in _DEBUG_PRINTED:
        jax.debug.print(
            "[{t}] " + ", ".join([f"{k}:{{{k}}}" for k in xs.keys()]),
            t=tag, **{k: v.dtype for k, v in xs.items()}
        )
        _DEBUG_PRINTED.add(tag)


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

class _PreLayerNorm(nn.Module):
    epsilon: float = 1e-6
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        d = x.shape[-1]
        scale = self.param('scale', nn.initializers.ones, (d,), jnp.bfloat16)
        offset = self.param('offset', nn.initializers.zeros, (d,), jnp.bfloat16)
        x32 = x.astype(jnp.float32)
        mu = jnp.mean(x32, axis=-1, keepdims=True)
        var = jnp.mean((x32 - mu) ** 2, axis=-1, keepdims=True)
        y = (x32 - mu) * jax.lax.rsqrt(var + jnp.array(self.epsilon, x32.dtype))
        y = y * scale.astype(jnp.float32) + offset.astype(jnp.float32)
        return y.astype(x.dtype)



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

    def setup(self):
        # ここで一度だけ submodule を作る。名前は leafspec に 100% 合わせる。
        D = self.cfg.d_model
        self.pre_ln   = _PreLayerNorm(name='norm')             # /transformer_layers_i/norm/{scale,offset}
        self.q_proj   = nn.Dense(D, use_bias=False, name='q')  # /transformer_layers_i/q/Dense_0/{kernel}
        self.k_proj   = nn.Dense(D, use_bias=False, name='k')  # /transformer_layers_i/k/Dense_0/{kernel}
        self.v_proj   = nn.Dense(D, use_bias=False, name='v')  # /transformer_layers_i/v/Dense_0/{kernel}
        self.o_proj   = nn.Dense(D, use_bias=False, name='o')  # /transformer_layers_i/o/Dense_0/{kernel}
        self.w1       = nn.Dense(4 * D, use_bias=True,  name='dense_proj')    # /transformer_layers_i/dense_proj/Dense_0/{kernel,bias}
        self.w2       = nn.Dense(D,     use_bias=True,  name='dense_proj_o')  # /transformer_layers_i/dense_proj_o/Dense_0/{kernel,bias}

    

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
    # compact は不要（setup() で pre_ln を作ってある）
    def norm(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.pre_ln(x)


    # --- 投影を公開（transformer_shard が直接呼べるように） ---
    @nn.compact
    def q(self, xBTD: jnp.ndarray) -> jnp.ndarray:
        D = xBTD.shape[-1]
        return nn.Dense(D, use_bias=False, name='q')(xBTD.astype(jnp.float32))

    @nn.compact
    def k(self, xBTD: jnp.ndarray) -> jnp.ndarray:
        D = xBTD.shape[-1]
        return nn.Dense(D, use_bias=False, name='k')(xBTD.astype(jnp.float32))

    @nn.compact
    def v(self, xBTD: jnp.ndarray) -> jnp.ndarray:
        D = xBTD.shape[-1]
        return nn.Dense(D, use_bias=False, name='v')(xBTD.astype(jnp.float32))

    @nn.compact
    def o(self, xBTD: jnp.ndarray) -> jnp.ndarray:
        D = xBTD.shape[-1]
        return nn.Dense(D, use_bias=False, name='o')(xBTD.astype(jnp.float32))

    @nn.compact
    def dense_proj(self, xBTD: jnp.ndarray) -> jnp.ndarray:
        D = xBTD.shape[-1]
        hidden = 4 * D
        return nn.Dense(hidden, use_bias=True, name='dense_proj')(xBTD.astype(jnp.float32))

    @nn.compact
    def dense_proj_o(self, xBTD: jnp.ndarray) -> jnp.ndarray:
        D = xBTD.shape[-1]
        return nn.Dense(D, use_bias=True, name='dense_proj_o')(xBTD.astype(jnp.float32))




    # -------- Full attention (prefix)
    @nn.compact
    def attend_full(self, xBTD: jnp.ndarray) -> jnp.ndarray:
        B, T, D = xBTD.shape
        H, Dh = self.n_heads, self.d_head
        assert D == H * Dh, "d_model must equal n_heads * d_head"

        # 既存パラメータを再利用（new しない）
        q = self.q_proj(_to_f32(xBTD))  # (B,T,D)
        k = self.k_proj(_to_f32(xBTD))
        v = self.v_proj(_to_f32(xBTD))

        # (B,T,D) -> (B,T,H,Dh)
        def to_BTHD(z): return z.reshape(B, T, H, Dh)
        qBTHD, kBTHD, vBTHD = to_BTHD(q), to_BTHD(k), to_BTHD(v)

        # Rotary (prefix)
        if self.pe == 'rotary' and self.pe_rotary_dims > 0:
            inv = _rope_freqs(self.pe_rotary_dims, dtype=jnp.float32)
            t_idx = jnp.arange(T, dtype=jnp.float32)
            cos, sin = _rope_angles(t_idx, inv)  # (T, half)
            cos = cos[None, :, None, :]
            sin = sin[None, :, None, :]
            q_rot = _apply_rope(qBTHD.astype(jnp.float32), cos, sin, self.pe_rotary_dims)
            k_rot = _apply_rope(kBTHD.astype(jnp.float32), cos, sin, self.pe_rotary_dims)
        else:
            q_rot = qBTHD.astype(jnp.float32)
            k_rot = kBTHD.astype(jnp.float32)

        # (B,T,H,Dh) -> (B,H,T,Dh)
        qBHtD = jnp.transpose(q_rot, (0, 2, 1, 3))
        kBHTD = jnp.transpose(k_rot, (0, 2, 1, 3))
        vBHTD = jnp.transpose(vBTHD.astype(jnp.float32), (0, 2, 1, 3))

        scale = jnp.array(1.0 / math.sqrt(Dh), dtype=jnp.float32)
        scores = jnp.einsum('BHTD,BHSD->BHTS', qBHtD, kBHTD) * scale  # (B,H,T,T)

        # debug dtype (一度だけ)
        #_print_once("attn", q=qBHtD, k=kBHTD, v=vBHTD, scores=scores)

        # causal mask
        m = jnp.tril(jnp.ones((T, T), dtype=bool))
        scores = jnp.where(m[None, None, :, :], scores, jnp.full_like(scores, -1e9, dtype=scores.dtype))
        probs = jax.nn.softmax(scores, axis=-1)

        ctxBHTD = jnp.einsum('BHTS,BHSD->BHTD', probs, vBHTD)      # (B,H,T,Dh)
        ctxBTD  = jnp.transpose(ctxBHTD, (0, 2, 1, 3)).reshape(B,T,D)  # (B,T,D)

        # ここも既存の O を使う（new しない）
        out = self.o_proj(ctxBTD.astype(jnp.float32))
        return out.astype(xBTD.dtype)  # 残差 dtype に合わせる（通常 bf16）



    # -------- MLP
    @nn.compact
    def mlp_block(self, xBTD: jnp.ndarray) -> jnp.ndarray:
        x32 = _to_f32(xBTD)
        h = self.w1(x32)     # FP32
        h = _gelu(h)         # FP32
        y = self.w2(h)       # FP32

        _print_once("ffn", x=x32, w1_out=h, w2_out=y)

        return _to_out_dtype(xBTD, y)  # 出力だけ bf16 に戻す




    # --- prefix の K/V を一括生成（RoPE 済み） ---
    @nn.compact
    def prefill_kv(self, xBTD: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        B, T, D = xBTD.shape
        H, Dh = self.n_heads, self.d_head
        kBTD = self.k(xBTD)  # (B,T,D)
        vBTD = self.v(xBTD)  # (B,T,D)
        kBTHD = kBTD.reshape(B, T, H, Dh)
        vBTHD = vBTD.reshape(B, T, H, Dh)

        if self.pe == 'rotary' and self.pe_rotary_dims > 0:
            inv = _rope_freqs(self.pe_rotary_dims, dtype=jnp.float32)
            t_idx = jnp.arange(T, dtype=jnp.float32)
            cos, sin = _rope_angles(t_idx, inv)  # (T, half)
            cos = cos[None, :, None, :]  # (1,T,1,half)
            sin = sin[None, :, None, :]  # (1,T,1,half)
            kBTHD = _apply_rope(kBTHD.astype(jnp.float32), cos, sin, self.pe_rotary_dims)
        # V は回転しない
        kTBHD = jnp.transpose(kBTHD, (1, 0, 2, 3))  # (T,B,H,Dh)
        vTBHD = jnp.transpose(vBTHD, (1, 0, 2, 3))  # (T,B,H,Dh)
        return kTBHD.astype(jnp.bfloat16), vTBHD.astype(jnp.bfloat16)



    # -------- KV-cache initializer (state dict: {"k": TBHD, "v": TBHD, "cur_index": int32})
    # 注意: ここは @nn.compact を付けない（パラメタ生成が不要なため）
    def init_decode_state(
        self,
        total_len: int,
        batch: int,
        *,
        start_index: Optional[int] = None,
        dtype = jnp.bfloat16,
    ) -> Dict[str, jnp.ndarray]:
        """KV キャッシュをゼロ初期化する。
        返す dict 仕様:
          - "k": (T, B, H, Dh)  bfloat16
          - "v": (T, B, H, Dh)  bfloat16
          - "cur_index": int32 スカラー（デコード済みの最後の位置。未設定なら -1）
        """
        H, Dh = self.n_heads, self.d_head
        kTBHD = jnp.zeros((total_len, batch, H, Dh), dtype=dtype)
        vTBHD = jnp.zeros((total_len, batch, H, Dh), dtype=dtype)
        start = -1 if start_index is None else int(start_index)
        cur = jnp.array(start, dtype=jnp.int32)
        return {"k": kTBHD, "v": vTBHD, "cur_index": cur}


    # -------- K/V projection for prefix fill (returns TBHD)
    @nn.compact
    def kv_for_prefix(self, xBTD: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """prefix T 分の K/V を一括計算して返す（RoPE は K のみに適用）。
        戻り値: (kTBHD, vTBHD) = ((T,B,H,Dh),(T,B,H,Dh))
        """
        B, T, D = xBTD.shape
        H, Dh = self.n_heads, self.d_head
        assert D == H * Dh

        k = self.k_proj(_to_f32(xBTD))
        v = self.v_proj(_to_f32(xBTD))
        kBTHD = k.reshape(B, T, H, Dh)
        vBTHD = v.reshape(B, T, H, Dh)

        if self.pe == 'rotary' and self.pe_rotary_dims > 0:
            inv = _rope_freqs(self.pe_rotary_dims, dtype=jnp.float32)
            t_idx = jnp.arange(T, dtype=jnp.float32)
            cos, sin = _rope_angles(t_idx, inv)
            cos = cos[None, :, None, :]
            sin = sin[None, :, None, :]
            kBTHD = _apply_rope(kBTHD.astype(jnp.float32), cos, sin, self.pe_rotary_dims)

        kTBHD = jnp.transpose(kBTHD, (1, 0, 2, 3)).astype(jnp.bfloat16)
        vTBHD = jnp.transpose(vBTHD, (1, 0, 2, 3)).astype(jnp.bfloat16)
        return kTBHD, vTBHD


    
    # -------- One-step decode using KV cache (state dict: {"k": TBHD, "v": TBHD, "cur_index": int32})
    @nn.compact
    def decode_once(
        self,
        decode_state: Dict[str, jnp.ndarray],
        xB1D: jnp.ndarray,
        attn_bias: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        B, one, D = xB1D.shape
        H, Dh = self.n_heads, self.d_head
        assert D == H * Dh

        # --- pre-LN -> Q/K/V（prefix と一致させる）---
        # ★ pre-LN を decode でも必ず通す（prefix と一致させる）
        xn = self.norm(xB1D)
        q = self.q_proj(_to_f32(xn))   # (B,1,D)
        k_new = self.k_proj(_to_f32(xn))
        v_new = self.v_proj(_to_f32(xn))


        qB1HD = q.reshape(B, 1, H, Dh)
        kB1HD = k_new.reshape(B, 1, H, Dh)
        vB1HD = v_new.reshape(B, 1, H, Dh)

        # RoPE 単ステップ（pos=cur_index）
        cur = decode_state['cur_index']          # 現在の書き込み位置（prefix T のとき T）
        pos_f = cur.astype(jnp.float32)          # RoPE 位置 = cur
        if self.pe == 'rotary' and self.pe_rotary_dims > 0:
            inv = _rope_freqs(self.pe_rotary_dims, dtype=jnp.float32)
            cos, sin = _rope_angles(pos_f[None], inv)   # (1, half)
            cos = cos[None, :, None, :]  # (1,1,1,half)
            sin = sin[None, :, None, :]  # (1,1,1,half)
            qB1HD = _apply_rope(qB1HD.astype(jnp.float32), cos, sin, self.pe_rotary_dims)
            kB1HD = _apply_rope(kB1HD.astype(jnp.float32), cos, sin, self.pe_rotary_dims)
        else:
            qB1HD = qB1HD.astype(jnp.float32)
            kB1HD = kB1HD.astype(jnp.float32)


        # 既存キャッシュ（TBHD）
        kTBHD = decode_state['k']
        vTBHD = decode_state['v']
        Tcache, Bc, Hc, Dhc = kTBHD.shape
        assert Bc == B and Hc == H and Dhc == Dh

        # cur 位置へ書き込み（dtype を合わせて set）
        k_cur = jnp.transpose(kB1HD, (1, 0, 2, 3))[0].astype(kTBHD.dtype)
        v_cur = jnp.transpose(vB1HD, (1, 0, 2, 3))[0].astype(vTBHD.dtype)
        kTBHD = kTBHD.at[cur].set(k_cur)
        vTBHD = vTBHD.at[cur].set(v_cur)

        # (B,H,1,Dh) × (B,H,T,Dh)
        qBH1D = jnp.transpose(qB1HD, (0, 2, 1, 3))
        kBHTD = jnp.transpose(kTBHD.astype(jnp.float32), (1, 2, 0, 3))
        vBHTD = jnp.transpose(vTBHD.astype(jnp.float32), (1, 2, 0, 3))

        scale = jnp.array(1.0 / math.sqrt(Dh), dtype=jnp.float32)
        scores = jnp.einsum('BH1D,BHTD->BH1T', qBH1D, kBHTD) * scale  # (B,H,1,T)

        _print_once("attn_step", q=qBH1D, k=kBHTD, v=vBHTD, scores=scores)

        # 未来を無効化（<= cur のみ許可）
        idx = jnp.arange(Tcache, dtype=cur.dtype)
        mask = idx[None, None, None, :] <= cur[None, None, None]
        scores = jnp.where(mask, scores, jnp.full_like(scores, -1e9, dtype=scores.dtype))
        probs = jax.nn.softmax(scores, axis=-1)

        ctxBH1D = jnp.einsum('BH1T,BHTD->BH1D', probs, vBHTD)
        ctxB1HD = jnp.transpose(ctxBH1D, (0, 2, 1, 3))
        ctxB1D  = ctxB1HD.reshape(B, 1, D)

        # 既存 O を使用（new しない）
        attn_out = self.o_proj(ctxB1D.astype(jnp.float32)).astype(xB1D.dtype)

        # FFN（pre-LN → FFN）
        x_after_attn = xB1D + attn_out
        xn2 = self.norm(x_after_attn)
        ff  = self.mlp_block(xn2)

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
