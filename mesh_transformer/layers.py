# Flax inference-only layers for GPT-J-like model (names/shapes match leafspec)

from __future__ import annotations
import math
from typing import Any, Dict, Tuple, List

import jax
import jax.numpy as jnp
import flax.linen as nn


# ---------- helpers: RoPE ----------
def _rotate_every_two(x):
    # x: (..., 2k) -> rotate pairs
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return jnp.stack((-x2, x1), axis=-1).reshape(x.shape)

def _rope_cos_sin(rot_dim: int, seq_len: int, base: float = 10000.0):
    # returns (cos, sin) with shape (1, seq_len, 1, rot_dim)
    inv_freq = 1.0 / (base ** (jnp.arange(0, rot_dim, 2, dtype=jnp.float32) / rot_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)[:, None]  # (T,1)
    freqs = t * inv_freq[None, :]                         # (T, rot_dim/2)
    # interleave
    emb = jnp.concatenate([freqs, freqs], axis=-1)        # (T, rot_dim)
    cos = jnp.cos(emb)[None, :, None, :]                  # (1, T, 1, rot_dim)
    sin = jnp.sin(emb)[None, :, None, :]
    return cos, sin

def apply_rope(q_or_k, rot_dim: int, positions: jnp.ndarray):
    """
    q_or_k: (B, T, H, Dh)
    positions: (...,) 位置（T あるいは 1）
    """
    if rot_dim == 0:
        return q_or_k
    B, T, H, Dh = q_or_k.shape
    rd = min(rot_dim, Dh)
    # 位置系列長
    if positions.ndim == 0:
        seq = 1
        offs = positions[None]
    else:
        seq = int(positions.shape[0])
        offs = positions
    # cos/sin (1, seq, 1, rd)
    cos, sin = _rope_cos_sin(rd, seq)
    # 末尾1ステップの場合: (1,1,1,rd) になる
    if seq == 1:
        cos = cos[:, :1, :, :]
        sin = sin[:, :1, :, :]
    # 先頭 rd のみ回転
    x_rot = q_or_k[..., :rd]
    x_pass = q_or_k[..., rd:]
    x_rot = (x_rot * cos) + (_rotate_every_two(x_rot) * sin)
    return jnp.concatenate([x_rot, x_pass], axis=-1)


# ---------- Embedding ----------
class EmbeddingShard(nn.Module):
    config: Dict[str, Any]
    name: str = "embed"

    @nn.compact
    def __call__(self, xBT: jnp.ndarray) -> jnp.ndarray:
        # xBT: (B,T) uint32
        n_vocab = int(self.config["n_vocab"])
        d_model = int(self.config["d_model"])
        emb = nn.Embed(
            num_embeddings=n_vocab,
            features=d_model,
            embedding_init=nn.initializers.normal(stddev=1.0),
            dtype=jnp.bfloat16,
            name="embed_layer",
        )
        x = emb(xBT)  # (B,T,D)
        return x


# ---------- Projection (final LN + Dense to vocab) ----------
class ProjectionShard(nn.Module):
    config: Dict[str, Any]
    name: str = "proj"

    @nn.compact
    def __call__(self, xBTD: jnp.ndarray) -> jnp.ndarray:
        d_model = int(self.config["d_model"])
        n_vocab = int(self.config["n_vocab"])
        # /proj/ReplicatedLayerNorm_0/{scale,offset}
        x = nn.LayerNorm(epsilon=1e-5, name="ReplicatedLayerNorm_0")(xBTD.astype(jnp.float32))
        # /proj/Dense_0/{kernel,bias}
        head = nn.Dense(
            features=n_vocab,
            use_bias=True,
            dtype=jnp.float32,          # compute in f32
            param_dtype=jnp.bfloat16,   # keep params as bf16 (leafspec dtypeは自由だが安全)
            name="Dense_0",
        )(x)
        return head


# ---------- Transformer layer (Pre-LN shared for Attn/MLP) ----------
class TransformerLayerShard(nn.Module):
    config: Dict[str, Any]
    name: str

    def setup(self):
        cfg = self.config
        D = int(cfg["d_model"])
        H = int(cfg["n_heads"])
        Dh = int(cfg["d_head"])
        assert H * Dh == D, "n_heads * d_head must equal d_model"

        # 1) shared pre-LN
        self.norm = nn.LayerNorm(epsilon=1e-5, name="norm")  # /transformer_layers_i/norm/{scale,offset}

        # 2) Attn projections: /q /k /v /o すべて bias なし（leafspec 準拠）
        dense_kwargs = dict(use_bias=False, dtype=jnp.float32, param_dtype=jnp.bfloat16)
        self.q = nn.Dense(features=D, name="q", **dense_kwargs)
        self.k = nn.Dense(features=D, name="k", **dense_kwargs)
        self.v = nn.Dense(features=D, name="v", **dense_kwargs)
        self.o = nn.Dense(features=D, name="o", **dense_kwargs)

        # 3) MLP: /dense_proj (D->4D, biasあり), /dense_proj_o(4D->D, biasあり)
        self.dense_proj   = nn.Dense(features=4*D, name="dense_proj",
                                     use_bias=True, dtype=jnp.float32, param_dtype=jnp.bfloat16)
        self.dense_proj_o = nn.Dense(features=D,   name="dense_proj_o",
                                     use_bias=True, dtype=jnp.float32, param_dtype=jnp.bfloat16)

        # cache/rope設定
        self.H = H
        self.Dh = Dh
        self.pe_rotary_dims = int(cfg.get("pe_rotary_dims", 64))
        self.scale = 1.0 / math.sqrt(self.Dh)

    # ---- attention (full) ----
    def _attn_full(self, x_norm: jnp.ndarray, attn_bias=0.0) -> jnp.ndarray:
        # x_norm: (B,T,D) float32
        B, T, D = x_norm.shape
        H, Dh = self.H, self.Dh

        q = self.q(x_norm).reshape(B, T, H, Dh) * self.scale
        k = self.k(x_norm).reshape(B, T, H, Dh)
        v = self.v(x_norm).reshape(B, T, H, Dh)

        # RoPE（先頭 pe_rotary_dims のみ）
        if self.pe_rotary_dims > 0:
            pos = jnp.arange(T, dtype=jnp.int32)
            q = apply_rope(q, self.pe_rotary_dims, pos)
            k = apply_rope(k, self.pe_rotary_dims, pos)

        # scores: (B,H,T,T)
        scores = jnp.einsum("bthd,bThd->bhtT", q, k)
        if not isinstance(attn_bias, (int, float)):
            scores = scores + attn_bias  # 形状はブロードキャスト前提
        w = jax.nn.softmax(scores.astype(jnp.float32), axis=-1)
        ctx = jnp.einsum("bhtT,bThd->bthd", w, v).reshape(B, T, D)
        out = self.o(ctx)
        return out.astype(jnp.bfloat16)

    # ---- MLP (f32演算→bf16戻し) ----
    def _mlp(self, x_norm: jnp.ndarray) -> jnp.ndarray:
        y = self.dense_proj(x_norm)                 # f32
        y = jax.nn.gelu(y)
        y = self.dense_proj_o(y)                    # f32
        return y.astype(jnp.bfloat16)

    # ---- 呼び出し（学習/推論のフルパス）----
    @nn.compact
    def __call__(self, xBTD: jnp.ndarray, attn_bias=0.0, layer_index: int = 0) -> jnp.ndarray:
        x_norm = self.norm(xBTD).astype(jnp.float32)
        attn_out = self._attn_full(x_norm, attn_bias)
        mlp_out  = self._mlp(x_norm)
        return (xBTD.astype(jnp.bfloat16) + attn_out + mlp_out)

    # ---- KV 初期化（prefix 一括）----
    def get_init_decode_state(self, xBTD: jnp.ndarray, given_length: int, attn_bias=0.0, mesh_manager=None):
        # xBTD: (B,T,D), given_length: 以降に生成する長さ (未使用だが互換のため保持)
        x_norm = self.norm(xBTD).astype(jnp.float32)
        B, T, D = x_norm.shape
        H, Dh = self.H, self.Dh

        # K,V全体を作ってキャッシュ
        k_full = self.k(x_norm).reshape(B, T, H, Dh)
        v_full = self.v(x_norm).reshape(B, T, H, Dh)
        if self.pe_rotary_dims > 0:
            pos = jnp.arange(T, dtype=jnp.int32)
            k_full = apply_rope(k_full, self.pe_rotary_dims, pos)

        # 初期の出力（prefix も順伝播して logits 用の x を得る）
        attn_out = self._attn_full(x_norm, attn_bias)
        mlp_out  = self._mlp(x_norm)
        res = (xBTD.astype(jnp.bfloat16) + attn_out + mlp_out)

        state = {"k": k_full, "v": v_full, "pos": jnp.array(T, dtype=jnp.int32)}
        return res, state

    # ---- 1トークン増分デコード ----
    def decode_once(self, decode_state: Dict[str, jnp.ndarray], x1BTD: jnp.ndarray, attn_bias=0.0):
        # x1BTD: (B,1,D)
        B, one, D = x1BTD.shape
        H, Dh = self.H, self.Dh

        x_norm = self.norm(x1BTD).astype(jnp.float32)
        q = (self.q(x_norm).reshape(B, one, H, Dh)) * self.scale
        k =  self.k(x_norm).reshape(B, one, H, Dh)
        v =  self.v(x_norm).reshape(B, one, H, Dh)

        # 位置 = 既存Kの長さ
        pos = decode_state["pos"]  # int32
        if self.pe_rotary_dims > 0:
            # 単発位置用 cos/sin（seq=1）
            # apply_rope は (B,1,H,Dh) + pos(1,) を想定
            q = apply_rope(q, self.pe_rotary_dims, jnp.array([pos]))
            k = apply_rope(k, self.pe_rotary_dims, jnp.array([pos]))

        # 既存キャッシュへ連結 (axis=1: T)
        k_full = jnp.concatenate([decode_state["k"], k], axis=1)
        v_full = jnp.concatenate([decode_state["v"], v], axis=1)
        Tcur = k_full.shape[1]

        # 得点: (B,H,1,Tcur)
        scores = jnp.einsum("b1hd,bThd->bh1T", q, k_full)
        w = jax.nn.softmax(scores.astype(jnp.float32), axis=-1)
        ctx = jnp.einsum("bh1T,bThd->b1hd", w, v_full).reshape(B, 1, D)

        out = self.o(ctx).astype(jnp.bfloat16)
        new_x = (x1BTD.astype(jnp.bfloat16) + out + self._mlp(self.norm(x1BTD).astype(jnp.float32)))
        new_state = {"k": k_full, "v": v_full, "pos": pos + 1}
        return new_x, new_state
