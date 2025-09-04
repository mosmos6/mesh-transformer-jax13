# mesh_transformer/transformer_shard.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from .layers import (
    EmbedBlock,
    LayerCfg,
    ProjectionShard,
    TransformerLayerShard,
    #ReplicatedLayerNorm,  # optional export
)


# ========== Flax Module 本体 ==========
class CausalTransformerShard(nn.Module):
    """
    Flax linen Module.
    params パスは leafspec に合わせて命名している。
    """
    layers: int
    d_model: int
    n_heads: int
    d_head: int
    n_vocab: int
    rotary_dim: int = 64  # GPT‑J 既定
    # 互換のためのフラグ（使わないが残す）
    norm: str = "layernorm"
    pe: str = "rotary"

    def setup(self):
        # /embed/*
        self.embed = EmbedBlock(self.n_vocab, self.d_model, name="embed")
        # /transformer_layers_*
        cfg = LayerCfg(self.d_model, self.n_heads, self.d_head, self.rotary_dim)
        self.layers_mod = [
            TransformerLayerShard(cfg, name=f"transformer_layers_{i}")
            for i in range(self.layers)
        ]
        # /proj/*
        self.proj = ProjectionShard(self.d_model, self.n_vocab, name="proj")

    # ---------- 通常 forward（B,T） ----------
    @nn.compact
    def __call__(self, idsBT: jnp.ndarray) -> jnp.ndarray:
        """
        idsBT: (B,T) uint32 → logits (B,T,V)
        """
        x = self.embed(idsBT)                        # (B,T,D)
        for layer in self.layers_mod:
            # pre-LN は Layer 内部
            x = layer(x)
        # 最終投影（直前 LN は ProjectionShard 内）
        logits = self.proj(x)                        # (B,T,V)
        return logits

    # ---------- 生成 (prefill) ----------
    def generate_initial(self, ctx_1d: jnp.ndarray, gen_len: int):
        """
        ctx_1d: (T,) uint32
        gen_len: 生成トークン数（静的引数で呼ぶこと）
        return: logits_last(B=1,T=1,V), (last_tok(1,), states(tuple), rng)
        """
        ids = ctx_1d.reshape((1, -1))               # (1,T)
        B, T = 1, ids.shape[1]
        total_len = T + int(gen_len)

        # full pass (KV prefill)
        x = self.embed(ids)                          # (1,T,D)
        # 各層で pre-LN → attn → 残差 → pre-LN → MLP → 残差
        states = []
        for layer in self.layers_mod:
            # pre-LN 内部
            # 1) フル注意
            xn = layer.norm(x)
            attn = layer.attend_full(xn)
            x = x + attn
            # 2) MLP
            xn2 = layer.norm(x)
            ff = layer.mlp_block(xn2)
            x = x + ff

            # KV を用意して prefix T を埋める
            st = layer.init_decode_state(total_len=total_len, batch=B)
            # Q/K/V をもう一度計算して T ぶんの K/V をキャッシュへ（RoPE 一致のため）
            H, Dh = layer.cfg.n_heads, layer.cfg.d_head
            q = layer.q(xn)   # (1,T,D)
            k = layer.k(xn)   # (1,T,D)
            v = layer.v(xn)   # (1,T,D)
            q = q.reshape(B, T, H, Dh)
            k = k.reshape(B, T, H, Dh)
            v = v.reshape(B, T, H, Dh)
            # RoPE
            pos = jnp.arange(T, dtype=jnp.int32)
            _, k = layer.apply({"params": self.variables["params"]},
                               k, k, pos, layer.cfg.rotary_dim,
                               method=TransformerLayerShard.apply_rope,
                               mutable=False) if False else (None, None)  # 占位
            # 直接呼べないので共通関数を再適用
            from .layers import apply_rope
            _, k = apply_rope(q, k, pos, layer.cfg.rotary_dim)
            # write
            st["k"] = jax.lax.dynamic_update_slice(st["k"], k, (0, 0, 0, 0))
            st["v"] = jax.lax.dynamic_update_slice(st["v"], v, (0, 0, 0, 0))
            st["cur_index"] = jnp.int32(T)
            states.append(st)

        # 最後のトークンの logits（(1,T,V) から末尾を抽出して (1,1,V)）
        logits_full = self.proj(x)                   # (1,T,V)
        logits_last = logits_full[:, -1:, :]         # (1,1,V)
        last_tok = ids[:, -1]                        # (1,)

        rng = jax.random.PRNGKey(0)
        return logits_last, (last_tok, tuple(states), rng)

    # ---------- 生成 (1 step decode) ----------
    def generate_once(self,
                      new_tok_1d: jnp.ndarray,
                      state_tuple: Tuple[jnp.ndarray, Tuple[Dict[str, Any], ...], Any],
                      mask: Optional[Any] = None):
        """
        new_tok_1d: (1,) uint32
        state_tuple: (last_tok(1,), states(tuple of dict), rng)
        return: logits(B=1,T=1,V), (last_tok(1,), new_states, rng)
        """
        _last, states, rng = state_tuple
        B = 1
        # xB1D
        x = self.embed(new_tok_1d.reshape((1, 1)))   # (1,1,D)
        # 各層: pre-LN → decode_once → 残差 → pre-LN → MLP → 残差
        new_states = []
        for layer, st in zip(self.layers_mod, states):
            xn = layer.norm(x)
            delta, st2 = layer.decode_once(st, xn)   # delta: (1,1,D)
            x = x + delta
            xn2 = layer.norm(x)
            ff = layer.mlp_block(xn2)
            x = x + ff
            new_states.append(st2)

        logits = self.proj(x)                        # (1,1,V)
        return logits, (new_tok_1d.reshape((1,)), tuple(new_states), rng)


# ========== “外向け” ラッパ（既存コードの呼び出し互換） ==========
class CausalTransformer:
    """
    既存の Colab/ノートからはこのクラスを使います。
    - __init__(cfg) だけを受け付ける（Haiku 互換の余剰引数は廃止）
    - init_shmap(params_fd, ctx_1d, gen_len=...) → (logits0, (last, states, rng))
    - decode_shmap(params_fd, new_tok_1d, state_tuple) → (logits, (last, states, rng))
    """
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = dict(cfg)
        self.shard_mod = CausalTransformerShard(
            layers=int(cfg["layers"]),
            d_model=int(cfg["d_model"]),
            n_heads=int(cfg["n_heads"]),
            d_head=int(cfg["d_head"]),
            n_vocab=int(cfg["n_vocab"]),
            rotary_dim=int(cfg.get("pe_rotary_dims", 64)),
            norm=str(cfg.get("norm", "layernorm")),
            pe=str(cfg.get("pe", "rotary")),
        )

    # xmap→shmap のインターフェイスを保つ（単一コアでは普通の apply）
    def init_shmap(self, params_fd, ctx_1d: jnp.ndarray, gen_len: int = 1):
        # gen_len は呼び出し元で static_argnames 付き @jit にして下さい
        return self.shard_mod.apply(params_fd, ctx_1d, int(gen_len),
                                    method=self.shard_mod.generate_initial)

    def decode_shmap(self, params_fd, new_tok_1d: jnp.ndarray, state_tuple):
        return self.shard_mod.apply(params_fd, new_tok_1d, state_tuple,
                                    method=self.shard_mod.generate_once)
