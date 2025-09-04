# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Optional

import math
import jax
import jax.numpy as jnp
from flax import linen as nn

from .layers import (
    EmbedBlock,
    ProjectionShard,
    TransformerLayerShard,
    LayerCfg,
)

# -----------------------------
# Flax Module (shard) 本体
# -----------------------------
class CausalTransformerShard(nn.Module):
    # モデル構成
    layers: int
    d_model: int
    n_heads: int
    d_head: int
    n_vocab: int
    pe: str = 'rotary'
    pe_rotary_dims: int = 64

    def setup(self):
        # Embedding
        self.embed = EmbedBlock(n_vocab=self.n_vocab, d_model=self.d_model, name='embed')
        # Transformer layers
        cfg = LayerCfg(self.d_model, self.n_heads, self.d_head, pe=self.pe, pe_rotary_dims=self.pe_rotary_dims)
        self.layers_mod = [
            TransformerLayerShard(cfg, name=f"transformer_layers_{i}")
            for i in range(self.layers)
        ]
        # LM Head
        self.proj = ProjectionShard(d_model=self.d_model, n_vocab=self.n_vocab, name='proj')

    # ---- prefix 実行 + KV 構築 ----
    @nn.compact
    def generate_initial(self, ctx_1d: jnp.ndarray, gen_len: int) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, Tuple[Dict[str, jnp.ndarray], ...], jnp.ndarray]]:
        """
        ctx_1d: (T,), uint32
        return:
          logits_last: (B=1, 1, V) float32
          (last_tok(1,), states(tuple of dict), rng_key)
        """
        assert ctx_1d.ndim == 1, "ctx_1d must be (T,)"
        B = 1
        T = ctx_1d.shape[0]
        total_len = T + int(gen_len)

        # Embed prefix
        xBT = ctx_1d[None, :]                        # (1,T)
        xBTD = self.embed(xBT)                       # (1,T,D) bf16

        states: List[Dict[str, jnp.ndarray]] = []

        # 層ごとに prefix のフル注意 + MLP を実行しつつ KV prefill
        for layer in self.layers_mod:
            # pre-LN attention
            xn = layer.norm(xBTD)
            attn = layer.attend_full(xn)
            xBTD = xBTD + attn
            # pre-LN MLP
            xn2 = layer.norm(xBTD)
            ff = layer.mlp_block(xn2)
            xBTD = xBTD + ff

            # KV 初期化 + prefix 分を詰める
            st = layer.init_decode_state(total_len=total_len, batch=B)     # {"k":(Ttot,B,H,Dh), "v":..., "cur_index":0}
            kTBHD, vTBHD = layer.prefill_kv(xn)                             # (T,B,H,Dh) for prefix (注意: xnで整合)
            st["k"] = st["k"].at[:T].set(kTBHD)
            st["v"] = st["v"].at[:T].set(vTBHD)
            st["cur_index"] = jnp.array(T, dtype=jnp.int32)                 # 次に書き込む位置
            states.append(st)

        # 直近位置のロジット（B,1,V）
        x_last = xBTD[:, -1:, :]                    # (1,1,D)
        logits_last = self.proj(x_last)             # (1,1,V) float32

        last_tok = ctx_1d[-1:].astype(jnp.uint32)   # (1,)
        rng = jax.random.PRNGKey(0)
        return logits_last, (last_tok, tuple(states), rng)

    # ---- 1 ステップ decode ----
    @nn.compact
    def generate_once(self, new_tok_1d: jnp.ndarray, state_tuple: Tuple[jnp.ndarray, Tuple[Dict[str, jnp.ndarray], ...], jnp.ndarray]
                      ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, Tuple[Dict[str, jnp.ndarray], ...], jnp.ndarray]]:
        """
        new_tok_1d: (1,), uint32
        state_tuple: (last_tok(1,), states(tuple of dict), rng_key)
        """
        last_tok, states, rng = state_tuple
        assert new_tok_1d.ndim == 1 and new_tok_1d.shape[0] == 1

        # Embed token → (1,1,D)
        xB1 = new_tok_1d[None, :]                    # (1,1)
        xB1D = self.embed(xB1)                       # (1,1,D) bf16

        new_states: List[Dict[str, jnp.ndarray]] = []
        # 各層で decode_once
        for layer, st in zip(self.layers_mod, states):
            xn = layer.norm(xB1D)
            delta, st2 = layer.decode_once(st, xn, attn_bias=None)
            xB1D = xB1D + delta
            new_states.append(st2)

        logits = self.proj(xB1D)                     # (1,1,V) float32
        return logits, (new_tok_1d, tuple(new_states), rng)

# -----------------------------
# 薄いラッパ（既存 call site 互換）
# -----------------------------
class CausalTransformer:
    def __init__(self, cfg: Dict[str, Any]):
        self.layers = int(cfg["layers"])
        self.d_model = int(cfg["d_model"])
        self.n_heads = int(cfg["n_heads"])
        self.d_head  = int(cfg["d_head"])
        self.n_vocab = int(cfg["n_vocab"])
        self.pe = str(cfg.get("pe", "rotary"))
        self.pe_rotary_dims = int(cfg.get("pe_rotary_dims", 64))

        self.shard_mod = CausalTransformerShard(
            layers=self.layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_head=self.d_head,
            n_vocab=self.n_vocab,
            pe=self.pe,
            pe_rotary_dims=self.pe_rotary_dims,
        )

    # 既存の init_shmap / decode_shmap API
    def init_shmap(self, params_fd, ctx_1d: jnp.ndarray, gen_len: int = 1):
        return self.shard_mod.apply(params_fd, ctx_1d, int(gen_len), method=self.shard_mod.generate_initial)

    def decode_shmap(self, params_fd, new_tok_1d: jnp.ndarray, state_tuple):
        return self.shard_mod.apply(params_fd, new_tok_1d, state_tuple, method=self.shard_mod.generate_once)
