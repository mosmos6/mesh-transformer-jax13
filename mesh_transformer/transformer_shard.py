# Copyright (c) 2025
# CausalTransformer wrapper + Flax core module.
#  - 名前/パスは model_leaves_new.json と互換:
#    /embed/embed_layer/embedding
#    /transformer_layers_*/norm/{scale,offset}
#    /transformer_layers_*/{q,k,v,o}/Dense_0/{kernel,bias?=なし}
#    /transformer_layers_*/dense_proj{,_o}/Dense_0/{kernel,bias}
#    /proj/ReplicatedLayerNorm_0/{scale,offset}
#    /proj/Dense_0/{kernel,bias}
#
# 推論:
#   generate_initial(context, gen_len) -> (logits(1,1,V), (last_tok(1,), states(list), rng))
#   generate_once(new_tok(1,), state_tuple)            -> (logits(1,1,V), (last_tok, states, rng))

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from .layers import (
    EmbedBlock,
    TransformerLayerShard,
    ProjectionShard,
    LayerCfg,
)

# --------------------------
# Flax core
# --------------------------

class _CausalTransformerCore(nn.Module):
    # モデル寸法
    layers: int
    d_model: int
    n_heads: int
    d_head: int
    n_vocab: int
    # 位置埋め
    pe: str = "rotary"
    pe_rotary_dims: int = 64

    def setup(self):
        # embed
        self.embed = EmbedBlock(n_vocab=self.n_vocab, d_model=self.d_model, name="embed")

        # transformer layers
        cfg = LayerCfg(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_head=self.d_head,
            pe=self.pe,
            pe_rotary_dims=self.pe_rotary_dims,
        )
        self.transformer_layers = [
            TransformerLayerShard(cfg, name=f"transformer_layers_{i}") for i in range(self.layers)
        ]

        # lm head
        self.proj = ProjectionShard(d_model=self.d_model, n_vocab=self.n_vocab, name="proj")

    # ============ prefix + KV プリフィル ============
    def generate_initial(self, ctx_1d: jnp.ndarray, gen_len: int):
        """
        ctx_1d: (T,) uint32
        返り値:
          logits_last: (1,1,V)
          (last_tok(1,), states(list of dict), rng)
        """
        assert ctx_1d.ndim == 1, "ctx_1d は (T,) を想定"
        T = ctx_1d.shape[0]
        B = 1

        # embed: (1,T,D) / bf16
        x = self.embed(ctx_1d[None, :])

        states: List[Dict[str, jnp.ndarray]] = []

        # レイヤ順伝播（pre-LN → attn → MLP）
        for layer in self.transformer_layers:
            # pre-LN 入力（全トークン）
            xn = layer.norm(x)  # (1,T,D), bf16

            # ---- KV プリフィル（RoPE は K のみに適用）----
            # ここで出るのは TBHD の prefix（T,B,H,Dh）; bf16 に落としてからキャッシュへ
            kTBHD_prefix, vTBHD_prefix = layer.kv_for_prefix(xn)  # bf16

            total_len = T + int(gen_len)
            H, Dh = layer.n_heads, layer.d_head

            k_cache = jnp.zeros((total_len, B, H, Dh), dtype=jnp.bfloat16)
            v_cache = jnp.zeros((total_len, B, H, Dh), dtype=jnp.bfloat16)

            # ★ ここで dtype を合わせてから scatter（FutureWarning 回避）
            k_cache = k_cache.at[:T].set(kTBHD_prefix.astype(k_cache.dtype))
            v_cache = v_cache.at[:T].set(vTBHD_prefix.astype(v_cache.dtype))

            # 「次に書く位置」を T で初期化（= 最初の decode は index=T に書く）
            cur = jnp.array(T, dtype=jnp.int32)
            states.append({"k": k_cache, "v": v_cache, "cur_index": cur})

            # ---- 本体の prefix attention + MLP（学習時と同じ）----
            attn = layer.attend_full(xn)  # (1,T,D)
            x = x + attn
            xn2 = layer.norm(x)
            ff = layer.mlp_block(xn2)
            x = x + ff

        # LM head → 末尾トークンの logits
        logitsBT = self.proj(x)                # (1,T,V) float32
        logits_last = logitsBT[:, -1:, :]      # (1,1,V)

        last_tok = jnp.array([ctx_1d[-1]], dtype=jnp.uint32)
        rng = jax.random.PRNGKey(0)  # dropout なしだが形だけ保持
        return logits_last, (last_tok, tuple(states), rng)

    

    # ============ 1 ステップ decode ============
    def generate_once(self, new_tok_1d: jnp.ndarray, state_tuple):
        """
        new_tok_1d: (1,) uint32
        state_tuple: (last(1,), states(tuple of dict), rng)
        返り値: (logits(1,1,V), (last, states, rng))
        """
        last_tok, states_in, rng = state_tuple
        assert new_tok_1d.shape == (1,), "new_tok_1d は (1,)"

        # embed current token: (1,1,D) / bf16
        x = self.embed(new_tok_1d[None, :])  # (1,1,D)

        new_states = []
        for layer, st in zip(self.transformer_layers, states_in):
            # decode_once: delta を返す（attn + mlp 合算）
            delta, st2 = layer.decode_once(st, x, attn_bias=None)
            x = x + delta
            new_states.append(st2)

        logits = self.proj(x)  # (1,1,V) float32
        return logits, (new_tok_1d, tuple(new_states), rng)


# --------------------------
# Python ラッパ (既存セル互換)
# --------------------------

class CausalTransformer:
    """既存 Colab セル互換の薄いラッパ。"""
    def __init__(self, cfg: Dict[str, Any]):
        # n_vocab は ckpt から実寸を取るのが安全だが、ここでは cfg をそのまま用いる
        self.layers = int(cfg["layers"])
        self.d_model = int(cfg["d_model"])
        self.n_heads = int(cfg["n_heads"])
        self.d_head = int(cfg["d_head"])
        self.n_vocab = int(cfg["n_vocab"])
        self.pe = cfg.get("pe", "rotary")
        self.pe_rotary_dims = int(cfg.get("pe_rotary_dims", 64))

        self.shard_mod = _CausalTransformerCore(
            layers=self.layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_head=self.d_head,
            n_vocab=self.n_vocab,
            pe=self.pe,
            pe_rotary_dims=self.pe_rotary_dims,
        )

    # 既存セルで呼ばれている API
    def init_shmap(self, params_fd, ctx_1d: jnp.ndarray, gen_len: int = 1):
        # gen_len は上位で static_argnames にすること
        return self.shard_mod.apply(
            params_fd, ctx_1d, int(gen_len), method=self.shard_mod.generate_initial
        )

    def decode_shmap(self, params_fd, new_tok_1d: jnp.ndarray, state_tuple):
        return self.shard_mod.apply(
            params_fd, new_tok_1d, state_tuple, method=self.shard_mod.generate_once
        )
