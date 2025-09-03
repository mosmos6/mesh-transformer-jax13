# mesh_transformer/transformer_shard.py
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import remat

# メッシュ / shard_map
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

# プロジェクト内
from mesh_transformer.util import to_f32, to_bf16, global_norm
from mesh_transformer.layers import (
    EmbeddingShard, TransformerLayerShard, RelativePositionEmbs, ProjectionShard
)
from mesh_transformer.mesh_context_manager import MeshContextManager
from mesh_transformer.rng_manager import RNGManager

Array = jnp.ndarray


def _ensure_BT(x: Array) -> Array:
    """(T,) -> (1,T) / (T,D)->(1,T,D) に正規化。その他はそのまま。"""
    if x.ndim == 1:  # (T,)
        return x[None, :]
    return x


def _ensure_BTD(x: Array) -> Array:
    """(T,D)->(1,T,D) に正規化。その他はそのまま。"""
    if x.ndim == 2:
        return x[None, :, :]
    return x


class CausalTransformerShard(nn.Module):
    """Flax Module 本体（1レプリカ=1シャードの計算定義）"""
    config: Dict[str, Any]
    mesh_manager: MeshContextManager
    rng_manager: RNGManager

    def setup(self):
        cfg = self.config
        self.layers = int(cfg["layers"])
        self.d_model = int(cfg["d_model"])
        self.n_heads = int(cfg["n_heads"])
        self.d_head  = int(cfg["d_head"])
        self.heads_per_shard = self.n_heads // int(cfg["cores_per_replica"])
        self.pe_type = cfg.get("pe", "rotary")
        self.pe_rotary_dims = int(cfg.get("pe_rotary_dims", 64))

        mesh = self.mesh_manager.get_mesh()
        # Embed / Blocks / LM head
        self.embed = EmbeddingShard(config=cfg, mesh=mesh)
        self.transformer_layers = [
            remat(TransformerLayerShard)(config=cfg, mesh=mesh)
            for _ in range(self.layers)
        ]
        self.proj = remat(ProjectionShard)(config=cfg, mesh=mesh)

        self.rpe = RelativePositionEmbs() if self.pe_type == "t5" else None

    # --------- forward (B,T) -> (B,T,V) ----------
    def __call__(self, xBT: Array, mask: float = 0.0) -> Array:
        # ids (B,T) -> (B,T,D)
        x = self.embed(xBT)
        x = _ensure_BTD(x)  # (B,T,D)
        B, T, _ = x.shape

        if self.rpe is not None:
            attn_bias = self.rpe(T, T, self.heads_per_shard)
        else:
            attn_bias = mask

        # ここは「層が delta を返す」契約で統一
        for li, layer in enumerate(self.transformer_layers):
            delta = layer(x, attn_bias, li)  # (B,T,D)
            x = x + delta

        return self.proj(x)  # (B,T,V)

    # --------- KV 初期化（prefix 一括） ----------
    def generate_initial(self, context_1d: Array, gen_len: int, mask: float = 0.0):
        """context_1d: (T,) uint32。戻り: (logits(B, T-1, V)), (last(1,), states, rng)"""
        assert isinstance(gen_len, int), "gen_len must be Python int (static)"
        ctx = jnp.asarray(context_1d, jnp.uint32)
        assert ctx.ndim == 1, "pass (T,) ids here"

        last = ctx[-1: ]           # (1,)
        prefix = ctx[:-1]          # (T-1,)
        x = self.embed(prefix)     # (T-1,D) もしくは (B=1,T-1,D)
        x = _ensure_BTD(x)         # -> (1,T-1,D)
        B, Tm1, _ = x.shape

        if self.rpe is not None:
            attn_bias = self.rpe(Tm1, Tm1, self.heads_per_shard)
        else:
            attn_bias = mask

        states = []
        for li, layer in enumerate(self.transformer_layers):
            # 契約: (residual_delta, layer_state)
            delta, st = layer.get_init_decode_state(x, gen_len - 1, attn_bias, self.mesh_manager)
            x = x + delta
            states.append(st)

        logits = self.proj(x)  # (1,T-1,V)
        rng = self.rng_manager.get_current_key()
        return logits, (last, tuple(states), rng)

    # --------- 単ステップ decode ----------
    def generate_once(self, new_tok_1d: Array, state_tuple, mask: float = 0.0):
        """new_tok_1d:(1,) uint32 / state_tuple=(last(1,), states, rng)"""
        last, states, rng = state_tuple  # 互換のため last/rng も保持

        x = self.embed(new_tok_1d)  # (1,D)
        if x.ndim == 2:
            x = x[:, None, :]       # -> (1,1,D)
        else:
            x = _ensure_BTD(x)

        # キャッシュ長は (B,T,H,Dh) の T 軸
        kv_len = int(states[0]["v"].shape[1]) if (len(states) > 0 and "v" in states[0]) else 0

        if self.rpe is not None:
            # ここで 1 ステップ分のバイアスに整形
            attn_bias_full = self.rpe(kv_len + 1, kv_len + 1, self.heads_per_shard)
            attn_bias = attn_bias_full[:, -1:, :]  # 末尾 1 ステップ
        else:
            attn_bias = mask

        new_states = []
        for li, (layer, st) in enumerate(zip(self.transformer_layers, states)):
            # 契約: (residual_delta, new_state)
            delta, st2 = layer.decode_once(st, x, attn_bias)
            x = x + delta
            new_states.append(st2)

        logits = self.proj(x)  # (1,1,V)
        return logits, (last, tuple(new_states), rng)


# =========================
#  外側ラッパ: shard_map を保持
# =========================
class CausalTransformer:
    """
    * dp×mp メッシュを張り、shard_map 付きの関数を公開する外側ラッパ。
    * device_count()==1 の時は JIT にフォールバック（コードとしては shard_map を保持）。
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config.copy()
        self.rng_manager = RNGManager(seed=0)

        # メッシュ準備
        cores_per_replica = int(self.config.get("cores_per_replica", 1))
        devs = jax.devices()
        dp = max(1, len(devs) // cores_per_replica)
        mp = cores_per_replica
        self.mesh_manager = MeshContextManager(dp, mp)
        self.mesh: Mesh = self.mesh_manager.get_mesh()

        # 本体 Module
        self.shard_mod = CausalTransformerShard(self.config, self.mesh_manager, self.rng_manager)

        # ===== shmap: generate_initial =====
        def _gen_init_fn(params, ctx_1d, gen_len: int):
            return self.shard_mod.apply(params, ctx_1d, int(gen_len), method=self.shard_mod.generate_initial)

        # バッチ分散を想定するなら in_specs に P("dp") 等を当てるが、
        # ここでは「形だけ保持」。単一コアでは JIT フォールバック。
        if jax.device_count() > 1:
            self.init_shmap = jax.jit(
                shard_map(
                    _gen_init_fn,
                    in_specs=(P(), P(), None),
                    out_specs=(P(), P()),
                    mesh=self.mesh
                ),
                static_argnames=("gen_len",),
            )
        else:
            self.init_shmap = jax.jit(_gen_init_fn, static_argnames=("gen_len",))

        # ===== shmap: decode_once =====
        def _gen_once_fn(params, new_tok_1d, state_tuple):
            return self.shard_mod.apply(params, new_tok_1d, state_tuple, method=self.shard_mod.generate_once)

        if jax.device_count() > 1:
            self.decode_shmap = jax.jit(
                shard_map(
                    _gen_once_fn,
                    in_specs=(P(), P(), P()),
                    out_specs=(P(), P()),
                    mesh=self.mesh
                )
            )
        else:
            self.decode_shmap = jax.jit(_gen_once_fn)




        
        
