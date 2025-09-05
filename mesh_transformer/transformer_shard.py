# Copyright (c) 2025
# Preserves shard_map-based API while working on single-core too.
# Matches parameter names/paths expected by model_leaves_new.json:
#  - /embed/embed_layer/embedding
#  - /transformer_layers_{i}/{q,k,v,o}/Dense_0/{kernel(,bias)}
#  - /transformer_layers_{i}/dense_proj/Dense_0/{kernel,bias}
#  - /transformer_layers_{i}/dense_proj_o/Dense_0/{kernel,bias}
#  - /transformer_layers_{i}/norm/{scale,offset}
#  - /proj/ReplicatedLayerNorm_0/{scale,offset}, /proj/Dense_0/{kernel,bias}

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

# ==== local modules (names kept) ====
from .layers import (
    EmbedBlock,
    ProjectionShard,
    TransformerLayerShard,
    LayerCfg,
)
from .mesh_context_manager import MeshContextManager  # axis_names: ('dp','mp','core') or ('single_core',) :contentReference[oaicite:1]{index=1}


# -----------------------------------------------------------------------------
# 内部ユーティリティ
# -----------------------------------------------------------------------------
def _ensure_uint32(x: jnp.ndarray) -> jnp.ndarray:
    return x.astype(jnp.uint32)

def _expand_ctx(ctx_1d: jnp.ndarray) -> jnp.ndarray:
    """(T,) -> (1,T) へ拡張"""
    if ctx_1d.ndim == 1:
        return ctx_1d[None, :]
    return ctx_1d

def _ensure_batched_ids(ids: jnp.ndarray) -> Tuple[jnp.ndarray, int, int]:
    """ids を (B,T) に正規化して (idsBT, B, T) を返す。"""
    if ids.ndim == 1:
        ids = ids[None, :]
    B, T = ids.shape
    return ids, B, T


# -----------------------------------------------------------------------------
# 1 shard ぶんの「推論ネットワーク」(Flax Module)
#  - ここが重みスコープの「正解」：/embed, /transformer_layers_*, /proj
# -----------------------------------------------------------------------------
class CausalTransformerShard(nn.Module):
    layers: int
    d_model: int
    n_heads: int
    d_head: int
    n_vocab: int
    rotary_dim: int  # pe_rotary_dims

    def setup(self):
        # /embed/...
        self.embed = EmbedBlock(n_vocab=self.n_vocab, d_model=self.d_model, name="embed")

        # /transformer_layers_{i}/...
        cfg = LayerCfg(d_model=self.d_model, n_heads=self.n_heads, d_head=self.d_head,
                       pe='rotary', pe_rotary_dims=self.rotary_dim)
        self.layers_mod: List[TransformerLayerShard] = [
            TransformerLayerShard(cfg, name=f"transformer_layers_{i}") for i in range(self.layers)
        ]

        # /proj/...
        self.proj = ProjectionShard(d_model=self.d_model, n_vocab=self.n_vocab, name="proj")

    # -------------------------------------------------------------------------
    # (B,T) コンテキストから初回ログイットと KV-state を作る
    # 返値: logitsB1V, (lastTokB, tuple(states_per_layer), rng_key)
    # -------------------------------------------------------------------------
    def generate_initial(self, ctx_BT: jnp.ndarray, gen_len: int
                         ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, Tuple[Dict[str, jnp.ndarray], ...], jnp.ndarray]]:
        """
        ctx_BT: (B,T) uint32
        """
        idsBT, B, T = _ensure_batched_ids(_ensure_uint32(ctx_BT))
        x = self.embed(idsBT)  # (B,T,D) bf16

        states: List[Dict[str, jnp.ndarray]] = []
        total_len = T + int(gen_len)

        # レイヤごとに pre-LN -> Attention -> 残差 -> pre-LN -> MLP -> 残差
        # かつ、Attention の入力 pre-LN から prefix の K/V を生成して KV キャッシュに格納。
        for layer in self.layers_mod:
            xn = layer.norm(x)                 # (B,T,D)  params: /transformer_layers_i/norm
            attn = layer.attend_full(xn)       # (B,T,D)
            x = x + attn

            xn2 = layer.norm(x)                # (B,T,D)
            ff = layer.mlp_block(xn2)          # (B,T,D)
            x = x + ff

            # --- KV prefill (K-only RoPE 済み) ---
            kTBHD, vTBHD = layer.kv_for_prefix(xn)   # (T,B,H,Dh)
            # preallocate to total_len and place prefix at [:T]
            H, Dh = layer.n_heads, layer.d_head
            k_all = jnp.zeros((total_len, B, H, Dh), dtype=jnp.bfloat16)
            v_all = jnp.zeros((total_len, B, H, Dh), dtype=jnp.bfloat16)
            k_all = k_all.at[:T].set(kTBHD)
            v_all = v_all.at[:T].set(vTBHD)
            # 次に書き込む位置（= prefix 長）
            cur = jnp.array(T, dtype=jnp.int32)
            states.append({"k": k_all, "v": v_all, "cur_index": cur})

        # 最終トークン位置のロジット (B,1,V) を返す（float32）
        logitsB1V = self.proj(x[:, -1:, :])  # (B,1,V) float32
        last_tokB = idsBT[:, -1]             # (B,) uint32
        rng = jax.random.PRNGKey(0)          # 生成側の契約維持（未使用でも形は返す）
        return logitsB1V, (last_tokB, tuple(states), rng)




    # -------------------------------------------------------------------------
    # 1 トークン step デコード（KV 使用）
    # 入力: new_tok_B (B,), states (tuple of dict)
    # 返値: logitsB1V, (new_tok_B, new_states, rng)
    # -------------------------------------------------------------------------
    def generate_once(self, new_tok_B: jnp.ndarray,
                      state_tuple: Tuple[jnp.ndarray, Tuple[Dict[str, jnp.ndarray], ...], jnp.ndarray]
                      ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, Tuple[Dict[str, jnp.ndarray], ...], jnp.ndarray]]:
        """
        new_tok_B: (B,) uint32
        state_tuple: (lastTokB (unused here), tuple(states per layer), rng)
        """
        new_tok_B = _ensure_uint32(new_tok_B)
        lastTokB, states_in, rng = state_tuple
        B = new_tok_B.shape[0]

        # 埋め込みして (B,1,D)
        x = self.embed(new_tok_B[:, None])  # bf16

        new_states: List[Dict[str, jnp.ndarray]] = []
        for layer, st in zip(self.layers_mod, states_in):
            # layer.decode_once は (deltaB1D, new_state) を返す
            delta, st2 = layer.decode_once(st, x, attn_bias=None)
            x = x + delta
            new_states.append(st2)

        logitsB1V = self.proj(x)  # float32
        return logitsB1V, (new_tok_B, tuple(new_states), rng)

        # 例: 1 回だけ
        jax.debug.print("[kv] cur:{c} k0:{k} v0:{v}", c=new_states[0]['cur_index'], k=new_states[0]['k'].shape, v=new_states[0]['v'].shape)



# -----------------------------------------------------------------------------
# ユーザーフレンドリなラッパ（shard_map を保持しつつ単一コアにも対応）
# -----------------------------------------------------------------------------
class CausalTransformer:
    """Public wrapper used by notebooks:
       - init_shmap(params_fd, ctx_1d, gen_len)
       - decode_shmap(params_fd, new_tok_1d, state_tuple)
       shard_map は B 次元を dp で割る設計。B 次元が無い/1 の場合はフォールバック。
    """
    def __init__(self, cfg: Dict[str, Any]):
        self.layers = int(cfg["layers"])
        self.d_model = int(cfg["d_model"])
        self.n_heads = int(cfg["n_heads"])
        self.d_head = int(cfg["d_head"])
        self.n_vocab = int(cfg["n_vocab"])
        self.rotary_dim = int(cfg.get("pe_rotary_dims", 0))

        # Flax Module （パラメタスコープの正解）
        self.shard_mod = CausalTransformerShard(
            layers=self.layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_head=self.d_head,
            n_vocab=self.n_vocab,
            rotary_dim=self.rotary_dim,
        )

        # メッシュ（dp, mp, core）: 単一コア時もコンテキストは保持
        dp = int(cfg.get("dp", 1))
        mp = int(cfg.get("mp", 1))
        core = int(cfg.get("cores_per_replica", 1))
        self.mesh_ctx = MeshContextManager(dp=dp, mp=mp, core=core)  # :contentReference[oaicite:2]{index=2}
        self.mesh = self.mesh_ctx.get_mesh()

    # ----------------------- 内部：生関数（shard_map で包む前） -----------------------
    def _init_fn(self, params_fd, ctx_BT, gen_len: int):
        # ctx は (B,T) でも (T,) でも可。内部で (B,T) に正規化
        return self.shard_mod.apply(params_fd, ctx_BT, int(gen_len),
                                    method=self.shard_mod.generate_initial)

    def _step_fn(self, params_fd, new_tok_B, state_tuple):
        return self.shard_mod.apply(params_fd, new_tok_B, state_tuple,
                                    method=self.shard_mod.generate_once)

    # ----------------------- 公開 API：shard_map を「保持」 -----------------------
    def init_shmap(self, params_fd, ctx_1d: jnp.ndarray, gen_len: int = 1):
        """ctx_1d: (T,) or (B,T) のトークン id。B>1 かつ複数デバイス時は shard_map 使用。"""
        idsBT = _expand_ctx(ctx_1d)  # (B,T) へ
        B = int(idsBT.shape[0])

        use_shmap = (B > 1) and (jax.device_count() > 1)
        if use_shmap:
            # B を dp に割る想定。簡易 spec（複雑なパーティショニングはここでは省略）
            fn = shard_map(
                lambda p, x, g: self._init_fn(p, x, g),
                mesh=self.mesh,
                in_specs=(P(), P('dp', None), P()),
                out_specs=(P('dp', None, None),  # logits (B,1,V) -> dp に沿って割る想定
                           (P('dp',),  # lastTokB
                            P(),        # states（木構造は replication）
                            P())),      # rng
            )
            return fn(params_fd, idsBT, int(gen_len))
        else:
            return self._init_fn(params_fd, idsBT, int(gen_len))

    def decode_shmap(self, params_fd, new_tok_1d: jnp.ndarray, state_tuple):
        """new_tok_1d: (B,) or (1,)。B>1 かつ複数デバイス時は shard_map 使用。"""
        new_tok_B = new_tok_1d if new_tok_1d.ndim == 1 else new_tok_1d.reshape(-1)
        B = int(new_tok_B.shape[0])

        use_shmap = (B > 1) and (jax.device_count() > 1)
        if use_shmap:
            fn = shard_map(
                lambda p, t, st: self._step_fn(p, t, st),
                mesh=self.mesh,
                in_specs=(P(), P('dp',), P()),
                out_specs=(P('dp', None, None),  # logits (B,1,V)
                           (P('dp',), P(), P())),  # (lastTokB, states, rng)
            )
            return fn(params_fd, new_tok_B, state_tuple)
        else:
            return self._step_fn(params_fd, new_tok_B, state_tuple)

#jax.debug.print("[kv] shapes k:{k} v:{v} cur:{c}", k=state[0]['k'].shape, v=state[0]['v'].shape, c=state[0]['cur_index'])

