# -*- coding: utf-8 -*-
# layers.py — Flax linen 実装（delta 返却 / TBHD KV / RoPE=64 / FFN f32）
import jax
import jax.numpy as jnp
import flax.linen as nn

# ------------------------------------------------------------
#  LayerNorm（パラメータ名を leafspec に合わせる: scale / offset）
#  - 計算は f32、出力は bf16（元 dtype を尊重）
# ------------------------------------------------------------
class OffsetScaleLayerNorm(nn.Module):
    features: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        x_dtype = x.dtype
        x = x.astype(jnp.float32)
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var  = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        inv_std = jax.lax.rsqrt(var + self.eps)

        scale  = self.param("scale",  nn.initializers.ones,  (self.features,), jnp.bfloat16)
        offset = self.param("offset", nn.initializers.zeros, (self.features,), jnp.bfloat16)

        y = (x - mean) * inv_std
        y = y * scale.astype(jnp.float32) + offset.astype(jnp.float32)
        return y.astype(x_dtype)


# ------------------------------------------------------------
#  Embedding
#  - /embed/embed_layer/embedding
# ------------------------------------------------------------
class EmbeddingShard(nn.Module):
    config: dict
    mesh: object | None = None

    def setup(self):
        self.embed_layer = nn.Embed(
            num_embeddings=self.config["n_vocab"],
            features=self.config["d_model"],
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            name="embed_layer",
        )

    def __call__(self, xBT):
        # 入力は (B,T) または (T,) を許容
        if xBT.ndim == 1:
            xBT = xBT[None, :]
        return self.embed_layer(xBT)  # -> (B,T,D) bf16


# ------------------------------------------------------------
#  出力投影（最終 LN + Dense）
#  - /proj/ReplicatedLayerNorm_0/{scale,offset}
#  - /proj/Dense_0/{kernel,bias}
# ------------------------------------------------------------
class ProjectionShard(nn.Module):
    config: dict
    mesh: object | None = None

    def setup(self):
        D = self.config["d_model"]
        V = self.config["n_vocab"]
        # leafspec と同名: /proj/ReplicatedLayerNorm_0/*
        self.ReplicatedLayerNorm_0 = OffsetScaleLayerNorm(D, name="ReplicatedLayerNorm_0")
        # 計算 dtype は f32（安定化）、param は bf16（leafspec）
        self.Dense_0 = nn.Dense(
            V, use_bias=True,
            dtype=jnp.float32,       # matmul 出力を f32
            param_dtype=jnp.bfloat16,
            name="Dense_0"
        )

    def __call__(self, xBTD):
        x = self.ReplicatedLayerNorm_0(xBTD)  # (B,T,D) bf16
        x = x.astype(jnp.float32)
        logits = self.Dense_0(x)              # (B,T,V) f32
        return logits


# ------------------------------------------------------------
#  相対位置（本件は rotary を使うのでダミー）
# ------------------------------------------------------------
class RelativePositionEmbs(nn.Module):
    @nn.compact
    def __call__(self, q_len, k_len, n_heads):
        # 互換のため 0.0 を返す
        return 0.0


# ------------------------------------------------------------
#  RoPE (rotary position embedding) ユーティリティ
#  - GPT-J 想定: 先頭 pe_rotary_dims (=64) のみ回転、偶奇ペア
#  - 事前計算形: (T, rotary/2) の cos/sin を作り、内部で (1,T,1,rot/2) にブロードキャスト
#  - 単ステップ形: (rot/2,) から (1,1,1,rot/2)
# ------------------------------------------------------------
def _rope_inv_freq(rotary_dims: int):
    dim = rotary_dims
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    return inv_freq  # (rot/2,)

def rope_table(T: int, rotary_dims: int):
    inv = _rope_inv_freq(rotary_dims)              # (rot/2,)
    pos = jnp.arange(T, dtype=jnp.float32)[:, None]  # (T,1)
    ang = pos * inv[None, :]                        # (T, rot/2)
    return jnp.cos(ang), jnp.sin(ang)               # (T, rot/2)

def rope_single(pos_index: int, rotary_dims: int):
    inv = _rope_inv_freq(rotary_dims)               # (rot/2,)
    ang = (jnp.float32(pos_index)) * inv            # (rot/2,)
    return jnp.cos(ang), jnp.sin(ang)               # (rot/2,)

def apply_rope_qk(q, k, cos, sin, rotary_dims: int, single_step: bool = False):
    """
    q,k: (..., H, Dh) ここで Dh>=rotary_dims
    cos,sin:
      - 連番時: (T, rot/2)
      - 単ステップ: (rot/2,)
    出力: q', k'（同形状）
    """
    R = rotary_dims
    q_rot, q_pass = q[..., :R], q[..., R:]
    k_rot, k_pass = k[..., :R], k[..., R:]

    # 偶奇分離
    def _even(x): return x[..., ::2]
    def _odd (x): return x[..., 1::2]

    if single_step:
        # (rot/2,) -> (1,1,1,rot/2)
        cos = cos.reshape((1, 1, 1, -1))
        sin = sin.reshape((1, 1, 1, -1))
    else:
        # (T,rot/2) -> (1,T,1,rot/2)  … (B,T,H,rot/2) へ自然ブロードキャスト
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]

    # q
    q_even = _even(q_rot)
    q_odd  = _odd (q_rot)
    q_rot2 = jnp.stack([
        q_even * cos - q_odd * sin,
        q_odd  * cos + q_even * sin
    ], axis=-1).reshape(q_rot.shape)

    # k
    k_even = _even(k_rot)
    k_odd  = _odd (k_rot)
    k_rot2 = jnp.stack([
        k_even * cos - k_odd * sin,
        k_odd  * cos + k_even * sin
    ], axis=-1).reshape(k_rot.shape)

    q_out = jnp.concatenate([q_rot2, q_pass], axis=-1)
    k_out = jnp.concatenate([k_rot2, k_pass], axis=-1)
    return q_out, k_out


# ------------------------------------------------------------
#  Transformer Layer（delta 返却 / KV=TBHD）
#  - 1LN 並列残差: delta = Attn(LN(x)) + MLP(LN(x))
#  - get_init_decode_state:  全長で delta を返しつつ、KV=TBHD と cur_index=T を返す
#  - decode_once:            1 トークン分の delta と KV 更新
# ------------------------------------------------------------
class TransformerLayerShard(nn.Module):
    config: dict
    mesh: object | None = None

    def setup(self):
        D  = self.config["d_model"]
        H  = self.config["n_heads"]
        Dh = self.config["d_head"]

        self.D, self.H, self.Dh = D, H, Dh
        self.pe_rotary_dims = int(self.config.get("pe_rotary_dims", 0))
        self.scale = jnp.float32(Dh) ** -0.5

        # 単一 LN（leafspec: /transformer_layers_*/norm/{scale,offset}）
        self.norm = OffsetScaleLayerNorm(D, name="norm")

        # Q K V O（leafspec に合わせて属性名を付ける）
        self.q = nn.Dense(
            D, use_bias=False, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name="q"
        )
        self.k = nn.Dense(
            D, use_bias=False, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name="k"
        )
        self.v = nn.Dense(
            D, use_bias=False, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name="v"
        )
        self.o = nn.Dense(
            D, use_bias=False, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, name="o"
        )

        # MLP（FFN は f32 計算で安定化）
        self.dense_proj = nn.Dense(
            4 * D, use_bias=True, dtype=jnp.float32, param_dtype=jnp.bfloat16, name="dense_proj"
        )
        self.dense_proj_o = nn.Dense(
            D, use_bias=True, dtype=jnp.float32, param_dtype=jnp.bfloat16, name="dense_proj_o"
        )

    # ---- 内部ユーティリティ ----
    def _split_heads(self, xBTD):
        B, T, D = xBTD.shape
        H, Dh = self.H, self.Dh
        return xBTD.reshape(B, T, H, Dh)

    def _merge_heads(self, xBTHD):
        B, T, H, Dh = xBTHD.shape
        return xBTHD.reshape(B, T, H * Dh)

    def _causal_mask(self, T):
        # (T,T) 上三角（対角より上）に -1e10
        m = jnp.triu(jnp.ones((T, T), dtype=bool), 1)
        return jnp.where(m, jnp.float32(-1e10), jnp.float32(0.0))  # (T,T)

    def _attn_parallel_delta(self, xBTD):
        """
        並列残差: delta = Attn(LN(x)) + MLP(LN(x))
        ここではフル長（B,T,D）を想定し、自己注意 & causal mask を適用。
        KV を返すのは get_init_decode_state 側。
        """
        B, T, _ = xBTD.shape
        D, H, Dh = self.D, self.H, self.Dh

        x_norm = self.norm(xBTD)                    # (B,T,D) bf16
        # --- QKV（bf16）→ f32 に上げて内積
        q = self._split_heads(self.q(x_norm))       # (B,T,H,Dh) bf16
        k = self._split_heads(self.k(x_norm))       # (B,T,H,Dh) bf16
        v = self._split_heads(self.v(x_norm))       # (B,T,H,Dh) bf16

        # RoPE（先頭 pe_rotary_dims のみ）
        if self.pe_rotary_dims > 0:
            cos, sin = rope_table(T, self.pe_rotary_dims)        # (T,rot/2)
            q, k = apply_rope_qk(q, k, cos, sin, self.pe_rotary_dims, single_step=False)

        q32 = q.astype(jnp.float32)
        k32 = k.astype(jnp.float32)
        v32 = v.astype(jnp.float32)

        # 注意スコア: (B,H,T,T)
        attn_scores = jnp.einsum("bthd,bshd->bhts", q32, k32) * self.scale
        attn_scores = attn_scores + self._causal_mask(T)[None, None, :, :]
        attn = jax.nn.softmax(attn_scores, axis=-1)

        ctx = jnp.einsum("bhts,bshd->bthd", attn, v32)           # (B,T,H,Dh) f32
        attn_out = self.o(self._merge_heads(ctx).astype(jnp.bfloat16))  # (B,T,D) bf16

        # MLP（f32 計算 → bf16 戻し）
        h = self.dense_proj(x_norm.astype(jnp.float32))          # (B,T,4D) f32
        h = jax.nn.gelu(h)
        h = self.dense_proj_o(h)                                 # (B,T,D) f32
        mlp_out = h.astype(jnp.bfloat16)

        delta = (attn_out.astype(jnp.float32) + mlp_out.astype(jnp.float32)).astype(jnp.bfloat16)
        return delta, (k, v)   # delta=(B,T,D), KV=(B,T,H,Dh)

    # ---- 呼び出し（全長）: delta を返す ----
    def __call__(self, xBTD, attn_bias, layer_index, state=None):
        delta, _ = self._attn_parallel_delta(xBTD)
        return delta  # (B,T,D) bf16

    # ---- 初期デコード: KV 構築（TBHD）+ delta 返却 ----
    def get_init_decode_state(self, xBTD, given_length: int, attn_bias, mesh_manager=None):
        """
        xBTD: (B,T,D)
        戻り値:
          delta: (B,T,D)
          state: {"k": (T,B,H,Dh), "v": (T,B,H,Dh), "cur_index": T}
        """
        delta, (kBTHD, vBTHD) = self._attn_parallel_delta(xBTD)      # (B,T,H,Dh)
        # KV を TBHD に並べ替え
        kTBHD = jnp.swapaxes(kBTHD, 0, 1)    # (T,B,H,Dh)
        vTBHD = jnp.swapaxes(vBTHD, 0, 1)
        B, T, _ = xBTD.shape
        state = {
            "k": kTBHD,
            "v": vTBHD,
            "cur_index": jnp.array(T, dtype=jnp.int32)   # 既存長
        }
        return delta, state

    # ---- 単ステップ decode: 1 トークンの delta と KV 追記 ----
    def decode_once(self, decode_state, xB1D, attn_bias):
        """
        decode_state: {"k": (T,B,H,Dh), "v": (T,B,H,Dh), "cur_index": int32}
        xB1D: (B,1,D)
        戻り値:
          delta: (B,1,D)
          new_state: 同上（T->T+1）
        """
        kTBHD, vTBHD = decode_state["k"], decode_state["v"]
        cur = int(decode_state["cur_index"])
        B = xB1D.shape[0]

        x_norm = self.norm(xB1D)                                 # (B,1,D)
        q = self._split_heads(self.q(x_norm))                    # (B,1,H,Dh)
        k = self._split_heads(self.k(x_norm))                    # (B,1,H,Dh)
        v = self._split_heads(self.v(x_norm))                    # (B,1,H,Dh)

        # RoPE: 単ステップ（現在位置=cur）
        if self.pe_rotary_dims > 0:
            cos1, sin1 = rope_single(cur, self.pe_rotary_dims)   # (rot/2,)
            q, k = apply_rope_qk(q, k, cos1, sin1, self.pe_rotary_dims, single_step=True)

        # KV 追記（TBHD）
        k_step_TBHD = jnp.swapaxes(k, 0, 1)  # (1,B,H,Dh)
        v_step_TBHD = jnp.swapaxes(v, 0, 1)
        kTBHD_new = jnp.concatenate([kTBHD, k_step_TBHD], axis=0)  # (T+1,B,H,Dh)
        vTBHD_new = jnp.concatenate([vTBHD, v_step_TBHD], axis=0)

        # 注意（q against 全キー）
        kBTHD = jnp.swapaxes(kTBHD_new, 0, 1).astype(jnp.float32)  # (B,T+1,H,Dh)
        vBTHD = jnp.swapaxes(vTBHD_new, 0, 1).astype(jnp.float32)  # (B,T+1,H,Dh)
        q32 = q.astype(jnp.float32)

        scores = jnp.einsum("b1hd,bthd->bh1t", q32, kBTHD) * self.scale  # (B,H,1,T+1)
        attn  = jax.nn.softmax(scores, axis=-1)
        ctx   = jnp.einsum("bh1t,bthd->b1hd", attn, vBTHD)               # (B,1,H,Dh)
        attn_out = self.o(self._merge_heads(ctx).astype(jnp.bfloat16))   # (B,1,D) bf16

        # MLP（f32）
        h = self.dense_proj(x_norm.astype(jnp.float32))
        h = jax.nn.gelu(h)
        h = self.dense_proj_o(h)                                          # (B,1,D) f32
        mlp_out = h.astype(jnp.bfloat16)

        delta = (attn_out.astype(jnp.float32) + mlp_out.astype(jnp.float32)).astype(jnp.bfloat16)

        new_state = {
            "k": kTBHD_new,
            "v": vTBHD_new,
            "cur_index": jnp.array(cur + 1, dtype=jnp.int32),
        }
        return delta, new_state
