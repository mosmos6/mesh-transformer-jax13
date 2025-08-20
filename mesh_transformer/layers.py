# mesh_transformer/layers.py
# Flax/shmap version faithful to GPT‑J (pre‑LN, rotary, scaled dot‑product, +O projection)

from functools import partial
import gc
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from einops import rearrange, repeat

from mesh_transformer.util import f_psum, g_psum  # degrade to identity on single-core
from jax.sharding import PartitionSpec as P

# -------------------- Norms --------------------

class ReplicatedLayerNorm(nn.Module):
    """LayerNorm with scale/offset replicated over model-parallel shards."""
    mesh: object
    offset: bool = True
    name: str = "layer_norm"

    @nn.compact
    def __call__(self, x):
        features = x.shape[-1]
        scale  = self.param("scale",  nn.initializers.ones,  (features,), x.dtype)
        if self.offset:
            offset = self.param("offset", nn.initializers.zeros, (features,), x.dtype)
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var  = jnp.var(x, axis=-1, keepdims=True)
        inv  = scale * jax.lax.rsqrt(var + self.eps)
        y = inv * (x - mean)
        if self.offset:
            y = y + offset
        return y


def getnorm(kind: str, *, mesh=None, name="norm"):
    if kind == "layernorm":
        return ReplicatedLayerNorm(mesh=mesh, name=name)
    elif kind == "layernorm-desync":
        return nn.LayerNorm(name=name)
    elif kind == "layernorm-nobias":
        return ReplicatedLayerNorm(mesh=mesh, offset=False, name=name)
    elif kind == "rmsnorm":
        # Simple RMS: scale only
        class _RMS(nn.Module):
            name: str = "rms_norm"
            @nn.compact
            def __call__(self, x):
                g = self.param("scale", nn.initializers.constant(x.shape[-1] ** 0.5), (x.shape[-1],))
                return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-5) * g
        return _RMS(name=name)
    else:
        raise NotImplementedError(f"norm kind '{kind}'")

# -------------------- Rotary helpers --------------------

def _fixed_pos_embedding(seq_len: int, dim: int):
    """Return sin,cos of shape (T, dim/2)."""
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.einsum("t,d->td", t, inv_freq)  # (T, dim/2)
    return jnp.sin(freqs), jnp.cos(freqs)

def _rotate_every_two(x):
    """(..., D) where D even -> rotation pairs (x0,x1)->(-x1,x0)."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = jnp.stack((-x2, x1), axis=-1)
    return rearrange(x_rot, "... d j -> ... (d j)")

def _apply_rotary(x, sin, cos):
    # x: (B, T, H, d_rotary), sin/cos: (T, 1, 1, d_rotary)
    return (x * cos) + (_rotate_every_two(x) * sin)

def _apply_rotary_qk(q, k, T, H, d_head, d_rotary):
    # q,k: (B, T, H, d_head) -> apply rotary on first d_rotary
    if d_rotary == 0:
        return q, k
    sin, cos = _fixed_pos_embedding(T, d_rotary)  # (T, d_rotary/2)
    # expand to (T, 1, 1, d_rotary)
    sin = repeat(sin, "t d -> t 1 1 (d j)", j=2)[:, :, :, :d_rotary]
    cos = repeat(cos, "t d -> t 1 1 (d j)", j=2)[:, :, :, :d_rotary]

    q_rot, q_pass = q[..., :d_rotary], q[..., d_rotary:]
    k_rot, k_pass = k[..., :d_rotary], k[..., d_rotary:]
    q_rot = _apply_rotary(q_rot, sin, cos)
    k_rot = _apply_rotary(k_rot, sin, cos)
    q = jnp.concatenate([q_rot, q_pass], axis=-1)
    k = jnp.concatenate([k_rot, k_pass], axis=-1)
    return q, k

# -------------------- Embedding --------------------

class EmbeddingShard(nn.Module):
    """Single unsharded embedding matrix (50400, 4096)."""
    config: dict
    mesh: object
    name: str = "embed"

    def setup(self):
        self.embed_layer = nn.Embed(
            num_embeddings=int(self.config["n_vocab"]),
            features=int(self.config["d_model"]),
            name="embed_layer"
        )

    def __call__(self, x):
        # x: (B, T) -> (B, T, D)
        return self.embed_layer(x)

# -------------------- Transformer layer --------------------

class TransformerLayerShard(nn.Module):
    """GPT‑J pre‑LN attention + MLP with learned O projection."""
    config: dict
    mesh: object
    init_scale: float = 1.0
    name: str = None  # let the caller assign "transformer_layers_{i}"

    def setup(self):
        self.D   = int(self.config["d_model"])
        self.H   = int(self.config["n_heads"])
        self.dh  = self.D // self.H
        self.pe_rotary_dims = int(self.config.get("pe_rotary_dims", self.dh))
        assert self.D % self.H == 0, "d_model must be divisible by n_heads"

        # Pre-LN
        self.norm = getnorm(self.config["norm"], mesh=self.mesh, name="norm")

        # Q/K/V and O projection – all (4096, 4096) kernels to match your leaf shapes
        self.q = nn.Dense(self.D, use_bias=False, name="q")
        self.k = nn.Dense(self.D, use_bias=False, name="k")
        self.v = nn.Dense(self.D, use_bias=False, name="v")
        self.o = nn.Dense(self.D, use_bias=False, name="o",
                          kernel_init=nn.initializers.truncated_normal(stddev=self.init_scale / np.sqrt(self.D)))

        # MLP: (4096, 16384) and (16384, 4096) + biases
        self.dense_proj   = nn.Dense(4 * self.D, name="dense_proj")
        self.dense_proj_o = nn.Dense(self.D, name="dense_proj_o",
                                     kernel_init=nn.initializers.truncated_normal(stddev=self.init_scale / np.sqrt(self.D)))

        self.is_rotary = (self.config.get("pe", "rotary") == "rotary")

    def _attn(self, x_norm, attn_bias):
        """x_norm: (B, T, D)  -> attn_out: (B, T, D)"""
        B, T, _ = x_norm.shape

        # Projections
        q = self.q(x_norm)  # (B,T,D)
        k = self.k(x_norm)
        v = self.v(x_norm)

        # Split heads
        q = q.reshape(B, T, self.H, self.dh)
        k = k.reshape(B, T, self.H, self.dh)
        v = v.reshape(B, T, self.H, self.dh)

        # Rotary (first pe_rotary_dims)
        if self.is_rotary and self.pe_rotary_dims > 0:
            q, k = _apply_rotary_qk(q, k, T=T, H=self.H, d_head=self.dh, d_rotary=self.pe_rotary_dims)

        # Scaled dot-product attention with causal mask
        attn_scores = jnp.einsum("bthd, bThd -> bhtT", q, k)  # (B,H,T,T)
        attn_scores = attn_scores / jnp.sqrt(jnp.asarray(self.dh, dtype=attn_scores.dtype))

        # causal mask (T,T), broadcast over B,H
        causal = np.tril(np.ones((T, T), dtype=np.float32))
        attn_scores = attn_scores + (jnp.log(causal + 1e-9) - jnp.log1p(-causal))  # 0 or -inf

        if attn_bias is not None and jnp.ndim(attn_bias) > 0:
            # Expecting bias broadcastable to (B,H,T,T) or (H,T,T)
            attn_scores = attn_scores + attn_bias

        attn_weights = jax.nn.softmax(attn_scores, axis=-1)
        ctx = jnp.einsum("bhtT, bThd -> bthd", attn_weights, v)  # (B,T,H,dh)
        ctx = ctx.reshape(B, T, self.D)  # merge heads

        # Learned O projection
        return self.o(ctx)

    def _ff(self, x_norm):
        y = self.dense_proj(x_norm)
        y = jax.nn.gelu(y)
        y = self.dense_proj_o(y)
        return y

    @nn.compact
    def __call__(self, x, attn_bias, layer_index=None, state=None):
        """x: (B, T, D) -> (B, T, D)"""
        # (optional) psum identity on single-core
        x = f_psum(x)

        # Pre-LN branches
        x_norm = self.norm(x)
        attn_out = self._attn(x_norm, attn_bias)
        ff_out   = self._ff(x_norm)

        out = x + attn_out + ff_out
        # (optional) psum identity on single-core
        out = g_psum(out)
        return out

    # --- Decoding helpers (kept compatible with your current call sites; refine in step #4) ---

    def get_init_decode_state(self, x, given_length, attn_bias, mesh_manager=None):
        """x: (T,B,D) or (B,T,D) in your current generator; we won’t rely on this path yet."""
        # Minimal stub: produce a KV cache consistent with decode_once
        if x.ndim == 3 and x.shape[0] != 1:
            # (T,B,D) -> (B,T,D)
            xBTD = jnp.swapaxes(x, 0, 1)
        else:
            xBTD = x  # assume (B,T,D)

        B, T, _ = xBTD.shape
        x_norm = self.norm(xBTD)
        k = self.k(x_norm).reshape(B, T, self.H, self.dh)
        v = self.v(x_norm).reshape(B, T, self.H, self.dh)
        return jnp.zeros_like(xBTD), {"k": k, "v": v, "tokens_decoded": given_length.astype(jnp.uint32)}

    def decode_once(self, decode_state, x, attn_bias):
        """x: (B=1,1,D). We’ll compute 1-step attention against cached KV."""
        x_norm = self.norm(x)
        q = self.q(x_norm).reshape(1, 1, self.H, self.dh)
        # append new kv
        k_prev, v_prev = decode_state["k"], decode_state["v"]  # (1,T,H,dh)
        k_new = self.k(x_norm).reshape(1, 1, self.H, self.dh)
        v_new = self.v(x_norm).reshape(1, 1, self.H, self.dh)
        k = jnp.concatenate([k_prev, k_new], axis=1)
        v = jnp.concatenate([v_prev, v_new], axis=1)

        # rotary for last token position only (optional)
        T = k.shape[1]
        if self.is_rotary and self.pe_rotary_dims > 0:
            sin, cos = _fixed_pos_embedding(T, self.pe_rotary_dims)
            sin = repeat(sin, "t d -> t 1 1 (d j)", j=2)[:, :, :, :self.pe_rotary_dims]
            cos = repeat(cos, "t d -> t 1 1 (d j)", j=2)[:, :, :, :self.pe_rotary_dims]
            # apply to all keys; apply to q of the last token
            k_rot, k_pass = k[..., :self.pe_rotary_dims], k[..., self.pe_rotary_dims:]
            k_rot = (k_rot * cos) + (_rotate_every_two(k_rot) * sin)
            k = jnp.concatenate([k_rot, k_pass], axis=-1)

            q_rot, q_pass = q[..., :self.pe_rotary_dims], q[..., self.pe_rotary_dims:]
            q_rot = (q_rot * cos[-1:, ...]) + (_rotate_every_two(q_rot) * sin[-1:, ...])
            q = jnp.concatenate([q_rot, q_pass], axis=-1)

        # scores against full K
        scores = jnp.einsum("bthd,bThd->bhtT", q, k) / jnp.sqrt(jnp.asarray(self.dh, scores.dtype))
        # causal for last step = mask future (none here), bias if provided
        if attn_bias is not None and jnp.ndim(attn_bias) > 0:
            scores = scores + attn_bias
        w = jax.nn.softmax(scores, axis=-1)
        ctx = jnp.einsum("bhtT,bThd->bthd", w, v).reshape(1, 1, self.D)
        out = self.o(ctx) + self._ff(x_norm)

        state = {"k": k, "v": v, "tokens_decoded": decode_state["tokens_decoded"] + 1}
        return out, state

# -------------------- Projection head (norm + 4096→4096) --------------------

class ProjectionShard(nn.Module):
    config: dict
    mesh: object
    name: str = "proj"

    def setup(self):
        self.dim = int(self.config["d_model"])
        self.layer_norm = getnorm(self.config["norm"], mesh=self.mesh, name="layer_norm")
        self.dense = nn.Dense(self.dim, name="dense")  # (4096,4096) + bias

    @nn.remat
    def forward(self, x):
        x = self.layer_norm(x)
        return self.dense(x)

    @nn.compact
    def __call__(self, x):
        # Match training: LayerNorm -> Linear(n_vocab) with bias
        x = ReplicatedLayerNorm(offset=True)(x)
        n_vocab = int(self.config["n_vocab"])           # 50400 in your JSON
        d_model = int(self.config["d_model"])           # 4096
        # init similar scale as Haiku default (ok to keep it simple)
        kernel_init = nn.initializers.normal(stddev=1.0/np.sqrt(d_model))
        logits = nn.Dense(features=n_vocab, use_bias=True, kernel_init=kernel_init)(x)
        return logits

class RelativePositionEmbs(nn.Module):
    num_buckets: int
    max_distance: int

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        n = np.maximum(n, 0)
        max_exact = num_buckets // 2
        is_small = (n < max_exact)
        val_if_large = max_exact + (
                np.log(n.astype(np.float32) / max_exact + np.finfo(np.float32).eps) /
                np.log(max_distance / max_exact) *
                (num_buckets - max_exact)).astype(np.int32)
        val_if_large = np.minimum(val_if_large, num_buckets - 1)
        ret += np.where(is_small, n, val_if_large)
        return ret

    @nn.compact
    def __call__(self, qlen, klen, heads):
        context_position = np.arange(qlen, dtype=jnp.int32)[:, None]
        memory_position = np.arange(klen, dtype=jnp.int32)[None, :]
        relative_position = memory_position - context_position
        rp_bucket = self._relative_position_bucket(relative_position, self.num_buckets, self.max_distance)
        relative_attention_bias = self.param('rel_embedding', nn.initializers.truncated_normal(stddev=0.02), [heads, self.num_buckets])
        bcast_iota = jax.lax.broadcasted_iota(jnp.int32, (self.num_buckets, 1, 1), 0)
        rp_bucket_one_hot = jnp.array(rp_bucket[jnp.newaxis, Ellipsis] == bcast_iota).astype(relative_attention_bias.dtype)
        values = jax.lax.dot_general(
            relative_attention_bias,
            rp_bucket_one_hot,
            (((1,), (0,)), ((), ())))
        return values

