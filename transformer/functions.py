import math
from dataclasses import dataclass

import numpy as np
import torch
from einops import einsum, rearrange

from .embedding import Embedding
from .linear import MLP, Linear

# import torch.nn.functional as F


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: str | None = None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (batch_size, sequence_length, d_model)
        assert x.ndim == 3
        in_dtype = x.dtype
        x = x.to(torch.float32)

        RMS = torch.sqrt(torch.sum(torch.pow(x, 2), dim=-1, keepdim=True) / self.d_model + self.eps)
        result = x / RMS * self.gain

        return result.to(dtype=in_dtype)


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        theta_pows = 1 / torch.pow(theta, torch.arange(0, d_k, step=2, dtype=torch.float32) / d_k)  # shape = (d_k/2)
        positions = torch.arange(0, max_seq_len, step=1)  # shape(max_seq_length)
        # we want cos, sin of shape(max_seq_length, d_k)
        angle = einsum(positions, theta_pows, "max_seq_length, d_k -> max_seq_length d_k")
        self.register_buffer("cos", torch.cos(angle), persistent=False)
        self.register_buffer("sin", torch.sin(angle), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        # x(..., seq_len, d_k)
        # q1 = x[0::2], q2 =x[1::2]
        # q1, q2 shape(..., seq_len, d_k / 2)
        # x[0::2], x[1::2] = q1 * cos - q2 * sin, q1 * sin + q2 * cos
        # sin, cos shape(..., seq_len, d_k / 2)
        q1 = x[..., 0::2]
        q2 = x[..., 1::2]
        cos_cached = self.cos[token_positions]
        sin_cached = self.sin[token_positions]
        out = torch.empty_like(x)

        # Assign the rotated pairs to the new tensor
        out[..., 0::2] = q1 * cos_cached - q2 * sin_cached
        out[..., 1::2] = q1 * sin_cached + q2 * cos_cached
        return out


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x - torch.max(x, dim=dim, keepdim=True).values
    exp = torch.exp(x)
    return exp / torch.sum(exp, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    """
    Require shape = "... q_len d_q", mask = "... q_shape k_shape" Output is "... q_len d_v"
    """
    assert q.shape[-1] == k.shape[-1]
    assert k.shape[-2] == v.shape[-2]
    if mask is not None:
        assert mask.shape[-2:] == torch.Size((q.shape[-2], k.shape[-2]))
    d_q = q.shape[-1]
    d_k = k.shape[-1]
    d_v = v.shape[-1]
    # d_q == d_k
    qk = einsum(q, k, "... q_len d_q, ... k_len d_q -> ... q_len k_len") / math.sqrt(d_q)
    if mask is not None:
        qk = qk.masked_fill(mask == 0, float("-inf"))
    # k_len == v_len
    return einsum(softmax(qk, -1), v, "... q_len k_len, ... k_len d_v -> ... q_len d_v")


class multihead_self_attention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float = 10000,
        rope: RotaryPositionalEmbedding | None = None,
        device: str | None = None,
        dtype=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert d_model / num_heads == d_model // num_heads
        self.d_model = d_model
        self.num_q_heads = num_heads
        self.num_kv_heads = num_heads
        self.d_head = d_model // num_heads
        self.rope = rope
        self.theta = theta
        self.max_seq_len = max_seq_len

        self.q = Linear(d_model, self.num_q_heads * self.d_head)
        self.k = Linear(d_model, self.num_kv_heads * self.d_head)
        self.v = Linear(d_model, self.num_kv_heads * self.d_head)
        self.proj = Linear(self.d_head * self.num_kv_heads, d_model)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        assert x.ndim == 3
        B, T, D = x.shape
        q = self.q(x).reshape(B, T, self.num_q_heads, self.d_head).transpose(1, 2)  # shape = B, H, T, D
        k = self.k(x).reshape(B, T, self.num_kv_heads, self.d_head).transpose(1, 2)
        v = self.v(x).reshape(B, T, self.num_kv_heads, self.d_head).transpose(1, 2)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        if self.rope is None:
            self.rope = RotaryPositionalEmbedding(self.theta, k.shape[-1], self.max_seq_len)
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        o = scaled_dot_product_attention(q, k, v, mask)  # B, H, T, D how to concat on head?
        o = o.transpose(1, 2)  # B, T, H_kv, D_kv
        o = o.reshape(B, T, -1)  # B, T, d_model
        return self.proj(o)


class transformer_block(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        d_ff: int,
        theta: float = 10000,
        rope: RotaryPositionalEmbedding | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.ffn = MLP(d_model, d_ff)
        self.mha = multihead_self_attention(d_model, num_heads, max_seq_len, theta, rope)
        self.rms_norm1 = RMSNorm(d_model)
        self.rms_norm2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is B, T, C
        pos_ids = torch.arange(0, x.shape[1])
        pos_ids = rearrange(pos_ids, "seq -> 1 seq")
        x = x + self.mha(self.rms_norm1(x), token_positions=pos_ids)
        x = x + self.ffn(self.rms_norm2(x))
        return x


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6  # number of query heads
    d_ff: int = 2048
    # n_kv_head: int = 6 # number of key/value heads
    n_embd: int = 768


class transformer_lm(torch.nn.Module):
    def __init__(self, config: GPTConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding = Embedding(config.vocab_size, config.n_embd)
        d_k = config.n_embd // config.n_head
        self.config = config
        self.layers = [
            transformer_block(
                config.n_embd,
                config.n_head,
                config.sequence_len,
                config.d_ff,
                rope=RotaryPositionalEmbedding(10000, d_k, config.sequence_len),
            )
            for i in range(config.n_layer)
        ]
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = Linear(config.n_embd, config.vocab_size)

    def forward(self, token_ids: torch.Tensor):
        x = self.embedding(token_ids)
        for i in range(self.config.n_layer):
            x = self.layers[i](x)
        x = self.norm(x)
        x = self.lm_head(x)
        return x


def cross_entropy(logits, target):  # Float[Tensor, " batch_size vocab_size"], Int[Tensor, " batch_size"] -> float
    bs, vs = logits.shape
    logits = logits - torch.max(logits, dim=-1, keepdim=True).values
    exp = torch.exp(logits)
    loss = -(logits[torch.arange(bs), target] - torch.log((torch.sum(exp, dim=-1))))
    return torch.sum(loss) / bs


def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str):
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    inputs = torch.tensor(
        np.stack([dataset[start : start + context_length] for start in starts]),
        dtype=torch.long,
        device=device,
    )
    outputs = torch.tensor(
        np.stack([dataset[start + 1 : start + context_length + 1] for start in starts]),
        dtype=torch.long,
        device=device,
    )
    return inputs, outputs
