"""Reusable building blocks. Pure PyTorch only -- no third-party layers,
so the whole stack installs from a local wheel cache with no internet."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


class SwiGLU(nn.Module):
    def __init__(self, d: int, mult: float = 8 / 3):
        super().__init__()
        hidden = int(d * mult / 8) * 8
        self.w12 = nn.Linear(d, 2 * hidden, bias=False)
        self.w3 = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, -1)
        return self.w3(F.silu(a) * b)


def rope_cache(seq: int, head_dim: int, device, dtype, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq, device=device).float()
    freqs = torch.outer(pos, inv)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def apply_rope(x, cos, sin):
    # x: (B, H, T, D)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None], sin[None, None]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.stack((o1, o2), -1).flatten(-2)


class SelfAttention(nn.Module):
    """Bidirectional self-attention with rotary positions.

    Bidirectional is correct here: the context window is entirely in the past
    at decision time, so there is no leakage -- every token the model attends
    to is already observed. Causal masking inside the window would only
    discard information.
    """

    def __init__(self, d: int, heads: int):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x, rope=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if rope is not None:
            cos, sin = rope
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(B, T, D))


class CrossAttention(nn.Module):
    def __init__(self, d: int, heads: int, d_kv: int | None = None):
        super().__init__()
        d_kv = d_kv or d
        self.h, self.dh = heads, d // heads
        self.q = nn.Linear(d, d, bias=False)
        self.kv = nn.Linear(d_kv, 2 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x, mem):
        B, T, D = x.shape
        S = mem.shape[1]
        q = self.q(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k, v = self.kv(mem).view(B, S, 2, self.h, self.dh).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(B, T, D))


class EncoderLayer(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.n1, self.attn = RMSNorm(d), SelfAttention(d, heads)
        self.n2, self.ff = RMSNorm(d), SwiGLU(d)

    def forward(self, x, rope=None):
        x = x + self.attn(self.n1(x), rope)
        return x + self.ff(self.n2(x))


class DecoderLayer(nn.Module):
    """Self-attention over the predictor's own tokens, then cross-attention
    into the Market Analyser's memory."""

    def __init__(self, d: int, heads: int, d_mem: int):
        super().__init__()
        self.n1, self.attn = RMSNorm(d), SelfAttention(d, heads)
        self.n2, self.xattn = RMSNorm(d), CrossAttention(d, heads, d_mem)
        self.n3, self.ff = RMSNorm(d), SwiGLU(d)

    def forward(self, x, mem, rope=None):
        x = x + self.attn(self.n1(x), rope)
        x = x + self.xattn(self.n2(x), mem)
        return x + self.ff(self.n3(x))


class ConvStem(nn.Module):
    """Strided depthwise-separable conv stack: (B, C_in, T) -> (B, d, T/factor).

    Downsampling before attention is what makes a 12 h / 43 200 s context
    tractable: each lane of 1800 steps becomes `1800/factor` tokens.
    """

    def __init__(self, c_in: int, d: int, factor: int = 8, layers: int | None = None):
        super().__init__()
        n = layers or int(math.log2(factor))
        assert 2**n == factor, "factor must be a power of two"
        mods, c = [], c_in
        for i in range(n):
            c_out = d if i == n - 1 else max(d // 2, 64)
            mods += [
                nn.Conv1d(c, c, 5, padding=2, groups=c) if c > 1 else nn.Identity(),
                nn.Conv1d(c, c_out, 1),
                nn.GELU(),
                nn.Conv1d(c_out, c_out, 4, stride=2, padding=1),
                nn.GroupNorm(8, c_out),
                nn.GELU(),
            ]
            c = c_out
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


class TCNBlock(nn.Module):
    """Dilated residual conv block (architecture variant B)."""

    def __init__(self, d: int, dilation: int, kernel: int = 5):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.norm = RMSNorm(d)
        self.dw = nn.Conv1d(d, d, kernel, padding=pad, dilation=dilation, groups=d)
        self.pw1 = nn.Linear(d, 2 * d)
        self.pw2 = nn.Linear(2 * d, d)

    def forward(self, x):  # (B, T, d)
        h = self.norm(x)
        h = self.dw(h.transpose(1, 2)).transpose(1, 2)
        return x + self.pw2(F.gelu(self.pw1(h)))


class GatedStateBlock(nn.Module):
    """Gated diagonal state-space mixer (architecture variant C).

    A learned per-channel exponential decay implements an unbounded-horizon
    EMA, giving long memory at O(T) cost with no attention matrix. Decays are
    parameterised in log space and clamped to (0, 1) for stability.
    """

    def __init__(self, d: int, chunk: int = 64):
        super().__init__()
        self.chunk = chunk
        self.norm = RMSNorm(d)
        self.in_proj = nn.Linear(d, 3 * d)
        self.out_proj = nn.Linear(d, d)
        # init decays spread over timescales from ~2 to ~500 steps
        tau = torch.logspace(math.log10(2.0), math.log10(500.0), d)
        self.log_decay = nn.Parameter(torch.log(torch.exp(-1.0 / tau)))
        self.conv = nn.Conv1d(d, d, 4, padding=3, groups=d)

    def forward(self, x):  # (B, T, d)
        B, T, D = x.shape
        h = self.norm(x)
        v, g, r = self.in_proj(h).chunk(3, -1)
        v = self.conv(v.transpose(1, 2))[..., :T].transpose(1, 2)
        v = F.silu(v)

        a = torch.exp(self.log_decay).clamp(1e-4, 0.9999).float()  # (D,)

        # Chunked associative scan. Within a chunk the recurrence is applied
        # as a lower-triangular decay matmul (only *multiplies* by a^k, so
        # fast-decaying channels underflow gracefully to zero instead of
        # blowing up a reciprocal); the inter-chunk carry is a short loop.
        C = min(self.chunk, T)
        pad = (-T) % C
        vf = F.pad(v.float(), (0, 0, 0, pad))            # (B, T+pad, D)
        L = vf.shape[1] // C
        vf = vf.view(B, L, C, D)

        k = torch.arange(C, device=x.device, dtype=torch.float32)
        # decay[i, j, d] = a_d^(i-j) for i >= j else 0
        delta = k[:, None] - k[None, :]
        mask = delta >= 0
        decay = torch.where(
            mask[..., None], a.pow(delta.clamp_min(0)[..., None]), torch.zeros((), device=x.device)
        )                                                 # (C, C, D)
        intra = torch.einsum("ijd,bljd->blid", decay, vf)

        state = torch.zeros(B, D, device=x.device)
        outs = []
        a_pow_C = a.pow(float(C))
        last_w = a.pow((C - 1.0 - k)[:, None])[None]      # (1, C, D)
        for l in range(L):
            outs.append(intra[:, l] + state[:, None, :] * a.pow((k + 1.0)[:, None])[None])
            state = state * a_pow_C + (vf[:, l] * last_w).sum(1)
        y = torch.cat(outs, 1)[:, :T]
        y = (y * (1.0 - a)[None, None]).to(x.dtype)       # normalise EMA gain

        y = y * torch.sigmoid(g) + F.silu(r) * 0.1
        return x + self.out_proj(y)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
