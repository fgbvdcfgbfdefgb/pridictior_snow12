"""
Price Predictor -- maps (Market Analyser output + raw 12 h context) to a
quantile forecast of the next 25 minutes.

Three interchangeable trunks are provided so several instances can be trained
concurrently against the *same* analyser and compared head-to-head:

    "xfmr"  bidirectional transformer with cross-attention into the analyser
    "tcn"   dilated depthwise conv stack (strong local inductive bias, cheapest)
    "ssm"   gated diagonal state-space mixer (O(T) long memory, no attn matrix)

All three share the input stem and the horizon-query head, so a comparison
isolates the trunk rather than confounding it with head capacity.

Output
------
(B, n_horizons, n_quantiles) log-returns relative to the anchor price.
Quantiles are emitted as a median plus cumulative positive offsets, which makes
crossing (q10 > q90) structurally impossible rather than merely penalised.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.features import N_FEATURES
from .blocks import (
    ConvStem,
    CrossAttention,
    DecoderLayer,
    GatedStateBlock,
    RMSNorm,
    TCNBlock,
    count_params,
    rope_cache,
)

QUANTILES = (0.1, 0.5, 0.9)


@dataclass
class PredictorConfig:
    name: str = "xfmr"
    trunk: str = "xfmr"  # xfmr | tcn | ssm
    d_model: int = 1024
    n_layers: int = 9
    n_heads: int = 16
    d_mem: int = 768
    n_lanes: int = 3
    lane_steps: int = 1800
    downsample: int = 8
    n_features: int = N_FEATURES
    n_horizons: int = 25
    quantiles: tuple[float, ...] = field(default_factory=lambda: QUANTILES)


class HorizonHead(nn.Module):
    """One learned query per horizon, cross-attending into the trunk.

    Queries are initialised along a smooth ramp and the final projection is
    shared across horizons, so neighbouring horizons start out correlated --
    this is a large part of why the forecast curve comes out smooth in time
    instead of jagged across the 25 steps.
    """

    def __init__(self, d: int, heads: int, n_h: int, n_q: int):
        super().__init__()
        self.n_h, self.n_q = n_h, n_q
        ramp = torch.linspace(-1, 1, n_h)[:, None]
        self.query = nn.Parameter(ramp * torch.randn(1, d) * 0.02 + torch.randn(n_h, d) * 0.01)
        self.norm = RMSNorm(d)
        self.attn = CrossAttention(d, heads, d)
        self.ff = nn.Sequential(RMSNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.out_med = nn.Linear(d, 1)
        self.out_spread = nn.Linear(d, n_q - 1)
        nn.init.zeros_(self.out_med.weight)
        nn.init.zeros_(self.out_med.bias)

    def forward(self, trunk):
        B = trunk.shape[0]
        q = self.query[None].expand(B, -1, -1)
        q = q + self.attn(self.norm(q), trunk)
        q = q + self.ff(q)
        med = self.out_med(q)                              # (B, H, 1)
        spread = F.softplus(self.out_spread(q)) + 1e-6     # (B, H, n_q-1)
        mid = (self.n_q - 1) // 2
        lower = med - torch.flip(torch.cumsum(torch.flip(spread[..., :mid], [-1]), -1), [-1])
        upper = med + torch.cumsum(spread[..., mid:], -1)
        return torch.cat([lower, med, upper], -1)          # (B, H, n_q)


class PricePredictor(nn.Module):
    def __init__(self, cfg: PredictorConfig | None = None):
        super().__init__()
        self.cfg = cfg or PredictorConfig()
        c = self.cfg
        d = c.d_model

        # The predictor re-reads the raw 12 h context itself rather than
        # relying solely on the analyser, as specified: it gets both.
        self.stems = nn.ModuleList(
            [ConvStem(c.n_features, d, c.downsample) for _ in range(c.n_lanes)]
        )
        self.lane_emb = nn.Parameter(torch.zeros(c.n_lanes, 1, d) + 0.02 * torch.randn(c.n_lanes, 1, d))
        self.summary_proj = nn.Linear(c.d_mem, d)

        if c.trunk == "xfmr":
            self.layers = nn.ModuleList([DecoderLayer(d, c.n_heads, c.d_mem) for _ in range(c.n_layers)])
        elif c.trunk == "tcn":
            self.mem_mix = nn.ModuleList(
                [CrossAttention(d, c.n_heads, c.d_mem) for _ in range(c.n_layers // 3)]
            )
            self.layers = nn.ModuleList(
                [TCNBlock(d, dilation=2 ** (i % 6)) for i in range(c.n_layers * 2)]
            )
        elif c.trunk == "ssm":
            self.mem_mix = nn.ModuleList(
                [CrossAttention(d, c.n_heads, c.d_mem) for _ in range(c.n_layers // 3)]
            )
            self.layers = nn.ModuleList([GatedStateBlock(d) for i in range(c.n_layers * 2)])
        else:
            raise ValueError(f"unknown trunk {c.trunk!r}")

        self.norm = RMSNorm(d)
        self.head = HorizonHead(d, c.n_heads, c.n_horizons, len(c.quantiles))
        self.register_buffer(
            "q_levels", torch.tensor(c.quantiles, dtype=torch.float32), persistent=False
        )

    def forward(self, lanes: torch.Tensor, mem: torch.Tensor, summary: torch.Tensor):
        B, L = lanes.shape[0], lanes.shape[1]
        toks = [self.stems[k](lanes[:, k].transpose(1, 2)).transpose(1, 2) + self.lane_emb[k]
                for k in range(L)]
        x = torch.cat(toks, 1)
        x = x + self.summary_proj(summary)[:, None, :]

        head_dim = self.cfg.d_model // self.cfg.n_heads
        rope = rope_cache(x.shape[1], head_dim, x.device, x.dtype)

        if self.cfg.trunk == "xfmr":
            for layer in self.layers:
                x = layer(x, mem, rope)
        else:
            every = max(1, len(self.layers) // max(1, len(self.mem_mix)))
            mi = 0
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i % every == every - 1 and mi < len(self.mem_mix):
                    x = x + self.mem_mix[mi](x, mem)
                    mi += 1

        return self.head(self.norm(x))

    @property
    def median_index(self) -> int:
        return (len(self.cfg.quantiles) - 1) // 2

    def n_params(self) -> int:
        return count_params(self)


def build_variants(d_mem: int, n_horizons: int, scale: float = 1.0) -> list[PredictorConfig]:
    """The default three-way architecture sweep, sharing one analyser."""
    def s(x):
        return max(64, int(round(x * scale / 64)) * 64)

    return [
        PredictorConfig(name="xfmr", trunk="xfmr", d_model=s(1024), n_layers=8,
                        n_heads=16, d_mem=d_mem, n_horizons=n_horizons),
        PredictorConfig(name="tcn", trunk="tcn", d_model=s(1024), n_layers=13,
                        n_heads=16, d_mem=d_mem, n_horizons=n_horizons),
        PredictorConfig(name="ssm", trunk="ssm", d_model=s(960), n_layers=15,
                        n_heads=16, d_mem=d_mem, n_horizons=n_horizons),
    ]
