"""
Market Analyser -- the shared feature/signal extractor.

One instance is trained for the whole system; every Price Predictor variant
consumes its output. It sees all three resolution lanes, downsamples each with
a conv stem, tags them with a learned lane embedding, and runs a bidirectional
transformer over the concatenated token sequence.

Outputs
-------
memory  (B, S, d_model)  per-token signal field for predictors to attend into
summary (B, d_model)     attention-pooled global market state

Default size is ~80 M parameters, tuned for a 23 GB A10 alongside a ~150 M
predictor under DDP with bf16 autocast.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..data.features import N_FEATURES
from .blocks import ConvStem, EncoderLayer, RMSNorm, count_params, rope_cache


@dataclass
class AnalyserConfig:
    d_model: int = 768
    n_layers: int = 10
    n_heads: int = 12
    n_lanes: int = 3
    lane_steps: int = 1800
    downsample: int = 8
    n_features: int = N_FEATURES

    @property
    def tokens_per_lane(self) -> int:
        return self.lane_steps // self.downsample

    @property
    def n_tokens(self) -> int:
        return self.tokens_per_lane * self.n_lanes


class MarketAnalyser(nn.Module):
    def __init__(self, cfg: AnalyserConfig | None = None):
        super().__init__()
        self.cfg = cfg or AnalyserConfig()
        c = self.cfg
        # One stem per lane: the statistics of a 1 s bar and a 24 s bar differ
        # enough that sharing weights measurably hurts.
        self.stems = nn.ModuleList(
            [ConvStem(c.n_features, c.d_model, c.downsample) for _ in range(c.n_lanes)]
        )
        self.lane_emb = nn.Parameter(torch.zeros(c.n_lanes, 1, c.d_model))
        nn.init.normal_(self.lane_emb, std=0.02)
        self.layers = nn.ModuleList(
            [EncoderLayer(c.d_model, c.n_heads) for _ in range(c.n_layers)]
        )
        self.norm = RMSNorm(c.d_model)
        self.pool_q = nn.Parameter(torch.randn(1, 1, c.d_model) * 0.02)
        self.pool = nn.MultiheadAttention(c.d_model, c.n_heads, batch_first=True)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, lanes: torch.Tensor):
        """lanes: (B, n_lanes, lane_steps, n_features)"""
        B, L, T, F_ = lanes.shape
        toks = []
        for k in range(L):
            z = self.stems[k](lanes[:, k].transpose(1, 2))  # (B, d, T/ds)
            toks.append(z.transpose(1, 2) + self.lane_emb[k])
        x = torch.cat(toks, 1)  # (B, S, d)

        head_dim = self.cfg.d_model // self.cfg.n_heads
        rope = rope_cache(x.shape[1], head_dim, x.device, x.dtype)
        for layer in self.layers:
            x = layer(x, rope)
        mem = self.norm(x)

        q = self.pool_q.expand(B, -1, -1)
        summary, _ = self.pool(q, mem, mem, need_weights=False)
        return mem, summary.squeeze(1)

    def n_params(self) -> int:
        return count_params(self)
