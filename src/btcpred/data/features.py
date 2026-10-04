"""
Multi-resolution feature construction over the 1-second bar store.

A literal 12 h x 1 s context is 43 200 steps -- far too long for attention and
mostly redundant, since BTC microstructure decorrelates in seconds while the
25-minute horizon is driven by much slower state. We therefore build three
equal-length "lanes" that tile the same 12 h with geometrically coarser
resolution, so recent history keeps full fidelity and distant history is
summarised rather than discarded:

    lane 0   last     30 min  @  1 s   -> 1800 steps
    lane 1   last      3 h    @  6 s   -> 1800 steps
    lane 2   last     12 h    @ 24 s   -> 1800 steps

Coarse lanes aggregate correctly (open=first, high=max, low=min, close=last,
volume/count=sum), so no trade is lost -- only resolved more coarsely.

Every feature is scale-free and anchored on the *last observed close* of the
window, which is the only quantity the model is allowed to know at decision
time. This makes the representation stationary across a 2020-2026 price range
spanning $4k to $120k, which raw prices emphatically are not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_FEATURES = 8
FEATURE_NAMES = (
    "logret_close",   # log(close_t / anchor)
    "logret_step",    # log(close_t / close_{t-1})  within the lane
    "hl_range",       # log(high/low)
    "oc_body",        # log(close/open)
    "log_volume",     # log1p(volume)
    "log_trades",     # log1p(n_trades)
    "buy_imbalance",  # 2*taker_buy/volume - 1  in [-1, 1], 0 when no volume
    "gap_flag",       # fraction of underlying seconds with zero trades
)


@dataclass(frozen=True)
class LaneSpec:
    stride: int  # seconds per step
    steps: int   # steps in the lane

    @property
    def span(self) -> int:
        return self.stride * self.steps


DEFAULT_LANES = (LaneSpec(1, 1800), LaneSpec(6, 1800), LaneSpec(24, 1800))


class BarStore:
    """Read-only memmap view over a resampled 1-second bar store."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        self.n = int(self.meta["n_seconds"])
        self.start_ts = int(self.meta["start_ts"])
        self.scale = float(self.meta.get("price_scale", 100))
        self.prices = np.memmap(
            self.root / "prices.i64", np.int64, "r", shape=(self.n, 4)
        )
        self.flows = np.memmap(
            self.root / "flows.f32", np.float32, "r", shape=(self.n, 4)
        )

    def __len__(self) -> int:
        return self.n

    def close(self, i: int | slice) -> np.ndarray:
        return self.prices[i, 3].astype(np.float64) / self.scale

    def ts(self, i: int) -> int:
        return self.start_ts + i


def _aggregate(
    prices: np.ndarray, flows: np.ndarray, stride: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fold a (steps*stride, 4) block down to (steps, 4) bars."""
    if stride == 1:
        return prices.astype(np.float64), flows.astype(np.float64)
    p = prices.reshape(-1, stride, 4).astype(np.float64)
    f = flows.reshape(-1, stride, 4).astype(np.float64)
    out_p = np.empty((p.shape[0], 4), np.float64)
    out_p[:, 0] = p[:, 0, 0]            # open  = first
    out_p[:, 1] = p[:, :, 1].max(1)     # high  = max
    out_p[:, 2] = p[:, :, 2].min(1)     # low   = min
    out_p[:, 3] = p[:, -1, 3]           # close = last
    out_f = np.empty((f.shape[0], 4), np.float64)
    out_f[:, 0] = f[:, :, 0].sum(1)     # volume
    out_f[:, 1] = f[:, :, 1].sum(1)     # n_trades
    out_f[:, 2] = f[:, :, 2].sum(1)     # taker buy volume
    out_f[:, 3] = f[:, -1, 3]           # vwap (last sub-bar)
    return out_p, out_f


def encode_lane(
    prices: np.ndarray, flows: np.ndarray, stride: int, anchor: float, scale: float
) -> np.ndarray:
    """Build the (steps, N_FEATURES) float32 tensor for one lane."""
    p, f = _aggregate(prices, flows, stride)
    o, h, lo, c = (p[:, k] / scale for k in range(4))
    vol, cnt, buy = f[:, 0], f[:, 1], f[:, 2]

    eps = 1e-12
    out = np.empty((p.shape[0], N_FEATURES), np.float64)
    out[:, 0] = np.log(np.maximum(c, eps) / anchor)
    prev = np.concatenate(([c[0]], c[:-1]))
    out[:, 1] = np.log(np.maximum(c, eps) / np.maximum(prev, eps))
    out[:, 2] = np.log(np.maximum(h, eps) / np.maximum(lo, eps))
    out[:, 3] = np.log(np.maximum(c, eps) / np.maximum(o, eps))
    out[:, 4] = np.log1p(np.maximum(vol, 0.0))
    out[:, 5] = np.log1p(np.maximum(cnt, 0.0))
    out[:, 6] = np.where(vol > 0, 2.0 * buy / np.maximum(vol, eps) - 1.0, 0.0)
    out[:, 7] = (cnt == 0).astype(np.float64) if stride == 1 else \
        1.0 - np.minimum(cnt, stride) / stride

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# Per-feature scale factors. Chosen so each channel lands at roughly unit
# variance on BTC 1 s data; fixed constants (not dataset statistics) keep the
# encoder identical between offline training and live inference.
FEATURE_GAIN = np.array(
    [200.0, 2000.0, 1000.0, 2000.0, 0.5, 0.5, 1.0, 1.0], np.float32
)


def build_context(
    store: BarStore,
    t: int,
    lanes: tuple[LaneSpec, ...] = DEFAULT_LANES,
) -> tuple[np.ndarray, float]:
    """Context tensor at decision time `t` (inclusive of bar t).

    Returns (lanes, steps, N_FEATURES) float32 and the anchor price.
    Caller must guarantee t >= max lane span.
    """
    anchor = float(store.prices[t, 3]) / store.scale
    feats = np.empty((len(lanes), lanes[0].steps, N_FEATURES), np.float32)
    for k, ln in enumerate(lanes):
        lo = t + 1 - ln.span
        feats[k] = encode_lane(
            np.asarray(store.prices[lo : t + 1]),
            np.asarray(store.flows[lo : t + 1]),
            ln.stride,
            anchor,
            store.scale,
        )
    feats *= FEATURE_GAIN
    return np.clip(feats, -16.0, 16.0, out=feats), anchor


def horizon_grid(minutes: int = 25, every_sec: int = 60) -> np.ndarray:
    """Target offsets in seconds: 60, 120, ... 1500 for the default 25 min."""
    return np.arange(every_sec, minutes * 60 + 1, every_sec, dtype=np.int64)


def build_target(
    store: BarStore, t: int, horizons: np.ndarray, anchor: float
) -> np.ndarray:
    """Realised log-returns from the anchor at each horizon.

    This reads the *actual recorded market data* -- the same ground truth the
    simulator is forbidden to show the model -- and is what every per-second
    prediction is scored against.
    """
    idx = t + horizons
    future = store.prices[idx, 3].astype(np.float64) / store.scale
    return np.log(np.maximum(future, 1e-12) / anchor).astype(np.float32)
