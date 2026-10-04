"""
Streaming, time-ordered sampling over the bar store.

There are deliberately **no epochs**. Training walks the market clock forward
one second at a time (optionally decimated by `step`), exactly as the live
system will experience it, and is scored against the realised future at every
cursor position. An "epoch" would require shuffling, which destroys the
temporal-consistency signal the stability loss depends on.

Rank sharding: the timeline is cut into `world_size` contiguous shards and each
rank streams its own. Contiguity preserves per-rank temporal order (so the
consistency loss is well defined), while different shards decorrelate the
gradients that DDP averages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from torch.utils.data import IterableDataset

from .features import (
    DEFAULT_LANES,
    BarStore,
    LaneSpec,
    build_context,
    build_target,
    horizon_grid,
)


@dataclass
class SplitSpec:
    """Train / validation boundary expressed in days from the end."""

    val_days: float = 14.0
    test_days: float = 7.0

    def bounds(self, n: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        test_n = int(self.test_days * 86400)
        val_n = int(self.val_days * 86400)
        test_lo = n - test_n
        val_lo = test_lo - val_n
        return (0, val_lo), (val_lo, test_lo), (test_lo, n)


class MarketStream(IterableDataset):
    """Yields (context, target, anchor, t) tuples in strict market-time order.

    Parameters
    ----------
    lookahead
        Seconds of future required for the target (max horizon). Cursors stop
        this far from the end of the range so every sample has a real label.
    step
        Cursor advance in seconds. 1 = every second (the production setting);
        larger values decimate for faster coverage of long histories.
    consecutive
        Number of back-to-back cursors emitted as one group. The stability
        loss compares predictions within a group, so this must be >= 2 for
        that term to be active.
    """

    def __init__(
        self,
        store: BarStore,
        lo: int,
        hi: int,
        lanes: tuple[LaneSpec, ...] = DEFAULT_LANES,
        horizons: np.ndarray | None = None,
        step: int = 1,
        consecutive: int = 2,
        rank: int = 0,
        world_size: int = 1,
        loop: bool = True,
        seed: int = 0,
    ):
        self.store = store
        self.lanes = lanes
        self.horizons = horizon_grid() if horizons is None else horizons
        self.step = step
        self.consecutive = max(1, consecutive)
        self.loop = loop
        self.seed = seed

        warmup = max(ln.span for ln in lanes)
        lookahead = int(self.horizons[-1])
        lo = max(lo, warmup - 1)
        hi = min(hi, len(store) - lookahead - 1)
        if hi - lo < self.consecutive * step:
            raise ValueError(
                f"range [{lo},{hi}) too small for warmup={warmup} "
                f"lookahead={lookahead}; need a longer history"
            )

        # Contiguous per-rank shard.
        width = (hi - lo) // world_size
        self.lo = lo + rank * width
        self.hi = self.lo + width if world_size > 1 else hi
        self.rank = rank

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.rank)
        # Random phase so restarts don't replay an identical cursor sequence.
        t = self.lo + int(rng.integers(0, max(1, self.step * self.consecutive)))
        while True:
            if t + self.consecutive * self.step >= self.hi:
                if not self.loop:
                    return
                t = self.lo
            for k in range(self.consecutive):
                tc = t + k * self.step
                ctx, anchor = build_context(self.store, tc, self.lanes)
                tgt = build_target(self.store, tc, self.horizons, anchor)
                yield ctx, tgt, np.float32(anchor), np.int64(tc)
            t += self.consecutive * self.step


def collate(batch):
    import torch

    ctx = torch.from_numpy(np.stack([b[0] for b in batch]))
    tgt = torch.from_numpy(np.stack([b[1] for b in batch]))
    anchor = torch.tensor([b[2] for b in batch])
    t = torch.tensor([b[3] for b in batch])
    return ctx, tgt, anchor, t
