"""
Live market simulator -- replays the recorded 1-second history as if it were
arriving in real time.

The simulator exposes exactly the interface the live feed will expose, so the
same model runner drives both backtest and production. Crucially it enforces
the information boundary: `observe()` returns only data up to and including the
current second, while `ground_truth()` reads ahead and is reserved for the
scorer. Nothing on the model path can reach future bars.

`speed` controls pacing:
    0     -> as fast as the CPU allows (training / backtests)
    1.0   -> true wall-clock real time, 1 simulated second per second
    60.0  -> 60x, useful for eyeballing the live dashboard quickly
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..data.features import (
    DEFAULT_LANES,
    BarStore,
    LaneSpec,
    build_context,
    build_target,
    horizon_grid,
)


@dataclass
class Tick:
    t: int              # index into the bar store
    ts: int             # unix seconds
    price: float        # last close -- the anchor
    context: np.ndarray # (lanes, steps, features)
    volume: float
    n_trades: float


class LiveSimulator:
    def __init__(
        self,
        store: BarStore,
        start: int | None = None,
        end: int | None = None,
        lanes: tuple[LaneSpec, ...] = DEFAULT_LANES,
        horizons: np.ndarray | None = None,
        speed: float = 0.0,
        step: int = 1,
    ):
        self.store = store
        self.lanes = lanes
        self.horizons = horizon_grid() if horizons is None else horizons
        self.speed = speed
        self.step = step

        warmup = max(ln.span for ln in lanes)
        self.lookahead = int(self.horizons[-1])
        self.start = max(warmup - 1, start if start is not None else warmup - 1)
        self.end = min(end if end is not None else len(store), len(store))
        self.t = self.start
        self._wall0 = None

    # -- model-visible ---------------------------------------------------
    def observe(self) -> Tick:
        ctx, anchor = build_context(self.store, self.t, self.lanes)
        return Tick(
            t=self.t,
            ts=self.store.ts(self.t),
            price=anchor,
            context=ctx,
            volume=float(self.store.flows[self.t, 0]),
            n_trades=float(self.store.flows[self.t, 1]),
        )

    # -- scorer-only -----------------------------------------------------
    def ground_truth(self, t: int | None = None) -> np.ndarray | None:
        """Realised log-returns at the horizon grid. None if not yet knowable."""
        t = self.t if t is None else t
        if t + self.lookahead >= len(self.store):
            return None
        anchor = float(self.store.prices[t, 3]) / self.store.scale
        return build_target(self.store, t, self.horizons, anchor)

    def future_path(self, t: int | None = None) -> np.ndarray | None:
        """Actual per-second close prices over the next 25 minutes."""
        t = self.t if t is None else t
        if t + self.lookahead >= len(self.store):
            return None
        return self.store.close(slice(t + 1, t + 1 + self.lookahead))

    def history(self, seconds: int) -> tuple[np.ndarray, np.ndarray]:
        """(unix_ts, close) for the trailing `seconds`, for plotting."""
        lo = max(0, self.t + 1 - seconds)
        idx = np.arange(lo, self.t + 1)
        return self.store.start_ts + idx, self.store.close(slice(lo, self.t + 1))

    # -- clock -----------------------------------------------------------
    def __iter__(self):
        self._wall0 = time.monotonic()
        t0 = self.t
        while self.t < self.end - self.lookahead - 1:
            if self.speed > 0:
                due = (self.t - t0) / self.speed
                lag = due - (time.monotonic() - self._wall0)
                if lag > 0:
                    time.sleep(lag)
            yield self.observe()
            self.t += self.step

    def reset(self, t: int | None = None) -> None:
        self.t = self.start if t is None else t
        self._wall0 = None
