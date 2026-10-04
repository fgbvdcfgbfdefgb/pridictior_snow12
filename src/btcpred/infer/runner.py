"""
Live model runner -- one object that drives both the simulator and the real
market, so the backtest and production paths cannot diverge.

Stability in production comes from two layers:

1. the `consistency` training term, which teaches the model not to revise
   violently in the first place; and
2. an inference-side EMA over the *absolute* predicted price path (not over
   the anchor-relative returns -- smoothing those would lag the anchor and
   systematically bias the forecast whenever price moves).

`alpha` is the EMA weight on each new forecast. 1.0 disables smoothing.
`max_revision_bp` additionally clamps how far the published path may move in
one second, which bounds the worst case even if the model spikes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..data.features import N_FEATURES, LaneSpec, encode_lane, FEATURE_GAIN
from ..models.analyser import AnalyserConfig, MarketAnalyser
from ..models.predictor import PredictorConfig, PricePredictor


@dataclass
class Forecast:
    ts: int                 # decision time, unix seconds
    anchor: float           # last observed price
    horizons: np.ndarray    # seconds ahead
    median: np.ndarray      # predicted price at each horizon
    lower: np.ndarray       # q10 price
    upper: np.ndarray       # q90 price
    raw_median: np.ndarray  # pre-smoothing price path
    revision_bp: float      # how far the published path moved this second


class LiveRunner:
    def __init__(
        self,
        ckpt_dir: str | Path,
        device: str | torch.device = "cpu",
        variant: str | None = None,
        alpha: float = 0.35,
        max_revision_bp: float = 25.0,
    ):
        ckpt = Path(ckpt_dir)
        self.device = torch.device(device)
        self.alpha = float(alpha)
        self.max_revision_bp = float(max_revision_bp)

        a = torch.load(ckpt / "analyser.pt", map_location="cpu", weights_only=False)
        self.analyser = MarketAnalyser(AnalyserConfig(**a["config"]))
        self.analyser.load_state_dict(a["state_dict"])
        self.analyser.to(self.device).eval()

        pfile = ckpt / (f"predictor_{variant}.pt" if variant else "best_predictor.pt")
        if not pfile.exists():
            cands = sorted(ckpt.glob("predictor_*.pt"))
            if not cands:
                raise FileNotFoundError(f"no predictor checkpoint in {ckpt}")
            pfile = cands[0]
        p = torch.load(pfile, map_location="cpu", weights_only=False)
        cfg = PredictorConfig(**p["config"])
        self.predictor = PricePredictor(cfg)
        self.predictor.load_state_dict(p["state_dict"])
        self.predictor.to(self.device).eval()
        self.variant = p.get("variant", cfg.name)
        self.metrics = p.get("metrics")

        import json
        state = json.loads((ckpt / "train_state.json").read_text())
        self.horizons = np.array(state["horizons"], dtype=np.int64)
        self.lanes = tuple(LaneSpec(**l) for l in state["lanes"])
        self.warmup = max(l.span for l in self.lanes)

        self._prev_path: np.ndarray | None = None
        self._prev_ts: int | None = None

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, context: np.ndarray, anchor: float, ts: int) -> Forecast:
        """context: (lanes, steps, N_FEATURES) float32, already gain-scaled."""
        x = torch.from_numpy(np.ascontiguousarray(context))[None].to(self.device)
        mem, summ = self.analyser(x)
        out = self.predictor(x, mem, summ).float()[0].cpu().numpy()  # (H, Q)

        mi = self.predictor.median_index
        raw = anchor * np.exp(out[:, mi])
        lower = anchor * np.exp(out[:, 0])
        upper = anchor * np.exp(out[:, -1])

        path, rev = self._stabilise(raw, ts)
        # Keep the band centred on the published path.
        shift = path - raw
        return Forecast(ts, anchor, self.horizons, path, lower + shift,
                        upper + shift, raw, rev)

    def _stabilise(self, raw: np.ndarray, ts: int) -> tuple[np.ndarray, float]:
        if self._prev_path is None or self._prev_ts is None:
            self._prev_path, self._prev_ts = raw.copy(), ts
            return raw, 0.0

        dt = max(1, ts - self._prev_ts)
        # Re-align the previous path onto the current target times: the old
        # forecast for t+h is now the forecast for (t+dt) + (h-dt).
        prev_aligned = np.interp(
            self.horizons + dt, self.horizons, self._prev_path,
            left=self._prev_path[0], right=self._prev_path[-1],
        )
        a = 1.0 - (1.0 - self.alpha) ** dt  # EMA weight compounded over the gap
        path = a * raw + (1 - a) * prev_aligned

        cap = self.max_revision_bp * 1e-4 * dt
        delta = np.log(np.maximum(path, 1e-9) / np.maximum(prev_aligned, 1e-9))
        clipped = np.clip(delta, -cap, cap)
        path = prev_aligned * np.exp(clipped)

        rev = float(np.abs(np.log(path / np.maximum(prev_aligned, 1e-9))).mean() * 1e4)
        self._prev_path, self._prev_ts = path.copy(), ts
        return path, rev

    def reset(self) -> None:
        self._prev_path = self._prev_ts = None


def context_from_closes(
    closes: np.ndarray,
    volumes: np.ndarray | None,
    trades: np.ndarray | None,
    lanes: tuple[LaneSpec, ...],
) -> tuple[np.ndarray, float]:
    """Build a model context from a plain 1-second close series.

    Used by the live runner when the only feed available is last-price ticks
    (e.g. a websocket miniTicker) rather than full aggTrades. Missing volume
    and trade-count channels are zero-filled, which the model tolerates but
    which does cost some accuracy versus the training distribution.
    """
    need = max(l.span for l in lanes)
    if closes.shape[0] < need:
        raise ValueError(f"need >= {need} seconds of history, got {closes.shape[0]}")
    closes = closes[-need:].astype(np.float64)
    vol = np.zeros(need) if volumes is None else volumes[-need:].astype(np.float64)
    cnt = np.zeros(need) if trades is None else trades[-need:].astype(np.float64)

    prices = np.empty((need, 4), np.float64)
    prices[:, 0] = prices[:, 1] = prices[:, 2] = prices[:, 3] = closes * 100.0
    flows = np.zeros((need, 4), np.float64)
    flows[:, 0], flows[:, 1] = vol, cnt
    flows[:, 2] = vol * 0.5
    flows[:, 3] = closes * 100.0

    anchor = float(closes[-1])
    out = np.empty((len(lanes), lanes[0].steps, N_FEATURES), np.float32)
    for k, ln in enumerate(lanes):
        out[k] = encode_lane(prices[-ln.span:], flows[-ln.span:], ln.stride, anchor, 100.0)
    out *= FEATURE_GAIN
    return np.clip(out, -16.0, 16.0, out=out), anchor
