"""
Real-time objective: scored every market second, no epochs.

Four terms, each addressing a requirement from the spec:

1. `pinball`      quantile (pinball) loss of the predicted 25-minute path
                  against the *actually recorded* market data. This is the
                  accuracy term. Quantile rather than MSE because trading
                  needs the distribution, and because the median under pinball
                  is robust to the fat tails of crypto returns.

2. `consistency`  the stability term. The forecast made at second t for wall
                  time T must agree with the forecast made at t+1 for the same
                  wall time T. Since the horizon grids of two adjacent cursors
                  are offset, pred(t+1) is linearly interpolated onto pred(t)'s
                  target times before comparison. This directly penalises the
                  erratic second-to-second jumps that would wreck a live
                  trading signal -- it is a constraint on the *revision
                  process*, not just on accuracy.

3. `smoothness`   second difference along the horizon axis, so the predicted
                  path is a plausible price trajectory rather than 25
                  independent guesses.

4. `reward`       a reported (not back-propagated) trading P&L proxy: take a
                  volatility-scaled position from the median forecast, mark it
                  against the realised move, and charge transaction cost on
                  position changes. This is the number used to rank models.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    pinball: float = 1.0
    consistency: float = 0.5
    smoothness: float = 0.05
    # Scale applied to all log-return quantities before the loss. Raw 25-min
    # log-returns are O(1e-3); without this the gradients are numerically tiny
    # and Adam's epsilon starts to dominate.
    target_gain: float = 1000.0


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """pred (B,H,Q), target (B,H), q (Q,) -> scalar."""
    err = target.unsqueeze(-1) - pred
    return torch.maximum(q * err, (q - 1.0) * err).mean()


def interp_horizons(
    pred: torch.Tensor, src_h: torch.Tensor, dst_h: torch.Tensor
) -> torch.Tensor:
    """Linearly resample (B,H,Q) predictions from src_h onto dst_h (seconds).

    Values outside src_h are clamped to the endpoints.
    """
    src = src_h.to(pred.dtype)
    dst = dst_h.to(pred.dtype).clamp(src[0], src[-1])
    idx = torch.searchsorted(src, dst).clamp(1, src.numel() - 1)
    lo, hi = idx - 1, idx
    w = ((dst - src[lo]) / (src[hi] - src[lo]).clamp_min(1e-6)).view(1, -1, 1)
    return pred[:, lo] * (1 - w) + pred[:, hi] * w


def consistency_loss(
    pred_a: torch.Tensor,
    anchor_a: torch.Tensor,
    t_a: torch.Tensor,
    pred_b: torch.Tensor,
    anchor_b: torch.Tensor,
    t_b: torch.Tensor,
    horizons: torch.Tensor,
) -> torch.Tensor:
    """Agreement between two forecasts of the same wall-clock future.

    Predictions are anchor-relative log-returns, so they are first lifted to
    absolute log-prices before comparison.
    """
    dt = (t_b - t_a).to(pred_a.dtype)  # (B,) seconds, >= 0
    log_a = torch.log(anchor_a.clamp_min(1e-9))[:, None, None]
    log_b = torch.log(anchor_b.clamp_min(1e-9))[:, None, None]

    abs_a = pred_a + log_a  # targets at t_a + horizons

    # pred_b's horizon h corresponds to wall time t_b + h = t_a + dt + h,
    # so to hit t_a + horizons we need pred_b at (horizons - dt).
    shifted = horizons[None, :].to(pred_a.dtype) - dt[:, None]  # (B,H)
    src = horizons.to(pred_a.dtype)
    dst = shifted.clamp(src[0], src[-1])
    idx = torch.searchsorted(src, dst.contiguous()).clamp(1, src.numel() - 1)
    lo, hi = idx - 1, idx
    s_lo, s_hi = src[lo], src[hi]
    w = ((dst - s_lo) / (s_hi - s_lo).clamp_min(1e-6)).unsqueeze(-1)
    b_lo = torch.gather(pred_b, 1, lo.unsqueeze(-1).expand(-1, -1, pred_b.shape[-1]))
    b_hi = torch.gather(pred_b, 1, hi.unsqueeze(-1).expand(-1, -1, pred_b.shape[-1]))
    abs_b = b_lo * (1 - w) + b_hi * w + log_b

    # Only compare where pred_b actually covers the target time.
    valid = (shifted >= src[0]).unsqueeze(-1).to(abs_a.dtype)
    diff = (abs_a - abs_b) * valid
    return (diff.pow(2).sum() / valid.sum().clamp_min(1.0)).sqrt()


def smoothness_loss(pred: torch.Tensor) -> torch.Tensor:
    """Second difference along the horizon axis of every quantile."""
    if pred.shape[1] < 3:
        return pred.new_zeros(())
    d2 = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
    return d2.pow(2).mean()


@torch.no_grad()
def trading_reward(
    median: torch.Tensor,
    target: torch.Tensor,
    horizon_idx: int = -1,
    vol: float = 1.7e-3,
    cost_bps: float = 1.0,
    prev_pos: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """P&L proxy at one horizon. `median`/`target` are raw log-returns.

    `vol` defaults to the empirical 25-minute BTC log-return sigma (~0.17 %),
    so a one-sigma forecast maps to a full-size position.
    """
    sig = median[:, horizon_idx]
    real = target[:, horizon_idx]
    pos = (sig / vol).clamp(-1.0, 1.0)
    gross = pos * real
    turn = (pos - prev_pos).abs() if prev_pos is not None else pos.abs()
    cost = turn * (cost_bps * 1e-4)
    return {
        "reward": (gross - cost).mean(),
        "gross": gross.mean(),
        "hit_rate": ((sig.sign() == real.sign()) & (real != 0)).float().mean(),
        "position": pos,
    }


def compute_losses(
    pred_a, tgt_a, anchor_a, t_a,
    pred_b, tgt_b, anchor_b, t_b,
    q_levels, horizons, w: LossWeights,
):
    """Full objective for a consecutive pair of cursors (t_a, t_b)."""
    g = w.target_gain
    acc = 0.5 * (
        pinball_loss(pred_a * g, tgt_a * g, q_levels)
        + pinball_loss(pred_b * g, tgt_b * g, q_levels)
    )
    cons = consistency_loss(pred_a, anchor_a, t_a, pred_b, anchor_b, t_b, horizons) * g
    smooth = 0.5 * (smoothness_loss(pred_a * g) + smoothness_loss(pred_b * g))

    total = w.pinball * acc + w.consistency * cons + w.smoothness * smooth
    return total, {"pinball": acc.detach(), "consistency": cons.detach(),
                   "smoothness": smooth.detach(), "total": total.detach()}
