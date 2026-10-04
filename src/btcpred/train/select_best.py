"""
Walk-forward evaluation and model selection.

Every variant is replayed through the held-out window second by second and
scored against the recorded market. Reported metrics:

skill            1 - MSE(model) / MSE(random walk) on the 25-min log-return.
                 THIS IS THE ONE THAT MATTERS. A 25-minute BTC forecast is
                 close to a martingale: predicting "no change" is a strong
                 baseline, and a model can reach a very low pinball loss
                 purely by collapsing to zero. Skill <= 0 means the model has
                 learned nothing beyond the random walk, however pretty the
                 training curve looked.

revision_bp      mean |forecast(t, T) - forecast(t+1, T)| in basis points:
                 how much the model changes its mind about the *same* future
                 instant from one second to the next. This is the stability
                 number; a live trading signal needs it small and it is
                 reported independently of accuracy.

hit_rate         directional accuracy at 25 min.
net_reward_bp    volatility-scaled position P&L after transaction cost.
sharpe           annualised Sharpe of the per-decision P&L series.
coverage_80      fraction of outcomes inside the q10-q90 band (target 0.80);
                 measures whether the uncertainty estimate is honest.

Selection ranks by net reward subject to a stability gate, because the model
drives real trades: an accurate but jittery model is worse than useless.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..data.features import BarStore, LaneSpec, build_context, build_target
from ..models.analyser import AnalyserConfig, MarketAnalyser
from ..models.predictor import PredictorConfig, PricePredictor
from ..utils.config import load_config

SEC_PER_YEAR = 365 * 24 * 3600


def load_models(ckpt: Path, device, variants=None):
    a_blob = torch.load(ckpt / "analyser.pt", map_location=device, weights_only=False)
    analyser = MarketAnalyser(AnalyserConfig(**a_blob["config"])).to(device).eval()
    analyser.load_state_dict(a_blob["state_dict"])

    preds = {}
    for f in sorted(ckpt.glob("predictor_*.pt")):
        name = f.stem.replace("predictor_", "")
        if variants and name not in variants:
            continue
        blob = torch.load(f, map_location=device, weights_only=False)
        cfg = PredictorConfig(**blob["config"])
        m = PricePredictor(cfg).to(device).eval()
        m.load_state_dict(blob["state_dict"])
        preds[name] = m
    return analyser, preds


@torch.no_grad()
def evaluate(analyser, predictors, store, lo, hi, lanes, horizons,
             device, step=60, max_points=2000, cost_bps=1.0, batch=8):
    """Walk-forward replay. `step` decimates cursors for speed; revision is
    always measured on a true 1-second gap."""
    cursors = list(range(lo, hi - int(horizons[-1]) - 2, step))[:max_points]
    names = list(predictors)
    acc = {n: {"err": [], "tgt": [], "rev": [], "cov": [], "pos": []} for n in names}

    for i in range(0, len(cursors), batch):
        chunk = cursors[i:i + batch]
        ctxs, tgts, anchors = [], [], []
        ctxs1, anchors1 = [], []
        for t in chunk:
            c, a = build_context(store, t, lanes)
            ctxs.append(c); anchors.append(a)
            tgts.append(build_target(store, t, horizons, a))
            c1, a1 = build_context(store, t + 1, lanes)   # one second later
            ctxs1.append(c1); anchors1.append(a1)

        X = torch.from_numpy(np.stack(ctxs)).to(device)
        X1 = torch.from_numpy(np.stack(ctxs1)).to(device)
        T = torch.from_numpy(np.stack(tgts)).to(device)
        A = torch.tensor(anchors, device=device, dtype=torch.float32)
        A1 = torch.tensor(anchors1, device=device, dtype=torch.float32)

        mem, summ = analyser(X)
        mem1, summ1 = analyser(X1)
        H = torch.tensor(horizons, device=device, dtype=torch.float32)

        for n, m in predictors.items():
            p = m(X, mem, summ).float()
            p1 = m(X1, mem1, summ1).float()
            mi = m.median_index
            med, med1 = p[..., mi], p1[..., mi]

            acc[n]["err"].append((med - T).cpu().numpy())
            acc[n]["tgt"].append(T.cpu().numpy())

            # Revision: same wall-clock target, forecasts 1 s apart.
            # p1's horizon h covers t+1+h; interpolate onto h-1.
            dst = (H - 1.0).clamp(H[0], H[-1])
            idx = torch.searchsorted(H, dst).clamp(1, H.numel() - 1)
            w = ((dst - H[idx - 1]) / (H[idx] - H[idx - 1])).unsqueeze(0)
            m1_i = med1[:, idx - 1] * (1 - w) + med1[:, idx] * w
            abs_now = med + torch.log(A)[:, None]
            abs_next = m1_i + torch.log(A1)[:, None]
            acc[n]["rev"].append((abs_now - abs_next).abs().cpu().numpy())

            inside = ((T >= p[..., 0]) & (T <= p[..., -1])).float()
            acc[n]["cov"].append(inside.cpu().numpy())
            acc[n]["pos"].append((med[:, -1] / 1.7e-3).clamp(-1, 1).cpu().numpy())

    out = {}
    for n in names:
        err = np.concatenate(acc[n]["err"])          # (N, H)
        tgt = np.concatenate(acc[n]["tgt"])
        rev = np.concatenate(acc[n]["rev"])
        cov = np.concatenate(acc[n]["cov"])
        pos = np.concatenate(acc[n]["pos"])

        mse_model = (err[:, -1] ** 2).mean()
        mse_rw = (tgt[:, -1] ** 2).mean()            # random walk predicts 0
        pnl = pos * tgt[:, -1] - np.abs(np.diff(pos, prepend=0.0)) * cost_bps * 1e-4
        per_year = SEC_PER_YEAR / max(step, 1)
        sharpe = (pnl.mean() / pnl.std() * np.sqrt(per_year)) if pnl.std() > 0 else 0.0

        out[n] = {
            "n_points": int(err.shape[0]),
            "skill": float(1 - mse_model / max(mse_rw, 1e-18)),
            "rmse_bp": float(np.sqrt(mse_model) * 1e4),
            "rmse_rw_bp": float(np.sqrt(mse_rw) * 1e4),
            "revision_bp": float(rev.mean() * 1e4),
            "hit_rate": float(((np.sign(err[:, -1] + tgt[:, -1]) == np.sign(tgt[:, -1]))
                               & (tgt[:, -1] != 0)).mean()),
            "net_reward_bp": float(pnl.mean() * 1e4),
            "sharpe": float(sharpe),
            "coverage_80": float(cov.mean()),
            "mean_abs_pos": float(np.abs(pos).mean()),
        }
    return out


MIN_POINTS_FOR_REWARD = 500


def select(metrics: dict, max_revision_bp: float = 5.0) -> tuple[str, str]:
    """Two gates, then rank.

    Gate 1 (skill > 0) comes first and is non-negotiable: reward and Sharpe are
    extremely noisy, and without it the ranking happily picks a model that is
    far worse than a random walk simply because its P&L noise landed positive.
    Gate 2 is stability. Only then do we rank by reward -- and if the sample is
    too small for reward to mean anything, we rank by skill instead.
    """
    skilled = {k: v for k, v in metrics.items() if v["skill"] > 0}
    if not skilled:
        best = max(metrics, key=lambda k: metrics[k]["skill"])
        return best, ("NO model beats the random walk (skill <= 0); "
                      "picked the least-bad by skill")

    stable = {k: v for k, v in skilled.items() if v["revision_bp"] <= max_revision_bp}
    if stable:
        pool, note = stable, "skill > 0 and passed stability gate"
    else:
        pool, note = skilled, (f"skill > 0 but NO model met revision_bp "
                               f"<= {max_revision_bp}")

    n = min(v["n_points"] for v in pool.values())
    if n < MIN_POINTS_FOR_REWARD:
        best = max(pool, key=lambda k: pool[k]["skill"])
        return best, f"{note}; only {n} eval points so ranked by skill, not reward"
    best = max(pool, key=lambda k: pool[k]["net_reward_bp"])
    return best, f"{note}; ranked by net reward over {n} points"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/a10x4.json"))
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints"))
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--step", type=int, default=60)
    ap.add_argument("--max-points", type=int, default=2000)
    ap.add_argument("--max-revision-bp", type=float, default=5.0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dc = cfg["data"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    store = BarStore(dc["bars"])
    lanes = tuple(LaneSpec(**l) for l in dc["lanes"])
    state = json.loads((args.ckpt / "train_state.json").read_text())
    horizons = np.array(state["horizons"], dtype=np.int64)

    from ..data.dataset import SplitSpec
    (_, _), (va_lo, va_hi), (te_lo, te_hi) = SplitSpec(
        dc["val_days"], dc["test_days"]).bounds(len(store))
    lo, hi = (va_lo, va_hi) if args.split == "val" else (te_lo, te_hi)

    analyser, preds = load_models(args.ckpt, device)
    print(f"evaluating {list(preds)} on {args.split} [{lo:,}:{hi:,}] "
          f"every {args.step}s", flush=True)

    metrics = evaluate(analyser, preds, store, lo, hi, lanes, horizons,
                       device, step=args.step, max_points=args.max_points)

    hdr = f"{'variant':8} {'skill':>8} {'rmse_bp':>8} {'rw_bp':>8} {'revis_bp':>9} " \
          f"{'hit':>6} {'rew_bp':>8} {'sharpe':>7} {'cov80':>6}"
    print("\n" + hdr); print("-" * len(hdr))
    for n, m in metrics.items():
        print(f"{n:8} {m['skill']:+8.4f} {m['rmse_bp']:8.2f} {m['rmse_rw_bp']:8.2f} "
              f"{m['revision_bp']:9.3f} {m['hit_rate']:6.3f} {m['net_reward_bp']:+8.3f} "
              f"{m['sharpe']:+7.2f} {m['coverage_80']:6.3f}")

    best, reason = select(metrics, args.max_revision_bp)
    print(f"\nselected: {best}  ({reason})")
    n = metrics[best]["n_points"]
    if n < MIN_POINTS_FOR_REWARD:
        print(f"NOTE: {n} evaluation points is far too few for net_reward_bp or "
              f"sharpe to be meaningful -- treat those two columns as noise "
              f"until you evaluate on >= {MIN_POINTS_FOR_REWARD} points.")
    if metrics[best]["skill"] <= 0:
        print("WARNING: best model has skill <= 0 -- it does not beat a random "
              "walk on the held-out window. Do not trade this. Train longer, "
              "add data, or accept that this horizon may not be predictable.")

    blob = torch.load(args.ckpt / f"predictor_{best}.pt", map_location="cpu",
                      weights_only=False)
    blob["metrics"] = metrics[best]
    blob["variant"] = best
    torch.save(blob, args.ckpt / "best_predictor.pt")
    (args.ckpt / "metrics.json").write_text(
        json.dumps({"split": args.split, "metrics": metrics, "best": best}, indent=2))
    print(f"wrote {args.ckpt/'best_predictor.pt'} and metrics.json")


if __name__ == "__main__":
    main()
