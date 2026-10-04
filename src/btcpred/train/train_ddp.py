"""
Distributed trainer: one shared Market Analyser, N Price Predictor variants.

Topology
--------
The analyser and *all* predictor variants live on every rank and are stepped
together on the same batch. Each gets its own DDP wrapper and its own
optimiser; the analyser receives the summed gradient from every predictor, so
it learns a representation that serves all architectures rather than
overfitting to one. Variants never exchange gradients with each other.

This is deliberately not a "one variant per GPU" layout: that would give each
variant its own analyser copy drifting apart, violating the "only one market
analyser" requirement and making the comparison unfair.

No epochs. The loop walks market time forward, scoring every cursor against
recorded reality, and checkpoints on a wall-clock/step budget.

Launch
------
    torchrun --nproc_per_node=4 -m btcpred.train.train_ddp --config configs/a10x4.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from ..data.dataset import MarketStream, SplitSpec, collate
from ..data.features import BarStore, DEFAULT_LANES, LaneSpec, horizon_grid
from ..models.analyser import AnalyserConfig, MarketAnalyser
from ..models.predictor import PricePredictor, build_variants
from .reward import LossWeights, compute_losses, trading_reward
from ..utils.config import load_config
from ..utils.hardware import describe_hardware


def setup_dist():
    if "RANK" in os.environ and torch.distributed.is_available():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
        return rank, world, local, True
    return 0, 1, 0, False


def log0(rank, *a):
    if rank == 0:
        print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/a10x4.json"))
    ap.add_argument("--bars", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--variants", type=str, default=None,
                    help="comma list, e.g. xfmr,tcn")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.bars:       cfg["data"]["bars"] = str(args.bars)
    if args.out:        cfg["train"]["out"] = str(args.out)
    if args.max_steps:  cfg["train"]["max_steps"] = args.max_steps
    if args.max_hours:  cfg["train"]["max_hours"] = args.max_hours
    if args.variants:   cfg["model"]["variants"] = args.variants.split(",")

    rank, world, local, distributed = setup_dist()
    dev = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        hw = describe_hardware()
        print(json.dumps(hw, indent=2), flush=True)

    tc, dc, mc = cfg["train"], cfg["data"], cfg["model"]
    out = Path(tc["out"]); out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(tc.get("seed", 0) + rank)
    use_amp = bool(tc.get("amp", True)) and dev.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

    # ---------------- data ------------------------------------------------
    store = BarStore(dc["bars"])
    lanes = tuple(LaneSpec(**l) for l in dc["lanes"]) if dc.get("lanes") else DEFAULT_LANES
    horizons = horizon_grid(dc.get("horizon_minutes", 25), dc.get("horizon_every_sec", 60))
    split = SplitSpec(dc.get("val_days", 14.0), dc.get("test_days", 7.0))
    (tr_lo, tr_hi), (va_lo, va_hi), _ = split.bounds(len(store))
    log0(rank, f"store {len(store):,}s  train[{tr_lo:,}:{tr_hi:,}] val[{va_lo:,}:{va_hi:,}]")

    stream = MarketStream(
        store, tr_lo, tr_hi, lanes, horizons,
        step=dc.get("cursor_step", 1), consecutive=2,
        rank=rank, world_size=world, seed=tc.get("seed", 0),
    )
    # consecutive=2 means each yielded pair must stay in one batch, so the
    # per-rank batch size must be even.
    bs = int(tc["batch_size"])
    assert bs % 2 == 0, "batch_size must be even (cursor pairs)"
    loader = DataLoader(
        stream, batch_size=bs, num_workers=tc.get("workers", 4),
        collate_fn=collate, pin_memory=(dev.type == "cuda"),
        persistent_workers=tc.get("workers", 4) > 0,
        prefetch_factor=4 if tc.get("workers", 4) > 0 else None,
    )

    # ---------------- models ----------------------------------------------
    acfg = AnalyserConfig(
        d_model=mc["analyser"]["d_model"], n_layers=mc["analyser"]["n_layers"],
        n_heads=mc["analyser"]["n_heads"], n_lanes=len(lanes),
        lane_steps=lanes[0].steps, downsample=mc["analyser"]["downsample"],
    )
    analyser = MarketAnalyser(acfg).to(dev)
    if tc.get("grad_checkpoint", False):
        analyser.gradient_checkpointing = True

    wanted = mc.get("variants", ["xfmr", "tcn", "ssm"])
    pcfgs = [c for c in build_variants(acfg.d_model, len(horizons),
                                       mc.get("scale", 1.0)) if c.name in wanted]
    predictors = {c.name: PricePredictor(c).to(dev) for c in pcfgs}

    log0(rank, f"analyser {analyser.n_params()/1e6:.1f}M  " +
         "  ".join(f"{k}:{v.n_params()/1e6:.1f}M" for k, v in predictors.items()))

    if distributed:
        analyser_d = DDP(analyser, device_ids=[local] if dev.type == "cuda" else None,
                         find_unused_parameters=False)
        predictors_d = {k: DDP(v, device_ids=[local] if dev.type == "cuda" else None)
                        for k, v in predictors.items()}
    else:
        analyser_d, predictors_d = analyser, predictors

    opt_a = torch.optim.AdamW(analyser.parameters(), lr=tc["lr_analyser"],
                              betas=(0.9, 0.95), weight_decay=tc.get("wd", 0.01))
    opts_p = {k: torch.optim.AdamW(v.parameters(), lr=tc["lr_predictor"],
                                   betas=(0.9, 0.95), weight_decay=tc.get("wd", 0.01))
              for k, v in predictors.items()}
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    warm = tc.get("warmup_steps", 500)
    total = tc.get("max_steps", 100_000)

    def lr_at(step):
        if step < warm:
            return step / max(1, warm)
        p = (step - warm) / max(1, total - warm)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    w = LossWeights(**tc.get("loss_weights", {}))
    q_levels = torch.tensor(pcfgs[0].quantiles, device=dev)
    h_t = torch.tensor(horizons, device=dev, dtype=torch.float32)

    # ---------------- loop -------------------------------------------------
    hist = {k: [] for k in predictors}
    t_start = time.monotonic()
    max_hours = tc.get("max_hours", 1e9)
    step = 0
    log_every = tc.get("log_every", 20)
    ckpt_every = tc.get("ckpt_every", 2000)

    for ctx, tgt, anchor, tt in loader:
        if step >= total or (time.monotonic() - t_start) / 3600 > max_hours:
            break
        ctx = ctx.to(dev, non_blocking=True)
        tgt = tgt.to(dev, non_blocking=True)
        anchor = anchor.to(dev, non_blocking=True)
        tt = tt.to(dev, non_blocking=True)

        scale = lr_at(step)
        for g in opt_a.param_groups:
            g["lr"] = tc["lr_analyser"] * scale
        for o in opts_p.values():
            for g in o.param_groups:
                g["lr"] = tc["lr_predictor"] * scale

        opt_a.zero_grad(set_to_none=True)
        for o in opts_p.values():
            o.zero_grad(set_to_none=True)

        with torch.autocast(dev.type, dtype=amp_dtype, enabled=use_amp):
            mem, summary = analyser_d(ctx)
            # even indices = cursor a, odd = cursor b (the next second)
            ia, ib = slice(0, None, 2), slice(1, None, 2)
            stats_all = {}
            loss_sum = 0.0
            for name, pm in predictors_d.items():
                pred = pm(ctx, mem, summary)
                loss, st = compute_losses(
                    pred[ia], tgt[ia], anchor[ia], tt[ia],
                    pred[ib], tgt[ib], anchor[ib], tt[ib],
                    q_levels, h_t, w,
                )
                loss_sum = loss_sum + loss
                med = pred[..., pm.module.median_index if distributed else pm.median_index]
                st.update(trading_reward(med.float(), tgt.float()))
                st.pop("position", None)
                stats_all[name] = st

        scaler.scale(loss_sum).backward()
        for o in (opt_a, *opts_p.values()):
            scaler.unscale_(o)
        clip = tc.get("grad_clip", 1.0)
        torch.nn.utils.clip_grad_norm_(analyser.parameters(), clip)
        for v in predictors.values():
            torch.nn.utils.clip_grad_norm_(v.parameters(), clip)
        for o in (opt_a, *opts_p.values()):
            scaler.step(o)
        scaler.update()

        for k, st in stats_all.items():
            hist[k].append({kk: float(vv) for kk, vv in st.items()})

        if rank == 0 and step % log_every == 0:
            el = time.monotonic() - t_start
            parts = []
            for k, st in stats_all.items():
                parts.append(f"{k} pin={float(st['pinball']):.4f} "
                             f"cons={float(st['consistency']):.4f} "
                             f"rew={float(st['reward'])*1e4:+.2f}bp "
                             f"hit={float(st['hit_rate']):.3f}")
            print(f"[{step:>7}] {el/60:6.1f}m lr={scale:.3f} | " + " | ".join(parts),
                  flush=True)

        if rank == 0 and step > 0 and step % ckpt_every == 0:
            save(out, analyser, predictors, acfg, pcfgs, step, hist, horizons, lanes)

        step += 1

    if rank == 0:
        save(out, analyser, predictors, acfg, pcfgs, step, hist, horizons, lanes)
        print(f"done: {step} steps in {(time.monotonic()-t_start)/60:.1f} min")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def save(out, analyser, predictors, acfg, pcfgs, step, hist, horizons, lanes):
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": analyser.state_dict(), "config": asdict(acfg),
                "step": step}, out / "analyser.pt")
    for c in pcfgs:
        torch.save({"state_dict": predictors[c.name].state_dict(),
                    "config": asdict(c), "step": step},
                   out / f"predictor_{c.name}.pt")
    meta = {
        "step": step,
        "horizons": [int(h) for h in horizons],
        "lanes": [{"stride": l.stride, "steps": l.steps} for l in lanes],
        "recent": {k: v[-200:] for k, v in hist.items()},
    }
    (out / "train_state.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
