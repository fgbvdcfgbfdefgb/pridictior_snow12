"""
Streaming, resumable build of the full 1-second bar store.

`resample.py` holds eight full-length accumulators in RAM. At 182 M seconds
that is 11.7 GB, and it needs all ~25 GB of archives on disk simultaneously:
~34 GB peak, which is fine on a 100 GB Snowflake box and impossible almost
anywhere else.

This module folds the dataset in **one archive at a time**:

    for each monthly archive, in chronological order:
        download it (if missing)  ->  accumulate into month-sized buffers
        write that month's slice straight into the output memmaps
        optionally delete the archive before moving on

Peak RAM becomes one month of accumulators (~170 MB) instead of 11.7 GB, and
peak disk becomes the output store plus one archive (~9 GB) instead of 34 GB.
Progress is journalled after every archive, so an interrupted run resumes
where it stopped rather than starting over.

Forward-filling gaps needs the previous bar, which may sit in the previous
month, so the last known close is carried across block boundaries in the
journal.

    python -m btcpred.data.build --start 2020-01 --out data/bars_1s --purge
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import os
import shutil
from pathlib import Path

import numpy as np

from .download_binance import Target, build_targets, fetch, already_ok
from .resample import CENTS, iter_chunks, normalise_ts

STATE = "build_state.json"


def month_bounds(ym: str) -> tuple[int, int]:
    y, m = (int(x) for x in ym.split("-"))
    days = calendar.monthrange(y, m)[1]
    lo = int(dt.datetime(y, m, 1, tzinfo=dt.timezone.utc).timestamp())
    return lo, lo + days * 86400


def day_bounds(ymd: str) -> tuple[int, int]:
    y, m, d = (int(x) for x in ymd.split("-"))
    lo = int(dt.datetime(y, m, d, tzinfo=dt.timezone.utc).timestamp())
    return lo, lo + 86400


def archive_span(path: Path) -> tuple[int, int]:
    """[lo, hi) unix-second range an archive covers, from its filename."""
    stem = path.stem                      # BTCUSDT-aggTrades-2020-01[-15]
    tail = stem.split("aggTrades-")[-1]
    parts = tail.split("-")
    return day_bounds(tail) if len(parts) == 3 else month_bounds(tail)


def plan(symbol: str, start: str, end: str, raw: Path) -> list[Target]:
    return build_targets(symbol, start, end, raw)


def estimate(targets: list[Target]) -> dict:
    lo = min(archive_span(t.dest)[0] for t in targets)
    hi = max(archive_span(t.dest)[1] for t in targets)
    n = hi - lo
    return {
        "start_ts": lo, "end_ts": hi, "n_seconds": n,
        "store_gb": round(n * 48 / 1e9, 2),
        "peak_ram_mb": round(31 * 86400 * 64 / 1e6),
        "archives": len(targets),
    }


def _fold_block(archive: Path, lo: int, hi: int, chunksize: int):
    """Accumulate one archive into month-sized buffers. Returns block arrays."""
    n = hi - lo
    o = np.zeros(n, np.int64)
    h = np.zeros(n, np.int64)
    l = np.full(n, np.iinfo(np.int64).max, np.int64)
    c = np.zeros(n, np.int64)
    cnt = np.zeros(n, np.int64)
    vol = np.zeros(n, np.float64)
    notional = np.zeros(n, np.float64)
    buy = np.zeros(n, np.float64)

    for df in iter_chunks(archive, chunksize):
        cols = list(df.columns)
        price = df[cols[0]].to_numpy(np.float64)
        qty = df[cols[1]].to_numpy(np.float64)
        ts = normalise_ts(df[cols[2]].to_numpy(np.int64))
        maker = df[cols[3]].to_numpy()

        idx = ts - lo
        keep = (idx >= 0) & (idx < n)
        if not keep.all():
            idx, price, qty, maker = idx[keep], price[keep], qty[keep], maker[keep]
        if idx.size == 0:
            continue

        pc = np.rint(price * CENTS).astype(np.int64)
        cnt += np.bincount(idx, minlength=n)
        vol += np.bincount(idx, weights=qty, minlength=n)
        notional += np.bincount(idx, weights=price * qty, minlength=n)
        buy += np.bincount(idx, weights=(~maker.astype(bool)) * qty, minlength=n)
        np.maximum.at(h, idx, pc)
        np.minimum.at(l, idx, pc)
        fresh = o[idx] == 0
        if fresh.any():
            o[idx[fresh]] = pc[fresh]
        c[idx] = pc

    return o, h, l, c, cnt, vol, notional, buy


def build(symbol: str, start: str, end: str, raw: Path, out: Path,
          chunksize: int = 1_000_000, purge: bool = False,
          keep_going: bool = False, verify: bool = True) -> dict:
    import requests

    targets = plan(symbol, start, end, raw)
    if not targets:
        raise SystemExit("nothing to build")
    est = estimate(targets)
    out.mkdir(parents=True, exist_ok=True)

    free = shutil.disk_usage(out).free / 1e9
    # store + room for one in-flight archive (~0.5 GB) + 10% slack
    need = est["store_gb"] * 1.1 + (0.0 if purge else 0.5) + 0.2
    print(f"span {est['start_ts']}..{est['end_ts']}  {est['n_seconds']:,} s "
          f"({est['n_seconds']/86400:.0f} days)")
    print(f"store {est['store_gb']} GB | peak RAM ~{est['peak_ram_mb']} MB | "
          f"free disk {free:.1f} GB")
    if free < need:
        raise SystemExit(f"need ~{need:.1f} GB free, have {free:.1f} GB "
                         f"(use --purge to delete archives as they are folded)")

    n = est["n_seconds"]
    prices = np.memmap(out / "prices.i64", np.int64, "r+" if
                       (out / "prices.i64").exists() else "w+", shape=(n, 4))
    flows = np.memmap(out / "flows.f32", np.float32, "r+" if
                      (out / "flows.f32").exists() else "w+", shape=(n, 4))

    spath = out / STATE
    state = json.loads(spath.read_text()) if spath.exists() else {
        "done": [], "carry_close": 0, "max_ts": 0, "start_ts": est["start_ts"]}
    if state.get("start_ts") != est["start_ts"]:
        raise SystemExit(f"{spath} is for a different span; delete it to rebuild")
    done = set(state["done"])

    session = requests.Session()
    for i, t in enumerate(targets, 1):
        name = t.dest.name
        if name in done:
            continue
        lo, hi = archive_span(t.dest)
        print(f"[{i}/{len(targets)}] {name}", flush=True)

        if not already_ok(session, t, verify=verify):
            if not fetch(session, t):
                if keep_going:
                    print(f"  !! skipped (download failed)")
                    continue
                raise SystemExit(f"download failed: {t.url}")

        o, h, l, c, cnt, vol, notional, buy = _fold_block(t.dest, lo, hi, chunksize)

        # Forward-fill this block's closes, seeded by the previous block.
        carry = int(state["carry_close"])
        nz = c != 0
        if not nz.any():
            c[:] = carry
        else:
            idx = np.where(nz, np.arange(c.size), -1)
            idx = np.maximum.accumulate(idx)
            first = int(np.argmax(nz))
            c = np.where(idx >= 0, c[idx.clip(0)], carry)
            if carry == 0:
                c[:first] = c[first]
        empty = cnt == 0
        o = np.where(empty | (o == 0), c, o)
        h = np.where(empty | (h == 0), c, h)
        l = np.where(empty | (l == np.iinfo(np.int64).max), c, l)

        sl = slice(lo - est["start_ts"], hi - est["start_ts"])
        prices[sl, 0], prices[sl, 1] = o, h
        prices[sl, 2], prices[sl, 3] = l, c
        vwap = np.where(vol > 0, notional / np.maximum(vol, 1e-12), c / CENTS) * CENTS
        flows[sl, 0], flows[sl, 1] = vol, cnt
        flows[sl, 2], flows[sl, 3] = buy, vwap
        prices.flush(); flows.flush()

        if cnt.any():
            state["max_ts"] = max(state["max_ts"], lo + int(np.max(np.nonzero(cnt))))
        state["carry_close"] = int(c[-1])
        state["done"].append(name)
        spath.write_text(json.dumps(state))

        if purge:
            t.dest.unlink(missing_ok=True)
            print(f"  purged {name}")

    # Trim to the last second that actually carried a trade.
    last = int(state["max_ts"]) if state["max_ts"] else est["end_ts"] - 1
    n_final = last - est["start_ts"] + 1
    del prices, flows
    if n_final < n:
        os.truncate(out / "prices.i64", n_final * 32)
        os.truncate(out / "flows.f32", n_final * 16)

    meta = {
        "symbol": symbol,
        "start_ts": est["start_ts"],
        "n_seconds": n_final,
        "price_scale": CENTS,
        "price_channels": ["open", "high", "low", "close"],
        "flow_channels": ["volume", "n_trades", "taker_buy_volume", "vwap"],
        "built_by": "btcpred.data.build (streaming)",
        "archives": len(state["done"]),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {out}  {n_final:,} seconds ({n_final/86400:.1f} days, "
          f"{n_final*48/1e9:.2f} GB)")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start", default="2020-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/bars_1s"))
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    ap.add_argument("--purge", action="store_true",
                    help="delete each archive once folded in (saves ~25 GB)")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    today = dt.date.today()
    end = args.end or f"{today.year:04d}-{today.month:02d}"

    if args.dry_run:
        est = estimate(plan(args.symbol, args.start, end, args.raw))
        print(json.dumps(est, indent=2))
        return 0

    build(args.symbol, args.start, end, args.raw, args.out,
          chunksize=args.chunksize, purge=args.purge,
          keep_going=args.keep_going, verify=not args.no_verify)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
