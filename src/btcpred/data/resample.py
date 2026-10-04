"""
Resample Binance aggTrades archives into a gap-free 1-second grid.

Output (a "bar store") is written to <out>/ as three files:

    prices.i64    memmap int64  (N, 4)  open, high, low, close   in cents
    flows.f32     memmap float32(N, 4)  volume, n_trades, taker_buy_vol, vwap_cents
    meta.json                           start_ts, n_seconds, symbol, channels

Prices are stored as integer cents so that no precision is lost (float32 can
only resolve ~0.008 USD at a 100k price level, which is comparable to the tick
size and would corrupt second-scale returns).

Seconds with no trades are *not* dropped. They are forward-filled on price and
zero-filled on flow, and a gap mask is derivable from `n_trades == 0`. Keeping
the grid contiguous is what lets the trainer slice a 12 h window with pure
pointer arithmetic instead of a timestamp search.

Memory: the per-file loop streams in chunks and aggregates with bincount, so
peak RSS stays a few hundred MB regardless of dataset size.

Usage
-----
    python -m btcpred.data.resample --raw data/raw --out data/bars_1s
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# aggTrades schema (headerless in older archives, headered in newer ones):
#   0 agg_trade_id  1 price  2 quantity  3 first_id  4 last_id
#   5 timestamp     6 is_buyer_maker    7 is_best_match
COLS = [1, 2, 5, 6]
NAMES = ["price", "qty", "ts", "is_buyer_maker"]
CENTS = 100

DATE_RE = re.compile(r"(\d{4})-(\d{2})(?:-(\d{2}))?\.zip$")


def archive_sort_key(p: Path) -> tuple:
    m = DATE_RE.search(p.name)
    if not m:
        return (9999, 99, 99)
    y, mo, d = m.group(1), m.group(2), m.group(3)
    return (int(y), int(mo), int(d) if d else 0)


def normalise_ts(ts: np.ndarray) -> np.ndarray:
    """Binance switched aggTrades timestamps from ms to us during 2025.

    Detect by magnitude and return seconds (int64).
    """
    probe = int(ts[0])
    if probe > 1_000_000_000_000_000:  # microseconds
        return ts // 1_000_000
    if probe > 1_000_000_000_000:  # milliseconds
        return ts // 1_000
    return ts  # already seconds


def iter_chunks(zpath: Path, chunksize: int):
    """Yield DataFrames from the single CSV inside a Binance archive."""
    with zipfile.ZipFile(zpath) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            head = fh.read(256)
        has_header = b"agg_trade_id" in head or b"transact_time" in head

        with zf.open(name) as fh:
            yield from pd.read_csv(
                fh,
                header=0 if has_header else None,
                usecols=COLS,
                names=None if has_header else NAMES,
                dtype={1: np.float64, 2: np.float64, 5: np.int64},
                chunksize=chunksize,
                memory_map=False,
            )


def accumulate(df: pd.DataFrame, store: dict, start_ts: int, n: int) -> None:
    """Fold one chunk of trades into the second-grid accumulators."""
    cols = list(df.columns)
    price = df[cols[0]].to_numpy(np.float64)
    qty = df[cols[1]].to_numpy(np.float64)
    ts = normalise_ts(df[cols[2]].to_numpy(np.int64))
    maker = df[cols[3]].to_numpy()

    idx = ts - start_ts
    keep = (idx >= 0) & (idx < n)
    if not keep.all():
        idx, price, qty, maker = idx[keep], price[keep], qty[keep], maker[keep]
    if idx.size == 0:
        return

    pc = np.rint(price * CENTS).astype(np.int64)

    cnt = np.bincount(idx, minlength=n)
    store["count"] += cnt
    store["vol"] += np.bincount(idx, weights=qty, minlength=n)
    store["notional"] += np.bincount(idx, weights=price * qty, minlength=n)

    # is_buyer_maker == True  =>  the aggressor was a *seller*.
    taker_buy = (~maker.astype(bool)).astype(np.float64) * qty
    store["buyvol"] += np.bincount(idx, weights=taker_buy, minlength=n)

    # High / low via maximum.at and minimum.at (order-independent).
    np.maximum.at(store["high"], idx, pc)
    np.minimum.at(store["low"], idx, pc)

    # Open = price of the first trade in the second, close = the last.
    # Chunks arrive in time order, so: first writer wins for open,
    # last writer wins for close.
    fresh = store["open"][idx] == 0
    if fresh.any():
        store["open"][idx[fresh]] = pc[fresh]
    store["close"][idx] = pc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/bars_1s"))
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    args = ap.parse_args()

    archives = sorted(args.raw.rglob("*.zip"), key=archive_sort_key)
    if not archives:
        raise SystemExit(f"no .zip archives under {args.raw}")
    print(f"{len(archives)} archive(s)")

    # --- pass 1: determine the global time span -------------------------
    def edge_ts(z: Path, last: bool) -> int:
        best = None
        for df in iter_chunks(z, 200_000):
            t = normalise_ts(df[df.columns[2]].to_numpy(np.int64))
            best = int(t[-1]) if last else int(t[0])
            if not last:
                break
        return best

    start_ts = edge_ts(archives[0], last=False)
    end_ts = edge_ts(archives[-1], last=True)
    n = int(end_ts - start_ts + 1)
    print(f"span {start_ts}..{end_ts}  ->  {n:,} seconds "
          f"({n / 86400:.1f} days)")

    args.out.mkdir(parents=True, exist_ok=True)

    store = {
        "open": np.zeros(n, np.int64),
        "high": np.zeros(n, np.int64),
        "low": np.full(n, np.iinfo(np.int64).max, np.int64),
        "close": np.zeros(n, np.int64),
        "count": np.zeros(n, np.int64),
        "vol": np.zeros(n, np.float64),
        "notional": np.zeros(n, np.float64),
        "buyvol": np.zeros(n, np.float64),
    }

    # --- pass 2: aggregate ----------------------------------------------
    for i, z in enumerate(archives, 1):
        print(f"[{i}/{len(archives)}] {z.name}", flush=True)
        for df in iter_chunks(z, args.chunksize):
            accumulate(df, store, start_ts, n)

    # --- fill gaps -------------------------------------------------------
    empty = store["count"] == 0
    print(f"empty seconds: {empty.sum():,} ({100 * empty.mean():.3f}%)")

    close = store["close"]
    if close[0] == 0:  # leading gap: back-fill from the first real print
        first = int(np.flatnonzero(close)[0])
        close[:first] = close[first]
    # Forward-fill close across gaps.
    nz = np.flatnonzero(close)
    fill_idx = np.maximum.accumulate(np.where(close != 0, np.arange(n), 0))
    close = close[fill_idx]

    for k in ("open", "high", "low"):
        store[k] = np.where(empty, close, store[k])
    store["low"] = np.where(store["low"] == np.iinfo(np.int64).max,
                            close, store["low"])
    store["close"] = close

    vwap = np.where(store["vol"] > 0,
                    store["notional"] / np.maximum(store["vol"], 1e-12),
                    close / CENTS) * CENTS

    prices = np.memmap(args.out / "prices.i64", np.int64, "w+", shape=(n, 4))
    prices[:, 0] = store["open"]
    prices[:, 1] = store["high"]
    prices[:, 2] = store["low"]
    prices[:, 3] = store["close"]
    prices.flush()

    flows = np.memmap(args.out / "flows.f32", np.float32, "w+", shape=(n, 4))
    flows[:, 0] = store["vol"]
    flows[:, 1] = store["count"]
    flows[:, 2] = store["buyvol"]
    flows[:, 3] = vwap
    flows.flush()

    meta = {
        "symbol": args.symbol,
        "start_ts": int(start_ts),
        "n_seconds": int(n),
        "price_scale": CENTS,
        "price_channels": ["open", "high", "low", "close"],
        "flow_channels": ["volume", "n_trades", "taker_buy_volume", "vwap"],
        "empty_seconds": int(empty.sum()),
        "source_archives": [z.name for z in archives],
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {args.out}  ({n:,} x 8)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
