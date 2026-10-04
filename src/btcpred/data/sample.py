"""
The committed smoke-test sample: 3 days of real BTCUSDT 1-second bars.

This exists so that `git clone` alone -- no Git LFS, no network, no Snowflake
stage -- is enough to run the tests, the simulator and a smoke training job.

It is stored as a single compressed .npz (~2.4 MB) rather than raw memmap
binaries (12.4 MB). Prices are delta-encoded to int32 before compression:
consecutive 1-second closes in cents differ by very little, so the deltas are
tiny integers that zip down to roughly a fifth of the raw size, and the whole
file lands far under GitHub's 100 MB per-file limit as an ordinary blob.

This is the SAMPLE, not the dataset. Three days is enough to prove the code
runs; it is nowhere near enough to train. The real store is ~8.7 GB and
travels by Snowflake stage (see btcpred.data.stage).

    python -m btcpred.data.sample --expand --out data/bars_1s
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_NPZ = Path("data/sample/bars_3d.npz")


def compress(bars: Path, out: Path) -> int:
    """Pack a bar store directory into a single compressed .npz."""
    meta = json.loads((bars / "meta.json").read_text())
    n = int(meta["n_seconds"])
    p = np.asarray(np.memmap(bars / "prices.i64", np.int64, "r", shape=(n, 4)))
    f = np.asarray(np.memmap(bars / "flows.f32", np.float32, "r", shape=(n, 4)))

    # Delta-encode: 1-second price changes are small, so int32 deltas
    # compress far better than absolute int64 cents. Prepend a ZERO row (not
    # p[:1]) so that deltas[0] == p[0] and cumsum reconstructs the absolute
    # series; prepending p[:1] would zero the first row and lose the base.
    deltas = np.diff(p, axis=0, prepend=np.zeros((1, 4), p.dtype)).astype(np.int32)
    if not np.array_equal(np.cumsum(deltas.astype(np.int64), axis=0), p):
        raise ValueError("delta encoding is lossy for this store "
                         "(price jump exceeds int32) - refusing to write")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, dp=deltas, flows=f, meta=json.dumps(meta))
    return out.stat().st_size


def expand(npz: Path, out: Path, force: bool = False) -> dict:
    """Materialise the .npz back into a memmap bar store."""
    if not npz.exists():
        raise FileNotFoundError(
            f"{npz} not found. It should be committed to the repo; "
            f"re-clone or run --compress to regenerate it."
        )
    out.mkdir(parents=True, exist_ok=True)
    if (out / "meta.json").exists() and not force:
        existing = json.loads((out / "meta.json").read_text())
        return {"skipped": True, "n_seconds": int(existing["n_seconds"])}

    with np.load(npz, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        prices = np.cumsum(z["dp"].astype(np.int64), axis=0)
        flows = z["flows"].astype(np.float32)

    n = int(meta["n_seconds"])
    if prices.shape != (n, 4) or flows.shape != (n, 4):
        raise ValueError(f"shape mismatch: prices {prices.shape}, "
                         f"flows {flows.shape}, meta says n={n}")

    mp = np.memmap(out / "prices.i64", np.int64, "w+", shape=(n, 4))
    mp[:] = prices
    mp.flush()
    del mp
    mf = np.memmap(out / "flows.f32", np.float32, "w+", shape=(n, 4))
    mf[:] = flows
    mf.flush()
    del mf
    meta["is_sample"] = True
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return {"skipped": False, "n_seconds": n, "days": round(n / 86400, 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compress", action="store_true",
                    help="pack an existing bar store into the committed .npz")
    ap.add_argument("--expand", action="store_true",
                    help="materialise the .npz into a usable bar store")
    ap.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    ap.add_argument("--bars", type=Path, default=Path("data/bars_1s"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.compress:
        size = compress(args.bars, args.npz)
        print(f"wrote {args.npz} ({size/1e6:.2f} MB)")
        return 0

    if args.expand:
        out = args.out or args.bars
        r = expand(args.npz, out, force=args.force)
        if r.get("skipped"):
            print(f"{out} already exists ({r['n_seconds']:,} seconds); "
                  f"use --force to overwrite")
        else:
            print(f"expanded sample -> {out}  "
                  f"({r['n_seconds']:,} seconds, {r['days']} days)")
        print("\nNOTE: this is the 3-day SAMPLE, for smoke tests only.\n"
              "      Training needs the full store (~2100 days, ~8.7 GB):\n"
              "        ./scripts/fetch_data.sh          # networked machine\n"
              "        python -m btcpred.data.stage --pack --out dist/\n"
              "        # PUT dist/* to a Snowflake stage, then --unpack there")
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
