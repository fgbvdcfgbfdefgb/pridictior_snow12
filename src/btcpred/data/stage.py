"""
Move the bar store into Snowflake without internet.

Why not Git LFS
---------------
LFS is the wrong transport for this dataset, for two independent reasons:

  * Size. The full 2020-today bar store is ~8.7 GB (prices.i64 5.8 GB +
    flows.f32 2.9 GB), plus ~25 GB of raw archives. GitHub's free LFS tier is
    1 GB of storage and 1 GB of bandwidth per month.
  * Offline. `git lfs pull` is a network call, separate from `git clone`.
    A Snowflake container with no egress cannot resolve pointers at all --
    it gets 132-byte stubs and numpy dies on the mmap.

So code travels by git; data travels by Snowflake stage.

Design
------
The store is split into fixed-size chunks with a SHA-256 manifest. Chunking
buys three things on a multi-gigabyte transfer: per-part resumability, early
corruption detection, and the ability to parallelise PUT. Reassembly verifies
every chunk and then the whole file, so a silent truncation becomes a loud
error rather than a bizarre training failure.

Usage
-----
On a networked machine, after ./scripts/fetch_data.sh:

    python -m btcpred.data.stage --pack --bars data/bars_1s --out dist/
    snowsql -q "CREATE STAGE IF NOT EXISTS BTCPRED_DATA"
    snowsql -q "PUT file://dist/* @BTCPRED_DATA AUTO_COMPRESS=FALSE OVERWRITE=TRUE"

Inside Snowflake (no internet needed):

    GET @BTCPRED_DATA file:///tmp/dist/;
    python -m btcpred.data.stage --unpack --src /tmp/dist --out data/bars_1s

Or, if you have the Python connector configured, --upload / --download wrap
the PUT/GET for you.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

CHUNK_BYTES = 256 * 1024 * 1024  # 256 MiB: big enough to be fast, small
                                 # enough that one retry is cheap
READ_BLOCK = 8 * 1024 * 1024
STORE_FILES = ("meta.json", "prices.i64", "flows.f32")
MANIFEST = "manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(READ_BLOCK):
            h.update(block)
    return h.hexdigest()


def pack(bars: Path, out: Path, chunk_bytes: int = CHUNK_BYTES) -> dict:
    """Split a bar store into verifiable chunks under `out`."""
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"chunk_bytes": chunk_bytes, "files": {}}

    for name in STORE_FILES:
        src = bars / name
        if not src.exists():
            raise FileNotFoundError(f"{src} missing - is the bar store built?")
        total = src.stat().st_size
        entry = {"bytes": total, "sha256": sha256_file(src), "chunks": []}

        with src.open("rb") as fh:
            idx = 0
            while True:
                buf = fh.read(chunk_bytes)
                if not buf:
                    break
                cname = f"{name}.{idx:05d}"
                cpath = out / cname
                cpath.write_bytes(buf)
                entry["chunks"].append({
                    "name": cname,
                    "bytes": len(buf),
                    "sha256": hashlib.sha256(buf).hexdigest(),
                })
                idx += 1
        # A zero-length file still needs one (empty) chunk to round-trip.
        if not entry["chunks"]:
            cname = f"{name}.00000"
            (out / cname).write_bytes(b"")
            entry["chunks"].append({"name": cname, "bytes": 0,
                                    "sha256": hashlib.sha256(b"").hexdigest()})
        manifest["files"][name] = entry

    (out / MANIFEST).write_text(json.dumps(manifest, indent=2))
    return manifest


def unpack(src: Path, out: Path, keep_chunks: bool = False) -> dict:
    """Reassemble and verify a packed bar store."""
    mpath = src / MANIFEST
    if not mpath.exists():
        raise FileNotFoundError(
            f"{mpath} not found. Did the GET from the stage complete? "
            f"Expected {MANIFEST} plus the *.NNNNN chunk files."
        )
    manifest = json.loads(mpath.read_text())
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    for name, entry in manifest["files"].items():
        dest = out / name
        with dest.open("wb") as fh:
            for c in entry["chunks"]:
                cpath = src / c["name"]
                if not cpath.exists():
                    raise FileNotFoundError(
                        f"chunk {c['name']} missing - transfer incomplete. "
                        f"Re-run the GET; packing is deterministic so you can "
                        f"fetch only the missing parts."
                    )
                buf = cpath.read_bytes()
                if len(buf) != c["bytes"]:
                    raise ValueError(
                        f"chunk {c['name']}: {len(buf)} bytes, expected "
                        f"{c['bytes']} - truncated transfer"
                    )
                got = hashlib.sha256(buf).hexdigest()
                if got != c["sha256"]:
                    raise ValueError(
                        f"chunk {c['name']}: sha256 mismatch - corrupted "
                        f"transfer (got {got[:16]}, want {c['sha256'][:16]})"
                    )
                fh.write(buf)

        size = dest.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{name}: assembled {size} bytes, "
                             f"expected {entry['bytes']}")
        digest = sha256_file(dest)
        if digest != entry["sha256"]:
            raise ValueError(f"{name}: whole-file sha256 mismatch after "
                             f"reassembly")
        report[name] = {"bytes": size, "sha256": digest}

    if not keep_chunks:
        for entry in manifest["files"].values():
            for c in entry["chunks"]:
                (src / c["name"]).unlink(missing_ok=True)

    return report


# --------------------------------------------------------------------------
# Optional Snowflake connector wrappers
# --------------------------------------------------------------------------
def _connect():
    try:
        import snowflake.connector as sf
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "snowflake-connector-python is not installed. Either\n"
            "  pip install snowflake-connector-python\n"
            "or skip --upload/--download and use SnowSQL PUT/GET manually "
            "with --pack / --unpack."
        ) from e

    missing = [k for k in ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER") if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"set {', '.join(missing)} (and SNOWFLAKE_PASSWORD "
                           f"or SNOWFLAKE_PRIVATE_KEY_PATH)")
    return sf.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ.get("SNOWFLAKE_PASSWORD"),
        role=os.environ.get("SNOWFLAKE_ROLE"),
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE"),
        database=os.environ.get("SNOWFLAKE_DATABASE"),
        schema=os.environ.get("SNOWFLAKE_SCHEMA"),
    )


def upload(local: Path, stage: str, parallel: int = 8) -> None:
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(f"CREATE STAGE IF NOT EXISTS {stage.lstrip('@')}")
        for f in sorted(local.iterdir()):
            if not f.is_file():
                continue
            print(f"PUT {f.name} ({f.stat().st_size/1e6:.1f} MB)", flush=True)
            cur.execute(
                f"PUT 'file://{f.as_posix()}' {stage} "
                f"AUTO_COMPRESS=FALSE OVERWRITE=TRUE PARALLEL={parallel}"
            )
    finally:
        conn.close()


def download(stage: str, local: Path, parallel: int = 8) -> None:
    local.mkdir(parents=True, exist_ok=True)
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(f"GET {stage} 'file://{local.as_posix()}/' PARALLEL={parallel}")
        for row in cur.fetchall():
            print(row, flush=True)
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", action="store_true")
    ap.add_argument("--unpack", action="store_true")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--bars", type=Path, default=Path("data/bars_1s"))
    ap.add_argument("--src", type=Path, default=Path("dist"))
    ap.add_argument("--out", type=Path, default=Path("dist"))
    ap.add_argument("--stage", default="@BTCPRED_DATA")
    ap.add_argument("--chunk-mb", type=int, default=CHUNK_BYTES // (1024 * 1024))
    ap.add_argument("--keep-chunks", action="store_true")
    args = ap.parse_args()

    if args.pack:
        m = pack(args.bars, args.out, args.chunk_mb * 1024 * 1024)
        total = sum(e["bytes"] for e in m["files"].values())
        nch = sum(len(e["chunks"]) for e in m["files"].values())
        print(f"packed {total/1e9:.2f} GB into {nch} chunk(s) under {args.out}")
        for name, e in m["files"].items():
            print(f"  {name:14} {e['bytes']:>15,} B  {len(e['chunks'])} chunk(s)")
        print(f"\nnext:\n  snowsql -q \"PUT file://{args.out}/* {args.stage} "
              f"AUTO_COMPRESS=FALSE OVERWRITE=TRUE\"")
        return 0

    if args.upload:
        upload(args.out, args.stage)
        return 0

    if args.download:
        download(args.stage, args.src)
        # fall through to unpack

    if args.unpack or args.download:
        rep = unpack(args.src, args.bars, keep_chunks=args.keep_chunks)
        print(f"reassembled into {args.bars}")
        for name, info in rep.items():
            print(f"  {name:14} {info['bytes']:>15,} B  sha256 "
                  f"{info['sha256'][:16]} OK")
        from .verify import check
        r = check(args.bars)
        print(f"\nverified: {r['n_seconds']:,} seconds ({r['days']} days)")
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
