"""
Dataset preflight: diagnose a bar store before anything tries to mmap it.

Motivation: a Git LFS pointer is a 132-byte text file sitting exactly where a
multi-gigabyte binary is expected. `np.memmap` reports that as

    ValueError: mmap length is greater than file size

which says nothing about the actual cause. This module turns every failure
mode into a message that names the problem and the fix.

    python -m btcpred.data.verify --bars data/bars_1s
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LFS_MAGIC = b"version https://git-lfs"
EXPECTED = {"prices.i64": 32, "flows.f32": 16}  # bytes per second of data


class DatasetError(RuntimeError):
    """Raised with an actionable message rather than a stack trace."""


def is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return fh.read(len(LFS_MAGIC)) == LFS_MAGIC
    except OSError:
        return False


def pointer_size(path: Path) -> int | None:
    """The true byte size recorded inside an LFS pointer file."""
    try:
        for line in path.read_text(errors="ignore").splitlines():
            if line.startswith("size "):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return None


def check(bars: str | Path) -> dict:
    """Inspect a bar store. Returns a report; raises DatasetError if unusable."""
    root = Path(bars)
    report: dict = {"root": str(root), "ok": False, "problems": [], "files": {}}

    if not root.exists():
        raise DatasetError(
            f"Dataset directory not found: {root}\n\n"
            f"Build it with:  ./scripts/fetch_data.sh\n"
            f"or load it from a Snowflake stage (see README section 'Getting "
            f"the dataset into Snowflake')."
        )

    meta_path = root / "meta.json"
    if not meta_path.exists():
        raise DatasetError(f"Missing {meta_path}. The bar store is incomplete.")
    meta = json.loads(meta_path.read_text())
    n = int(meta["n_seconds"])
    report["n_seconds"] = n
    report["days"] = round(n / 86400, 2)

    pointers, truncated = [], []
    for name, bpr in EXPECTED.items():
        f = root / name
        want = n * bpr
        if not f.exists():
            report["problems"].append(f"{name}: missing")
            continue
        got = f.stat().st_size
        report["files"][name] = {"bytes": got, "expected": want}
        if is_lfs_pointer(f):
            pointers.append((name, got, pointer_size(f) or want))
        elif got != want:
            truncated.append((name, got, want))

    if pointers:
        lines = "\n".join(
            f"    {nm:14} {got:>10,} B on disk  ->  should be {real:>14,} B"
            for nm, got, real in pointers
        )
        raise DatasetError(
            "The dataset files are Git LFS POINTERS, not real data.\n\n"
            f"{lines}\n\n"
            "A pointer is a ~132-byte text stub. numpy reports this as\n"
            "  'ValueError: mmap length is greater than file size'.\n\n"
            "Fix, depending on where you are:\n\n"
            "  WITH internet (your laptop, molab):\n"
            "      git lfs install && git lfs pull\n\n"
            "  WITHOUT internet (Snowflake):\n"
            "      git lfs pull CANNOT work here - it needs network.\n"
            "      Load the dataset from a Snowflake stage instead:\n"
            "          python -m btcpred.data.stage --download \\\n"
            "              --stage @BTCPRED_DATA --out data/bars_1s\n"
            "      See README -> 'Getting the dataset into Snowflake'."
        )

    if truncated:
        lines = "\n".join(
            f"    {nm:14} {got:>14,} B  ->  expected {want:>14,} B"
            for nm, got, want in truncated
        )
        raise DatasetError(
            "Dataset files are the wrong size - the transfer was truncated or "
            "meta.json does not match the binaries.\n\n"
            f"{lines}\n\n"
            "Re-copy the bar store, or rebuild it:\n"
            "    python -m btcpred.data.resample --raw data/raw --out "
            f"{root}"
        )

    if report["problems"]:
        raise DatasetError(
            "Bar store incomplete: " + "; ".join(report["problems"])
        )

    report["ok"] = True
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bars", default="data/bars_1s")
    ap.add_argument("--warmup-seconds", type=int, default=43200,
                    help="context the model needs (12 h default)")
    ap.add_argument("--min-days", type=float, default=40.0)
    args = ap.parse_args()

    try:
        rep = check(args.bars)
    except DatasetError as exc:
        print(f"\nDATASET NOT USABLE\n\n{exc}\n")
        return 1

    print(f"bar store : {rep['root']}")
    print(f"seconds   : {rep['n_seconds']:,}  ({rep['days']} days)")
    for nm, info in rep["files"].items():
        print(f"  {nm:14} {info['bytes']:>15,} B  OK")

    warn = []
    need_days = (args.warmup_seconds + 1500) / 86400
    if rep["days"] < need_days:
        warn.append(
            f"only {rep['days']} days: a single sample needs "
            f"{need_days:.2f} days (12 h warm-up + 25 min lookahead)"
        )
    if rep["days"] < args.min_days:
        warn.append(
            f"{rep['days']} days is far too little to train on; the val/test "
            f"split alone reserves 21 days. Expect ~2100 days for 2020-today."
        )
    if warn:
        print("\nWARNINGS")
        for w in warn:
            print(f"  ! {w}")
        print("\n  This looks like the 3-day SAMPLE shipped for smoke tests,\n"
              "  not the full dataset. Run ./scripts/fetch_data.sh.")
        return 2

    print("\nready to train")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
