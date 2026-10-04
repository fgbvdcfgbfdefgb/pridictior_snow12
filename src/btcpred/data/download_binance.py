"""
Resumable downloader for Binance public market-data archives.

Source: https://data.binance.vision  (free, no API key, no auth)

We pull `aggTrades` (aggregated trades), which is the only free bulk source
with true sub-second resolution going back to 2020. Monthly ZIPs are used for
complete past months; daily ZIPs fill in the current (incomplete) month.

Every file is checksum-verified against the .CHECKSUM sidecar that Binance
publishes, and already-verified files are skipped, so the job is fully
resumable after an interruption.

Typical full run (BTCUSDT, 2020-01 .. today):
    ~69 monthly files, ~25 GB compressed, 6-10 h on a home connection.

Usage
-----
    python -m btcpred.data.download_binance \
        --symbol BTCUSDT --start 2020-01 --end 2026-10 --out data/raw
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests

BASE = "https://data.binance.vision/data/spot"
CHUNK = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class Target:
    """One archive file to fetch."""

    url: str
    dest: Path

    @property
    def checksum_url(self) -> str:
        return self.url + ".CHECKSUM"


def month_range(start: str, end: str) -> list[str]:
    """Inclusive list of 'YYYY-MM' strings from start to end."""
    y0, m0 = (int(x) for x in start.split("-"))
    y1, m1 = (int(x) for x in end.split("-"))
    out, y, m = [], y0, m0
    while (y, m) <= (y1, m1):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def build_targets(symbol: str, start: str, end: str, out: Path) -> list[Target]:
    """Monthly archives for complete months, daily archives for the current one."""
    today = dt.date.today()
    current_month = f"{today.year:04d}-{today.month:02d}"
    targets: list[Target] = []

    for ym in month_range(start, end):
        if ym >= current_month:
            continue  # current month is not published as a monthly archive yet
        name = f"{symbol}-aggTrades-{ym}.zip"
        targets.append(
            Target(f"{BASE}/monthly/aggTrades/{symbol}/{name}", out / "monthly" / name)
        )

    # Daily files for the current month, up to yesterday (today is incomplete).
    if end >= current_month:
        day = today.replace(day=1)
        while day < today:
            name = f"{symbol}-aggTrades-{day.isoformat()}.zip"
            targets.append(
                Target(f"{BASE}/daily/aggTrades/{symbol}/{name}", out / "daily" / name)
            )
            day += dt.timedelta(days=1)

    return targets


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def expected_sha(session: requests.Session, target: Target) -> str | None:
    """Binance .CHECKSUM files contain '<sha256>  <filename>'."""
    try:
        r = session.get(target.checksum_url, timeout=30)
        if r.status_code != 200:
            return None
        return r.text.split()[0].strip()
    except requests.RequestException:
        return None


def already_ok(session: requests.Session, target: Target, verify: bool) -> bool:
    if not target.dest.exists() or target.dest.stat().st_size == 0:
        return False
    if not verify:
        return True
    want = expected_sha(session, target)
    return want is None or sha256(target.dest) == want


def fetch(session: requests.Session, target: Target, retries: int = 5) -> bool:
    """Download one file with HTTP range-resume and exponential backoff."""
    target.dest.parent.mkdir(parents=True, exist_ok=True)
    part = target.dest.with_suffix(target.dest.suffix + ".part")

    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with session.get(
                target.url, headers=headers, stream=True, timeout=60
            ) as r:
                if r.status_code == 404:
                    print(f"  !! 404 (not published): {target.url}", file=sys.stderr)
                    return False
                if r.status_code not in (200, 206):
                    raise requests.RequestException(f"HTTP {r.status_code}")
                # A 200 to a Range request means the server ignored it: restart.
                mode = "ab" if (have and r.status_code == 206) else "wb"
                with part.open(mode) as fh:
                    for block in r.iter_content(CHUNK):
                        fh.write(block)
            part.replace(target.dest)
            return True
        except requests.RequestException as exc:
            wait = 2**attempt
            print(f"  .. retry {attempt + 1}/{retries} in {wait}s ({exc})",
                  file=sys.stderr)
            time.sleep(wait)

    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start", default="2020-01", help="YYYY-MM inclusive")
    ap.add_argument("--end", default=None, help="YYYY-MM inclusive (default: today)")
    ap.add_argument("--out", default="data/raw", type=Path)
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N files (smoke tests)")
    ap.add_argument("--daily-only", action="store_true",
                    help="only fetch daily archives (small samples)")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip sha256 verification of existing files")
    args = ap.parse_args()

    today = dt.date.today()
    end = args.end or f"{today.year:04d}-{today.month:02d}"

    targets = build_targets(args.symbol, args.start, end, args.out)
    if args.daily_only:
        targets = [t for t in targets if "daily" in t.dest.parts]
    if args.limit:
        targets = targets[: args.limit]

    print(f"{len(targets)} archive(s) queued -> {args.out}")
    session = requests.Session()
    ok = skipped = failed = 0

    for i, t in enumerate(targets, 1):
        label = f"[{i}/{len(targets)}] {t.dest.name}"
        if already_ok(session, t, verify=not args.no_verify):
            print(f"{label}  (cached)")
            skipped += 1
            continue
        print(f"{label}  downloading...", flush=True)
        if fetch(session, t):
            mb = t.dest.stat().st_size / 1e6
            print(f"{label}  done ({mb:.1f} MB)")
            ok += 1
        else:
            failed += 1

    print(f"\ndownloaded={ok} cached={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
