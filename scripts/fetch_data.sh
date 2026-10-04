#!/usr/bin/env bash
# Build the full 1-second dataset. Run on a NETWORKED machine, then commit
# the result (Git LFS) so Snowflake can train with no internet.
#
#   ~25 GB of downloads, 6-10 h. Fully resumable: re-run after any interruption.
#   Needs ~60 GB free disk (25 GB archives + ~9 GB bar store + headroom).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src

START="${START:-2020-01}"
END="${END:-$(date -u +%Y-%m)}"
SYMBOL="${SYMBOL:-BTCUSDT}"

echo ">> downloading ${SYMBOL} aggTrades ${START} .. ${END}"
python -m btcpred.data.download_binance \
    --symbol "$SYMBOL" --start "$START" --end "$END" --out data/raw

echo ">> resampling to a gap-free 1-second grid"
python -m btcpred.data.resample --raw data/raw --out data/bars_1s

echo ">> done"
du -sh data/raw data/bars_1s
