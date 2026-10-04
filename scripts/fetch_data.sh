#!/usr/bin/env bash
# Build the full 1-second dataset, streaming and resumable.
#
# Folds one archive at a time: download -> accumulate -> write -> purge.
#   peak RAM   ~200 MB   (not 11.7 GB)
#   peak disk  ~10 GB    (not ~34 GB)   with --purge
#   time       6-10 h, fully resumable - just re-run after any interruption
#
# Drop --purge if you want to keep the raw archives (needs ~25 GB more).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src

START="${START:-2020-01}"
END="${END:-$(date -u +%Y-%m)}"
SYMBOL="${SYMBOL:-BTCUSDT}"
PURGE="${PURGE:---purge}"

echo ">> plan"
python -m btcpred.data.build --symbol "$SYMBOL" --start "$START" --end "$END" --dry-run

echo ">> building (resumable; safe to re-run)"
python -m btcpred.data.build \
    --symbol "$SYMBOL" --start "$START" --end "$END" \
    --raw data/raw --out data/bars_1s $PURGE --keep-going

echo ">> verifying"
python -m btcpred.data.verify --bars data/bars_1s

echo
echo ">> next: get it into Snowflake (no internet needed there)"
echo "   python -m btcpred.data.stage --pack --bars data/bars_1s --out dist/"
echo "   snowsql -q \"PUT file://dist/* @BTCPRED_DATA AUTO_COMPRESS=FALSE OVERWRITE=TRUE\""
