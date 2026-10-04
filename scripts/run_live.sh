#!/usr/bin/env bash
# Launch the live molab dashboard.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export BTCPRED_CKPT="${BTCPRED_CKPT:-checkpoints}"
exec marimo edit notebooks/molab_live.py --host 0.0.0.0 --port "${PORT:-2718}"
