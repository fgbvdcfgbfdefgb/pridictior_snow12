#!/usr/bin/env bash
# Distributed training on 4x A10. Offline-safe.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

NGPU="${NGPU:-$(python -c 'import torch;print(torch.cuda.device_count())')}"
echo ">> detected ${NGPU} GPU(s)"
python -m btcpred.utils.hardware

torchrun --standalone --nproc_per_node="${NGPU}" \
    -m btcpred.train.train_ddp --config configs/a10x4.json "$@"

python -m btcpred.train.select_best --config configs/a10x4.json \
    --ckpt checkpoints --split val --step 30 --max-points 4000
