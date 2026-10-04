"""
Compute discovery and automatic model sizing.

Run this *before* training. It inventories GPUs, VRAM, CPU cores and RAM, then
proposes the largest analyser/predictor configuration that fits, using an
explicit VRAM budget rather than trial-and-error OOM.

Memory model per GPU (DDP replicates everything on every rank):

    params      P * 2 bytes   (bf16 compute copy)
    master/opt  P * 12 bytes  (fp32 master + Adam m + v)
    gradients   P * 2 bytes
    activations ~ batch * tokens * d_model * layers * k

so the optimiser state, not the weights, dominates. At ~16 bytes/param a
23 GB A10 can hold roughly 600 M total parameters before activations, which is
why the default 85 M + 3x155 M (~550 M) sits near the practical ceiling when
all three variants train concurrently on one GPU.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess


def _ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError):
        return 0.0


def describe_hardware() -> dict:
    import torch

    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_count": os.cpu_count(),
        "cpu_affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "ram_gb": round(_ram_gb(), 1),
        "cuda_available": torch.cuda.is_available(),
        "gpus": [],
        "disk_free_gb": round(shutil.disk_usage(".").free / 1e9, 1),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["bf16"] = torch.cuda.is_bf16_supported()
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            info["gpus"].append({
                "index": i, "name": p.name,
                "vram_gb": round(p.total_memory / 1e9, 1),
                "sm": f"{p.major}.{p.minor}",
                "multi_processor_count": p.multi_processor_count,
            })
    else:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10)
            if out.returncode == 0 and out.stdout.strip():
                info["nvidia_smi"] = out.stdout.strip().splitlines()
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
    return info


BYTES_PER_PARAM_TRAIN = 16  # bf16 weights + fp32 master + Adam m/v + grads


def autoscale(info: dict, n_variants: int = 3, activation_reserve_gb: float = 6.0) -> dict:
    """Pick analyser/predictor widths that fit the detected VRAM."""
    gpus = info.get("gpus") or []
    if not gpus:
        return {"scale": 0.35, "variants": ["tcn"], "batch_size": 2,
                "amp": False, "grad_checkpoint": False,
                "note": "no GPU detected - CPU smoke configuration"}

    vram = min(g["vram_gb"] for g in gpus)
    budget_gb = max(1.0, vram - activation_reserve_gb)
    param_budget = budget_gb * 1e9 / BYTES_PER_PARAM_TRAIN

    # Baseline totals at scale=1.0 (see models/): 85M analyser + ~155M each.
    base_total = 85e6 + 155e6 * n_variants
    # Parameters grow ~quadratically in width, so scale ~ sqrt(ratio).
    scale = (param_budget / base_total) ** 0.5
    scale = max(0.5, min(scale, 2.0))

    batch = 8 if vram >= 40 else (4 if vram >= 22 else 2)
    return {
        "scale": round(scale, 2),
        "variants": ["xfmr", "tcn", "ssm"][:n_variants],
        "batch_size": batch,
        "amp": True,
        "grad_checkpoint": scale > 1.2,
        "param_budget_m": round(param_budget / 1e6),
        "vram_gb": vram,
        "n_gpus": len(gpus),
    }


if __name__ == "__main__":
    import json

    info = describe_hardware()
    print(json.dumps(info, indent=2))
    print("\nrecommended:")
    print(json.dumps(autoscale(info), indent=2))
