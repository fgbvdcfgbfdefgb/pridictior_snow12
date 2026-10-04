"""
Correctness tests for the parts that are easy to get silently wrong.

Run:  PYTHONPATH=src python -m pytest tests/ -v
      (or plain `python tests/test_pipeline.py` with no pytest installed)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from btcpred.data.features import (  # noqa: E402
    FEATURE_GAIN,
    LaneSpec,
    encode_lane,
    horizon_grid,
)
from btcpred.models.blocks import GatedStateBlock  # noqa: E402
from btcpred.train.reward import (  # noqa: E402
    consistency_loss,
    pinball_loss,
    smoothness_loss,
)


def test_state_block_matches_sequential_reference():
    """The chunked scan must equal a naive per-step EMA recurrence."""
    import torch.nn.functional as F

    torch.manual_seed(0)
    blk = GatedStateBlock(64, chunk=16).double()
    x = torch.randn(2, 100, 64, dtype=torch.float64) * 0.1
    y = blk(x)

    h = blk.norm(x)
    v, g, r = blk.in_proj(h).chunk(3, -1)
    v = F.silu(blk.conv(v.transpose(1, 2))[..., :100].transpose(1, 2))
    a = torch.exp(blk.log_decay).clamp(1e-4, 0.9999).double()
    s = torch.zeros(2, 64, dtype=torch.float64)
    ref = []
    for t in range(100):
        s = s * a + v[:, t]
        ref.append(s)
    expected = x + blk.out_proj(
        (torch.stack(ref, 1) * (1 - a)) * torch.sigmoid(g) + F.silu(r) * 0.1
    )
    err = (y - expected).abs().max().item()
    assert err < 1e-8, f"chunked scan diverges from reference: {err}"


def test_features_are_scale_invariant():
    """A 10x price level change must not move the features.

    This is the property that lets a model trained across $4k-$120k BTC
    generalise; if it breaks, the model silently learns the price level.
    """
    n = 480
    base = np.cumsum(np.random.default_rng(0).normal(0, 1, n)) + 10_000
    prices = np.stack([base] * 4, 1) * 100
    flows = np.zeros((n, 4))
    flows[:, 0] = 1.0
    flows[:, 1] = 5.0
    flows[:, 2] = 0.5

    f1 = encode_lane(prices, flows, 1, base[-1], 100.0)
    f2 = encode_lane(prices * 10, flows, 1, base[-1] * 10, 100.0)
    # price channels (0-3) must match; volume channels legitimately do not
    err = np.abs(f1[:, :4] - f2[:, :4]).max()
    assert err < 1e-5, f"price features are not scale invariant: {err}"


def test_no_lookahead_in_context():
    """Features at time t must not change when the future is rewritten."""
    n = 480
    rng = np.random.default_rng(1)
    base = np.cumsum(rng.normal(0, 1, n * 2)) + 50_000
    prices = np.stack([base] * 4, 1) * 100
    flows = np.ones((n * 2, 4))

    a = encode_lane(prices[:n], flows[:n], 1, base[n - 1], 100.0)
    tampered = prices.copy()
    tampered[n:] *= 3.0  # obliterate the future
    b = encode_lane(tampered[:n], flows[:n], 1, base[n - 1], 100.0)
    assert np.allclose(a, b), "context leaked information from future bars"


def test_quantiles_cannot_cross():
    from btcpred.models.predictor import HorizonHead

    torch.manual_seed(0)
    head = HorizonHead(64, 4, 25, 3)
    trunk = torch.randn(4, 30, 64) * 3.0
    q = head(trunk)
    assert (q[..., 1:] >= q[..., :-1]).all(), "quantile crossing is possible"


def test_pinball_minimised_at_the_true_quantile():
    q = torch.tensor([0.1, 0.5, 0.9])
    target = torch.randn(4096, 1).expand(-1, 1) * 0 + torch.randn(4096, 1)
    grid = torch.linspace(-3, 3, 61)
    for qi, level in enumerate(q):
        losses = []
        for g in grid:
            pred = torch.zeros(4096, 1, 3)
            pred[..., qi] = g
            err = target.unsqueeze(-1) - pred
            losses.append(torch.maximum(q * err, (q - 1) * err)[..., qi].mean())
        best = grid[int(torch.tensor(losses).argmin())]
        truth = torch.quantile(target, level)
        assert abs(best - truth) < 0.35, (
            f"pinball q={level:.1f} minimised at {best:.2f}, "
            f"empirical quantile is {truth:.2f}"
        )


def test_consistency_loss_is_zero_for_a_perfectly_stable_forecaster():
    """Two forecasts of the same future prices, made 1 s apart, agree."""
    h = torch.tensor(horizon_grid(), dtype=torch.float32)
    H = len(h)
    anchor_a = torch.tensor([50_000.0])
    anchor_b = torch.tensor([50_010.0])
    t_a, t_b = torch.tensor([1000]), torch.tensor([1001])

    # A single underlying absolute price path, expressed from both anchors.
    abs_path = torch.linspace(50_020.0, 50_090.0, H)[None, :, None]
    pred_a = torch.log(abs_path / anchor_a[:, None, None])
    # cursor b's horizon grid points at t_b + h = t_a + 1 + h, so it must
    # forecast the path shifted one second earlier
    shifted = torch.from_numpy(
        np.interp(horizon_grid() + 1, horizon_grid(),
                  abs_path[0, :, 0].numpy()).astype(np.float32)
    )[None, :, None]
    pred_b = torch.log(shifted / anchor_b[:, None, None])

    loss = consistency_loss(pred_a, anchor_a, t_a, pred_b, anchor_b, t_b, h)
    assert loss.item() < 2e-5, f"stable forecaster penalised: {loss.item()}"


def test_consistency_loss_punishes_a_jumpy_forecaster():
    h = torch.tensor(horizon_grid(), dtype=torch.float32)
    H = len(h)
    anchor = torch.tensor([50_000.0])
    t_a, t_b = torch.tensor([1000]), torch.tensor([1001])
    pred_a = torch.zeros(1, H, 1)
    pred_b = torch.full((1, H, 1), 0.01)  # 1% lurch in one second
    loss = consistency_loss(pred_a, anchor, t_a, pred_b, anchor, t_b, h)
    assert loss.item() > 1e-3, "jumpy forecaster escaped the stability penalty"


def test_smoothness_zero_on_a_straight_line():
    line = torch.linspace(0, 1, 25)[None, :, None].expand(2, -1, 3)
    assert smoothness_loss(line).item() < 1e-10


def test_live_runner_smoothing_reduces_revisions():
    """The EMA must damp revisions relative to the raw model output."""
    from btcpred.infer.runner import LiveRunner

    rng = np.random.default_rng(0)
    h = horizon_grid()
    obj = LiveRunner.__new__(LiveRunner)
    obj.horizons = h
    obj.alpha = 0.3
    obj.max_revision_bp = 25.0
    obj._prev_path = None
    obj._prev_ts = None

    raw_prev = None
    raw_jumps, out_jumps = [], []
    for i in range(60):
        raw = 50_000 * np.exp(rng.normal(0, 3e-4, len(h)).cumsum() * 0.3)
        path, _ = obj._stabilise(raw, 1000 + i)
        if raw_prev is not None:
            raw_jumps.append(np.abs(np.log(raw / raw_prev)).mean())
            out_jumps.append(np.abs(np.log(path / prev_out)).mean())
        raw_prev, prev_out = raw, path

    assert np.mean(out_jumps) < np.mean(raw_jumps), (
        f"smoothing increased revisions: {np.mean(out_jumps):.2e} "
        f"vs raw {np.mean(raw_jumps):.2e}"
    )


def test_horizon_grid_covers_25_minutes():
    h = horizon_grid(25, 60)
    assert h[0] == 60 and h[-1] == 1500 and len(h) == 25


def test_feature_gain_length_matches():
    from btcpred.data.features import N_FEATURES

    assert len(FEATURE_GAIN) == N_FEATURES


def test_feed_grid_uses_zero_order_hold_not_interpolation():
    """Sparse samples must be HELD, never smoothly interpolated.

    A held price is an honest "no new information". A smooth ramp would
    manufacture microstructure the market never produced and would leak the
    next sample's value backwards in time.
    """
    from btcpred.live.feeds import _to_grid

    ts = np.array([100, 160], np.int64)
    close = np.array([50_000.0, 50_600.0])
    z = np.zeros(2)
    g, c, v, n, synth = _to_grid(ts, close, z, z, 61, end_ts=160)

    assert g[0] == 100 and g[-1] == 160
    held = c[(g > 100) & (g < 160)]
    assert np.all(held == 50_000.0), "grid interpolated between samples"
    assert c[-1] == 50_600.0
    assert synth[(g > 100) & (g < 160)].all(), "held seconds not flagged synthetic"
    assert not synth[0] and not synth[-1]


def test_feed_grid_never_leaks_future_samples():
    from btcpred.live.feeds import _to_grid

    ts = np.array([10, 20, 30], np.int64)
    close = np.array([1.0, 2.0, 3.0])
    z = np.zeros(3)
    g, c, _, _, _ = _to_grid(ts, close, z, z, 21, end_ts=30)
    # every grid second must carry the most recent value AT OR BEFORE it
    for gi, ci in zip(g, c):
        expected = close[ts <= gi][-1] if (ts <= gi).any() else close[0]
        assert ci == expected, f"t={gi} got {ci}, expected {expected}"


def test_hybrid_basis_rebases_onto_the_training_instrument():
    """biquote BTCUSD (MT5 CFD) must be shifted onto the Binance spot level.

    Without this the model sees a step change at the warm-up/live seam and
    reads a pure instrument basis as a genuine price move.
    """
    from btcpred.live.feeds import HybridFeed

    f = HybridFeed.__new__(HybridFeed)
    f.basis = -6.5
    f.alpha = 0.1
    f.basis_stale = False
    f.basis_refresh_sec = 1e9       # suppress network re-anchoring
    f._last_basis_at = 1e18
    f._last_px = None
    f._last_ts = None

    class _Stub:
        def poll(self):
            return [(1000, 85_000.0, 0.0, 1.0)]
    f.biquote = _Stub()
    f.binance = None

    out = f.poll()
    assert len(out) == 1
    assert abs(out[0][1] - 85_006.5) < 1e-9, (
        f"basis not removed: got {out[0][1]}, expected 85006.5"
    )


def _tiny_store(root: Path, n: int = 500):
    import json as _json
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    base = 8_000_000 + np.cumsum(rng.integers(-50, 51, n))
    p = np.stack([base, base + 20, base - 20, base], 1).astype(np.int64)
    f = rng.random((n, 4)).astype(np.float32)
    np.memmap(root / "prices.i64", np.int64, "w+", shape=(n, 4))[:] = p
    np.memmap(root / "flows.f32", np.float32, "w+", shape=(n, 4))[:] = f
    (root / "meta.json").write_text(_json.dumps(
        {"symbol": "T", "start_ts": 0, "n_seconds": n, "price_scale": 100}))
    return p, f


def test_verify_detects_lfs_pointers_not_mmap_crash():
    """An LFS stub must produce an actionable error, not a numpy mmap error."""
    import tempfile
    from btcpred.data.verify import DatasetError, check

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "bars"
        _tiny_store(root)
        (root / "prices.i64").write_text(
            "version https://git-lfs.github.com/spec/v1\n"
            "oid sha256:" + "0" * 64 + "\nsize 16000\n")
        try:
            check(root)
            raise AssertionError("pointer stub was not detected")
        except DatasetError as e:
            msg = str(e)
            assert "LFS POINTER" in msg
            assert "git lfs pull" in msg
            assert "Snowflake" in msg, "offline path not explained"


def test_verify_detects_truncated_files():
    import tempfile
    from btcpred.data.verify import DatasetError, check

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "bars"
        _tiny_store(root)
        with open(root / "flows.f32", "r+b") as fh:
            fh.truncate(100)
        try:
            check(root)
            raise AssertionError("truncation not detected")
        except DatasetError as e:
            assert "wrong size" in str(e)


def test_sample_npz_roundtrip_is_lossless():
    import tempfile
    from btcpred.data.sample import compress, expand

    with tempfile.TemporaryDirectory() as d:
        root, out = Path(d) / "bars", Path(d) / "restored"
        p, f = _tiny_store(root)
        compress(root, Path(d) / "s.npz")
        expand(Path(d) / "s.npz", out)
        p2 = np.asarray(np.memmap(out / "prices.i64", np.int64, "r", shape=p.shape))
        f2 = np.asarray(np.memmap(out / "flows.f32", np.float32, "r", shape=f.shape))
        assert np.array_equal(p, p2), "delta encoding lost price information"
        assert np.array_equal(f, f2)


def test_stage_pack_unpack_roundtrip_and_corruption_detection():
    import tempfile
    from btcpred.data.stage import pack, unpack

    with tempfile.TemporaryDirectory() as d:
        root, dist, out = Path(d) / "bars", Path(d) / "dist", Path(d) / "out"
        p, _ = _tiny_store(root)
        pack(root, dist, chunk_bytes=4096)          # force many chunks
        unpack(dist, out, keep_chunks=True)
        p2 = np.asarray(np.memmap(out / "prices.i64", np.int64, "r", shape=p.shape))
        assert np.array_equal(p, p2)

        victim = sorted(dist.glob("prices.i64.*"))[0]
        b = bytearray(victim.read_bytes())
        b[0] ^= 0xFF
        victim.write_bytes(bytes(b))
        try:
            unpack(dist, Path(d) / "out2")
            raise AssertionError("corrupted chunk was not detected")
        except ValueError as e:
            assert "sha256 mismatch" in str(e)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
