# Bitcoin 25-Minute Price Predictor

Second-by-second BTC forecasting: a shared **Market Analyser** feeds three
competing **Price Predictor** architectures that each forecast the next
25 minutes of price, update every second, and are scored in real market time
against recorded reality.

Built to train offline on **4 × A10 (Snowflake)** and run live on an
**RTX Pro 6000 (molab)**.

---

## Read this first

Two things about this problem are worth knowing before you spend GPU hours.

**1. A 25-minute BTC forecast is close to a martingale.** On the 1-second data
in this repo, the log-return over 25 minutes has σ ≈ 17 bp. "Predict no
change" is a genuinely strong baseline. A model can reach a very low training
loss by simply collapsing to zero — the loss curve will look beautiful and the
model will be worthless.

Everything in the evaluation path is built around that fact. The headline
metric is **skill** = `1 − MSE_model / MSE_random-walk`, and the selector
refuses to recommend a model with skill ≤ 0 without shouting about it. Treat a
positive, *test-split* skill as the only evidence that anything was learned.

**2. Costs dominate at this horizon.** A 17 bp move against a ~1 bp round-trip
cost means a mediocre signal is a losing strategy. `net_reward_bp` is computed
after cost — set `--cost-bps` to your real fees before believing it.

---

## Architecture

```
            12 h of 1-second bars
                     │
        ┌────────────┴────────────┐
        │   multi-resolution      │   lane 0:  30 min @ 1 s   → 1800 steps
        │   lane encoder          │   lane 1:   3 h   @ 6 s   → 1800 steps
        └────────────┬────────────┘   lane 2:  12 h   @ 24 s  → 1800 steps
                     │
         conv stem, stride 8  →  675 tokens
                     │
        ┌────────────▼────────────┐
        │   MARKET ANALYSER       │   85 M params, one shared instance
        │   12-layer transformer  │   → memory (675 × 768) + summary
        └────────────┬────────────┘
                     │  (memory + summary + the raw 12 h context)
      ┌──────────────┼──────────────┐
      ▼              ▼              ▼
  ┌────────┐    ┌────────┐    ┌────────┐
  │ xfmr   │    │ tcn    │    │ ssm    │   3 × ~155 M params
  │ 160 M  │    │ 155 M  │    │ 154 M  │   trained concurrently
  └────┬───┘    └────┬───┘    └────┬───┘
       └─────────────┼─────────────┘
                     ▼
        25 horizons × 3 quantiles
        (+60 s … +1500 s, q10/q50/q90)
```

**Why three lanes instead of 43 200 raw steps.** A literal 12 h × 1 s context
is 43 200 tokens — intractable for attention, and mostly redundant since
microstructure decorrelates in seconds. The lanes tile the same 12 h at
geometrically coarser resolution, so recent history keeps full fidelity and
distant history is *summarised* (open/high/low/close/sum-volume), not dropped.

**Why one analyser, three predictors.** All variants are stepped on the same
batch and the analyser receives the summed gradient from all of them, so it
learns a representation that serves every architecture. Giving each variant
its own analyser would let them drift apart and make the comparison
meaningless. The three trunks are capacity-matched (154–160 M) so a win
reflects the architecture, not the parameter count.

**Why bidirectional attention inside the window.** Every token in the context
is already in the past at decision time, so there is no leakage; causal
masking would only throw information away. `tests/test_no_lookahead_in_context`
enforces the actual boundary.

---

## Stability

The spec calls for predictions stable enough to drive real trades. Three
mechanisms, at different layers:

| Layer | Mechanism |
|---|---|
| Loss | **Consistency term.** The forecast made at second *t* for wall-clock time *T* must match the forecast made at *t+1* for that same *T*. Because the two horizon grids are offset, the later forecast is linearly interpolated onto the earlier one's target times before comparison. This constrains the *revision process*, not just accuracy. |
| Loss | **Smoothness term.** Second difference along the horizon axis, so the output is a plausible price path rather than 25 independent guesses. |
| Architecture | Quantiles are emitted as a median plus cumulative `softplus` offsets, making q10 > q90 **structurally impossible** rather than merely penalised. |
| Inference | EMA over the *absolute* price path (smoothing anchor-relative returns would lag the anchor and bias the forecast), plus a hard per-second revision clamp. |

Measured by `revision_bp`: mean |forecast(t,T) − forecast(t+1,T)| in basis
points. Reported independently of accuracy, and a gate during selection.

---

## Quick start

```bash
# Install git-lfs BEFORE cloning. Without it the dataset arrives as 132-byte
# pointer stubs and BarStore dies with:
#   ValueError: mmap length is greater than file size
# (already cloned? just run `git lfs install && git lfs pull` to recover)
git lfs install

git clone https://github.com/fgbvdcfgbfdefgb/pridictior_snow12.git
cd pridictior_snow12
git lfs pull                             # 1s bar store + raw archives

pip install -r requirements.txt
export PYTHONPATH=src

python -m btcpred.utils.hardware          # what have we got?
python tests/test_pipeline.py             # 14 correctness tests
```

### 1. Build the dataset (networked machine)

```bash
./scripts/fetch_data.sh                   # 2020-01 → today
```

Pulls `aggTrades` from `data.binance.vision` — free, keyless, and the only
bulk source with true sub-second resolution back to 2020 — then resamples to a
gap-free 1-second grid.

| | |
|---|---|
| Archives | ~69 monthly ZIPs, **~25 GB** |
| Download | 6–10 h, fully resumable (sha256-verified, HTTP range resume) |
| Bar store | ~9 GB (`int64` cents × 4 + `float32` × 4) |
| Disk needed | **~60 GB** |
| Resample | ~45 min CPU |

Prices are stored as **integer cents**, not float32: at a $100 k price level
float32 resolves only ~0.8 ¢, comparable to the tick size, which would corrupt
second-scale returns.

Empty seconds (~11 % of the grid — real, BTC does not trade every second) are
forward-filled on price, zero-filled on flow, and flagged to the model via a
dedicated `gap_flag` feature.

> **Note on `api.binance.com`:** it returns **HTTP 451** from India and several
> cloud regions. Everything here uses `data-api.binance.vision` (the public
> mirror) first and falls back to other hosts. No API key is needed anywhere
> in this repo.

### 2. Train (Snowflake, offline)

```bash
./scripts/train_a10x4.sh
```
or open `notebooks/snowflake_train.ipynb` and run it top to bottom.

Fully offline — no network call anywhere in the training path. If the
container has no PyPI either:

```bash
# on a networked box
pip download -r requirements.txt -d wheels/
# then commit wheels/ ; the notebook installs with --no-index
```

**No epochs.** The loop walks market time forward one second at a time and is
scored every second against the recorded future. The timeline is cut into
`world_size` contiguous shards, one per rank: contiguity keeps the consistency
loss well defined, while different shards decorrelate the DDP gradients.
Budget by wall clock (`max_hours`), not epoch count.

**Sizing is detected, not hard-coded.** `utils/hardware.py` inventories VRAM
and solves a budget of ~16 bytes/param (bf16 weights + fp32 master + Adam
moments + grads — optimiser state dominates, not weights), then sets `scale`.
On 23 GB A10s with all three variants resident, 85 M + 3×155 M ≈ 550 M is near
the practical ceiling.

### 3. Select

```bash
python -m btcpred.train.select_best --ckpt checkpoints --split val \
    --step 30 --max-points 4000
```

```
variant     skill  rmse_bp    rw_bp  revis_bp    hit   rew_bp  sharpe  cov80
xfmr      +0.0183     6.13     6.19     0.020  0.667   +0.021 +107.18  0.427
```

Selection is gated, then ranked:

1. **skill > 0** — non-negotiable. Reward and Sharpe are so noisy that without
   this gate the ranking will cheerfully pick a model far worse than a random
   walk because its P&L noise happened to land positive.
2. **revision_bp ≤ threshold** — an accurate but jittery model is worse than
   useless when it drives trades.
3. Rank by `net_reward_bp` — but only with ≥ 500 evaluation points; below that
   it ranks by skill and tells you why.

Run `--split test` **once**, after selection is locked.

### 4. Run live (molab)

```bash
BTCPRED_CKPT=checkpoints marimo edit notebooks/molab_live.py
```

CoinMarketCap-styled dashboard: dark `#0D1421` canvas, `#16C784`/`#EA3943`
price line with area fill, dotted `#3861FB` forecast with an 80 % band, live
price header, and a realised-error panel that scores matured forecasts against
a random walk once 25 minutes have actually elapsed.

#### Live data sources

Selectable in the notebook (`src/btcpred/live/feeds.py`):

| Mode | Warm-up (12 h) | Live ticks | Use when |
|---|---|---|---|
| **`hybrid`** *(default)* | Binance mirror, true 1 s | **biquote.io** | Normal operation |
| `biquote` | biquote OHLC, **~97 % synthetic** | biquote.io | Binance unreachable |
| `binance` | Binance mirror, true 1 s | Binance mirror | Binance reachable |
| `sim` | — | recorded replay | Offline / demo |

**What biquote.io actually provides** (measured 2026-10-04, not assumed):

```
GET /api/BTCUSD           latest tick: bid, ask, mid, spread   ✓ excellent
GET /api/BTCUSD/history   capped at 100 ticks (~1.8 min) — `limit` is ignored
GET /api/BTCUSD/ohlc      intervals 1m 5m 15m 30m 1h 4h 1d — NO 1s
                          rolling windows, NO pagination:
                            1m → 301 bars (5 h)
                            5m → 289 bars (24 h)
                           15m → 193 bars (48 h)
```

It is a genuinely good live source — free, no key, sub-100 ms, and **not
geo-blocked**, which `api.binance.com` is from India. But four properties
decide the architecture:

1. **No 1-second history, and none deeper than 5 h at 1 m.** The model needs
   43 200 one-second bars for its 12 h context; biquote can supply at most 720
   one-minute bars over that span. Measured: a biquote-only warm-up is
   **96.8 % synthetic**. The notebook reports this figure in red rather than
   hiding it.
2. **No volume.** `volume` is 0 on every tick — it is an MT5 CFD feed, not an
   exchange tape. Only `tickVolume` exists, in OHLC bars. The volume and
   taker-buy-imbalance channels get zero-filled, which is off-distribution.
3. **~0.5 new quotes/second**, so roughly every other second repeats.
4. **Different instrument.** biquote `BTCUSD` is an MT5 broker CFD; the model
   trains on Binance `BTCUSDT` spot. Measured basis **−6.5 USD (−0.8 bp),
   sd 3.2**, and it drifts. `HybridFeed` tracks it with an EMA (re-anchoring
   against Binance every 60 s) and subtracts it, so the live series stays on
   the price level the model was trained on. Splicing raw would hand the model
   a step change it reads as a real move.

Synthetic warm-up seconds are **held, never interpolated** — a held price is an
honest "no new information", whereas a smooth ramp manufactures microstructure
and leaks the next sample backwards in time. Two tests enforce this.

**On the "one chart that updates, not a new chart each tick" requirement** —
this notebook cannot stack charts, structurally:

- In marimo a cell has exactly one output and re-running **replaces** it. The
  plot lives in its own cell whose only job is to render current state, so
  there is nowhere for a second chart to accumulate. (The usual failure is a
  Jupyter loop calling `plt.plot()` / `fig.show()` per tick, which appends.)
- The figure sets a constant **`uirevision`**, so Plotly patches the existing
  canvas and preserves zoom, pan and hover across updates instead of
  rebuilding — it reads as one continuously-moving chart.

Works with no checkpoint, no internet, or neither — each cell degrades to an
explanatory callout rather than a traceback. Switch *Data source* to
**Replay recorded history** to drive the same dashboard from the simulator.

---

## Layout

```
src/btcpred/
  data/download_binance.py   resumable, checksum-verified archive fetcher
  data/resample.py           aggTrades → gap-free 1 s bar store (memmap)
  data/features.py           3-lane multi-resolution encoder, scale-free
  data/dataset.py            streaming time-ordered sampler, rank sharding
  sim/simulator.py           replay-as-live; enforces the information boundary
  models/blocks.py           RMSNorm, SwiGLU, RoPE attention, TCN, state-space
  models/analyser.py         Market Analyser (85 M)
  models/predictor.py        Price Predictor ×3 trunks + quantile head
  train/reward.py            pinball + consistency + smoothness + P&L reward
  train/train_ddp.py         DDP trainer, no epochs
  train/select_best.py       walk-forward eval, gated selection
  live/feeds.py              biquote.io / Binance / hybrid live feeds
  infer/runner.py            live runner with revision smoothing
  utils/hardware.py          compute discovery + automatic model sizing
notebooks/molab_live.py        marimo live dashboard (CMC style)
notebooks/snowflake_train.ipynb offline training notebook
configs/a10x4.json             4 × A10 production config
configs/smoke_cpu.json         tiny CPU config for correctness runs
tests/test_pipeline.py         14 tests
```

## The simulator's information boundary

`LiveSimulator.observe()` returns only data up to and including the current
second. `ground_truth()` and `future_path()` read ahead and are reserved for
the scorer. Nothing on the model path can reach a future bar — and
`test_no_lookahead_in_context` verifies that rewriting the future leaves the
features at time *t* bit-identical.

Scoring always uses the **recorded** market data, never simulator output.

## Verified

Smoke-trained on 3 days of real BTCUSDT (2026-10-01 → 10-03, 259 198 seconds,
11.4 % empty) on CPU:

- all three variants train; pinball 43.9 → 0.2, consistency 9.78 → 0.01
- walk-forward evaluation, gated selection, and `best_predictor.pt` export run
- live path verified end-to-end against real data: 6 144 s warm-up → forecast
  with an 80 % band, mean revision **0.035 bp/s**
- all three live feeds exercised against the real APIs: `binance` 0 % synthetic,
  `biquote` 96.8 % synthetic (correctly flagged), `hybrid` 0 % synthetic with
  live basis tracking and no duplicate bars at the warm-up seam
- `notebooks/molab_live.py` executes headless with zero cell errors
- 14/14 tests pass

Not verified here (no GPU in the build environment): multi-GPU DDP throughput
and bf16 numerics. The DDP path is standard `torchrun`, but budget time for a
short 4-GPU shakeout before committing to a long run.

## Known limitations

- **Spot only, one venue.** No order-book depth, no perps/funding, no
  cross-exchange flow — all of which carry most of the short-horizon signal.
  The ~17 bp 25-minute σ is mostly noise and this feature set may simply not
  be enough to beat the random walk.
- **`context_from_closes`** (used in live mode) zero-fills the volume and
  trade-count channels, which is off the training distribution and costs
  accuracy. With biquote this is unavoidable — the MT5 feed has no volume.
- **biquote-only mode is a fallback, not a peer of hybrid.** A 97 % synthetic
  context is far outside anything the model saw in training; treat its
  forecasts as indicative only.
- **`cost_bps=1.0` is a placeholder.** Substitute your real fees and slippage.
- **No regime handling.** 2020–2026 spans wildly different volatility regimes;
  a single walk-forward split does not test regime robustness.
