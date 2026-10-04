import marimo

__generated_with = "0.9.0"
app = marimo.App(width="full", app_title="BTC 25-Minute Predictor")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Bitcoin 25-Minute Price Predictor — Live

        Real-time inference on the live Binance feed, CoinMarketCap-styled.

        **The single-updating-chart requirement.** The usual failure mode is a
        notebook that *appends* a new chart on every tick. This notebook cannot
        do that, for two structural reasons:

        1. In marimo a cell has exactly one output, and re-running **replaces**
           it. The plot lives in its own cell whose only job is to render
           current state, so there is physically nowhere for a second chart to
           accumulate.
        2. The figure sets `uirevision` to a constant. Plotly then reuses the
           existing WebGL canvas and keeps your zoom, pan and hover across
           updates, so it reads as one continuously-moving chart rather than a
           redraw.
        """
    )
    return


@app.cell
def _(mo):
    import os
    from pathlib import Path

    ckpt_dir = mo.ui.text(
        value=os.environ.get("BTCPRED_CKPT", "checkpoints"),
        label="Checkpoint directory", full_width=True,
    )
    source = mo.ui.radio(
        options={
            "Live Binance feed": "live",
            "Replay recorded history (offline)": "sim",
        },
        value="Live Binance feed",
        label="Data source",
    )
    device = mo.ui.dropdown(
        options=["cuda", "cpu"], value="cuda",
        label="Device (RTX Pro 6000 → cuda)",
    )
    alpha = mo.ui.slider(
        0.05, 1.0, value=0.35, step=0.05,
        label="Forecast smoothing α (1.0 = raw model output)",
    )
    window_min = mo.ui.slider(
        15, 240, value=90, step=15, label="Chart history window (minutes)",
    )
    mo.vstack([ckpt_dir, mo.hstack([source, device], justify="start"),
               alpha, window_min])
    return Path, alpha, ckpt_dir, device, os, source, window_min


@app.cell
def _(Path, alpha, ckpt_dir, device, mo):
    import sys

    _repo = Path(__file__).resolve().parents[1]
    if str(_repo / "src") not in sys.path:
        sys.path.insert(0, str(_repo / "src"))

    import numpy as np
    import torch

    from btcpred.infer.runner import LiveRunner, context_from_closes

    _dev = device.value
    if _dev == "cuda" and not torch.cuda.is_available():
        _dev = "cpu"

    try:
        runner = LiveRunner(ckpt_dir.value, device=_dev, alpha=alpha.value)
        gpu = torch.cuda.get_device_name(0) if _dev == "cuda" else "CPU"
        load_status = mo.md(
            f"**Model loaded** — variant `{runner.variant}`, "
            f"{sum(p.numel() for p in runner.predictor.parameters())/1e6:.0f}M params, "
            f"on `{gpu}`. Needs **{runner.warmup/3600:.1f} h** of warm-up history."
        ).callout(kind="success")
    except Exception as exc:  # noqa: BLE001
        runner = None
        load_status = mo.md(
            f"**Could not load the model** from `{ckpt_dir.value}`.\n\n```\n{exc}\n```\n\n"
            "Train first, or point the checkpoint directory at a trained run."
        ).callout(kind="danger")

    load_status
    return (
        LiveRunner,
        context_from_closes,
        gpu,
        load_status,
        np,
        runner,
        sys,
        torch,
    )


@app.cell
def _(mo, np, runner, source):
    import time
    from collections import deque

    import requests

    # api.binance.com returns HTTP 451 from several regions (India included,
    # and some cloud ranges). data-api.binance.vision is the public, keyless,
    # un-geo-blocked market-data mirror and serves identical klines, so it is
    # tried FIRST and the main host is only a fallback.
    BINANCE_HOSTS = [
        "https://data-api.binance.vision/api/v3",
        "https://api-gcp.binance.com/api/v3",
        "https://api.binance.com/api/v3",
    ]
    SYMBOL = "BTCUSDT"

    def binance_get(path: str, params: dict, timeout: int = 20):
        """GET from the first Binance host that is reachable from here."""
        errs = []
        for host in BINANCE_HOSTS:
            try:
                r = requests.get(f"{host}{path}", params=params, timeout=timeout)
                if r.status_code == 451:
                    errs.append(f"{host}: 451 geo-restricted")
                    continue
                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:
                errs.append(f"{host}: {e}")
        raise RuntimeError("all Binance hosts failed -> " + "; ".join(errs))

    def fetch_warmup_1s(seconds: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pull `seconds` of 1-second klines, paginating backwards (1000/req)."""
        end = int(time.time() * 1000)
        ts, close, vol, trades = [], [], [], []
        remaining = seconds
        while remaining > 0:
            n = min(1000, remaining)
            rows = binance_get("/klines", {"symbol": SYMBOL, "interval": "1s",
                                           "endTime": end, "limit": n})
            if not rows:
                break
            ts = [int(x[0]) // 1000 for x in rows] + ts
            close = [float(x[4]) for x in rows] + close
            vol = [float(x[5]) for x in rows] + vol
            trades = [float(x[8]) for x in rows] + trades
            end = int(rows[0][0]) - 1
            remaining -= len(rows)
        return (np.array(ts, np.int64), np.array(close, np.float64),
                np.array(vol, np.float64), np.array(trades, np.float64))

    def regrid(ts, close, vol, trades, need):
        """Binance omits empty seconds; rebuild a contiguous 1 s grid."""
        if len(ts) == 0:
            raise RuntimeError("no data returned from Binance")
        full = np.arange(ts[-1] - need + 1, ts[-1] + 1, dtype=np.int64)
        idx = np.searchsorted(ts, full).clip(0, len(ts) - 1)
        c = close[idx]
        exact = ts[idx] == full
        v = np.where(exact, vol[idx], 0.0)
        n = np.where(exact, trades[idx], 0.0)
        return full, c, v, n

    warm_status = mo.md("Model not loaded — nothing to warm up.").callout("warn")
    feed = None

    if runner is not None and source.value == "live":
        try:
            _ts, _c, _v, _n = fetch_warmup_1s(runner.warmup + 120)
            _ts, _c, _v, _n = regrid(_ts, _c, _v, _n, runner.warmup)
            feed = {"ts": deque(_ts.tolist(), maxlen=runner.warmup + 7200),
                    "close": deque(_c.tolist(), maxlen=runner.warmup + 7200),
                    "vol": deque(_v.tolist(), maxlen=runner.warmup + 7200),
                    "trades": deque(_n.tolist(), maxlen=runner.warmup + 7200)}
            warm_status = mo.md(
                f"**Warm-up complete** — {len(_c):,} seconds "
                f"({len(_c)/3600:.1f} h) of BTCUSDT loaded, "
                f"last price **${_c[-1]:,.2f}**."
            ).callout("success")
        except Exception as exc:  # noqa: BLE001
            warm_status = mo.md(
                f"**Live warm-up failed** (no internet, or Binance is "
                f"geo-blocked here).\n\n```\n{exc}\n```\n\n"
                "Switch the data source to *Replay recorded history*."
            ).callout("danger")

    warm_status
    return (
        BINANCE_HOSTS,
        binance_get,
        SYMBOL,
        deque,
        feed,
        fetch_warmup_1s,
        regrid,
        requests,
        time,
        warm_status,
    )


@app.cell
def _(Path, mo, np, runner, source):
    sim = None
    sim_status = mo.md("").callout("neutral") if source.value == "live" else None

    if runner is not None and source.value == "sim":
        try:
            from btcpred.data.features import BarStore
            from btcpred.sim.simulator import LiveSimulator

            _store = BarStore(Path(__file__).resolve().parents[1] / "data" / "bars_1s")
            sim = LiveSimulator(_store, lanes=runner.lanes,
                                horizons=runner.horizons, speed=0.0)
            sim.reset(runner.warmup + 10)
            sim_status = mo.md(
                f"**Replay ready** — {len(_store):,} recorded seconds."
            ).callout("success")
        except Exception as exc:  # noqa: BLE001
            sim_status = mo.md(
                f"**Replay unavailable**\n\n```\n{exc}\n```"
            ).callout("danger")

    sim_status
    return BarStore, LiveSimulator, sim, sim_status


@app.cell
def _(mo):
    # The heartbeat. Everything downstream re-runs on each tick, replacing its
    # output in place -- this is what makes the chart update rather than stack.
    refresher = mo.ui.refresh(
        options=["1s", "2s", "5s", "10s"], default_interval="2s",
        label="Live update",
    )
    running = mo.ui.switch(value=True, label="Streaming")
    mo.hstack([refresher, running], justify="start")
    return refresher, running


@app.cell
def _(mo):
    get_hist, set_hist = mo.state([])      # published (ts, price) for the chart
    get_fc, set_fc = mo.state(None)        # latest Forecast
    get_track, set_track = mo.state([])    # past forecasts, for the error panel
    return get_fc, get_hist, get_track, set_fc, set_hist, set_track


@app.cell
def _(
    binance_get,
    context_from_closes,
    feed,
    get_hist,
    get_track,
    np,
    refresher,
    requests,
    runner,
    set_fc,
    set_hist,
    set_track,
    sim,
    source,
    time,
    running,
):
    refresher  # dependency: re-run on every tick

    tick_error = None
    if runner is not None and running.value:
        try:
            if source.value == "live" and feed is not None:
                klines = binance_get("/klines", {"symbol": "BTCUSDT",
                                                 "interval": "1s", "limit": 60},
                                     timeout=10)
                last_ts = feed["ts"][-1]
                for row in klines:
                    t = int(row[0]) // 1000
                    if t <= last_ts:
                        continue
                    # Fill any seconds Binance skipped (no trades printed).
                    while feed["ts"][-1] + 1 < t:
                        feed["ts"].append(feed["ts"][-1] + 1)
                        feed["close"].append(feed["close"][-1])
                        feed["vol"].append(0.0)
                        feed["trades"].append(0.0)
                    feed["ts"].append(t)
                    feed["close"].append(float(row[4]))
                    feed["vol"].append(float(row[5]))
                    feed["trades"].append(float(row[8]))

                closes = np.fromiter(feed["close"], np.float64)
                vols = np.fromiter(feed["vol"], np.float64)
                trs = np.fromiter(feed["trades"], np.float64)
                ctx, anchor = context_from_closes(closes, vols, trs, runner.lanes)
                now_ts = int(feed["ts"][-1])
                hist_ts = np.fromiter(feed["ts"], np.int64)
                hist_px = closes

            elif sim is not None:
                tick = sim.observe()
                sim.t += 1
                ctx, anchor, now_ts = tick.context, tick.price, tick.ts
                hist_ts, hist_px = sim.history(8 * 3600)
            else:
                raise RuntimeError("no data source is ready")

            fc = runner.predict(ctx, anchor, now_ts)
            set_fc(fc)
            set_hist(list(zip(hist_ts[-28800:].tolist(), hist_px[-28800:].tolist())))

            # Retain forecasts so realised error can be scored once the
            # 25-minute horizon actually elapses.
            tr = get_track()[-900:]
            tr.append((now_ts, float(fc.median[-1]), float(anchor)))
            set_track(tr)

        except Exception as exc:  # noqa: BLE001
            tick_error = f"{type(exc).__name__}: {exc}"

    tick_error
    return anchor, ctx, fc, hist_px, hist_ts, klines, now_ts, tick_error, tr


@app.cell
def _():
    # CoinMarketCap palette.
    CMC = dict(
        bg="#0D1421",        # page / plot background
        panel="#171924",
        grid="#1F2735",
        text="#EAECEF",
        muted="#808A9D",
        up="#16C784",        # CMC green
        down="#EA3943",      # CMC red
        accent="#3861FB",    # CMC blue
        band="rgba(56,97,251,0.16)",
    )
    return (CMC,)


@app.cell
def _(CMC, get_fc, get_hist, mo, np, window_min):
    import plotly.graph_objects as go

    _hist = get_hist()
    _fc = get_fc()

    if not _hist:
        chart = mo.md("Waiting for the first tick…").callout("neutral")
    else:
        h = np.array(_hist, dtype=np.float64)
        keep = int(window_min.value) * 60
        h = h[-keep:]
        t_ms = (h[:, 0] * 1000).astype("datetime64[ms]")
        px = h[:, 1]

        first, last = px[0], px[-1]
        up = last >= first
        line_col = CMC["up"] if up else CMC["down"]
        fill_col = ("rgba(22,199,132,0.14)" if up else "rgba(234,57,67,0.14)")

        fig = go.Figure()

        # --- realised price: CMC's signature line + soft area fill ----------
        fig.add_trace(go.Scatter(
            x=t_ms, y=px, mode="lines", name="BTC/USDT",
            line=dict(color=line_col, width=2, shape="linear"),
            fill="tozeroy", fillcolor=fill_col,
            hovertemplate="<b>$%{y:,.2f}</b><br>%{x|%H:%M:%S}<extra></extra>",
        ))

        if _fc is not None:
            f_ms = ((_fc.ts + _fc.horizons) * 1000).astype("datetime64[ms]")
            # stitch the forecast onto the last real point so there is no gap
            fx = np.concatenate([[t_ms[-1]], f_ms])
            fmed = np.concatenate([[px[-1]], _fc.median])
            flo = np.concatenate([[px[-1]], _fc.lower])
            fhi = np.concatenate([[px[-1]], _fc.upper])

            fig.add_trace(go.Scatter(
                x=np.concatenate([fx, fx[::-1]]),
                y=np.concatenate([fhi, flo[::-1]]),
                fill="toself", fillcolor=CMC["band"],
                line=dict(width=0), hoverinfo="skip",
                name="80% interval",
            ))
            fig.add_trace(go.Scatter(
                x=fx, y=fmed, mode="lines", name="25-min forecast",
                line=dict(color=CMC["accent"], width=2, dash="dot"),
                hovertemplate="forecast <b>$%{y:,.2f}</b><br>%{x|%H:%M:%S}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=[f_ms[-1]], y=[_fc.median[-1]], mode="markers+text",
                marker=dict(color=CMC["accent"], size=9,
                            line=dict(color=CMC["bg"], width=2)),
                text=[f"  ${_fc.median[-1]:,.0f}"], textposition="middle right",
                textfont=dict(color=CMC["accent"], size=12),
                hoverinfo="skip", showlegend=False,
            ))
            # "now" divider between observed and predicted
            fig.add_vline(x=t_ms[-1], line=dict(color=CMC["muted"],
                                                width=1, dash="dot"))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=CMC["bg"], plot_bgcolor=CMC["bg"],
            font=dict(family="Inter, -apple-system, Segoe UI, sans-serif",
                      color=CMC["text"], size=12),
            margin=dict(l=8, r=64, t=8, b=8),
            height=520,
            hovermode="x unified",
            hoverlabel=dict(bgcolor=CMC["panel"], bordercolor=CMC["grid"],
                            font=dict(color=CMC["text"])),
            legend=dict(orientation="h", yanchor="bottom", y=1.01,
                        x=0, bgcolor="rgba(0,0,0,0)"),
            showlegend=True,
            # Constant uirevision => plotly patches the EXISTING figure and
            # preserves zoom/pan/hover instead of rebuilding a new chart.
            uirevision="btc-live-chart",
            xaxis=dict(showgrid=False, color=CMC["muted"],
                       showline=False, zeroline=False,
                       rangeslider=dict(visible=False)),
            yaxis=dict(side="right", gridcolor=CMC["grid"], griddash="dot",
                       color=CMC["muted"], zeroline=False,
                       tickprefix="$", tickformat=",.0f",
                       range=[px.min() - (px.max() - px.min()) * 0.45,
                              px.max() + (px.max() - px.min()) * 0.25]),
        )
        chart = mo.ui.plotly(fig)

    chart
    return chart, fig, fill_col, first, fx, go, h, keep, last, line_col, px, t_ms, up


@app.cell
def _(CMC, get_fc, get_hist, mo, np):
    _h, _f = get_hist(), get_fc()
    if not _h or _f is None:
        header = mo.md("")
    else:
        _px = np.array([p for _, p in _h], np.float64)
        _spot = _px[-1]
        _ref = _px[max(0, len(_px) - 3600)]
        _chg = (_spot / _ref - 1) * 100
        _pred = float(_f.median[-1])
        _pchg = (_pred / _spot - 1) * 100
        _c = CMC["up"] if _chg >= 0 else CMC["down"]
        _pc = CMC["up"] if _pchg >= 0 else CMC["down"]
        _w = _f.upper[-1] - _f.lower[-1]

        header = mo.Html(f"""
        <div style="display:flex;gap:40px;align-items:flex-end;flex-wrap:wrap;
                    background:{CMC['bg']};padding:18px 22px;border-radius:10px;
                    border:1px solid {CMC['grid']};
                    font-family:Inter,-apple-system,Segoe UI,sans-serif;">
          <div>
            <div style="color:{CMC['muted']};font-size:12px;">Bitcoin · BTC/USDT</div>
            <div style="color:{CMC['text']};font-size:34px;font-weight:700;
                        line-height:1.2;">${_spot:,.2f}</div>
            <div style="color:{_c};font-size:13px;font-weight:600;">
              {_chg:+.2f}% <span style="color:{CMC['muted']};font-weight:400;">1h</span>
            </div>
          </div>
          <div>
            <div style="color:{CMC['muted']};font-size:12px;">Forecast · +25 min</div>
            <div style="color:{CMC['accent']};font-size:28px;font-weight:700;
                        line-height:1.3;">${_pred:,.2f}</div>
            <div style="color:{_pc};font-size:13px;font-weight:600;">{_pchg:+.3f}%</div>
          </div>
          <div>
            <div style="color:{CMC['muted']};font-size:12px;">80% interval</div>
            <div style="color:{CMC['text']};font-size:15px;">
              ${_f.lower[-1]:,.0f} — ${_f.upper[-1]:,.0f}</div>
            <div style="color:{CMC['muted']};font-size:12px;">width ${_w:,.0f}</div>
          </div>
          <div>
            <div style="color:{CMC['muted']};font-size:12px;">Revision / s</div>
            <div style="color:{CMC['text']};font-size:15px;">{_f.revision_bp:.3f} bp</div>
            <div style="color:{CMC['muted']};font-size:12px;">lower = steadier</div>
          </div>
        </div>
        """)

    header
    return (header,)


@app.cell
def _(get_hist, get_track, mo, np):
    _tr, _h = get_track(), get_hist()
    if len(_tr) < 30 or not _h:
        realised = mo.md(
            "_Realised-error panel appears once forecasts are old enough "
            "(25 min) to be scored against what actually happened._"
        )
    else:
        _px = {int(t): float(p) for t, p in _h}
        rows = []
        for ts, pred, anch in _tr:
            tgt = ts + 1500
            if tgt in _px:
                rows.append((_px[tgt], pred, anch))
        if len(rows) < 10:
            realised = mo.md(
                f"_Scoring {len(rows)} matured forecasts — need ≥ 10._")
        else:
            a = np.array(rows)
            actual, pred, anch = a[:, 0], a[:, 1], a[:, 2]
            mae = np.abs(pred - actual).mean()
            mae_rw = np.abs(anch - actual).mean()
            hit = (np.sign(pred - anch) == np.sign(actual - anch)).mean()
            realised = mo.md(f"""
            **Realised accuracy** over {len(rows)} matured forecasts

            | metric | model | random walk |
            |---|---|---|
            | MAE (USD) | **{mae:,.2f}** | {mae_rw:,.2f} |
            | directional hit rate | **{hit:.1%}** | 50.0% |

            Skill vs. random walk: **{(1 - mae/max(mae_rw,1e-9)):+.2%}**
            (positive means the model is adding information; near zero or
            negative means it is not, which is the honest outcome for most
            25-minute crypto forecasts).
            """)
    realised
    return a, actual, anch, hit, mae, mae_rw, pred, realised, rows


if __name__ == "__main__":
    app.run()
