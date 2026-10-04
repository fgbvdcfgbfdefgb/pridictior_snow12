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
    source = mo.ui.dropdown(
        options={
            "Hybrid — Binance 1s warm-up + biquote.io live (recommended)": "hybrid",
            "biquote.io only — live quotes, synthetic warm-up": "biquote",
            "Binance only — true 1s throughout": "binance",
            "Replay recorded history (offline)": "sim",
        },
        value="Hybrid — Binance 1s warm-up + biquote.io live (recommended)",
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
def _(mo, runner, source):
    from btcpred.live.feeds import make_feed

    feed = None
    hist0 = None
    warm_status = mo.md("Model not loaded — nothing to warm up.").callout("warn")

    if runner is not None and source.value != "sim":
        try:
            feed = make_feed(source.value)
            hist0 = feed.warmup(runner.warmup)
            _notes = "\n".join(f"- {n}" for n in hist0.notes)
            _synth = hist0.synthetic_fraction
            _kind = "danger" if _synth > 0.5 else ("warn" if _synth > 0.05 else "success")
            warm_status = mo.md(
                f"**Warm-up complete — `{hist0.source}`**\n\n"
                f"{len(hist0.ts):,} seconds ({len(hist0.ts)/3600:.1f} h), "
                f"last **${hist0.close[-1]:,.2f}**, "
                f"**{_synth*100:.1f}% synthetic**\n\n{_notes}"
            ).callout(_kind)
        except Exception as exc:  # noqa: BLE001
            feed = None
            warm_status = mo.md(
                f"**Feed `{source.value}` failed**\n\n```\n{exc}\n```\n\n"
                "Try another source, or switch to *Replay recorded history*."
            ).callout("danger")

    warm_status
    return feed, hist0, make_feed, warm_status


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
    context_from_closes,
    feed,
    get_track,
    hist0,
    np,
    refresher,
    runner,
    running,
    set_fc,
    set_hist,
    set_track,
    sim,
    source,
):
    from collections import deque

    refresher  # dependency: re-run on every tick

    tick_error = None
    if runner is not None and running.value:
        try:
            if source.value != "sim" and feed is not None:
                if not hasattr(feed, "_buf"):
                    feed._buf = {
                        "ts": deque(hist0.ts.tolist(), maxlen=runner.warmup + 7200),
                        "close": deque(hist0.close.tolist(), maxlen=runner.warmup + 7200),
                        "vol": deque(hist0.volume.tolist(), maxlen=runner.warmup + 7200),
                        "trades": deque(hist0.trades.tolist(), maxlen=runner.warmup + 7200),
                    }
                buf = feed._buf
                for sec, pxv, volv, trdv in feed.poll():
                    # Fill any seconds the feed skipped, so the 1 s grid the
                    # model expects stays contiguous.
                    while buf["ts"][-1] + 1 < sec:
                        buf["ts"].append(buf["ts"][-1] + 1)
                        buf["close"].append(buf["close"][-1])
                        buf["vol"].append(0.0)
                        buf["trades"].append(0.0)
                    if sec > buf["ts"][-1]:
                        buf["ts"].append(sec)
                        buf["close"].append(pxv)
                        buf["vol"].append(volv)
                        buf["trades"].append(trdv)

                closes = np.fromiter(buf["close"], np.float64)
                ctx, anchor = context_from_closes(
                    closes, np.fromiter(buf["vol"], np.float64),
                    np.fromiter(buf["trades"], np.float64), runner.lanes)
                now_ts = int(buf["ts"][-1])
                hist_ts = np.fromiter(buf["ts"], np.int64)
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

            tr = get_track()[-900:]
            tr.append((now_ts, float(fc.median[-1]), float(anchor)))
            set_track(tr)

        except Exception as exc:  # noqa: BLE001
            tick_error = f"{type(exc).__name__}: {exc}"

    tick_error
    return (anchor, buf, closes, ctx, deque, fc, hist_px, hist_ts, now_ts,
            pxv, sec, tick_error, tr, trdv, volv)


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
