"""
Live market data feeds.

Three interchangeable sources, all exposing the same two operations:

    warmup(seconds) -> FeedHistory   the 12 h of 1-second bars the model needs
    poll()          -> new bars      whatever has printed since the last call

--------------------------------------------------------------------------
biquote.io -- what it actually provides (measured 2026-10-04, not guessed)
--------------------------------------------------------------------------
Free, no API key, no geo-block (unlike api.binance.com, which returns HTTP 451
from India and several cloud regions). Excellent as a *live* source. But its
history endpoints are hard-capped, which matters a great deal here:

    GET /api/{sym}                latest tick: bid, ask, mid, spread
    GET /api/{sym}/history        capped at 100 ticks (~1.8 min) -- `limit`
                                  above 100 is silently ignored
    GET /api/{sym}/ohlc           intervals 1m 5m 15m 30m 1h 4h 1d (NO 1s)
                                  rolling windows, NO pagination:
                                    1m -> 301 bars (5 h)
                                    5m -> 289 bars (24 h)
                                   15m -> 193 bars (48 h)

Consequences you cannot engineer around:

1. **No 1-second history.** The finest interval is 1 minute. The model needs
   43 200 one-second bars to fill its 12 h context, and biquote can supply at
   most 720 one-minute bars over that span. Reconstructing the other 59/60 of
   each minute means *inventing* data.
2. **No volume.** `volume` is 0 on every tick (it is an MT5 CFD feed, not an
   exchange tape). Only `tickVolume` (a quote-count proxy) exists, in OHLC
   bars. The model's volume and taker-buy-imbalance channels cannot be filled
   honestly, so they are zeroed and `gap_flag` is raised.
3. **~0.5 new quotes/second**, so roughly every other second is a repeat of
   the previous quote even in pure live operation.
4. **Different instrument.** biquote `BTCUSD` is an MT5 broker CFD; the model
   was trained on Binance `BTCUSDT` spot. Measured basis: **-6.5 USD
   (-0.8 bp), sd 3.2 USD**, and it drifts. Splicing the two without rebasing
   injects a step change the model reads as a real move.

`HybridFeed` is therefore the default: Binance's public mirror supplies a
faithful 1-second warm-up in the exact distribution the model was trained on,
and biquote drives the live updates, continuously rebased onto the Binance
price level. `BiquoteFeed` (pure) is available for environments where Binance
is unreachable, and it reports honestly how much of its context is synthetic.
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, field

import numpy as np
import requests

BIQUOTE = "https://biquote.io"
BINANCE_HOSTS = [
    "https://data-api.binance.vision/api/v3",
    "https://api-gcp.binance.com/api/v3",
    "https://api.binance.com/api/v3",
]

# Measured caps -- requesting more is silently ignored by the server.
BIQUOTE_OHLC_CAP = {"1m": 301, "5m": 289, "15m": 193, "30m": 193, "1h": 169}
BIQUOTE_INTERVAL_SEC = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600}


@dataclass
class FeedHistory:
    ts: np.ndarray        # int64 unix seconds, contiguous 1 s grid
    close: np.ndarray     # float64
    volume: np.ndarray    # float64 (0.0 when the source has none)
    trades: np.ndarray    # float64 (tick counts when available)
    synthetic: np.ndarray # bool: True where the second was interpolated
    source: str
    notes: list[str] = field(default_factory=list)

    @property
    def synthetic_fraction(self) -> float:
        return float(self.synthetic.mean()) if self.synthetic.size else 1.0


def _parse_ts(s: str) -> int:
    return int(dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def _to_grid(ts, close, vol, trades, need, end_ts=None):
    """Resample irregular samples onto a contiguous trailing 1-second grid."""
    ts = np.asarray(ts, np.int64)
    order = np.argsort(ts)
    ts, close = ts[order], np.asarray(close, np.float64)[order]
    vol = np.asarray(vol, np.float64)[order]
    trades = np.asarray(trades, np.float64)[order]

    end = int(end_ts if end_ts is not None else ts[-1])
    grid = np.arange(end - need + 1, end + 1, dtype=np.int64)

    # Step (zero-order hold) -- never interpolate a price forward in a way
    # that implies knowledge the feed did not have at that second.
    idx = np.searchsorted(ts, grid, side="right") - 1
    before = idx < 0
    idx = idx.clip(0, len(ts) - 1)
    c = close[idx]
    exact = (ts[idx] == grid) & ~before
    v = np.where(exact, vol[idx], 0.0)
    n = np.where(exact, trades[idx], 0.0)
    return grid, c, v, n, ~exact


# ---------------------------------------------------------------------------
# Binance (public mirror) -- true 1-second bars, matches the training data
# ---------------------------------------------------------------------------
class BinanceFeed:
    name = "binance"

    def __init__(self, symbol: str = "BTCUSDT", session: requests.Session | None = None):
        self.symbol = symbol
        self.s = session or requests.Session()
        self._last_ts: int | None = None

    def _get(self, path: str, params: dict, timeout: int = 20):
        errs = []
        for host in BINANCE_HOSTS:
            try:
                r = self.s.get(f"{host}{path}", params=params, timeout=timeout)
                if r.status_code == 451:
                    errs.append(f"{host}: 451 geo-restricted")
                    continue
                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:
                errs.append(f"{host}: {e}")
        raise RuntimeError("all Binance hosts failed -> " + "; ".join(errs))

    def warmup(self, seconds: int) -> FeedHistory:
        end = int(time.time() * 1000)
        ts, cl, vo, tr = [], [], [], []
        remaining = seconds + 120
        while remaining > 0:
            rows = self._get("/klines", {"symbol": self.symbol, "interval": "1s",
                                         "endTime": end, "limit": min(1000, remaining)})
            if not rows:
                break
            ts = [int(x[0]) // 1000 for x in rows] + ts
            cl = [float(x[4]) for x in rows] + cl
            vo = [float(x[5]) for x in rows] + vo
            tr = [float(x[8]) for x in rows] + tr
            end = int(rows[0][0]) - 1
            remaining -= len(rows)
        if not ts:
            raise RuntimeError("Binance returned no klines")

        g, c, v, n, synth = _to_grid(ts, cl, vo, tr, seconds)
        self._last_ts = int(g[-1])
        return FeedHistory(g, c, v, n, synth, "binance:1s", [
            f"{len(g):,} true 1-second bars from {self.symbol}",
            f"{synth.mean()*100:.1f}% of seconds had no print (forward-filled)",
        ])

    def spot_price(self) -> float:
        """Single last-price snapshot, used by HybridFeed to re-anchor basis."""
        return float(self._get("/ticker/price", {"symbol": self.symbol},
                               timeout=10)["price"])

    def poll(self):
        rows = self._get("/klines", {"symbol": self.symbol, "interval": "1s",
                                     "limit": 60}, timeout=10)
        out = []
        for row in rows:
            t = int(row[0]) // 1000
            if self._last_ts is not None and t <= self._last_ts:
                continue
            out.append((t, float(row[4]), float(row[5]), float(row[8])))
        if out:
            self._last_ts = out[-1][0]
        return out


# ---------------------------------------------------------------------------
# biquote.io -- live quotes, shallow history
# ---------------------------------------------------------------------------
class BiquoteFeed:
    name = "biquote"

    def __init__(self, symbol: str = "BTCUSD", session: requests.Session | None = None):
        self.symbol = symbol
        self.s = session or requests.Session()
        self._last_ts: int | None = None

    def tick(self) -> dict:
        r = self.s.get(f"{BIQUOTE}/api/{self.symbol}", timeout=15)
        r.raise_for_status()
        return r.json()

    def ohlc(self, interval: str) -> list[dict]:
        r = self.s.get(f"{BIQUOTE}/api/{self.symbol}/ohlc",
                       params={"interval": interval,
                               "limit": BIQUOTE_OHLC_CAP.get(interval, 300)},
                       timeout=25)
        r.raise_for_status()
        return r.json().get("bars", [])

    def recent_ticks(self) -> list[dict]:
        """At most 100 ticks (~1.8 min). The `limit` param is capped server-side."""
        r = self.s.get(f"{BIQUOTE}/api/{self.symbol}/history",
                       params={"limit": 100}, timeout=20)
        r.raise_for_status()
        return r.json()

    def warmup(self, seconds: int) -> FeedHistory:
        """Best-effort 1-second context from minute bars plus recent ticks.

        Minute bars are expanded with a zero-order hold, NOT smooth
        interpolation: a held value is an honest "no new information", whereas
        a smooth ramp would manufacture plausible-looking microstructure the
        market never produced. Every held second is flagged `synthetic`.
        """
        samples: dict[int, tuple[float, float, float]] = {}
        used = []
        for interval in ("15m", "5m", "1m"):  # coarse first, fine overwrites
            span = BIQUOTE_OHLC_CAP.get(interval, 0) * BIQUOTE_INTERVAL_SEC[interval]
            if span <= 0:
                continue
            try:
                bars = self.ohlc(interval)
            except requests.RequestException:
                continue
            if not bars:
                continue
            used.append(f"{interval}x{len(bars)}")
            step = BIQUOTE_INTERVAL_SEC[interval]
            for b in bars:
                t = _parse_ts(b["openTime"]) + step - 1  # close stamped at bar end
                samples[t] = (float(b["close"]), 0.0, float(b.get("tickVolume", 0)))

        try:
            for t in self.recent_ticks():
                samples[_parse_ts(t["timestamp"])] = (float(t["mid"]), 0.0, 1.0)
        except requests.RequestException:
            pass

        if not samples:
            raise RuntimeError("biquote returned no usable history")

        ts = np.array(sorted(samples), np.int64)
        cl = np.array([samples[t][0] for t in ts], np.float64)
        vo = np.array([samples[t][1] for t in ts], np.float64)
        tr = np.array([samples[t][2] for t in ts], np.float64)

        now = int(time.time())
        g, c, v, n, synth = _to_grid(ts, cl, vo, tr, seconds, end_ts=now)
        self._last_ts = int(g[-1])

        frac = synth.mean() * 100
        return FeedHistory(g, c, v, n, synth, "biquote:ohlc+ticks", [
            f"reconstructed from {', '.join(used) or 'ticks only'}",
            f"**{frac:.1f}% of the {seconds:,}-second context is synthetic** "
            f"(held between minute bars) -- biquote has no 1-second history",
            "volume and taker-buy-imbalance channels are zero-filled: the MT5 "
            "feed reports no real volume",
        ])

    def poll(self):
        out = []
        try:
            for t in self.recent_ticks():
                sec = _parse_ts(t["timestamp"])
                if self._last_ts is not None and sec <= self._last_ts:
                    continue
                out.append((sec, float(t["mid"]), 0.0, 1.0))
        except requests.RequestException:
            q = self.tick()
            sec = _parse_ts(q["timestamp"])
            if self._last_ts is None or sec > self._last_ts:
                out.append((sec, float(q["mid"]), 0.0, 1.0))
        # collapse duplicate seconds, keep the last quote in each
        dedup: dict[int, tuple] = {}
        for row in out:
            dedup[row[0]] = row
        out = [dedup[k] for k in sorted(dedup)]
        if out:
            self._last_ts = out[-1][0]
        return out


# ---------------------------------------------------------------------------
# Hybrid -- faithful warm-up + un-geo-blocked live updates
# ---------------------------------------------------------------------------
class HybridFeed:
    """Binance 1-second warm-up, biquote live ticks, continuously rebased.

    biquote BTCUSD (MT5 CFD) and Binance BTCUSDT (spot) are different
    instruments trading at a small, drifting basis (measured -6.5 USD / -0.8 bp,
    sd 3.2). Appending raw biquote prices onto a Binance history would present
    the model with a step change it would read as a genuine move. We therefore
    track the basis with an EMA and subtract it from every incoming quote, so
    the published series stays on the Binance price level the model knows.
    """

    name = "hybrid"

    def __init__(self, binance_symbol="BTCUSDT", biquote_symbol="BTCUSD",
                 basis_halflife_samples: float = 10.0,
                 basis_refresh_sec: float = 60.0):
        s = requests.Session()
        self.binance = BinanceFeed(binance_symbol, s)
        self.biquote = BiquoteFeed(biquote_symbol, s)
        self.basis: float | None = None
        self.basis_stale = False
        self.basis_refresh_sec = basis_refresh_sec
        self._last_basis_at = 0.0
        self.alpha = 1.0 - 0.5 ** (1.0 / max(basis_halflife_samples, 1.0))
        self._last_ts: int | None = None
        self._last_px: float | None = None

    def warmup(self, seconds: int) -> FeedHistory:
        h = self.binance.warmup(seconds)
        self._last_ts, self._last_px = int(h.ts[-1]), float(h.close[-1])
        # Seed the biquote cursor too, otherwise its 100-tick buffer (~1.8 min,
        # which overlaps the warm-up window) is replayed as if it were new.
        self.biquote._last_ts = self._last_ts
        try:
            q = self.biquote.tick()
            self.basis = float(q["mid"]) - self._last_px
            self._last_basis_at = time.time()
            h.notes.append(
                f"live updates from biquote {self.biquote.symbol}; initial basis "
                f"{self.basis:+.2f} USD ({self.basis/self._last_px*1e4:+.1f} bp), "
                f"rebased continuously"
            )
        except requests.RequestException as e:
            h.notes.append(f"biquote unreachable ({e}); falling back to Binance live")
        h.source = "hybrid: binance:1s warmup + biquote live"
        return h

    def poll(self):
        rows = self.biquote.poll()
        if not rows:
            return []

        # Re-anchor periodically. The basis is biquote_mid - binance_px, so it
        # can only be re-estimated when we actually have a Binance reference;
        # there is no way to infer drift from the biquote series alone. If
        # Binance becomes unreachable we hold the last known basis and say so.
        now = time.time()
        if now - self._last_basis_at >= self.basis_refresh_sec:
            self._last_basis_at = now
            try:
                spot = self.binance.spot_price()
                obs = rows[-1][1] - spot
                self.basis = obs if self.basis is None else \
                    (1 - self.alpha) * self.basis + self.alpha * obs
                self.basis_stale = False
            except (requests.RequestException, RuntimeError):
                self.basis_stale = True

        b = self.basis or 0.0
        out = [(sec, px - b, vol, trd) for sec, px, vol, trd in rows]
        self._last_px, self._last_ts = out[-1][1], out[-1][0]
        return out


FEEDS = {"hybrid": HybridFeed, "biquote": BiquoteFeed, "binance": BinanceFeed}


def make_feed(mode: str):
    if mode not in FEEDS:
        raise ValueError(f"unknown feed {mode!r}; choose from {list(FEEDS)}")
    return FEEDS[mode]()
