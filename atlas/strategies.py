"""Strategy library. Each strategy: f(prices, **params) -> target weights (dates x assets).
Strategies must be causal (use only data up to each row's date). Long-only by default
(Indian cash equities: no overnight shorting). Unallocated weight is cash."""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd


def _cap(w: pd.DataFrame) -> pd.DataFrame:
    gross = w.abs().sum(axis=1).clip(lower=1.0)
    return w.div(gross, axis=0).fillna(0.0)


def buy_hold(prices, **_):
    n = prices.shape[1]
    return pd.DataFrame(1.0 / n, index=prices.index, columns=prices.columns)


def tsmom(prices, lookback=126):
    """Time-series momentum: hold assets whose trailing return is positive."""
    sig = (prices.pct_change(lookback, fill_method=None) > 0).astype(float)
    return _cap(sig / prices.shape[1])


def xsmom(prices, lookback=126, skip=21, top_frac=0.3):
    """Cross-sectional momentum: top fraction by trailing return (skipping recent month)."""
    mom = prices.shift(skip) / prices.shift(skip + lookback) - 1
    rank = mom.rank(axis=1, pct=True)
    sel = (rank > 1 - top_frac).astype(float)
    return sel.div(sel.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def trend_ma(prices, window=100):
    sig = (prices > prices.rolling(window).mean()).astype(float)
    return _cap(sig / prices.shape[1])


def mean_reversion(prices, window=20, entry=-1.5, exit_=0.0):
    """Buy oversold (z < entry), hold until z crosses exit_. Stateful via forward-fill."""
    z = (prices - prices.rolling(window).mean()) / prices.rolling(window).std()
    state = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    state[z < entry] = 1.0
    state[z > exit_] = 0.0
    pos = state.ffill().fillna(0.0)
    return _cap(pos / prices.shape[1])


def rsi_reversion(prices, period=14, low=30, high=55):
    d = prices.diff()
    up = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    state = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    state[rsi < low] = 1.0
    state[rsi > high] = 0.0
    return _cap(state.ffill().fillna(0.0) / prices.shape[1])


def low_vol(prices, window=60, top_frac=0.3):
    vol = prices.pct_change(fill_method=None).rolling(window).std()
    rank = vol.rank(axis=1, pct=True)
    sel = (rank <= top_frac).astype(float).where(vol.notna(), 0.0)
    return sel.div(sel.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def _grid(**kw):
    keys = list(kw)
    return [dict(zip(keys, v)) for v in itertools.product(*kw.values())]


# name -> (function, parameter grid searched during walk-forward)
REGISTRY = {
    "tsmom": (tsmom, _grid(lookback=[63, 126, 252])),
    "xsmom": (xsmom, _grid(lookback=[63, 126, 252], top_frac=[0.2, 0.3])),
    "trend_ma": (trend_ma, _grid(window=[50, 100, 200])),
    "mean_reversion": (mean_reversion, _grid(window=[10, 20, 40], entry=[-1.0, -1.5, -2.0])),
    "rsi_reversion": (rsi_reversion, _grid(period=[7, 14], low=[25, 30, 35])),
    "low_vol": (low_vol, _grid(window=[60, 120], top_frac=[0.2, 0.3])),
}
