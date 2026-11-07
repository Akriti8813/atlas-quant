"""Phase 1: market data ingestion, caching, validation, synthetic data."""
from __future__ import annotations
import hashlib
import os
import numpy as np
import pandas as pd

# Liquid NIFTY 50 constituents (Yahoo Finance NSE tickers).
# NOTE: using today's constituents for past periods has survivorship bias.
NIFTY_UNIVERSE = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "WIPRO.NS", "ONGC.NS",
]
BENCHMARK = "^NSEI"


def load_prices(tickers, start="2012-01-01", end=None, cache_dir=".cache") -> pd.DataFrame:
    """Adjusted close prices (dates x tickers). Cached to CSV. Research use only."""
    import yfinance as yf
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.md5(f"{sorted(tickers)}{start}{end}".encode()).hexdigest()[:10]
    path = os.path.join(cache_dir, f"prices_{key}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    px = px.sort_index()
    px.to_csv(path)
    return px


def validate_prices(px: pd.DataFrame, max_missing=0.10, max_abs_ret=0.40):
    """Clean prices and return (clean_df, report dict). Never silently hides problems."""
    report = {"dropped_columns": [], "extreme_moves": [], "ffilled_cells": 0}
    px = px.copy()
    px = px[~px.index.duplicated(keep="last")].sort_index()
    px = px.dropna(how="all")
    miss = px.isna().mean()
    bad = miss[miss > max_missing].index.tolist()
    report["dropped_columns"] = bad
    px = px.drop(columns=bad)
    before = px.isna().sum().sum()
    px = px.ffill(limit=3)
    report["ffilled_cells"] = int(before - px.isna().sum().sum())
    r = px.pct_change(fill_method=None)
    stacked = r.stack()
    ext = stacked[stacked.abs() > max_abs_ret]
    report["extreme_moves"] = [(str(d.date()), t, round(float(v), 3)) for (d, t), v in ext.items()]
    report["nonpositive_prices"] = int((px <= 0).sum().sum())
    return px, report


def synthetic_prices(n_assets=15, n_days=2500, seed=7) -> pd.DataFrame:
    """Regime-switching market with a weak momentum factor. For tests/demos only."""
    rng = np.random.default_rng(seed)
    regime = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        regime[t] = regime[t - 1] if rng.random() > 0.01 else rng.integers(0, 3)
    mu = np.array([0.0006, 0.0000, -0.0004])[regime]
    sig = np.array([0.008, 0.012, 0.022])[regime]
    mkt = rng.normal(mu, sig)
    rets = np.zeros((n_days, n_assets))
    drift = rng.normal(0, 0.0003, n_assets)
    for j in range(n_assets):
        idio = rng.normal(0, 0.010, n_days)
        ar = np.zeros(n_days)
        for t in range(1, n_days):
            ar[t] = 0.04 * ar[t - 1] + idio[t]
        rets[:, j] = drift[j] + (0.8 + 0.4 * rng.random()) * mkt + ar
    idx = pd.bdate_range("2014-01-01", periods=n_days)
    return pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx,
                        columns=[f"ASSET{j:02d}" for j in range(n_assets)])
