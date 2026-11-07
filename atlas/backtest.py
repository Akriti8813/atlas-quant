"""Vectorised backtest engine with transaction costs and a causality check."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    net_returns: pd.Series
    gross_returns: pd.Series
    turnover: pd.Series
    weights: pd.DataFrame


def backtest(prices: pd.DataFrame, weights: pd.DataFrame,
             cost_bps: float = 10.0, slippage_bps: float = 5.0) -> BacktestResult:
    """Weights decided at close of day t earn the return of day t+1 (shift by one).
    Costs are charged on traded notional: (cost_bps + slippage_bps) per unit turnover.
    Cash earns 0. Weights are target portfolio fractions (sum |w| <= 1 for no leverage)."""
    rets = prices.pct_change(fill_method=None).fillna(0.0)
    w = weights.reindex(index=prices.index, columns=prices.columns).fillna(0.0)
    held = w.shift(1).fillna(0.0)
    gross = (held * rets).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1)
    turnover.iloc[0] = held.iloc[0].abs().sum()
    net = gross - turnover * (cost_bps + slippage_bps) / 1e4
    return BacktestResult(net, gross, turnover, w)


def check_causality(func, prices: pd.DataFrame, params: dict, n_checks: int = 5,
                    min_history: int = 300, seed: int = 0) -> bool:
    """Look-ahead detector: weights computed on a truncated history must equal the
    weights the full-history run produced for the same date."""
    full = func(prices, **params)
    rng = np.random.default_rng(seed)
    cuts = rng.integers(min_history, len(prices) - 1, size=n_checks)
    for c in cuts:
        part = func(prices.iloc[: c + 1], **params)
        a = part.iloc[-1].fillna(0.0).values
        b = full.iloc[c].fillna(0.0).values
        if not np.allclose(a, b, atol=1e-9):
            return False
    return True
