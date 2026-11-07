"""Walk-forward validation: pick parameters on a trailing train window, apply to the
next unseen test window, roll forward. Only out-of-sample returns are reported."""
from __future__ import annotations
import pandas as pd
from .backtest import backtest
from .metrics import sharpe


def walk_forward(prices, func, grid, train_days=756, test_days=126,
                 cost_bps=10.0, slippage_bps=5.0):
    # Strategies are causal, so each param set is backtested once over the full sample
    # and then sliced; the selection step only ever looks at the train slice.
    runs = [backtest(prices, func(prices, **p), cost_bps, slippage_bps) for p in grid]
    n = len(prices)
    oos, turn, log = [], [], []
    start = train_days
    while start + test_days <= n:
        tr = slice(start - train_days, start)
        te = slice(start, start + test_days)
        scores = [sharpe(r.net_returns.iloc[tr]) for r in runs]
        best = max(range(len(grid)), key=lambda i: scores[i])
        oos.append(runs[best].net_returns.iloc[te])
        turn.append(runs[best].turnover.iloc[te])
        log.append({"test_start": prices.index[start].date(), "params": grid[best],
                    "train_sharpe": round(scores[best], 2)})
        start += test_days
    if not oos:
        raise ValueError("Not enough data for one walk-forward fold")
    return pd.concat(oos), pd.concat(turn), pd.DataFrame(log), len(grid)
