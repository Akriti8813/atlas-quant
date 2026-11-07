"""Strategy competition: identical data, costs and validation for every strategy."""
from __future__ import annotations
import pandas as pd
from .backtest import backtest
from .metrics import summarize, equity_curve
from .strategies import REGISTRY, buy_hold
from .walkforward import walk_forward


def run_competition(prices, train_days=756, test_days=126, cost_bps=10.0, slippage_bps=5.0):
    rows, curves, logs = {}, {}, {}
    for name, (func, grid) in REGISTRY.items():
        r, t, log, n_trials = walk_forward(prices, func, grid, train_days, test_days,
                                           cost_bps, slippage_bps)
        rows[name] = {**summarize(r, t), "Trials": n_trials}
        curves[name], logs[name] = equity_curve(r), log
    # Benchmark over the identical out-of-sample window
    first_oos = prices.index[train_days]
    last_oos = max(c.index[-1] for c in curves.values())  # same window as the strategies
    bh = backtest(prices, buy_hold(prices), cost_bps, slippage_bps)
    bh_r = bh.net_returns.loc[first_oos:last_oos]
    rows["BUY_HOLD (equal wt)"] = {**summarize(bh_r, bh.turnover), "Trials": 1}
    curves["BUY_HOLD (equal wt)"] = equity_curve(bh_r)
    table = pd.DataFrame(rows).T.sort_values("Sharpe", ascending=False)
    return table, pd.DataFrame(curves), logs
