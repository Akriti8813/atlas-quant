import numpy as np
import pandas as pd
import pytest
from atlas.data import synthetic_prices, validate_prices
from atlas.backtest import backtest, check_causality
from atlas.strategies import REGISTRY, buy_hold
from atlas.metrics import max_drawdown, sharpe
from atlas.walkforward import walk_forward

PX = synthetic_prices(n_assets=8, n_days=1200)


@pytest.mark.parametrize("name", list(REGISTRY))
def test_strategy_is_causal(name):
    func, grid = REGISTRY[name]
    assert check_causality(func, PX, grid[0])


def test_checker_catches_lookahead():
    cheat = lambda p: (p.pct_change().shift(-1) > 0).astype(float) / p.shape[1]
    assert not check_causality(cheat, PX, {})


def test_weights_never_exceed_gross_1():
    for name, (func, grid) in REGISTRY.items():
        w = func(PX, **grid[0])
        assert w.abs().sum(axis=1).max() <= 1.0 + 1e-9, name


def test_costs_reduce_returns():
    func, grid = REGISTRY["mean_reversion"]
    w = func(PX, **grid[0])
    free = backtest(PX, w, 0, 0).net_returns.sum()
    paid = backtest(PX, w, 10, 5).net_returns.sum()
    assert paid < free


def test_buy_hold_matches_manual():
    res = backtest(PX, buy_hold(PX), 0, 0)
    manual = PX.pct_change().fillna(0).mean(axis=1)
    assert np.allclose(res.net_returns.iloc[2:], manual.iloc[2:], atol=1e-12)


def test_walk_forward_oos_only():
    func, grid = REGISTRY["tsmom"]
    r, t, log, n = walk_forward(PX, func, grid, train_days=500, test_days=100)
    assert r.index.is_unique and r.index[0] == PX.index[500]
    assert n == len(grid)


def test_validate_flags_problems():
    px = PX.copy()
    px.iloc[10, 0] = px.iloc[9, 0] * 2
    px["BAD"] = np.nan
    clean, rep = validate_prices(px)
    assert "BAD" in rep["dropped_columns"] and rep["extreme_moves"]


def test_metrics_sanity():
    r = pd.Series([0.01, -0.02, 0.015, 0.0, 0.01] * 50)
    assert max_drawdown(r) < 0 and sharpe(r) > 0


def test_competition_curves_share_window():
    from atlas.compete import run_competition
    table, curves, _ = run_competition(PX, train_days=500, test_days=100)
    assert not curves.isna().any().any()
