"""Performance metrics. All inputs are daily simple-return Series."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


def sharpe(r: pd.Series, rf_annual=0.0) -> float:
    ex = r - rf_annual / TRADING_DAYS
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0


def sortino(r: pd.Series, rf_annual=0.0) -> float:
    ex = r - rf_annual / TRADING_DAYS
    dd = np.sqrt((np.minimum(ex, 0) ** 2).mean())
    return float(ex.mean() / dd * np.sqrt(TRADING_DAYS)) if dd > 0 else 0.0


def equity_curve(r: pd.Series) -> pd.Series:
    return (1 + r.fillna(0)).cumprod()


def max_drawdown(r: pd.Series) -> float:
    eq = equity_curve(r)
    return float((eq / eq.cummax() - 1).min())


def probabilistic_sharpe(r: pd.Series, benchmark_sr_annual=0.0) -> float:
    """P(true Sharpe > benchmark) given sample skew/kurtosis (Bailey & Lopez de Prado)."""
    n = len(r)
    if n < 30 or r.std() == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)
    sr_b = benchmark_sr_annual / np.sqrt(TRADING_DAYS)
    sk, ku = stats.skew(r), stats.kurtosis(r, fisher=False)
    denom = np.sqrt(max(1e-12, 1 - sk * sr + (ku - 1) / 4 * sr ** 2))
    return float(stats.norm.cdf((sr - sr_b) * np.sqrt(n - 1) / denom))


def summarize(r: pd.Series, turnover: pd.Series | None = None, rf_annual=0.0) -> dict:
    r = r.dropna()
    n = len(r)
    if n == 0:
        return {}
    eq = equity_curve(r)
    cagr = eq.iloc[-1] ** (TRADING_DAYS / n) - 1
    mdd = max_drawdown(r)
    out = {
        "CAGR": cagr,
        "AnnVol": r.std(ddof=1) * np.sqrt(TRADING_DAYS),
        "Sharpe": sharpe(r, rf_annual),
        "Sortino": sortino(r, rf_annual),
        "MaxDD": mdd,
        "Calmar": cagr / abs(mdd) if mdd < 0 else float("nan"),
        "WinRate": float((r > 0).mean()),
        "PSR(>0)": probabilistic_sharpe(r),
    }
    if turnover is not None:
        out["AnnTurnover"] = float(turnover.reindex(r.index).fillna(0).mean() * TRADING_DAYS)
    return out
