# ATLAS QUANT — Phase 1–2: data layer + quant research engine

Option A (institutional-grade quant platform). This repo covers roadmap Phases 1–2.

## Run
    pip install -r requirements.txt
    python -m pytest -q                 # 13 tests, incl. look-ahead detection
    python run.py --synthetic           # offline demo
    python run.py                       # NIFTY large caps via Yahoo Finance
    python run.py --tickers TCS.NS INFY.NS --start 2015-01-01

Outputs `leaderboard.csv` and `equity_curves.png`.

## Live dashboard
Deployed automatically to GitHub Pages by `.github/workflows/pages.yml` (tests, then real NIFTY run, then deploy; refreshes every Monday).
Build locally: `python build_site.py` then open `site/index.html`.

## What is in it
- `atlas/data.py` — Yahoo ingestion + CSV cache, validation report (missing data, extreme moves), synthetic market
- `atlas/strategies.py` — 6 strategies (time-series & cross-sectional momentum, MA trend, z-score & RSI mean reversion, low-vol) + parameter grids
- `atlas/backtest.py` — vectorised engine, 1-day execution lag, cost + slippage, `check_causality` look-ahead detector
- `atlas/walkforward.py` — rolling train/test; parameters chosen on train only, only out-of-sample returns reported
- `atlas/metrics.py` — Sharpe, Sortino, max drawdown, Calmar, win rate, turnover, Probabilistic Sharpe Ratio
- `build_site.py` + `web/template.html` — static dashboard generator
- `atlas/compete.py` — strategy competition vs equal-weight buy & hold on identical data and costs

## Known limitations (state these honestly in the README of the final project)
- Yahoo data + today's NIFTY constituents = survivorship bias. Fine for research, not for claims.
- `Trials` is reported per strategy but no deflated-Sharpe correction for multiple testing yet.
- Synthetic results mean nothing; they only prove the pipeline runs.

## Next (Phase 3+)
Pairs/cointegration, HMM regime detection, portfolio optimiser, deterministic risk engine, LangGraph agents, FastAPI, paper trading.
