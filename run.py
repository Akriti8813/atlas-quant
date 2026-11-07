"""Usage:
  python run.py --synthetic                 # offline demo
  python run.py                             # NIFTY universe via Yahoo Finance
  python run.py --tickers TCS.NS INFY.NS ... --start 2015-01-01
"""
import argparse
import pandas as pd
from atlas.data import NIFTY_UNIVERSE, load_prices, validate_prices, synthetic_prices
from atlas.compete import run_competition

ap = argparse.ArgumentParser()
ap.add_argument("--synthetic", action="store_true")
ap.add_argument("--tickers", nargs="*", default=NIFTY_UNIVERSE)
ap.add_argument("--start", default="2012-01-01")
ap.add_argument("--cost-bps", type=float, default=10)
ap.add_argument("--slippage-bps", type=float, default=5)
a = ap.parse_args()

px = synthetic_prices() if a.synthetic else load_prices(a.tickers, a.start)
px, rep = validate_prices(px)
print(f"Data: {px.shape[1]} assets, {len(px)} days "
      f"({px.index[0].date()} -> {px.index[-1].date()})")
print("Validation:", {k: v for k, v in rep.items() if v})

table, curves, logs = run_competition(px, cost_bps=a.cost_bps, slippage_bps=a.slippage_bps)
pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format)
print("\nOUT-OF-SAMPLE LEADERBOARD (walk-forward, after costs)\n")
print(table)
table.to_csv("leaderboard.csv"); curves.to_csv("equity_curves.csv")
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ax = curves.plot(figsize=(11, 6), title="Out-of-sample equity curves (after costs)")
    ax.set_ylabel("Growth of 1"); plt.tight_layout(); plt.savefig("equity_curves.png", dpi=120)
except Exception as e:
    print("Plot skipped:", e)
