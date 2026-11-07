"""Build the static dashboard into ./site (run by GitHub Actions, or locally).
  python build_site.py              # real NIFTY data (fails loudly if data is thin)
  python build_site.py --synthetic  # labelled demo data
"""
import argparse, json, os, sys
import pandas as pd
from atlas.data import NIFTY_UNIVERSE, load_prices, validate_prices, synthetic_prices
from atlas.compete import run_competition

ap = argparse.ArgumentParser()
ap.add_argument("--synthetic", action="store_true")
ap.add_argument("--start", default="2012-01-01")
a = ap.parse_args()

px = synthetic_prices() if a.synthetic else load_prices(NIFTY_UNIVERSE, a.start)
px, rep = validate_prices(px)
if not a.synthetic and (px.shape[1] < 10 or len(px) < 1500):
    sys.exit(f"Not enough real data ({px.shape[1]} assets, {len(px)} days)")

table, curves, _ = run_competition(px)
c = curves.iloc[::21]
data = {
    "source": "synthetic" if a.synthetic else "real",
    "asof": px.index[-1].strftime("%d %b %Y"),
    "assets": int(px.shape[1]),
    "rows": [{"name": n, **{k: (None if pd.isna(v) else float(v)) for k, v in r.items()}}
             for n, r in table.iterrows()],
    "dates": [d.strftime("%Y-%m") for d in c.index],
    "series": {k: [round(float(x), 4) for x in c[k]] for k in c.columns},
}
os.makedirs("site", exist_ok=True)
html = open("web/template.html", encoding="utf-8").read().replace("/*DATA*/null", json.dumps(data))
open("site/index.html", "w", encoding="utf-8").write(html)
table.to_csv("site/leaderboard.csv"); curves.to_csv("site/equity_curves.csv")
print(f"Built site/index.html [{data['source']}] {data['assets']} assets through {data['asof']}")
