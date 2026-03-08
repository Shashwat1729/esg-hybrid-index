"""Compute short numeric summaries from Thesis_report/Tables CSVs and print a JSON
that can be copied into the Results chapter.

Run from the repo root:
    python scripts/13_compute_summaries.py
"""
import json
from pathlib import Path
import pandas as pd

root = Path('Thesis_report') / 'Tables'
out = {}

def read_csv(name):
    p = root / name
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception as e:
            return None
    return None

# timesplit
df = read_csv('timesplit_evaluation_synthetic_summary.csv')
if df is not None and len(df):
    out['timesplit'] = df.iloc[0].to_dict()

# benchmark
df = read_csv('benchmark_summary.csv')
if df is not None and len(df):
    try:
        row = df[df['index'].str.contains('Our Multi-Factor', na=False)].iloc[0]
        out['benchmark'] = {'our_avg_esg': float(row.get('avg_ESG', row.get('Avg_ESG', 0)))}
    except Exception:
        out['benchmark'] = {'note': 'Our Multi-Factor row not found; table available.'}

# pca
df = read_csv('advanced_pca_variance.csv')
if df is not None and len(df):
    out['pca'] = {
        'pc1_var': float(df.loc[0, 'variance_explained']) if 'variance_explained' in df.columns else None,
        'pc6_cum': float(df.loc[5, 'cumulative_variance']) if 'cumulative_variance' in df.columns and len(df) > 5 else None,
    }

# weight grid
df = read_csv('weight_grid_search.csv')
if df is not None and len(df):
    # try common column names
    cand = [c for c in df.columns if 'return' in c.lower() and 'sharpe' in c.lower()]
    if cand:
        best = df.loc[df[cand[0]].idxmax()]
    else:
        # fall back to 'return_sharpe' or 'sharpe'
        if 'return_sharpe' in df.columns:
            best = df.loc[df['return_sharpe'].idxmax()]
        elif 'sharpe' in df.columns:
            best = df.loc[df['sharpe'].idxmax()]
        else:
            best = None
    if best is not None:
        out['weight_grid'] = {'best_return_sharpe': float(best.get('return_sharpe', best.get('returnSharpe', best.get('sharpe', 0))))}

# efficient frontier
df = read_csv('advanced_efficient_frontier.csv')
if df is not None and len(df):
    if 'sharpe' in df.columns:
        best = df.loc[df['sharpe'].idxmax()]
        out['efront'] = {'max_sharpe': float(best['sharpe']), 'ret': float(best.get('return', best.get('ret', None))), 'risk': float(best.get('risk', None))}

print(json.dumps(out, indent=2))
