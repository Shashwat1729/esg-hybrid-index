"""Compute short numeric summaries from pipeline CSV outputs and print JSON
that can be referenced in the Results chapter.

Searches reports/tables/ first, then Thesis_report/Tables/ as fallback.

Run from the repo root:
    python scripts/13_compute_summaries.py
"""
import json
from pathlib import Path
import pandas as pd

dirs = [Path('reports') / 'tables', Path('Thesis_report') / 'Tables']
out = {}

def read_csv(name):
    for d in dirs:
        p = d / name
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                pass
    return None

# --- Predictive validation (new) ---
df = read_csv('predictive_validation_summary.csv')
if df is not None and len(df):
    ic_vals = df['ic_spearman'] if 'ic_spearman' in df.columns else pd.Series()
    out['predictive_validation'] = {
        'n_pairs': len(df),
        'mean_ic': float(ic_vals.mean()) if len(ic_vals) else None,
        'max_abs_ic': float(ic_vals.abs().max()) if len(ic_vals) else None,
        'n_significant': int(df['ic_significant'].sum()) if 'ic_significant' in df.columns else None,
    }

# --- Bootstrap stability ---
df = read_csv('predictive_validation_bootstrap.csv')
if df is not None and len(df):
    tau_row = df[df['metric'] == 'kendall_tau_mean']
    if len(tau_row):
        out['bootstrap'] = {'kendall_tau_mean': float(tau_row['value'].iloc[0])}

# --- Benchmark ---
df = read_csv('benchmark_summary.csv')
if df is not None and len(df):
    try:
        row = df[df['index'].str.contains('Our Multi-Factor', na=False)].iloc[0]
        out['benchmark'] = {
            'our_avg_esg': float(row.get('avg_ESG', 0)),
            'our_avg_financial': float(row.get('avg_financial', 0)),
        }
    except Exception:
        out['benchmark'] = {'note': 'Our Multi-Factor row not found; table available.'}

# --- PCA ---
df = read_csv('advanced_pca_variance.csv')
if df is not None and len(df):
    out['pca'] = {
        'pc1_var': float(df.loc[0, 'variance_explained']) if 'variance_explained' in df.columns else None,
        'pc6_cum': float(df.loc[5, 'cumulative_variance']) if 'cumulative_variance' in df.columns and len(df) > 5 else None,
    }

# --- Weight grid ---
df = read_csv('weight_grid_search.csv')
if df is not None and len(df):
    col = None
    if 'return_sharpe' in df.columns:
        col = 'return_sharpe'
    elif 'sharpe' in df.columns:
        col = 'sharpe'
    if col:
        best = df.loc[df[col].idxmax()]
        out['weight_grid'] = {'best_return_sharpe': float(best[col])}

# --- Factor tilt sensitivity (renamed from efficient frontier) ---
df = read_csv('advanced_factor_tilt_sensitivity.csv')
if df is None:
    df = read_csv('advanced_efficient_frontier.csv')  # legacy fallback
if df is not None and len(df):
    csir_col = 'cross_sectional_ir' if 'cross_sectional_ir' in df.columns else 'sharpe'
    if csir_col in df.columns:
        best = df.loc[df[csir_col].idxmax()]
        out['factor_tilt'] = {
            'best_csir': float(best[csir_col]),
            'ret': float(best.get('return', best.get('ret', 0))),
            'risk': float(best.get('risk', 0)),
        }

# --- Quintile spreads ---
df = read_csv('predictive_validation_spreads.csv')
if df is not None and len(df):
    pos = df[df['spread'] > 0]
    out['quintile_spreads'] = {
        'n_positive': len(pos),
        'n_total': len(df),
        'best_factor': df.loc[df['spread'].idxmax(), 'factor'] if len(df) else None,
        'best_spread': float(df['spread'].max()) if len(df) else None,
    }

print(json.dumps(out, indent=2, default=str))
