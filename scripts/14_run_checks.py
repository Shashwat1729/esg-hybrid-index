"""Simple iterative checks (no pytest) to validate presence and basic sanity of results CSVs.

Run from the repo root:
    python scripts/14_run_checks.py
"""
import sys
from pathlib import Path
import pandas as pd

root = Path('Thesis_report') / 'Tables'
errors = []

def must_exist(pname):
    p = root / pname
    if not p.exists():
        errors.append(f'MISSING: {p}')
    return p

ts = must_exist('timesplit_evaluation_synthetic_summary.csv')
bm = must_exist('benchmark_summary.csv')
pc = must_exist('advanced_pca_variance.csv')
wg = must_exist('weight_grid_search.csv')
ef = must_exist('advanced_efficient_frontier.csv')

if errors:
    print('\n'.join(errors))
    sys.exit(2)

def safe_read(p):
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f'ERROR reading {p}: {e}')
        sys.exit(2)

df_ts = safe_read(ts)
df_ef = safe_read(ef)
df_wg = safe_read(wg)

# Basic sanity checks
ok = True
if 'sharpe' in df_ts.columns:
    s = float(df_ts.iloc[0]['sharpe'])
    print(f"timesplit sharpe = {s:.6f}")
    if not (-10 < s < 10):
        print('UNUSUAL: timesplit Sharpe outside reasonable bounds')
        ok = False

if 'sharpe' in df_ef.columns:
    max_s = float(df_ef['sharpe'].max())
    print(f"efficient frontier max sharpe = {max_s:.6f}")
    if max_s <= 0:
        print('UNUSUAL: frontier max Sharpe <= 0')
        ok = False

if 'return_sharpe' in df_wg.columns or 'sharpe' in df_wg.columns:
    col = 'return_sharpe' if 'return_sharpe' in df_wg.columns else 'sharpe'
    best = float(df_wg[col].max())
    print(f"best grid {col} = {best:.6f}")
    if best < -5 or best > 10:
        print('UNUSUAL: grid best value outside wide bounds')
        ok = False

if ok:
    print('ALL CHECKS PASS')
    sys.exit(0)
else:
    print('SOME CHECKS FAILED')
    sys.exit(3)
