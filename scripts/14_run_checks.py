"""Simple iterative checks (no pytest) to validate presence and basic sanity of results CSVs.

Run from the repo root:
    python scripts/14_run_checks.py
"""
import sys
from pathlib import Path
import pandas as pd

# Pipeline outputs to reports/tables; 07_time_split copies key files to Thesis_report/Tables
root = Path('reports') / 'tables'
thesis_root = Path('Thesis_report') / 'Tables'
errors = []

def must_exist(pname, search_dirs=None):
    """Look for a file in multiple directories."""
    if search_dirs is None:
        search_dirs = [root, thesis_root]
    for d in search_dirs:
        p = d / pname
        if p.exists():
            return p
    errors.append(f'MISSING: {pname} (checked {[str(d) for d in search_dirs]})')
    return None

# Core validation tables (produced by 07b_cross_sectional_validation.py)
pv = must_exist('predictive_validation_summary.csv')
boot = must_exist('predictive_validation_bootstrap.csv')

# Other pipeline tables
bm = must_exist('benchmark_summary.csv')
pc = must_exist('advanced_pca_variance.csv')
wg = must_exist('weight_grid_search.csv')
ef = must_exist('advanced_factor_tilt_sensitivity.csv')

# Legacy table (may still exist from older runs)
ts = must_exist('timesplit_evaluation_synthetic_summary.csv')

if errors:
    print('\n'.join(errors))
    # Don't hard-fail if only legacy/optional tables are missing
    critical_missing = [e for e in errors if 'predictive_validation' in e or 'benchmark' in e]
    if critical_missing:
        print('CRITICAL tables missing — run the pipeline first.')
        sys.exit(2)
    else:
        print('Some optional tables missing (non-critical).')

def safe_read(p):
    if p is None:
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f'ERROR reading {p}: {e}')
        sys.exit(2)

df_pv = safe_read(pv)
df_boot = safe_read(boot)
df_ef = safe_read(ef)
df_wg = safe_read(wg)

# Basic sanity checks
ok = True


# Basic sanity checks
ok = True

# Predictive validation: IC values should be in reasonable range
if df_pv is not None and 'ic_spearman' in df_pv.columns:
    max_ic = float(df_pv['ic_spearman'].abs().max())
    print(f"max |IC| = {max_ic:.4f}")
    if max_ic > 1.0:
        print('UNUSUAL: IC > 1.0 (impossible for Spearman)')
        ok = False
    n_sig = 0
    if 'ic_significant' in df_pv.columns:
        n_sig = int(df_pv['ic_significant'].sum())
    print(f"significant IC pairs = {n_sig}")

# Bootstrap: Kendall tau should be positive and < 1
if df_boot is not None and 'value' in df_boot.columns:
    tau_row = df_boot[df_boot['metric'] == 'kendall_tau_mean']
    if len(tau_row) > 0:
        tau = float(tau_row['value'].iloc[0])
        print(f"bootstrap Kendall tau = {tau:.4f}")
        if not (0 < tau <= 1):
            print('UNUSUAL: bootstrap tau outside (0, 1]')
            ok = False

# Efficient frontier
if df_ef is not None and 'sharpe' in df_ef.columns:
    max_s = float(df_ef['sharpe'].max())
    print(f"efficient frontier max sharpe = {max_s:.6f}")
    if max_s <= 0:
        print('UNUSUAL: frontier max Sharpe <= 0')
        ok = False

# Weight grid
if df_wg is not None:
    col = None
    if 'return_sharpe' in df_wg.columns:
        col = 'return_sharpe'
    elif 'sharpe' in df_wg.columns:
        col = 'sharpe'
    if col:
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
