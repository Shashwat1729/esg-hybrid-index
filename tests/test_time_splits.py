"""
test_time_splits.py — Smoke tests for the cross-sectional predictive validation module.

These tests use a small synthetic DataFrame to verify that every public
function in 07b_cross_sectional_validation.py runs without error and produces
output in the expected shape.  They do NOT validate statistical correctness
(that is the job of the pipeline itself and 14_run_checks.py).

Run:  python tests/test_time_splits.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_synthetic_df(n=50, seed=42):
    """Create a small DataFrame that mimics indexed_data.csv columns."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "ticker": [f"T{i:03d}" for i in range(n)],
        "sector": rng.choice(["Tech", "Health", "Finance", "Energy"], n),
        "country": rng.choice(["US", "IN"], n),
        "ESG_composite": rng.uniform(30, 80, n),
        "financial_score": rng.uniform(20, 90, n),
        "market_score": rng.uniform(25, 75, n),
        "operational_score": rng.uniform(30, 70, n),
        "risk_adjusted_score": rng.uniform(20, 80, n),
        "growth_score": rng.uniform(10, 90, n),
        "value_score": rng.uniform(15, 85, n),
        "stability_score": rng.uniform(25, 75, n),
        "pref_balanced": rng.uniform(30, 70, n),
        "pref_esg_first": rng.uniform(30, 70, n),
        "pref_financial_first": rng.uniform(30, 70, n),
        "pref_balanced_ex_market": rng.uniform(30, 70, n),
        "pref_esg_first_ex_market": rng.uniform(30, 70, n),
        "pref_financial_first_ex_market": rng.uniform(30, 70, n),
        "price_momentum_1m": rng.normal(0, 10, n),
        "price_momentum_3m": rng.normal(2, 15, n),
        "price_momentum_6m": rng.normal(5, 20, n),
    })
    return df


# ---------------------------------------------------------------------------
# Import functions under test
# ---------------------------------------------------------------------------
import importlib.util

_script_path = str(PROJECT_ROOT / "scripts" / "07b_cross_sectional_validation.py")
_spec = importlib.util.spec_from_file_location("cross_sectional_validation", _script_path)
mod = importlib.util.module_from_spec(_spec)

# Prevent the script from executing main() or writing files at import time
# by temporarily overriding __name__
mod.__name__ = "cross_sectional_validation"
_spec.loader.exec_module(mod)

compute_ic_table = mod.compute_ic_table
compute_quintile_returns = mod.compute_quintile_returns
compute_quintile_spreads = mod.compute_quintile_spreads
bootstrap_rank_stability = mod.bootstrap_rank_stability
kruskal_wallis_quintiles = mod.kruskal_wallis_quintiles
build_summary_table = mod.build_summary_table


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_compute_ic_table():
    df = _make_synthetic_df()
    ic = compute_ic_table(df)
    assert isinstance(ic, pd.DataFrame), "IC table should be a DataFrame"
    assert len(ic) > 0, "IC table should not be empty"
    assert "ic_spearman" in ic.columns, "IC table must have ic_spearman column"
    assert all(ic["ic_spearman"].abs() <= 1.0), "IC values must be in [-1, 1]"
    print("  [PASS] test_compute_ic_table")


def test_compute_quintile_returns():
    df = _make_synthetic_df()
    q = compute_quintile_returns(df)
    assert isinstance(q, pd.DataFrame), "Quintile table should be a DataFrame"
    assert len(q) > 0, "Quintile table should not be empty"
    assert "mean_return" in q.columns, "Must have mean_return column"
    assert "quintile" in q.columns, "Must have quintile column"
    print("  [PASS] test_compute_quintile_returns")


def test_compute_quintile_spreads():
    df = _make_synthetic_df()
    q = compute_quintile_returns(df)
    spreads = compute_quintile_spreads(q)
    assert isinstance(spreads, pd.DataFrame), "Spread table should be a DataFrame"
    assert len(spreads) > 0, "Spread table should not be empty"
    assert "spread" in spreads.columns, "Must have spread column"
    print("  [PASS] test_compute_quintile_spreads")


def test_bootstrap_rank_stability():
    df = _make_synthetic_df()
    boot = bootstrap_rank_stability(df, n_bootstrap=20, seed=99)
    assert isinstance(boot, pd.DataFrame), "Bootstrap result should be a DataFrame"
    assert len(boot) > 0, "Bootstrap result should not be empty"
    tau_row = boot[boot["metric"] == "kendall_tau_mean"]
    assert len(tau_row) == 1, "Should have exactly one kendall_tau_mean row"
    tau = float(tau_row["value"].iloc[0])
    assert -1 <= tau <= 1, f"Kendall tau must be in [-1, 1], got {tau}"
    print("  [PASS] test_bootstrap_rank_stability")


def test_kruskal_wallis():
    df = _make_synthetic_df()
    kw = kruskal_wallis_quintiles(df)
    assert isinstance(kw, pd.DataFrame), "KW table should be a DataFrame"
    assert "kruskal_H" in kw.columns, "Must have kruskal_H column"
    print("  [PASS] test_kruskal_wallis")



def test_build_summary_table():
    df = _make_synthetic_df()
    ic = compute_ic_table(df)
    q = compute_quintile_returns(df)
    spreads = compute_quintile_spreads(q)
    kw = kruskal_wallis_quintiles(df)
    summary = build_summary_table(ic, spreads, kw)
    assert isinstance(summary, pd.DataFrame), "Summary should be a DataFrame"
    assert len(summary) > 0, "Summary should not be empty"
    assert "ic_spearman" in summary.columns, "Summary must have ic_spearman"
    assert "spread" in summary.columns, "Summary must have spread"
    print("  [PASS] test_build_summary_table")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 50)
    print("PREDICTIVE VALIDATION SMOKE TESTS")
    print("=" * 50)
    tests = [
        test_compute_ic_table,
        test_compute_quintile_returns,
        test_compute_quintile_spreads,
        test_bootstrap_rank_stability,
        test_kruskal_wallis,
        test_build_summary_table,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed out of {len(tests)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
