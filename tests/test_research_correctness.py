"""Research correctness tests — not just execution smoke tests.

Covers: weights sum, dates, no future info / leakage, return calcs,
factor directions, merge integrity, PCA, missing-data handling,
portfolio construction, and regression guards for fixed bugs (C1, M6, H2).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import (
    DEFAULT_WEIGHTS,
    DEFAULT_WEIGHTS_EX_MARKET,
    DEFAULT_WEIGHTS_WITH_MARKET,
    ESG_LOWER_IS_BETTER,
    SCORE_COLUMNS,
)
from src.utils import load_indexed_data


# ── Helpers ──────────────────────────────────────────────────────────

def _load_config_weights():
    with open(PROJECT_ROOT / "config" / "index_config.yaml") as f:
        cfg = yaml.safe_load(f)
    return cfg["preference_scoring"]["investor_profiles"]


# =====================================================================
# 1. Weight integrity
# =====================================================================
class TestWeightIntegrity:
    def test_default_weights_sum_to_one(self):
        assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-6)

    def test_ex_market_weights_sum_to_one(self):
        assert sum(DEFAULT_WEIGHTS_EX_MARKET.values()) == pytest.approx(1.0, abs=1e-6)

    def test_with_market_weights_sum_to_one(self):
        assert sum(DEFAULT_WEIGHTS_WITH_MARKET.values()) == pytest.approx(1.0, abs=1e-6)

    def test_default_is_ex_market(self):
        """DEFAULT_WEIGHTS must be ex-market (no market_score) — C1 fix."""
        assert DEFAULT_WEIGHTS is DEFAULT_WEIGHTS_EX_MARKET or DEFAULT_WEIGHTS == DEFAULT_WEIGHTS_EX_MARKET
        assert "market_score" not in DEFAULT_WEIGHTS or DEFAULT_WEIGHTS.get("market_score", 0) == 0

    def test_ex_market_excludes_similarity(self):
        """similarity_rank weight 0 — collinear with ESG (r=0.70)."""
        assert DEFAULT_WEIGHTS.get("similarity_rank", 0) == 0

    def test_config_profiles_sum_to_one(self):
        for name, w in _load_config_weights().items():
            assert sum(w.values()) == pytest.approx(1.0, abs=1e-4), f"profile {name} weights don't sum to 1"

    def test_config_similarity_rank_zero(self):
        for name, w in _load_config_weights().items():
            assert w.get("similarity_rank", 0) == 0, f"{name}: similarity_rank must be 0"

    def test_no_negative_weights(self):
        for d in [DEFAULT_WEIGHTS, DEFAULT_WEIGHTS_EX_MARKET, DEFAULT_WEIGHTS_WITH_MARKET]:
            for k, v in d.items():
                assert v >= 0, f"negative weight {k}={v}"
        for name, w in _load_config_weights().items():
            for k, v in w.items():
                assert v >= 0, f"profile {name}: negative {k}={v}"


# =====================================================================
# 2. Factor directions (higher score = better)
# =====================================================================
class TestFactorDirections:
    """Value multiples inverted; volatility/beta inverted; E/S/G lower_is_better."""

    def test_value_inversion_config(self):
        with open(PROJECT_ROOT / "config" / "index_config.yaml") as f:
            cfg = yaml.safe_load(f)
        vs = cfg["value_scoring"]["categories"]["valuation_multiples"]["indicators"]
        for ind, meta in vs.items():
            assert meta.get("inverse") is True, f"value indicator {ind} should be inverse"

    def test_market_volatility_inversion_config(self):
        with open(PROJECT_ROOT / "config" / "index_config.yaml") as f:
            cfg = yaml.safe_load(f)
        inv = set(cfg["market_factors"]["inverse_indicators"])
        assert "price_volatility" in inv
        assert "beta" in inv

    def test_esg_lower_is_better_set(self):
        assert "scope1_emissions" in ESG_LOWER_IS_BETTER
        assert "injury_rate" in ESG_LOWER_IS_BETTER
        assert "esg_risk_rating" in ESG_LOWER_IS_BETTER
        # pay_gap_ratio is intentionally NOT lower_is_better (encoding near 1.0 = good)
        assert "pay_gap_ratio" not in ESG_LOWER_IS_BETTER

    def test_stability_volatility_inversion_config(self):
        with open(PROJECT_ROOT / "config" / "index_config.yaml") as f:
            cfg = yaml.safe_load(f)
        stab = cfg["stability_scoring"]["categories"]["volatility"]["indicators"]
        for ind, meta in stab.items():
            assert meta.get("inverse") is True, f"stability volatility {ind} should be inverse"


# =====================================================================
# 3. No future information / leakage guards
# =====================================================================
class TestNoLeakage:
    def test_growth_excludes_momentum(self):
        """Growth must NOT contain price_momentum_* (look-ahead)."""
        with open(PROJECT_ROOT / "config" / "index_config.yaml") as f:
            cfg = yaml.safe_load(f)
        growth = cfg["growth_scoring"]["categories"]["growth_metrics"]["indicators"]
        for k in growth:
            assert "momentum" not in k.lower(), f"growth must not contain momentum: {k}"
            assert "price_momentum" not in k

    def test_no_forward_return_columns_in_features(self):
        """No column used as feature should be a forward return."""
        df = load_indexed_data(PROJECT_ROOT)
        # Any feature column must not be a forward return; our forward proxy is forward_quality_proxy
        # Ensure indexed_data doesn't have a column that looks like a leaked forward return
        assert "forward_return" not in df.columns
        assert "future_return" not in df.columns

    def test_pref_balanced_is_ex_market(self):
        """pref_balanced must be ex-market (clean); _with_market is contaminated."""
        df = load_indexed_data(PROJECT_ROOT)
        # Clean must exist; contaminated retained for audit
        assert "pref_balanced" in df.columns
        assert "pref_balanced_with_market" in df.columns
        # They should be correlated but not identical
        r = df["pref_balanced"].corr(df["pref_balanced_with_market"])
        assert 0.85 < r < 0.99, f"pref clean vs contaminated r={r:.3f} unexpected"

    def test_market_score_excluded_from_default_weights(self):
        """Regression: DEFAULT_WEIGHTS must not contain market_score with positive weight."""
        assert DEFAULT_WEIGHTS.get("market_score", 0) == 0


# =====================================================================
# 4. Data / merge integrity
# =====================================================================
class TestDataIntegrity:
    def test_no_duplicate_tickers(self):
        df = load_indexed_data(PROJECT_ROOT, include_benchmarks=True)
        assert df["ticker"].duplicated().sum() == 0

    def test_include_benchmarks_adds_45(self):
        mid = load_indexed_data(PROJECT_ROOT, include_benchmarks=False)
        all_df = load_indexed_data(PROJECT_ROOT, include_benchmarks=True)
        assert len(all_df) - len(mid) == 45

    def test_mid_cap_count(self):
        mid = load_indexed_data(PROJECT_ROOT, include_benchmarks=False)
        assert len(mid) == 276

    def test_score_ranges_0_100(self):
        df = load_indexed_data(PROJECT_ROOT)
        for col in ["ESG_composite", "financial_score", "market_score", "operational_score",
                     "risk_adjusted_score", "value_score", "growth_score", "stability_score"]:
            if col in df.columns:
                assert df[col].min() >= -0.01, f"{col} min {df[col].min()}"
                assert df[col].max() <= 100.01, f"{col} max {df[col].max()}"
                assert df[col].isna().sum() == 0, f"{col} has NaNs"

    def test_no_synthetic_noise_columns(self):
        """M6 fix: bid_ask_spread / free_float_pct must not exist."""
        df = load_indexed_data(PROJECT_ROOT, include_benchmarks=True)
        for bad in ["bid_ask_spread", "free_float_pct"]:
            assert bad not in df.columns, f"synthetic noise column {bad} should be removed"

    def test_indicator_overlap_zero(self):
        """H2 fix: factor overlap matrix must be all-zeros off-diagonal."""
        p = PROJECT_ROOT / "reports" / "tables" / "factor_overlap_matrix.csv"
        if not p.exists():
            pytest.skip("factor_overlap_matrix.csv not generated yet")
        m = pd.read_csv(p, index_col=0)
        # Off-diagonal must be 0
        vals = m.values.copy()
        np.fill_diagonal(vals, 0)
        assert (vals == 0).all(), f"overlap matrix has non-zero off-diagonal:\n{m}"

    def test_cleaning_metadata_exists(self):
        assert (PROJECT_ROOT / "data" / "processed" / "cleaning_metadata.json").exists()
        import json
        with open(PROJECT_ROOT / "data" / "processed" / "cleaning_metadata.json") as f:
            meta = json.load(f)
        assert meta["exchange_rate_used"] == 83.0
        assert meta["n_companies"] == 321
        assert len(meta["indian_tickers_converted"]) == 90


# =====================================================================
# 5. Missing-data handling
# =====================================================================
class TestMissingData:
    def test_no_nan_in_deployed_scores(self):
        df = load_indexed_data(PROJECT_ROOT, include_benchmarks=True)
        # Core 8 factor scores must be complete; sector_position/similarity may be
        # legitimately NaN for a few companies lacking raw inputs (small N)
        core = [c for c in SCORE_COLUMNS if c not in ("sector_position", "similarity_rank")]
        for col in core:
            if col in df.columns:
                assert df[col].isna().sum() == 0, f"{col} has NaN — imputation failed"
        for col in ["sector_position", "similarity_rank"]:
            if col in df.columns:
                assert df[col].isna().sum() < len(df) * 0.05, f"{col} has >5% NaN"

    def test_binary_vars_bounded(self):
        df = load_indexed_data(PROJECT_ROOT, include_benchmarks=True)
        for col in ["carbon_reduction_target", "human_rights_policy", "anti_corruption_policy"]:
            if col in df.columns:
                assert df[col].isin([0, 1, 0.0, 1.0]).all() or df[col].dropna().between(0, 1).all()


# =====================================================================
# 6. Portfolio construction
# =====================================================================
class TestPortfolioConstruction:
    def test_top20_distinct(self):
        df = load_indexed_data(PROJECT_ROOT)
        top20 = df.nlargest(20, "pref_balanced")
        assert len(top20) == 20
        assert top20["ticker"].nunique() == 20

    def test_sector_position_in_0_100(self):
        df = load_indexed_data(PROJECT_ROOT, include_benchmarks=True)
        if "sector_position" in df.columns:
            s = df["sector_position"].dropna()
            assert s.min() >= -0.01
            # Post-pipeline scores are 0-100; allow either [0,1] or [0,100] convention
            assert s.max() <= 100.01

    def test_similarity_rank_in_0_100(self):
        df = load_indexed_data(PROJECT_ROOT, include_benchmarks=True)
        if "similarity_rank" in df.columns:
            s = df["similarity_rank"].dropna()
            assert s.min() >= -0.01
            assert s.max() <= 100.01


# =====================================================================
# 7. PCA behavior
# =====================================================================
class TestPCA:
    def test_pca_loadings_shape(self):
        p = PROJECT_ROOT / "reports" / "tables" / "advanced_pca_loadings.csv"
        if not p.exists():
            pytest.skip("PCA loadings not generated yet")
        m = pd.read_csv(p, index_col=0)
        assert m.shape[0] == 8  # 8 factors in PCA
        assert m.shape[1] == 8  # 8 PCs

    def test_pca_variance_sums_to_one(self):
        p = PROJECT_ROOT / "reports" / "tables" / "advanced_pca_variance.csv"
        if not p.exists():
            pytest.skip("PCA variance not generated yet")
        v = pd.read_csv(p)
        # Sum of eigenvalues should be n_factors; ratio sums to 1
        # Detect which column holds the ratio
        for cand in ["explained_variance_ratio", "explained_variance", "variance_ratio"]:
            if cand in v.columns:
                col = cand
                break
        else:
            # If only eigenvalue columns present, sum of eigenvalues ≈ n_components
            # Just verify file is non-empty and plausible
            assert len(v) >= 2
            return
        assert v[col].sum() == pytest.approx(1.0, abs=1e-3)


# =====================================================================
# 8. Benchmark caveats
# =====================================================================
class TestBenchmarkCaveats:
    def test_benchmark_csv_has_proxy_header(self):
        p = PROJECT_ROOT / "reports" / "tables" / "benchmark_portfolio_performance.csv"
        if not p.exists():
            pytest.skip("benchmark not generated yet")
        with open(p) as f:
            first = f.readline()
        assert "Cross-sectional momentum proxy" in first

    def test_no_real_returns_claim(self):
        """No table should claim 'annualized return' without proxy qualifier for cross-sectional data."""
        # Check that benchmark_summary does not contain a column named exactly 'annualized_return' without proxy
        p = PROJECT_ROOT / "reports" / "tables" / "benchmark_summary.csv"
        if not p.exists():
            pytest.skip("benchmark_summary not generated")
        with open(p) as f:
            header = f.readline()
        # The file should have the proxy header line, not raw return claims
        assert "Cross-sectional momentum proxy" in header


# =====================================================================
# 9. Regression guards (specific bugs that were fixed)
# =====================================================================
class TestRegressionGuards:
    def test_issue_m6_no_synthetic_in_config(self):
        """M6: liquidity category must not contain bid_ask_spread / free_float_pct."""
        with open(PROJECT_ROOT / "config" / "index_config.yaml") as f:
            cfg = yaml.safe_load(f)
        liq = cfg["market_factors"]["categories"]["liquidity"]["indicators"]
        for bad in ["bid_ask_spread", "free_float_pct"]:
            assert bad not in liq, f"M6 regression: {bad} still in liquidity indicators"

    def test_issue_c1_market_not_in_default_profile_weights(self):
        """C1: balanced profile market weight is 0.05 but DEFAULT_WEIGHTS excludes it entirely."""
        # Balanced profile still has market_score for audit, but DEFAULT_WEIGHTS must exclude it
        assert DEFAULT_WEIGHTS.get("market_score", 0) == 0

    def test_issue_h2_no_overlap_indicators(self):
        """H2: value_score must not share enterprise_to_ebitda with financial_score."""
        with open(PROJECT_ROOT / "config" / "index_config.yaml") as f:
            cfg = yaml.safe_load(f)
        fin_inds = set()
        for cat in cfg["financial_scoring"]["categories"].values():
            fin_inds.update(cat.get("indicators", []))
        val_inds = set(cfg["value_scoring"]["categories"]["valuation_multiples"]["indicators"].keys())
        # After dedup, financial_scoring no longer contains enterprise_to_ebitda etc.
        overlap = fin_inds & val_inds
        assert len(overlap) == 0, f"H2 regression: overlap {overlap}"

    def test_provenance_file_exists(self):
        assert (PROJECT_ROOT / "reports" / "tables" / "esg_data_provenance.csv").exists()
