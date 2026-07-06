"""
Step 20: Subsampling Rank Stability Analysis
==============================================
Tests whether company rankings are stable under random subsampling.
For each of 500 iterations:
  1. Randomly subsample 80 % of companies
  2. Re-normalise all factor scores on the subsample
  3. Compute Kendall tau between full-sample and subsample rankings
  4. Track top-20 membership stability (Jaccard-style overlap)

This provides evidence that the index is not overly sensitive to
the specific composition of the mid-cap universe.

Input:  data/processed/indexed_data.csv
Outputs:
  reports/tables/subsampling_stability_summary.csv
  reports/tables/subsampling_stability_per_factor.csv
  reports/tables/subsampling_stability_top20.csv
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*divide by zero.*")
warnings.filterwarnings("ignore", message=".*invalid value.*")

from src.utils import load_indexed_data
from src.constants import SCORE_COLUMNS, RANDOM_SEED

TABLES = PROJECT_ROOT / "reports" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_ITERATIONS = 500
SUBSAMPLE_FRAC = 0.80
SEED = RANDOM_SEED
TOP_K = 20


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def subsampling_stability(
    df: pd.DataFrame,
    score_cols: list[str],
    n_iterations: int = N_ITERATIONS,
    subsample_frac: float = SUBSAMPLE_FRAC,
    top_k: int = TOP_K,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run random-subsampling stability analysis.

    Parameters
    ----------
    df : DataFrame
        Indexed data with factor score columns.
    score_cols : list[str]
        Score columns to evaluate.
    n_iterations : int
        Number of bootstrap-style subsamples.
    subsample_frac : float
        Fraction of companies retained per subsample.
    top_k : int
        Size of the "top-K" set for overlap analysis.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    per_factor : DataFrame
        Per-factor Kendall-tau and top-K overlap statistics.
    top20 : DataFrame
        Per-company frequency of appearing in the top-K across iterations.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    n_sub = int(n * subsample_frac)

    # ------------------------------------------------------------------
    # Full-sample reference rankings & top-K sets
    # ------------------------------------------------------------------
    full_ranks: dict[str, pd.Series] = {}
    full_topk: dict[str, set] = {}
    for col in score_cols:
        full_ranks[col] = df[col].rank(ascending=False)
        full_topk[col] = set(df.nlargest(top_k, col).index)

    # ------------------------------------------------------------------
    # Iteration storage
    # ------------------------------------------------------------------
    tau_results: dict[str, list[float]] = {col: [] for col in score_cols}
    topk_overlap: dict[str, list[float]] = {col: [] for col in score_cols}
    # Count how often each company lands in the top-K per factor
    topk_counts: dict[str, dict[int, int]] = {
        col: {i: 0 for i in df.index} for col in score_cols
    }

    for _ in range(n_iterations):
        idx = rng.choice(df.index, size=n_sub, replace=False)
        sub_df = df.loc[idx].copy()

        for col in score_cols:
            # Re-normalise on the subsample (z-score → 0-100 scale)
            sub_mean = sub_df[col].mean()
            sub_std = sub_df[col].std()
            if sub_std == 0 or pd.isna(sub_std):
                continue

            renorm = (sub_df[col] - sub_mean) / sub_std * 10 + 50
            renorm = renorm.clip(0, 100)

            # Subsample rank
            sub_rank = renorm.rank(ascending=False)

            # Kendall tau vs. full-sample ranks (same companies only)
            common = sub_rank.index
            full_sub = full_ranks[col].loc[common]
            if len(common) >= 10:
                tau, _ = kendalltau(full_sub, sub_rank)
                tau_results[col].append(tau)

            # Top-K overlap (Jaccard-style against full top-K)
            sub_topk = set(sub_df.nlargest(top_k, col).index)
            available = full_topk[col] & set(common)
            overlap = len(available & sub_topk)
            max_possible = min(top_k, len(available))
            jaccard = overlap / max_possible if max_possible > 0 else 0.0
            topk_overlap[col].append(jaccard)

            # Track individual membership
            for company_idx in sub_topk:
                topk_counts[col][company_idx] += 1

    # ------------------------------------------------------------------
    # Compile per-factor results
    # ------------------------------------------------------------------
    rows = []
    for col in score_cols:
        taus = tau_results[col]
        overlaps = topk_overlap[col]
        if not taus:
            continue
        rows.append({
            "factor": col,
            "mean_kendall_tau": round(np.mean(taus), 4),
            "std_kendall_tau": round(np.std(taus), 4),
            "p5_kendall_tau": round(np.percentile(taus, 5), 4),
            "p95_kendall_tau": round(np.percentile(taus, 95), 4),
            "mean_top20_overlap": round(np.mean(overlaps), 4),
            "std_top20_overlap": round(np.std(overlaps), 4),
            "n_iterations": len(taus),
        })
    per_factor = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Top-K membership frequency table
    # ------------------------------------------------------------------
    topk_rows = []
    for col in score_cols:
        for company_idx, count in topk_counts[col].items():
            if count > 0:
                topk_rows.append({
                    "factor": col,
                    "company_index": company_idx,
                    "ticker": df.loc[company_idx, "ticker"]
                        if "ticker" in df.columns else company_idx,
                    "times_in_top20": count,
                    "pct_iterations": round(count / n_iterations * 100, 1),
                })
    top20 = (
        pd.DataFrame(topk_rows)
        .sort_values(["factor", "times_in_top20"], ascending=[True, False])
        .reset_index(drop=True)
    )

    return per_factor, top20


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load mid-cap universe only
    df = load_indexed_data(PROJECT_ROOT, include_benchmarks=False)

    # Identify available score columns
    score_cols = [c for c in SCORE_COLUMNS if c in df.columns]

    # Also include preference-profile composite scores if present
    pref_cols = sorted(c for c in df.columns if c.startswith("pref_"))
    score_cols_all = score_cols + pref_cols

    print(f"Running subsampling stability "
          f"({N_ITERATIONS} iterations, {SUBSAMPLE_FRAC:.0%} subsample)...")
    print(f"  Companies : {len(df)}")
    print(f"  Factors   : {len(score_cols_all)}")
    print()

    per_factor, top20 = subsampling_stability(
        df, score_cols_all,
        n_iterations=N_ITERATIONS,
        subsample_frac=SUBSAMPLE_FRAC,
    )

    # ---- Save outputs ------------------------------------------------
    per_factor.to_csv(TABLES / "subsampling_stability_per_factor.csv", index=False)
    top20.to_csv(TABLES / "subsampling_stability_top20.csv", index=False)

    summary = pd.DataFrame([{
        "n_companies": len(df),
        "n_iterations": N_ITERATIONS,
        "subsample_fraction": SUBSAMPLE_FRAC,
        "overall_mean_tau": round(per_factor["mean_kendall_tau"].mean(), 4),
        "overall_min_tau": round(per_factor["mean_kendall_tau"].min(), 4),
        "overall_mean_top20_overlap": round(
            per_factor["mean_top20_overlap"].mean(), 4
        ),
    }])
    summary.to_csv(TABLES / "subsampling_stability_summary.csv", index=False)

    # ---- Console report ----------------------------------------------
    print("=" * 60)
    print("  Subsampling Stability Results")
    print("=" * 60)
    for row in per_factor.itertuples():
        print(
            f"  {row.factor:30s}  tau={row.mean_kendall_tau:.3f} "
            f"[{row.p5_kendall_tau:.3f}, {row.p95_kendall_tau:.3f}]  "
            f"top-{TOP_K} overlap={row.mean_top20_overlap:.1%}"
        )

    overall_tau = per_factor["mean_kendall_tau"].mean()
    print()
    if overall_tau > 0.90:
        verdict = "HIGHLY STABLE"
    elif overall_tau > 0.75:
        verdict = "MODERATELY STABLE"
    else:
        verdict = "UNSTABLE — investigate further"
    print(f"VERDICT: Rankings are {verdict} under subsampling "
          f"(mean tau = {overall_tau:.3f})")

    print()
    print(f"[OK] Saved  {TABLES / 'subsampling_stability_per_factor.csv'}")
    print(f"[OK] Saved  {TABLES / 'subsampling_stability_top20.csv'}")
    print(f"[OK] Saved  {TABLES / 'subsampling_stability_summary.csv'}")


if __name__ == "__main__":
    main()
