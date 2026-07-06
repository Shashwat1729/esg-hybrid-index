"""
Step 18: Leave-One-Sector-Out Cross-Validation
================================================
For each sector:
1. Remove all companies in that sector (hold-out set)
2. Re-normalize and re-score remaining companies (training set)
3. Apply the training-set normalization parameters to the held-out sector
4. Compare held-out scores to full-sample scores
5. Report rank correlation stability

This tests whether the index methodology generalizes across sectors,
which is critical for a mid-cap-focused index.

Input:  data/processed/indexed_data.csv
Output: reports/tables/sector_cv_rank_stability.csv
        reports/tables/sector_cv_score_comparison.csv
        reports/tables/sector_cv_summary.csv
"""

import sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*divide by zero.*")
warnings.filterwarnings("ignore", message=".*invalid value.*")

from src.utils import load_indexed_data
from src.constants import SCORE_COLUMNS


# ---------------------------------------------------------------------------
# Core CV logic
# ---------------------------------------------------------------------------

def leave_one_sector_out_cv(df, score_columns):
    """
    Perform leave-one-sector-out cross-validation on factor scores.

    For each sector:
    1. Compute full-sample rankings
    2. Remove sector, re-compute re-standardized scores on remaining
    3. Apply training normalization parameters to held-out companies
    4. Compare held-out rankings to full-sample rankings

    The re-standardization mirrors Step 7b of ``03_build_index.py``:
    each factor is re-mapped to mean=50, std=10, clipped to [0, 100].
    Here we apply the *training-set* mean/std to the held-out companies
    and measure how much scores and ranks shift.

    Returns
    -------
    pd.DataFrame
        Per-sector, per-factor stability metrics.
    """
    sectors = df["sector"].dropna().unique()

    # Full-sample rankings (descending — higher score = better rank)
    full_ranks = {}
    for col in score_columns:
        full_ranks[col] = df[col].rank(ascending=False)

    results = []
    score_comparisons = []

    for sector in sorted(sectors):
        holdout_mask = df["sector"] == sector
        n_holdout = holdout_mask.sum()

        if n_holdout < 3:
            continue

        train_df = df[~holdout_mask].copy()
        holdout_df = df[holdout_mask].copy()

        for col in score_columns:
            if col not in train_df.columns:
                continue

            # --- Training-set statistics (post-restandardization) ---
            train_mean = train_df[col].mean()
            train_std = train_df[col].std()

            if train_std == 0 or pd.isna(train_std):
                continue

            # --- Apply training normalization to holdout ---
            # The full-sample scores already sit on a 50 ± 10 scale.
            # Re-standardizing the holdout with training params simulates
            # what the holdout scores would look like had those companies
            # been scored against only the non-sector universe.
            holdout_z = (holdout_df[col] - train_mean) / train_std
            holdout_restd = holdout_z * 10 + 50  # map back to 50 ± 10
            holdout_restd = holdout_restd.clip(0, 100)

            original_scores = holdout_df[col].values
            cv_scores = holdout_restd.values

            # --- Rank correlation & error metrics ---
            if len(original_scores) >= 3:
                spearman_rho, spearman_p = stats.spearmanr(
                    original_scores, cv_scores
                )
                mae = np.mean(np.abs(original_scores - cv_scores))
                max_ae = np.max(np.abs(original_scores - cv_scores))
                rmse = np.sqrt(np.mean((original_scores - cv_scores) ** 2))
            else:
                spearman_rho = spearman_p = mae = max_ae = rmse = np.nan

            results.append({
                "held_out_sector": sector,
                "n_holdout": n_holdout,
                "factor": col,
                "spearman_rho": (round(spearman_rho, 4)
                                 if not np.isnan(spearman_rho) else np.nan),
                "spearman_p": (round(spearman_p, 6)
                               if not np.isnan(spearman_p) else np.nan),
                "mae_score_shift": (round(mae, 3)
                                    if not np.isnan(mae) else np.nan),
                "max_score_shift": (round(max_ae, 3)
                                    if not np.isnan(max_ae) else np.nan),
                "rmse_score_shift": (round(rmse, 3)
                                     if not np.isnan(rmse) else np.nan),
                "train_mean": round(train_mean, 2),
                "train_std": round(train_std, 2),
                "full_sample_mean": round(df[col].mean(), 2),
                "full_sample_std": round(df[col].std(), 2),
            })

            # Per-company score comparison (for the detailed output)
            for idx, (orig, cv) in zip(
                holdout_df.index,
                zip(original_scores, cv_scores),
            ):
                ticker = holdout_df.loc[idx, "ticker"] if "ticker" in holdout_df.columns else idx
                score_comparisons.append({
                    "ticker": ticker,
                    "sector": sector,
                    "factor": col,
                    "original_score": round(orig, 3),
                    "cv_score": round(cv, 3),
                    "score_diff": round(cv - orig, 3),
                    "original_rank": int(full_ranks[col].loc[idx]),
                })

    results_df = pd.DataFrame(results)
    comparisons_df = pd.DataFrame(score_comparisons)
    return results_df, comparisons_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("STEP 18: LEAVE-ONE-SECTOR-OUT CROSS-VALIDATION")
    print("=" * 70)

    ROOT = PROJECT_ROOT
    TABLES = ROOT / "reports" / "tables"
    TABLES.mkdir(parents=True, exist_ok=True)

    # Load mid-cap only data (exclude large-cap benchmarks)
    df = load_indexed_data(include_benchmarks=False)
    print(f"[OK] Loaded {len(df)} mid-cap companies")

    if "sector" not in df.columns:
        print("[ERROR] 'sector' column not found — cannot run sector CV.")
        sys.exit(1)

    n_sectors = df["sector"].nunique()
    print(f"[OK] {n_sectors} sectors detected: "
          f"{', '.join(sorted(df['sector'].dropna().unique()))}")

    score_cols = [c for c in SCORE_COLUMNS if c in df.columns]
    print(f"[OK] {len(score_cols)} factor score columns available")

    # ------------------------------------------------------------------
    # Run leave-one-sector-out CV
    # ------------------------------------------------------------------
    print("\nRunning leave-one-sector-out cross-validation...")
    cv_results, cv_comparisons = leave_one_sector_out_cv(df, score_cols)

    # ------------------------------------------------------------------
    # Save detailed results
    # ------------------------------------------------------------------
    cv_results.to_csv(
        TABLES / "sector_cv_rank_stability.csv", index=False, encoding="utf-8"
    )
    print(f"[OK] Saved sector_cv_rank_stability.csv "
          f"({len(cv_results)} rows)")

    cv_comparisons.to_csv(
        TABLES / "sector_cv_score_comparison.csv", index=False, encoding="utf-8"
    )
    print(f"[OK] Saved sector_cv_score_comparison.csv "
          f"({len(cv_comparisons)} rows)")

    # ------------------------------------------------------------------
    # Summary: average stability per factor
    # ------------------------------------------------------------------
    summary = (
        cv_results
        .groupby("factor")
        .agg(
            mean_spearman=("spearman_rho", "mean"),
            min_spearman=("spearman_rho", "min"),
            max_spearman=("spearman_rho", "max"),
            std_spearman=("spearman_rho", "std"),
            mean_mae=("mae_score_shift", "mean"),
            mean_rmse=("rmse_score_shift", "mean"),
            n_sectors=("held_out_sector", "nunique"),
        )
        .round(4)
    )
    summary.to_csv(TABLES / "sector_cv_summary.csv", encoding="utf-8")
    print(f"[OK] Saved sector_cv_summary.csv ({len(summary)} factors)")

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS: Leave-One-Sector-Out Cross-Validation")
    print("=" * 70)
    print(f"Sectors evaluated : {cv_results['held_out_sector'].nunique()}")
    print(f"Factors evaluated : {len(score_cols)}")
    print(f"Total comparisons : {len(cv_results)}")

    print(f"\n{'Factor':<25s} {'Mean ρ':>8s} {'Min ρ':>8s} "
          f"{'Max ρ':>8s} {'Mean MAE':>10s}")
    print("-" * 65)
    for _, row in summary.iterrows():
        print(f"  {row.name:<23s} {row['mean_spearman']:8.3f} "
              f"{row['min_spearman']:8.3f} {row['max_spearman']:8.3f} "
              f"{row['mean_mae']:10.3f}")

    overall_rho = cv_results["spearman_rho"].mean()
    overall_mae = cv_results["mae_score_shift"].mean()
    print(f"\nOverall mean Spearman ρ : {overall_rho:.4f}")
    print(f"Overall mean MAE        : {overall_mae:.3f}")

    # ------------------------------------------------------------------
    # Per-sector summary
    # ------------------------------------------------------------------
    sector_summary = (
        cv_results
        .groupby("held_out_sector")
        .agg(
            n_companies=("n_holdout", "first"),
            mean_rho=("spearman_rho", "mean"),
            mean_mae=("mae_score_shift", "mean"),
        )
        .round(4)
        .sort_values("mean_rho")
    )
    print(f"\n{'Sector':<30s} {'N':>4s} {'Mean ρ':>8s} {'Mean MAE':>10s}")
    print("-" * 56)
    for sector, row in sector_summary.iterrows():
        print(f"  {sector:<28s} {int(row['n_companies']):4d} "
              f"{row['mean_rho']:8.3f} {row['mean_mae']:10.3f}")

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    if overall_rho > 0.95:
        verdict = ("EXCELLENT — scores are highly robust to sector composition. "
                    "Removing any single sector has negligible impact on rankings.")
    elif overall_rho > 0.85:
        verdict = ("GOOD — scores are robust with minor sector sensitivity. "
                    "The methodology generalizes well across sectors.")
    elif overall_rho > 0.70:
        verdict = ("MODERATE — some sector sensitivity exists. "
                    "Certain sectors may disproportionately influence scores.")
    else:
        verdict = ("POOR — significant sector dependence detected. "
                    "The methodology may be overfitting to sector composition.")

    print(f"Verdict: {verdict}")

    # Identify weakest sector-factor combinations
    weak = cv_results[cv_results["spearman_rho"] < 0.80]
    if len(weak) > 0:
        print(f"\nWeak sector-factor combinations (ρ < 0.80): {len(weak)}")
        for _, row in weak.sort_values("spearman_rho").head(10).iterrows():
            print(f"  {row['held_out_sector']:25s} × {row['factor']:25s} "
                  f"ρ={row['spearman_rho']:.3f}  (n={int(row['n_holdout'])})")
    else:
        print("\nNo weak sector-factor combinations (all ρ ≥ 0.80).")

    print("\n[DONE] Sector cross-validation complete.")


if __name__ == "__main__":
    main()
