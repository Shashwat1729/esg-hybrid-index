"""
Master Pipeline: Run All Steps
================================
Executes the complete analysis pipeline in order:
  01. Download data
  02. Clean data
  03. Build index
  04. Statistical tests (75+ tables)
  05. Weight sensitivity & optimization
  06. Benchmark comparison (multi-horizon, alpha/beta)
  08. Advanced analysis (PCA, clustering, bootstrap, regime)
  07. Visualizations (30 figures)
  09. Generate research summary report (22 sections)
  10. ESG benchmarking & external provider validation
  11. Profile justification analysis
  15. Robustness analysis: high-cap generalization
  16. Financial validation (turnover, capacity, transaction costs)
  17. Proxy validation (provenance audit & alternative proxies)
  18. Sector-based cross-validation (leave-one-sector-out)
  19. Synthetic sensitivity (noise injection & factor dropout)
  20. Subsampling stability (rolling-window & bootstrap stability)
  13. Compute summary statistics
  14. Pipeline health checks (final validation)

Note: Step 08 runs before 07 so that advanced tables are available for figures.

Usage: python scripts/run_all.py
       python scripts/run_all.py --skip-download  (skip data download if already done)
"""

import sys
import os
import time
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

SCRIPTS = [
    ("01_download_data.py", "Download data from Yahoo Finance, SEC EDGAR, NSE"),
    ("02_clean_data.py", "Clean, standardize, and impute missing data"),
    ("03_build_index.py", "Build multi-factor composite index"),
    ("04_statistical_tests.py", "Run comprehensive statistical tests"),
    ("05_weight_sensitivity.py", "Weight sensitivity and optimization analysis"),
    ("06_benchmark_comparison.py", "Compare with benchmark indices"),
    ("07b_cross_sectional_validation.py", "Cross-sectional validation (IC, quintiles, bootstrap, circularity check)"),
    ("08_advanced_analysis.py", "PCA, clustering, bootstrap, efficient frontier"),
    ("07_visualizations.py", "Generate all research figures"),
    ("09_generate_report.py", "Compile summary report and key findings"),
    ("10_esg_benchmarking.py", "ESG benchmarking & external provider validation"),
    ("11_profile_justification.py", "Profile justification analysis"),
    ("15_robustness_highcap.py", "Robustness analysis: high-cap generalization"),
    ("16_financial_validation.py", "Financial validation: turnover, capacity, transaction costs"),
    ("17_proxy_validation.py", "Proxy validation: provenance audit & alternative proxies"),
    ("18_sector_cv.py", "Sector cross-validation: leave-one-sector-out stability"),
    ("19_synthetic_sensitivity.py", "Synthetic sensitivity: noise injection & factor dropout"),
    ("20_subsampling_stability.py", "Subsampling stability: rolling-window & bootstrap stability"),
    ("13_compute_summaries.py", "Compute summary statistics across all analyses"),
    ("14_run_checks.py", "Pipeline health checks and final validation"),
]


def run_script(script_name, description):
    """Run a script and capture output."""
    script_path = PROJECT_ROOT / "scripts" / script_name
    print(f"\n{'='*70}")
    print(f"Running: {script_name}")
    print(f"  {description}")
    print(f"{'='*70}")
    start = time.time()

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=False,
    )

    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"\n  [OK] {script_name} completed in {elapsed:.1f}s")
    else:
        print(f"\n  [ERROR] {script_name} failed (exit code {result.returncode})")
        return False
    return True


def main():
    skip_download = "--skip-download" in sys.argv

    print("=" * 70)
    print("MULTI-FACTOR ESG-INTEGRATED INVESTMENT INDEX")
    print("Complete Analysis Pipeline")
    print("=" * 70)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Skip download: {skip_download}")

    # Ensure directories exist
    for d in ["data/raw", "data/processed", "reports/tables", "reports/figures"]:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)

    total_start = time.time()
    success_count = 0
    total_count = 0

    for script, desc in SCRIPTS:
        if skip_download and script == "01_download_data.py":
            print(f"\n  [SKIP] {script} (--skip-download)")
            continue

        total_count += 1
        ok = run_script(script, desc)
        if ok:
            success_count += 1
        else:
            print(f"\n  [WARN] Continuing despite error in {script}...")

    total_elapsed = time.time() - total_start

    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE")
    print(f"  Successful: {success_count}/{total_count} scripts")
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*70}")
    print(f"\nResults:")
    print(f"  Data:    data/processed/indexed_data.csv")
    print(f"  Tables:  reports/tables/")
    print(f"  Figures: reports/figures/")
    print(f"  Report:  reports/research_summary.txt")


if __name__ == "__main__":
    main()
