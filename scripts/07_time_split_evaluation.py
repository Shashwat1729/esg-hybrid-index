"""
Step 07: Time-Split Evaluation Framework
----------------------------------------
Provides utilities to generate train/holdout splits (fixed, rolling, expanding)
and to evaluate index/portfolio performance across those splits.

This file contains importable functions for testing and a small CLI that
creates synthetic data when --synthetic is passed and runs an example
evaluation saving CSVs/figures into reports/.

Usage (synthetic example):
  python scripts/07_time_split_evaluation.py --synthetic

The module API is intentionally small and testable:
  - generate_splits(panel_df, date_col, holdout_years, train_years, stride, mode)
  - evaluate_splits(panel_df, splits, id_col, return_col, top_n)

"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)


def generate_splits(panel_df, date_col="date", holdout_years=2, train_years=8, stride=1, mode="fixed"):
    """Generate time-splits from a panel DataFrame.

    panel_df: DataFrame with a date column (datetime-like) and an id column.
    mode: 'fixed' (single final train/holdout), 'rolling' (sliding window), 'expanding' (expanding train window).

    Returns a list of dicts: [{'train_start', 'train_end', 'holdout_start', 'holdout_end'}]
    Dates are inclusive/exclusive in the usual pandas slicing sense.
    """
    df = panel_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    years = df[date_col].dt.year.unique()
    years = np.sort(years)
    if len(years) < (train_years + holdout_years):
        raise ValueError("Not enough years in data to generate requested splits")

    splits = []
    max_start = years.max() - holdout_years - train_years + 1
    if mode == "fixed":
        train_end = years.max() - holdout_years
        train_start = train_end - train_years + 1
        splits.append({
            "train_start": int(train_start),
            "train_end": int(train_end),
            "holdout_start": int(train_end + 1),
            "holdout_end": int(train_end + holdout_years),
        })
        return splits

    # rolling or expanding
    for s in range(int(max_start), int(years.max() - train_years + 1) + 1, stride):
        train_start = s
        train_end = s + train_years - 1
        holdout_start = train_end + 1
        holdout_end = train_end + holdout_years
        if holdout_end > years.max():
            break
        if mode == "expanding":
            train_start = years.min()

        splits.append({
            "train_start": int(train_start),
            "train_end": int(train_end),
            "holdout_start": int(holdout_start),
            "holdout_end": int(holdout_end),
        })

    return splits


def evaluate_splits(panel_df, splits, id_col="ticker", date_col="date", return_col="ret", top_n=20):
    """Evaluate each split by selecting top_n by score (if present) or by market cap
    and computing average return, sharpe, pct_positive for the holdout period.

    Expects that `panel_df` has columns for id, date, and return_col. If `score`
    column exists it will use the latest available score in the train period to
    rank companies; otherwise falls back to market cap.
    """
    results = []
    df = panel_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["year"] = df[date_col].dt.year

    for sp in splits:
        train_mask = (df["year"] >= sp["train_start"]) & (df["year"] <= sp["train_end"]) 
        hold_mask = (df["year"] >= sp["holdout_start"]) & (df["year"] <= sp["holdout_end"]) 

        train_df = df[train_mask]
        hold_df = df[hold_mask]

        # derive ranking from last available train-year snapshot
        train_last_year = sp["train_end"]
        train_last = train_df[train_df["year"] == train_last_year]

        if "score" in train_last.columns:
            winners = train_last.nlargest(top_n, "score")[id_col].tolist()
        elif "market_cap" in train_last.columns:
            winners = train_last.nlargest(top_n, "market_cap")[id_col].tolist()
        else:
            winners = train_last[id_col].unique().tolist()[:top_n]

        port_hold = hold_df[hold_df[id_col].isin(winners)]
        rets = port_hold[return_col].dropna()
        avg_ret = float(rets.mean()) if len(rets) > 0 else np.nan
        std_ret = float(rets.std()) if len(rets) > 0 else np.nan
        median_ret = float(rets.median()) if len(rets) > 0 else np.nan
        count_rets = int(len(rets))
        sharpe = float(avg_ret / (std_ret + 1e-10)) if len(rets) > 0 else np.nan
        pct_pos = float((rets > 0).mean() * 100) if len(rets) > 0 else np.nan

        results.append({
            "train_start": sp["train_start"],
            "train_end": sp["train_end"],
            "holdout_start": sp["holdout_start"],
            "holdout_end": sp["holdout_end"],
            "n_winners": len(winners),
            "avg_return": avg_ret,
            "median_return": median_ret,
            "std_return": std_ret,
            "n_holdout_obs": count_rets,
            "sharpe": sharpe,
            "pct_positive": pct_pos,
        })

    res_df = pd.DataFrame(results)
    return res_df


def _plot_results(res_df, out_path=None):
    if res_df.empty:
        return None
    fig, ax1 = plt.subplots(figsize=(7, 3.5))
    x = res_df["holdout_end"].astype(int)
    ax1.plot(x, res_df["sharpe"], marker="o", color="#1f77b4", label="Sharpe")
    ax1.set_xlabel("Holdout end year")
    ax1.set_ylabel("Sharpe (holdout)", color="#1f77b4")
    ax1.tick_params(axis='y', labelcolor="#1f77b4")
    ax2 = ax1.twinx()
    ax2.bar(x, res_df["avg_return"], alpha=0.3, color="#ff7f0e", label="Avg return")
    ax2.set_ylabel("Average return", color="#ff7f0e")
    ax2.tick_params(axis='y', labelcolor="#ff7f0e")
    ax1.set_title("Holdout performance by split")
    ax1.grid(True, alpha=0.25)
    fig.tight_layout()
    if out_path:
        # save both PNG and PDF if path has extension png, also save pdf
        fig.savefig(out_path)
        try:
            out_pdf = Path(str(out_path).rsplit('.', 1)[0] + '.pdf')
            fig.savefig(out_pdf)
        except Exception:
            pass
    plt.close(fig)
    return out_path


def make_synthetic_panel(start_year=2012, end_year=2023, n_tickers=50, seed=42):
    rng = np.random.RandomState(seed)
    years = np.arange(start_year, end_year + 1)
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    rows = []
    for y in years:
        for t in tickers:
            # synthetic market cap and a simple return signal with noise
            mcap = max(1e7, rng.lognormal(10, 1))
            # create a persistent quality 'score' per ticker
            base_score = rng.normal(50, 10)
            # return is somewhat correlated with base_score plus yearly noise
            ret = (base_score - 50) * 0.02 + rng.normal(0, 5)
            rows.append({"date": f"{y}-06-30", "ticker": t, "year": y,
                         "market_cap": mcap, "score": base_score + rng.normal(0, 3),
                         "ret": ret})
    df = pd.DataFrame(rows)
    return df


def main_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true", help="Run example on synthetic data")
    p.add_argument("--train-years", type=int, default=8)
    p.add_argument("--holdout-years", type=int, default=2)
    p.add_argument("--mode", choices=["fixed", "rolling", "expanding"], default="rolling")
    args = p.parse_args()

    if args.synthetic:
        print("[INFO] Generating synthetic panel data...")
        panel = make_synthetic_panel(start_year=2012, end_year=2023, n_tickers=60)
        splits = generate_splits(panel, date_col="date", holdout_years=args.holdout_years,
                                 train_years=args.train_years, stride=1, mode=args.mode)
        print(f"[INFO] Generated {len(splits)} splits")
        res = evaluate_splits(panel, splits, id_col="ticker", date_col="date", return_col="ret", top_n=20)
        out_csv = TABLES / "timesplit_evaluation_synthetic.csv"
        res.to_csv(out_csv, index=False)
        # also save a rounded summary CSV and a LaTeX table
        out_csv_round = TABLES / "timesplit_evaluation_synthetic_summary.csv"
        res.round(6).to_csv(out_csv_round, index=False)
        out_tex = TABLES / "timesplit_evaluation_synthetic.tex"
        try:
            with open(out_tex, 'w', encoding='utf8') as f:
                f.write(res.head(10).to_latex(index=False))
        except Exception:
            pass
        print(f"[OK] Saved {out_csv} and summary {out_csv_round}")
        fig_path = FIGURES / "timesplit_sharpe_synthetic.png"
        _plot_results(res, fig_path)
        print(f"[OK] Saved {fig_path} and PDF")
    else:
        print("Run with --synthetic for an example, or import functions in tests.")


if __name__ == "__main__":
    main_cli()
