"""
Multi-Factor ESG Investment Index — Interactive Explorer
=========================================================
A Gradio-based web interface for exploring, comparing, and screening
companies using the 10-factor ESG-integrated investment index.

Launch:  python app.py
Deploy:  Push to a HuggingFace Space (see README_APP.md)
"""

from __future__ import annotations

import math
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yaml

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "indexed_data.csv"
CONFIG_PATH = ROOT / "config" / "index_config.yaml"

df = pd.read_csv(DATA_PATH)

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

# The 10 score factors used in preference scoring
SCORE_FACTORS: list[str] = [
    "ESG_composite",
    "financial_score",
    "market_score",
    "operational_score",
    "risk_adjusted_score",
    "growth_score",
    "value_score",
    "stability_score",
    "similarity_rank",
    "sector_position",
]

FACTOR_LABELS: dict[str, str] = {
    "ESG_composite": "ESG Composite",
    "financial_score": "Financial",
    "market_score": "Market",
    "operational_score": "Operational",
    "risk_adjusted_score": "Risk-Adjusted",
    "growth_score": "Growth",
    "value_score": "Value",
    "stability_score": "Stability",
    "similarity_rank": "Peer Similarity",
    "sector_position": "Sector Position",
}

# Weight keys used in config (map score column -> config key)
SCORE_TO_WEIGHT_KEY: dict[str, str] = {
    "ESG_composite": "esg_score",
    "financial_score": "financial_score",
    "market_score": "market_score",
    "operational_score": "operational_score",
    "risk_adjusted_score": "risk_adjusted_score",
    "growth_score": "growth_score",
    "value_score": "value_score",
    "stability_score": "stability_score",
    "similarity_rank": "similarity_rank",
    "sector_position": "sector_position",
}

INVESTOR_PROFILES = config["preference_scoring"]["investor_profiles"]

# Defragment DataFrame and add derived columns in bulk
_new_cols: dict[str, pd.Series] = {}
for _col in ["similarity_rank", "sector_position"]:
    if df[_col].max() <= 1.0:
        _new_cols[f"{_col}_display"] = df[_col] * 100
    else:
        _new_cols[f"{_col}_display"] = df[_col]
for _pname in INVESTOR_PROFILES:
    _pcol = f"pref_{_pname}"
    _new_cols[f"{_pcol}_rank"] = df[_pcol].rank(ascending=False).astype(int)


def _esg_rating(score: float) -> str:
    if score >= 60:
        return "AAA"
    elif score >= 57:
        return "AA"
    elif score >= 54:
        return "A"
    elif score >= 50:
        return "BBB"
    elif score >= 46:
        return "BB"
    elif score >= 42:
        return "B"
    else:
        return "CCC"


_new_cols["esg_rating"] = df["ESG_composite"].apply(_esg_rating)
df = pd.concat([df, pd.DataFrame(_new_cols, index=df.index)], axis=1)
del _new_cols


# Build a display-ready value (0-100 scale) for radar charts
def _score_display(row: pd.Series, col: str) -> float:
    if col in ("similarity_rank", "sector_position"):
        return row.get(f"{col}_display", row[col] * 100)
    return row[col]


# Company list for dropdowns
COMPANIES = sorted(df["ticker"].tolist())
SECTORS = ["All"] + sorted(df["sector"].unique().tolist())


# ═══════════════════════════════════════════════════════════════════════════
# Helper: Radar chart
# ═══════════════════════════════════════════════════════════════════════════
def make_radar(ticker: str, color: str = "#636EFA", name: str | None = None) -> go.Scatterpolar:
    row = df[df["ticker"] == ticker].iloc[0]
    values = [_score_display(row, c) for c in SCORE_FACTORS]
    labels = [FACTOR_LABELS[c] for c in SCORE_FACTORS]
    values.append(values[0])  # close the polygon
    labels.append(labels[0])
    return go.Scatterpolar(
        r=values,
        theta=labels,
        fill="toself",
        name=name or ticker,
        line=dict(color=color),
        opacity=0.6,
    )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: Company Explorer
# ═══════════════════════════════════════════════════════════════════════════
def company_explorer(ticker: str):
    row = df[df["ticker"] == ticker].iloc[0]

    # Radar chart
    fig = go.Figure(data=[make_radar(ticker)])
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title=f"{ticker} — 10-Factor Score Profile",
        showlegend=False,
        height=500,
    )

    # Info card
    rank_bal = df["pref_balanced"].rank(ascending=False).astype(int)
    my_rank = int(rank_bal[df["ticker"] == ticker].values[0])
    info = (
        f"**{ticker}**\n\n"
        f"| Attribute | Value |\n|---|---|\n"
        f"| Sector | {row['sector']} |\n"
        f"| Country | {row['country']} |\n"
        f"| ESG Rating | {row['esg_rating']} |\n"
        f"| ESG Composite | {row['ESG_composite']:.2f} |\n"
        f"| Financial Score | {row['financial_score']:.2f} |\n"
        f"| Balanced Pref Score | {row['pref_balanced']:.2f} |\n"
        f"| **Rank (Balanced)** | **{my_rank} / {len(df)}** |\n"
    )

    # Score detail table
    detail_rows = []
    for c in SCORE_FACTORS:
        val = _score_display(row, c)
        col_for_rank = f"{c}_display" if c in ("similarity_rank", "sector_position") else c
        if col_for_rank in df.columns:
            rnk = int(df[col_for_rank].rank(ascending=False)[df["ticker"] == ticker].values[0])
        else:
            rnk = "—"
        detail_rows.append({"Factor": FACTOR_LABELS[c], "Score": round(val, 2), "Rank": rnk})
    detail_df = pd.DataFrame(detail_rows)

    return fig, info, detail_df


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: Company Comparison
# ═══════════════════════════════════════════════════════════════════════════
def company_comparison(ticker_a: str, ticker_b: str, profile: str):
    row_a = df[df["ticker"] == ticker_a].iloc[0]
    row_b = df[df["ticker"] == ticker_b].iloc[0]

    fig = go.Figure(
        data=[
            make_radar(ticker_a, "#636EFA", ticker_a),
            make_radar(ticker_b, "#EF553B", ticker_b),
        ]
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title=f"{ticker_a} vs {ticker_b}",
        height=520,
    )

    # Difference table
    rows = []
    for c in SCORE_FACTORS:
        va = _score_display(row_a, c)
        vb = _score_display(row_b, c)
        rows.append({
            "Factor": FACTOR_LABELS[c],
            ticker_a: round(va, 2),
            ticker_b: round(vb, 2),
            "Difference (A-B)": round(va - vb, 2),
        })
    diff_df = pd.DataFrame(rows)

    # Recommendation
    profile_key = profile.lower().replace("-", "_").replace(" ", "_")
    pref_col = f"pref_{profile_key}"
    if pref_col in df.columns:
        score_a = row_a[pref_col]
        score_b = row_b[pref_col]
    else:
        score_a = row_a["pref_balanced"]
        score_b = row_b["pref_balanced"]

    if score_a > score_b:
        winner, loser = ticker_a, ticker_b
        w_score, l_score = score_a, score_b
    else:
        winner, loser = ticker_b, ticker_a
        w_score, l_score = score_b, score_a

    # Find top differentiating factor
    weights = INVESTOR_PROFILES.get(profile_key, INVESTOR_PROFILES["balanced"])
    best_factor = max(
        SCORE_FACTORS,
        key=lambda c: abs(_score_display(row_a if winner == ticker_a else row_b, c)
                         - _score_display(row_b if winner == ticker_a else row_a, c))
                      * weights.get(SCORE_TO_WEIGHT_KEY[c], 0),
    )

    rec = (
        f"### Recommendation ({profile})\n\n"
        f"**{winner}** (score {w_score:.2f}) is recommended over "
        f"**{loser}** (score {l_score:.2f}).\n\n"
        f"Key differentiator: **{FACTOR_LABELS[best_factor]}** — "
        f"{winner} scores "
        f"{_score_display(df[df['ticker']==winner].iloc[0], best_factor):.1f} vs "
        f"{_score_display(df[df['ticker']==loser].iloc[0], best_factor):.1f}."
    )

    return fig, diff_df, rec


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: Investment Screener
# ═══════════════════════════════════════════════════════════════════════════
def investment_screener(
    min_esg: float,
    min_financial: float,
    max_risk: float,
    sector: str,
    profile: str,
):
    filtered = df.copy()

    filtered = filtered[filtered["ESG_composite"] >= min_esg]
    filtered = filtered[filtered["financial_score"] >= min_financial]
    # Risk: lower risk_adjusted_score means higher risk; use inverted logic
    # Actually higher risk_adjusted_score = better risk-adjusted return, so max_risk
    # means we keep companies where price_volatility <= max_risk
    if "price_volatility" in filtered.columns:
        filtered = filtered[filtered["price_volatility"] <= max_risk]

    if sector != "All":
        filtered = filtered[filtered["sector"] == sector]

    profile_key = profile.lower().replace("-", "_").replace(" ", "_")
    pref_col = f"pref_{profile_key}"
    if pref_col not in filtered.columns:
        pref_col = "pref_balanced"

    filtered = filtered.sort_values(pref_col, ascending=False)

    display_cols = [
        "ticker", "sector", "country", "esg_rating",
        "ESG_composite", "financial_score", "market_score",
        "risk_adjusted_score", "stability_score", pref_col,
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]
    result_df = filtered[display_cols].copy()
    result_df.columns = [c.replace("_", " ").title() for c in result_df.columns]

    # Round numeric columns
    for c in result_df.columns:
        if result_df[c].dtype in (np.float64, np.float32):
            result_df[c] = result_df[c].round(2)

    top5 = filtered.head(5)
    if len(top5) == 0:
        rec = "No companies match the selected criteria. Try relaxing the filters."
    else:
        lines = [f"### Top {min(5, len(top5))} Recommendations ({profile})\n"]
        for i, (_, r) in enumerate(top5.iterrows(), 1):
            lines.append(
                f"{i}. **{r['ticker']}** ({r['sector']}) — "
                f"ESG {r['ESG_composite']:.1f}, Financial {r['financial_score']:.1f}, "
                f"Pref Score {r[pref_col]:.2f}"
            )
        rec = "\n".join(lines)

    n_match = len(filtered)
    summary = f"**{n_match}** companies match your criteria out of {len(df)}."

    return result_df, rec, summary


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: Portfolio Builder
# ═══════════════════════════════════════════════════════════════════════════
def portfolio_builder(tickers: list[str]):
    if not tickers:
        empty = pd.DataFrame()
        fig_empty = go.Figure()
        fig_empty.update_layout(title="Select at least one company")
        return fig_empty, "", empty, fig_empty

    port = df[df["ticker"].isin(tickers)].copy()

    # Aggregate scores (equal-weighted portfolio)
    agg = {}
    for c in SCORE_FACTORS:
        col = f"{c}_display" if c in ("similarity_rank", "sector_position") and f"{c}_display" in df.columns else c
        agg[FACTOR_LABELS[c]] = port[col].mean()
    universe_agg = {}
    for c in SCORE_FACTORS:
        col = f"{c}_display" if c in ("similarity_rank", "sector_position") and f"{c}_display" in df.columns else c
        universe_agg[FACTOR_LABELS[c]] = df[col].mean()

    # Radar comparison: portfolio vs universe
    labels = list(agg.keys())
    port_vals = list(agg.values()) + [list(agg.values())[0]]
    univ_vals = list(universe_agg.values()) + [list(universe_agg.values())[0]]
    labels_closed = labels + [labels[0]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=port_vals, theta=labels_closed, fill="toself",
        name="Your Portfolio", line=dict(color="#636EFA"), opacity=0.6,
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=univ_vals, theta=labels_closed, fill="toself",
        name="Universe Average", line=dict(color="#FECB52"), opacity=0.4,
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title=f"Portfolio ({len(port)} companies) vs Universe",
        height=500,
    )

    # Summary markdown
    summary_lines = [
        "### Portfolio Aggregate Scores\n",
        "| Factor | Portfolio | Universe | Delta |",
        "|---|---|---|---|",
    ]
    for label in labels:
        pv = agg[label]
        uv = universe_agg[label]
        delta = pv - uv
        arrow = "▲" if delta > 0 else "▼" if delta < 0 else "—"
        summary_lines.append(f"| {label} | {pv:.2f} | {uv:.2f} | {arrow} {delta:+.2f} |")
    summary = "\n".join(summary_lines)

    # Holdings table
    display_cols = ["ticker", "sector", "country", "ESG_composite", "financial_score", "pref_balanced"]
    display_cols = [c for c in display_cols if c in port.columns]
    holdings_df = port[display_cols].copy()
    holdings_df.columns = [c.replace("_", " ").title() for c in holdings_df.columns]
    for c in holdings_df.columns:
        if holdings_df[c].dtype in (np.float64, np.float32):
            holdings_df[c] = holdings_df[c].round(2)

    # Sector diversification pie
    sector_counts = port["sector"].value_counts().reset_index()
    sector_counts.columns = ["Sector", "Count"]
    fig_pie = px.pie(
        sector_counts, values="Count", names="Sector",
        title="Sector Diversification", hole=0.35,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_pie.update_layout(height=400)

    return fig_radar, summary, holdings_df, fig_pie


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5: Index Methodology — Weight Adjuster
# ═══════════════════════════════════════════════════════════════════════════
def methodology_rerank(
    w_esg: float,
    w_fin: float,
    w_mkt: float,
    w_ops: float,
    w_risk: float,
    w_growth: float,
    w_value: float,
    w_stab: float,
    w_sim: float,
    w_sect: float,
):
    raw_weights = {
        "ESG_composite": w_esg,
        "financial_score": w_fin,
        "market_score": w_mkt,
        "operational_score": w_ops,
        "risk_adjusted_score": w_risk,
        "growth_score": w_growth,
        "value_score": w_value,
        "stability_score": w_stab,
        "similarity_rank": w_sim,
        "sector_position": w_sect,
    }

    total = sum(raw_weights.values())
    if total == 0:
        total = 1.0
    weights = {k: v / total for k, v in raw_weights.items()}

    # Compute custom composite
    temp = df.copy()
    temp["custom_score"] = sum(
        weights[c] * (temp[f"{c}_display"] if c in ("similarity_rank", "sector_position") and f"{c}_display" in temp.columns else temp[c])
        for c in SCORE_FACTORS
    )
    temp["custom_rank"] = temp["custom_score"].rank(ascending=False).astype(int)

    # Original balanced rank
    temp["original_rank"] = temp["pref_balanced"].rank(ascending=False).astype(int)
    temp["rank_change"] = temp["original_rank"] - temp["custom_rank"]  # positive = improved

    display = temp[["ticker", "sector", "original_rank", "custom_rank", "rank_change",
                     "pref_balanced", "custom_score"]].copy()
    display = display.sort_values("custom_rank")
    display.columns = ["Ticker", "Sector", "Original Rank", "New Rank", "Rank Change",
                        "Original Score", "New Score"]
    display["Original Score"] = display["Original Score"].round(2)
    display["New Score"] = display["New Score"].round(2)

    # Weight summary
    weight_lines = ["### Normalised Weights\n", "| Factor | Weight |", "|---|---|"]
    for c in SCORE_FACTORS:
        weight_lines.append(f"| {FACTOR_LABELS[c]} | {weights[c]:.3f} |")
    weight_lines.append(f"\n**Sum: {sum(weights.values()):.3f}** (auto-normalised)")
    weight_md = "\n".join(weight_lines)

    # Methodology description
    factors_desc = [
        "### 10-Factor Index Methodology\n",
        "The composite score is a weighted sum of 10 normalised factor scores:\n",
        "| # | Factor | Description | Default Weight |",
        "|---|---|---|---|",
    ]
    defaults = config["preference_scoring"]["investor_profiles"]["balanced"]
    descs = {
        "esg_score": "Environmental, Social & Governance composite",
        "financial_score": "Profitability, growth, efficiency, stability, valuation",
        "market_score": "Liquidity, volatility, momentum",
        "operational_score": "Operating efficiency, innovation, market position",
        "risk_adjusted_score": "Sharpe/Sortino risk-adjusted return quality",
        "growth_score": "Revenue and earnings growth trajectory",
        "value_score": "Valuation discipline (PE, PB metrics)",
        "stability_score": "Balance sheet resilience, low leverage",
        "similarity_rank": "Peer-group alignment for diversification",
        "sector_position": "Within-sector relative strength",
    }
    for i, c in enumerate(SCORE_FACTORS, 1):
        wk = SCORE_TO_WEIGHT_KEY[c]
        factors_desc.append(f"| {i} | {FACTOR_LABELS[c]} | {descs[wk]} | {defaults[wk]:.2f} |")
    method_md = "\n".join(factors_desc)

    return weight_md, display, method_md


# ═══════════════════════════════════════════════════════════════════════════
# Build Gradio Interface
# ═══════════════════════════════════════════════════════════════════════════
THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
)

with gr.Blocks(title="Multi-Factor ESG Index Explorer") as demo:

    gr.Markdown(
        f"""
        # Multi-Factor ESG Investment Index Explorer
        **Explore, compare, and screen {len(df)} companies** across 10 integrated factors:
        ESG, Financial, Market, Operational, Risk-Adjusted, Growth, Value, Stability,
        Peer Similarity, and Sector Position.
        """
    )

    # ─── Tab 1: Company Explorer ──────────────────────────────────────────
    with gr.Tab("Company Explorer"):
        with gr.Row():
            explorer_ticker = gr.Dropdown(
                choices=COMPANIES, value=COMPANIES[0],
                label="Select Company", scale=2,
            )
        with gr.Row():
            explorer_radar = gr.Plot(label="10-Factor Radar Chart")
        with gr.Row():
            with gr.Column(scale=1):
                explorer_info = gr.Markdown(label="Company Info")
            with gr.Column(scale=1):
                explorer_table = gr.Dataframe(label="Score Details")

        explorer_ticker.change(
            company_explorer, inputs=[explorer_ticker],
            outputs=[explorer_radar, explorer_info, explorer_table],
        )
        demo.load(
            company_explorer, inputs=[explorer_ticker],
            outputs=[explorer_radar, explorer_info, explorer_table],
        )

    # ─── Tab 2: Company Comparison ────────────────────────────────────────
    with gr.Tab("Company Comparison"):
        with gr.Row():
            comp_a = gr.Dropdown(choices=COMPANIES, value="AAPL", label="Company A", scale=1)
            comp_b = gr.Dropdown(choices=COMPANIES, value="MSFT", label="Company B", scale=1)
            comp_profile = gr.Dropdown(
                choices=["esg_first", "balanced", "financial_first"],
                value="balanced", label="Investor Profile", scale=1,
            )
        comp_btn = gr.Button("Compare", variant="primary")
        comp_radar = gr.Plot(label="Radar Comparison")
        with gr.Row():
            with gr.Column():
                comp_table = gr.Dataframe(label="Score Differences")
            with gr.Column():
                comp_rec = gr.Markdown(label="Recommendation")

        comp_btn.click(
            company_comparison,
            inputs=[comp_a, comp_b, comp_profile],
            outputs=[comp_radar, comp_table, comp_rec],
        )

    # ─── Tab 3: Investment Screener ───────────────────────────────────────
    with gr.Tab("Investment Screener"):
        with gr.Row():
            scr_esg = gr.Slider(30, 70, value=45, step=1, label="Min ESG Composite Score")
            scr_fin = gr.Slider(30, 75, value=45, step=1, label="Min Financial Score")
            scr_risk = gr.Slider(5, 80, value=50, step=1, label="Max Price Volatility")
        with gr.Row():
            scr_sector = gr.Dropdown(choices=SECTORS, value="All", label="Sector Filter")
            scr_profile = gr.Dropdown(
                choices=["esg_first", "balanced", "financial_first"],
                value="balanced", label="Investor Profile",
            )
        scr_btn = gr.Button("Screen Companies", variant="primary")
        scr_summary = gr.Markdown()
        scr_rec = gr.Markdown()
        scr_table = gr.Dataframe(label="Matching Companies")

        scr_btn.click(
            investment_screener,
            inputs=[scr_esg, scr_fin, scr_risk, scr_sector, scr_profile],
            outputs=[scr_table, scr_rec, scr_summary],
        )

    # ─── Tab 4: Portfolio Builder ─────────────────────────────────────────
    with gr.Tab("Portfolio Builder"):
        port_tickers = gr.Dropdown(
            choices=COMPANIES, value=["AAPL", "MSFT", "GOOGL"],
            multiselect=True, label="Select Companies for Portfolio",
        )
        port_btn = gr.Button("Build Portfolio", variant="primary")
        with gr.Row():
            port_radar = gr.Plot(label="Portfolio vs Universe")
            port_pie = gr.Plot(label="Sector Diversification")
        port_summary = gr.Markdown()
        port_table = gr.Dataframe(label="Holdings")

        port_btn.click(
            portfolio_builder,
            inputs=[port_tickers],
            outputs=[port_radar, port_summary, port_table, port_pie],
        )

    # ─── Tab 5: Index Methodology ─────────────────────────────────────────
    with gr.Tab("Index Methodology"):
        gr.Markdown("### Adjust Factor Weights\nMove the sliders to re-weight the 10 factors. "
                     "Weights are auto-normalised to sum to 1.0.")
        defaults = INVESTOR_PROFILES["balanced"]
        with gr.Row():
            mw_esg = gr.Slider(0, 0.5, value=defaults["esg_score"], step=0.01, label="ESG")
            mw_fin = gr.Slider(0, 0.5, value=defaults["financial_score"], step=0.01, label="Financial")
            mw_mkt = gr.Slider(0, 0.5, value=defaults["market_score"], step=0.01, label="Market")
            mw_ops = gr.Slider(0, 0.5, value=defaults["operational_score"], step=0.01, label="Operational")
            mw_risk = gr.Slider(0, 0.5, value=defaults["risk_adjusted_score"], step=0.01, label="Risk-Adj")
        with gr.Row():
            mw_growth = gr.Slider(0, 0.5, value=defaults["growth_score"], step=0.01, label="Growth")
            mw_value = gr.Slider(0, 0.5, value=defaults["value_score"], step=0.01, label="Value")
            mw_stab = gr.Slider(0, 0.5, value=defaults["stability_score"], step=0.01, label="Stability")
            mw_sim = gr.Slider(0, 0.3, value=defaults["similarity_rank"], step=0.01, label="Peer Similarity")
            mw_sect = gr.Slider(0, 0.3, value=defaults["sector_position"], step=0.01, label="Sector Position")

        meth_btn = gr.Button("Recompute Rankings", variant="primary")
        meth_weights = gr.Markdown()
        meth_method = gr.Markdown()
        meth_table = gr.Dataframe(label="Before / After Rankings (sorted by New Rank)")

        meth_btn.click(
            methodology_rerank,
            inputs=[mw_esg, mw_fin, mw_mkt, mw_ops, mw_risk,
                    mw_growth, mw_value, mw_stab, mw_sim, mw_sect],
            outputs=[meth_weights, meth_table, meth_method],
        )

    # Footer
    gr.Markdown(
        f"""
        ---
        *Multi-Factor ESG Investment Index* — Built for academic research.
        Data covers {len(df)} companies (US & India) across 10 sectors.
        Scores normalised to 50 ± σ scale. See methodology tab for details.
        """
    )

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=THEME)
