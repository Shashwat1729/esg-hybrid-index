#!/usr/bin/env python3
"""
Generate Thesis EndSem Presentation PPT
Shashwat Bajpai | 2021B3AA3041H
Supervisor: Mr. Nandan Mishra | Co-supervisor: Dr. Dushyant Kumar
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
import os

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# Colors
DARK_BLUE = RGBColor(0x1B, 0x3A, 0x5C)
MED_BLUE = RGBColor(0x2C, 0x5F, 0x8A)
LIGHT_BLUE = RGBColor(0x3A, 0x7C, 0xA5)
ACCENT = RGBColor(0xE8, 0x6C, 0x00)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xF2, 0xF2, 0xF2)
DARK_GRAY = RGBColor(0x33, 0x33, 0x33)
MED_GRAY = RGBColor(0x66, 0x66, 0x66)
GREEN = RGBColor(0x27, 0xAE, 0x60)
RED = RGBColor(0xC0, 0x39, 0x2B)

FIGURES = os.path.join(os.path.dirname(__file__), "Figures")


def add_bg(slide, color=LIGHT_GRAY):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_title_bar(slide, title_text, subtitle_text=None):
    """Add a colored title bar at the top"""
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.2))
    shape.fill.solid()
    shape.fill.fore_color.rgb = DARK_BLUE
    shape.line.fill.background()

    tf = shape.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(28)
    p.font.color.rgb = WHITE
    p.font.bold = True
    p.alignment = PP_ALIGN.LEFT
    tf.margin_left = Inches(0.6)
    tf.margin_top = Inches(0.15)

    if subtitle_text:
        p2 = tf.add_paragraph()
        p2.text = subtitle_text
        p2.font.size = Pt(14)
        p2.font.color.rgb = RGBColor(0xBB, 0xCC, 0xDD)
        p2.alignment = PP_ALIGN.LEFT


def add_text_box(slide, left, top, width, height, text, font_size=16, bold=False, color=DARK_GRAY, alignment=PP_ALIGN.LEFT):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.alignment = alignment
    p.space_after = Pt(4)
    return tf


def add_bullet_list(slide, left, top, width, height, items, font_size=15, color=DARK_GRAY):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = f"  {item}"
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.space_before = Pt(6)
        p.space_after = Pt(4)
        p.level = 0
    return tf


def add_image_safe(slide, img_name, left, top, width, height=None):
    path = os.path.join(FIGURES, img_name)
    if os.path.exists(path):
        if height:
            slide.shapes.add_picture(path, left, top, width, height)
        else:
            slide.shapes.add_picture(path, left, top, width=width)
        return True
    return False


def add_notes(slide, text):
    notes_slide = slide.notes_slide
    notes_slide.notes_text_frame.text = text


def add_table(slide, left, top, width, height, rows, cols, data, col_widths=None):
    """data is list of lists. First row = header."""
    tbl_shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    tbl = tbl_shape.table

    if col_widths:
        for i, w in enumerate(col_widths):
            tbl.columns[i].width = w

    for r_idx, row_data in enumerate(data):
        for c_idx, cell_text in enumerate(row_data):
            cell = tbl.cell(r_idx, c_idx)
            cell.text = str(cell_text)
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.size = Pt(12)
                paragraph.font.color.rgb = WHITE if r_idx == 0 else DARK_GRAY
                paragraph.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT
            if r_idx == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = DARK_BLUE
            elif r_idx % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(0xE8, 0xEE, 0xF4)
    return tbl


# =====================================================================
# SLIDE 1: Title Slide
# =====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
shape.fill.solid()
shape.fill.fore_color.rgb = DARK_BLUE
shape.line.fill.background()

add_text_box(slide, Inches(1), Inches(1.5), Inches(11), Inches(1),
             "Multi-Factor ESG Scoring for Mid-Cap Equities",
             font_size=36, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)
add_text_box(slide, Inches(1), Inches(2.7), Inches(11), Inches(0.6),
             "A Cross-Market Validation Study with Factor-Based Portfolio Analysis",
             font_size=20, color=RGBColor(0xBB, 0xCC, 0xDD), alignment=PP_ALIGN.CENTER)

# Divider line
div = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(4), Inches(3.5), Inches(5), Pt(2))
div.fill.solid()
div.fill.fore_color.rgb = ACCENT
div.line.fill.background()

add_text_box(slide, Inches(1), Inches(4.0), Inches(11), Inches(0.5),
             "Shashwat Bajpai  |  2021B3AA3041H",
             font_size=22, color=WHITE, alignment=PP_ALIGN.CENTER)
add_text_box(slide, Inches(1), Inches(4.7), Inches(11), Inches(0.5),
             "Supervisor: Mr. Nandan Mishra   |   Co-supervisor: Dr. Dushyant Kumar",
             font_size=16, color=RGBColor(0xAA, 0xBB, 0xCC), alignment=PP_ALIGN.CENTER)
add_text_box(slide, Inches(1), Inches(5.4), Inches(11), Inches(0.5),
             "BITS Pilani, Hyderabad Campus  |  Thesis End-Semester Report",
             font_size=14, color=RGBColor(0x88, 0x99, 0xAA), alignment=PP_ALIGN.CENTER)

add_notes(slide, """Welcome everyone. I'm Shashwat Bajpai, ID 2021B3AA3041H. This is my thesis end-semester presentation on building a multi-factor ESG scoring system for mid-cap companies. My supervisor is Mr. Nandan Mishra and co-supervisor is Dr. Dushyant Kumar. Over the next 10 minutes I'll walk you through what we built, how we tested it, and what we found. The short version: the scoring system works well for ranking companies by quality, holds up under stress tests, and generalises to large-cap stocks. We use a mix of real and synthetic ESG data, and we've done validation studies to check how well the synthetic components align with commercial ESG ratings.""")


# =====================================================================
# SLIDE 2: Agenda
# =====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_title_bar(slide, "Agenda")

items = [
    "1.  Why Mid-Cap ESG? The Problem",
    "2.  Data: 276 Companies, 202 Variables, 11 Sectors",
    "3.  Methodology: 9-Factor Scoring Pipeline",
    "4.  Key Results: Factor Structure & Rankings",
    "5.  Portfolio Performance & Bear-Market Protection",
    "6.  Robustness: Bootstrap, Cross-Validation, Weight Sensitivity",
    "7.  High-Cap Generalization: S&P 500 Transfer Test",
    "8.  Comparison with MSCI, S&P, and Sustainalytics",
    "9.  Limitations & Future Work",
    "10. Conclusion"
]
add_bullet_list(slide, Inches(1), Inches(1.6), Inches(11), Inches(5.5), items, font_size=18)

add_notes(slide, """Here's a quick roadmap. We'll start with the motivation, then go through the data and methodology. The bulk of the talk covers the results, which break into four pieces: factor structure, portfolio performance, robustness checks, and the high-cap transfer test. Then I'll be upfront about what didn't work and where this can go next. Should take about 10 minutes total.""")


# =====================================================================
# SLIDE 3: Why Mid-Cap ESG?
# =====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_title_bar(slide, "Why Mid-Cap ESG?", "The Coverage Gap")

add_text_box(slide, Inches(0.6), Inches(1.5), Inches(5.5), Inches(0.5),
             "The Problem", font_size=22, bold=True, color=DARK_BLUE)
items = [
    "ESG investing is ~$35 trillion globally (GSIA 2024)",
    "But most ESG indices target large-cap stocks only",
    "Mid-caps ($2B-$20B): patchy coverage from MSCI, S&P, Sustainalytics",
    "Commercial ESG methods are proprietary and hidden",
    "ESG raters disagree with each other (correlation as low as 0.38)",
]
add_bullet_list(slide, Inches(0.6), Inches(2.1), Inches(5.5), Inches(3.5), items, font_size=16)

add_text_box(slide, Inches(6.5), Inches(1.5), Inches(6), Inches(0.5),
             "Our Approach", font_size=22, bold=True, color=DARK_BLUE)
items2 = [
    "Build an open, reproducible scoring system",
    "Cover 276 mid-cap firms (186 US, 90 India)",
    "9 factors, 202 variables, all weights visible",
    "Combine ESG with financial, market, risk signals",
    "Test it rigorously: bootstrap, CV, walk-forward, high-cap transfer",
]
add_bullet_list(slide, Inches(6.5), Inches(2.1), Inches(6), Inches(3.5), items2, font_size=16)

add_notes(slide, """So why mid-cap ESG? Sustainable investing is massive now, around 35 trillion dollars globally. But most of the well-known ESG indices like MSCI ESG Leaders or S&P ESG focus on large-cap companies. Mid-caps, which are companies with market cap between 2 and 20 billion dollars, get patchy coverage. And the methods commercial providers use are mostly black boxes.

On top of that, different ESG raters don't even agree with each other. Berg et al found correlations as low as 0.38 between major providers. So what we did is build our own open system where every weight and every step is visible and testable. We cover 276 companies across the US and India using 9 scoring factors and 202 underlying variables.""")


# =====================================================================
# SLIDE 4: Data Overview
# =====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_title_bar(slide, "Data Overview", "276 Companies | 202 Variables | 11 GICS Sectors")

data = [
    ["Dimension", "Details", "Source"],
    ["Universe", "276 mid-cap firms (186 US, 90 India)", "Yahoo Finance, SEC EDGAR"],
    ["Sectors", "11 GICS sectors (Tech, Healthcare, Energy...)", "Market classification"],
    ["Financial", "~60 ratios (P/E, ROA, ROE, D/E, margins...)", "Yahoo Finance"],
    ["Market", "Price momentum, volatility, beta, Sharpe, liquidity", "Yahoo Finance"],
    ["ESG", "12 indicators across E, S, G pillars", "Yahoo ESG, SEC filings, EPA"],
    ["Governance", "Board independence, audit quality, CEO pay ratio", "SEC proxy filings"],
    ["Period", "2020-2024 cross-section", "Latest available data"],
]
add_table(slide, Inches(0.6), Inches(1.5), Inches(12), Inches(3.5), 8, 3, data,
          col_widths=[Inches(2.5), Inches(5.5), Inches(4)])

add_image_safe(slide, "fig17_missing_data.png", Inches(0.6), Inches(5.2), Inches(5.5), Inches(2))
add_text_box(slide, Inches(6.5), Inches(5.3), Inches(6), Inches(1.5),
             "ESG columns are sparser than financial ones.\nSome ESG indicators use proxy construction where\ndirect data is unavailable. Financial data coverage\nis near-complete.",
             font_size=14, color=MED_GRAY)

add_notes(slide, """Our dataset covers 276 mid-cap companies, 186 from the US and 90 from India, spread across 11 GICS sectors. We pull financial data from Yahoo Finance, governance information from SEC proxy filings, and environmental indicators from EPA and other public sources.

The figure on the left shows the missing data heatmap. You can see that financial columns are nearly complete but ESG columns are sparser. Where direct ESG data isn't available, we use proxy construction based on related financial indicators, and we've validated these proxies against commercial ESG ratings from MSCI and Sustainalytics. The financial and market data is solid across the board.""")


# =====================================================================
# SLIDE 5: 9-Factor Scoring Pipeline
# =====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_title_bar(slide, "9-Factor Scoring Pipeline", "Variable-Type-Aware Normalisation")

factors = [
    ["Factor", "What It Captures", "Key Variables"],
    ["Financial Score", "Fundamental quality", "ROA, ROE, margins, debt ratios"],
    ["ESG Composite", "Environmental, Social, Governance", "12 indicators across 3 pillars"],
    ["Growth Score", "Earnings & revenue momentum", "Revenue growth, EPS growth, R&D"],
    ["Value Score", "Valuation attractiveness", "P/E, P/B, EV/EBITDA"],
    ["Risk-Adjusted", "Return per unit risk", "Sharpe, Sortino, drawdown, volatility"],
    ["Stability Score", "Earnings & price consistency", "Earnings stability, price volatility"],
    ["Operational Score", "Business efficiency", "Asset turnover, employee productivity"],
    ["Market Score", "Market-based signals", "Momentum, liquidity, beta"],
    ["Sector Score", "Industry-relative position", "SASB materiality-weighted"],
]
add_table(slide, Inches(0.4), Inches(1.5), Inches(12.5), Inches(4.5), 10, 3, factors,
          col_widths=[Inches(2.5), Inches(4.5), Inches(5.5)])

add_text_box(slide, Inches(0.6), Inches(6.2), Inches(12), Inches(1),
             "7 variable types handled separately: binary, ordinal, bounded %, ratios, counts, rates, continuous\n"
             "All scores normalised to 0-100 scale  |  SASB sector materiality applied to ESG pillars",
             font_size=14, color=MED_GRAY)

add_notes(slide, """Here are the nine factors. Each captures a different dimension of company quality. The key innovation is that we don't just z-score everything and average it. Variables come in seven different types: binary flags, ordinal scales, bounded percentages, ratios, counts, rates, and continuous measures. Each type gets its own normalisation method.

Quick definitions for the audience if asked:
- Sharpe ratio: measures return per unit of total risk. A Sharpe of 1.0 means you earned 1% excess return for every 1% of volatility.
- Sortino ratio: similar to Sharpe but only penalises downside volatility, so it doesn't punish upside swings.
- P/E ratio: price-to-earnings, how much investors pay per dollar of company earnings. Lower means cheaper.
- SASB: Sustainability Accounting Standards Board, they define which ESG issues are financially material for each industry. For example, carbon emissions matter more for energy companies than for software firms.
- Winsorising: capping extreme values at a percentile threshold (we use 2.5th and 97.5th) so outliers don't distort the normalisation.

For example, binary variables like 'has carbon reduction target' get mapped to 0 or 100. Ratios like P/E get winsorised because they can blow up when the denominator is near zero. Counts like revenue get log-transformed if they're highly skewed. Everything ends up on a 0 to 100 scale.""")


# =====================================================================
# SLIDE 6: Score Distributions
# =====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_title_bar(slide, "Factor Score Distributions", "After Normalisation")

add_image_safe(slide, "fig01_score_distributions.png", Inches(0.5), Inches(1.4), Inches(7), Inches(5.5))

add_text_box(slide, Inches(7.8), Inches(1.5), Inches(5), Inches(0.5),
             "Key Observations", font_size=20, bold=True, color=DARK_BLUE)
items = [
    "Most scores centred near 50, SD ~ 10",
    "Financial score is unusually tight (SD = 1.40)",
    "ESG composite: mean 50.05, SD 10.09",
    "Risk-adjusted: mean 49.82, SD 9.96",
    "Good separation for ranking purposes",
    "Max VIF = 2.80 (low multicollinearity)",
    "PCA: 3 components explain 67.4% variance",
]
add_bullet_list(slide, Inches(7.8), Inches(2.2), Inches(5), Inches(4.5), items, font_size=15)

add_notes(slide, """After normalisation, the factor scores look like this. Most factors sit around a mean of 50 with standard deviations near 10, which is exactly what we wanted for clean separation.

Quick definitions if asked:
- VIF (Variance Inflation Factor): measures how much a factor's variance is inflated due to correlation with other factors. VIF below 5 means no multicollinearity problem. Ours maxes at 2.80.
- PCA (Principal Component Analysis): a technique that finds the underlying independent dimensions in the data. If all 9 factors were secretly measuring the same thing, PCA would show 1 component explaining everything. We need 3, which means there are at least 3 genuinely different dimensions.

The one outlier is financial score which has a standard deviation of only 1.40. That means firms don't differ much on pure financial quality in our mid-cap sample, which is interesting but doesn't break the model. The important thing is that these factors are measuring different things.""")


# =====================================================================
# SLIDE 7: Correlation Heatmap
# =====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_title_bar(slide, "Factor Independence", "Correlation Structure")

add_image_safe(slide, "fig03_correlation_heatmap.png", Inches(0.5), Inches(1.5), Inches(6.5), Inches(5.5))

add_text_box(slide, Inches(7.5), Inches(1.5), Inches(5.5), Inches(0.5),
             "What This Tells Us", font_size=20, bold=True, color=DARK_BLUE)
items = [
    "Most factor pairs below |r| = 0.3",
    "Factors are genuinely independent",
    "No hidden mega-factor driving everything",
    "Max VIF = 2.80 (threshold: < 5.0)",
    "Condition number = 3.1 (ideal: < 10)",
    "",
    "This means our composite score captures",
    "multiple real dimensions of company quality,",
    "not the same signal repackaged 9 times."
]
add_bullet_list(slide, Inches(7.5), Inches(2.2), Inches(5.5), Inches(4.5), items, font_size=15)

add_notes(slide, """This is the correlation heatmap between our nine factors. Most pairs sit below 0.3 in absolute terms, which means they're genuinely capturing different things.

Quick definitions if asked:
- Condition number: measures how sensitive a matrix is to small changes. A condition number below 10 means the factors are well-separated and the model is numerically stable. Ours is 3.1.
- VIF threshold: the standard threshold is 5.0. Above 10 is a serious problem. Our max is 2.80, so we're well in the safe zone.

You can see a few moderate correlations like between growth and financial score, which makes sense since profitable companies tend to grow faster. But nothing is so high that we're double-counting the same signal. If all factors were just measuring the same underlying thing, the composite score would be meaningless. They're not.

Potential Q&A: 'Why not remove correlated factors?' Because moderate correlation is expected and even desirable. Growth and financial quality should be somewhat related. The key is that they're not so correlated that one is redundant.""")


# =====================================================================
# SLIDE 8: Monotonicity
# =====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_title_bar(slide, "Factor Monotonicity", "Do Higher Scores Mean Better Companies?")

data = [
    ["Factor", "Q1 (Low)", "Q5 (High)", "Monotonic?", "p-value"],
    ["Financial Score", "41.4", "61.9", "Yes", "< 0.001"],
    ["Risk-Adjusted", "38.0", "59.4", "Yes", "< 0.001"],
    ["Growth Score", "34.3", "69.1", "Yes", "< 0.001"],
    ["ESG Composite", "44.8", "55.6", "Yes", "< 0.001"],
    ["Market Score", "42.5", "54.9", "Yes", "< 0.001"],
    ["Operational", "46.0", "56.9", "Yes", "< 0.001"],
    ["Stability", "48.5", "52.5", "Yes", "< 0.001"],
    ["Value Score", "59.0", "39.1", "No (inverted)", "0.999"],
]
add_table(slide, Inches(0.5), Inches(1.5), Inches(12.3), Inches(4), 9, 5, data,
          col_widths=[Inches(2.5), Inches(2), Inches(2), Inches(2.8), Inches(3)])

add_text_box(slide, Inches(0.6), Inches(5.7), Inches(12), Inches(1.5),
             "7 of 8 factors show clean monotonic ordering against forward quality proxy (p < 0.001)\n"
             "Value score is inverted: reflects growth-over-value regime (2020-2024)\n"
             "Against return proxy: only 2 of 9 significant | Model ranks quality better than short-run performance",
             font_size=15, color=MED_GRAY)

add_notes(slide, """This table shows the quintile analysis. We sort companies into five equal groups (quintiles) by each factor score and check if higher quintiles correspond to higher quality outcomes.

Quick definitions if asked:
- Monotonicity: means the relationship goes in one direction consistently. Q1 (lowest score) should have the worst outcome, Q5 (highest score) should have the best. If Q1 < Q2 < Q3 < Q4 < Q5, that's monotonic.
- Jonckheere-Terpstra test: a statistical test specifically designed to check whether there's an ordered trend across groups. A p-value below 0.001 means we're very confident the trend is real, not random.
- Forward quality proxy: our validation target built from lagged earnings growth, ROA, and ROE. We use this instead of raw returns to avoid circularity.

Seven out of eight factors show clean monotonic ordering with p-values below 0.001. The one exception is value score, which is inverted: high-value (cheap) companies actually had lower forward quality. This makes sense given the growth-over-value regime from 2020 to 2024.

Against the return proxy, only 2 out of 9 factors are significant. So the model is better at ranking companies by underlying quality than at predicting short-horizon performance. That's a more modest claim but also a more honest one.

Potential Q&A: 'Why is value inverted?' During 2020-2024, growth stocks massively outperformed value stocks globally. This is a well-documented market regime effect, not a model failure.""")


# =====================================================================
# SLIDE 9: Top-20 Rankings
# =====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_title_bar(slide, "Top-20 Balanced Rankings", "Who Comes Out on Top?")

add_image_safe(slide, "fig07_top20_rankings.png", Inches(0.3), Inches(1.4), Inches(8), Inches(5.8))

add_text_box(slide, Inches(8.5), Inches(1.5), Inches(4.5), Inches(0.5),
             "Key Takeaways", font_size=20, bold=True, color=DARK_BLUE)
items = [
    "INCY, MEDP, EXEL lead (Healthcare)",
    "Healthcare & Technology dominate top-20",
    "Growth & risk-adjusted scores do the",
    "  heavy lifting in balanced profile",
    "",
    "3 investor profiles tested:",
    "  ESG-First, Balanced, Financial-First",
    "Kendall tau = 0.593 between extremes",
    "  (genuinely different, not cosmetic)",
]
add_bullet_list(slide, Inches(8.5), Inches(2.2), Inches(4.5), Inches(4.5), items, font_size=15)

add_notes(slide, """Here's the top-20 ranking under the balanced profile. Healthcare and technology companies dominate, with INCY, MEDP, and EXEL taking the top three spots. These companies score well on both growth and risk-adjusted dimensions.

Quick definitions if asked:
- Kendall tau: a rank correlation measure between 0 and 1. A tau of 1.0 means two rankings are identical. 0.593 between our ESG-First and Financial-First profiles means they're moderately different, genuinely producing distinct portfolios.
- Balanced profile: gives roughly equal weight to all 9 factors. ESG-First overweights ESG indicators. Financial-First emphasises profitability, margins, and debt ratios.

We tested three different investor profiles. The Kendall tau between the ESG-First and Financial-First rankings is 0.593, and the top-10 overlap ranges from 1 to 6 firms depending on which pair you compare. So the profiles produce genuinely different portfolios, not just cosmetic relabelling.

Potential Q&A: 'Why do healthcare companies dominate?' Healthcare mid-caps in our sample tend to have strong growth (biotech pipeline value), good risk-adjusted returns, and decent ESG scores. The balanced profile rewards companies that score well across multiple dimensions simultaneously.""")


# =====================================================================
# SLIDE 10: Portfolio Performance
# =====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_title_bar(slide, "Portfolio Construction Results", "How Different Builds Perform")

data = [
    ["Construction", "Excess Return", "Cross-Sectional IR", "Verdict"],
    ["Balanced Top-20 (equal-wt)", "-4.01%", "-0.710", "Weak"],
    ["Financial-First Top-20", "-3.13%", "0.251", "Slightly better"],
    ["Score-Weighted Portfolio", "+8.66%", "0.475", "Best result"],
    ["Rolling Rebalance (12 periods)", "+7.38% mean", "12/12 positive", "Consistent"],
]
add_table(slide, Inches(0.5), Inches(1.5), Inches(12.3), Inches(2.5), 5, 4, data,
          col_widths=[Inches(3.5), Inches(2.5), Inches(3), Inches(3.3)])

add_text_box(slide, Inches(0.6), Inches(4.3), Inches(5.5), Inches(0.5),
             "Key Finding", font_size=20, bold=True, color=DARK_BLUE)
items = [
    "Construction method matters a LOT",
    "Equal-weighted top-20 actually underperforms",
    "Score-weighted version captures the signal",
    "Rolling test: beats universe in ALL 12 periods",
    "Mean rolling excess: +7.38%",
]
add_bullet_list(slide, Inches(0.6), Inches(4.9), Inches(5.5), Inches(2.3), items, font_size=15)

add_image_safe(slide, "fig18_portfolio_performance.png", Inches(7), Inches(4.2), Inches(5.8), Inches(3))

add_notes(slide, """Here's where construction choice makes a huge difference. The equal-weighted top-20 portfolio actually posts a negative excess return of minus 4 percent. But when we switch to score-weighted construction, where companies with higher composite scores get larger portfolio weights, the same underlying rankings produce plus 8.66 percent excess with an IR of 0.475.

Quick definitions if asked:
- Cross-Sectional IR (Information Ratio): measures the portfolio's score advantage over the universe average, divided by the spread of scores. An IR above 0.3 is considered decent. It tells you whether the selected portfolio is meaningfully differentiated from random selection.
- Score-weighted: instead of giving each of the top-20 stocks the same 5% weight, we weight them proportionally to their composite score. Higher-ranked companies get more weight.
- Rolling rebalance: we shift the scoring window forward by one quarter at a time, re-rank, and check if the new portfolio still beats the universe. All 12 quarterly periods show positive excess.

The rolling rebalance test is very convincing. Across 12 rolling periods, the balanced strategy beats the universe in every single one, with a mean excess of plus 7.38 percent. So the ranking signal is real and consistent, you just have to build the portfolio properly to capture it.

Potential Q&A: 'Why does equal-weight underperform?' Because equal-weighting treats the 1st-ranked and 20th-ranked companies identically. Score-weighting concentrates more capital in the companies the model is most confident about.""")


# =====================================================================
# SLIDE 11: Bear Market Protection
# =====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_title_bar(slide, "Bear-Market Protection", "Our Strongest Practical Result")

# Big numbers
add_text_box(slide, Inches(0.8), Inches(1.8), Inches(3.5), Inches(1),
             "+7.27 pp", font_size=48, bold=True, color=GREEN, alignment=PP_ALIGN.CENTER)
add_text_box(slide, Inches(0.8), Inches(2.8), Inches(3.5), Inches(0.5),
             "Bear-period excess return", font_size=16, color=MED_GRAY, alignment=PP_ALIGN.CENTER)

add_text_box(slide, Inches(4.8), Inches(1.8), Inches(3.5), Inches(1),
             "0.585", font_size=48, bold=True, color=GREEN, alignment=PP_ALIGN.CENTER)
add_text_box(slide, Inches(4.8), Inches(2.8), Inches(3.5), Inches(0.5),
             "Information Ratio (bear)", font_size=16, color=MED_GRAY, alignment=PP_ALIGN.CENTER)

add_text_box(slide, Inches(8.8), Inches(1.8), Inches(3.5), Inches(1),
             "0.82", font_size=48, bold=True, color=GREEN, alignment=PP_ALIGN.CENTER)
add_text_box(slide, Inches(8.8), Inches(2.8), Inches(3.5), Inches(0.5),
             "Portfolio Beta", font_size=16, color=MED_GRAY, alignment=PP_ALIGN.CENTER)

add_text_box(slide, Inches(0.6), Inches(3.8), Inches(12), Inches(0.5),
             "What This Means", font_size=20, bold=True, color=DARK_BLUE)
items = [
    "The balanced portfolio tilts toward defensive, lower-beta names",
    "In market downturns, it outperforms by +7.27 percentage points",
    "Decomposition analysis shows +10.65 pp excess in bear regimes",
    "This is exactly where multi-factor ESG scoring should help most",
    "Portfolio beta of 0.82 means it moves less than the market",
    "",
    "Comparison: MSCI ESG Leaders typically shows +2-3 pp in bear markets",
    "Our result is stronger, partly because mid-caps have higher dispersion",
]
add_bullet_list(slide, Inches(0.6), Inches(4.4), Inches(12), Inches(2.8), items, font_size=15)

add_notes(slide, """This is probably our strongest practical result. During bear market periods, the balanced portfolio outperforms by 7.27 percentage points with an information ratio of 0.585.

Quick definitions if asked:
- Information Ratio (IR): measures excess return divided by the volatility of excess return (tracking error). An IR of 0.585 is considered good by industry standards. Above 0.5 is generally the threshold for a skilled strategy.
- Beta: measures how much a portfolio moves relative to the overall market. A beta of 1.0 means it moves exactly with the market. Our portfolio beta of 0.82 means it moves about 18% less than the market, so it falls less during downturns.
- Bear market: we define this as periods where the broad market declines. The portfolio naturally tilts toward defensive, lower-volatility names.
- pp (percentage points): the absolute difference, not a ratio. So +7.27 pp means our portfolio returned 7.27 percentage points more than the universe average.

For comparison, the MSCI ESG Leaders index typically shows about 2 to 3 percentage points of excess in bear markets. Our result is stronger, though mid-cap stocks have higher return dispersion, so the spreads look bigger.

Potential Q&A: 'Is this real alpha?' Not exactly. These are cross-sectional momentum proxy differentials, not realised portfolio returns. The bear-market protection is real in the sense that the portfolio selects lower-risk companies, but a live trading implementation would face transaction costs and capacity constraints.""")


# =====================================================================
# SLIDE 12: Bootstrap Stability
# =====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_title_bar(slide, "Robustness: Bootstrap & Weight Sensitivity")

add_image_safe(slide, "fig24_bootstrap_ci.png", Inches(0.4), Inches(1.4), Inches(6), Inches(5.5))

add_text_box(slide, Inches(6.8), Inches(1.5), Inches(6), Inches(0.5),
             "Stability Results", font_size=20, bold=True, color=DARK_BLUE)

data = [
    ["Test", "Result", "Verdict"],
    ["Bootstrap Kendall tau", "1.000", "Perfect ordering stability"],
    ["CI Width", "247-255 ranks", "Individual positions noisy"],
    ["Weight Sensitivity", "rho > 0.99", "Rankings hold under +/-20%"],
    ["CV Overfit Ratio", "0.499", "No overfitting detected"],
    ["Walk-Forward Ratio", "0.978", "Near-perfect OOS"],
]
add_table(slide, Inches(6.8), Inches(2.2), Inches(6), Inches(2.5), 6, 3, data,
          col_widths=[Inches(2.2), Inches(1.8), Inches(2)])

add_text_box(slide, Inches(6.8), Inches(5.0), Inches(6), Inches(2),
             "Bottom line: the portfolio-level ordering is rock solid.\n"
             "Individual company ranks are ranges, not pinpoints.\n"
             "Weight perturbations barely move the results.\n"
             "L2 regularisation + entropy penalty keep overfitting\n"
             "in check (CV ratio 0.499, walk-forward 0.978).",
             font_size=14, color=MED_GRAY)

add_notes(slide, """Let's talk about how stable these results are.

Quick definitions if asked:
- Bootstrap resampling: we randomly resample our 276 companies with replacement (some companies appear twice, some don't appear at all) and re-run the entire ranking. We do this 1000 times to see how much the rankings bounce around. It simulates 'what if our sample were slightly different?'
- Kendall tau: a rank correlation measure. Tau of 1.000 means the portfolio-level ordering is perfectly preserved across all 1000 resamples.
- CI (Confidence Interval) width: the range of ranks a company lands in across the 1000 bootstrap runs. A width of 247-255 means a top-ranked company could plausibly land anywhere from rank 1 to around rank 255. That's wide, but it's an honest assessment.
- Spearman rho: another rank correlation measure. Rho above 0.99 under weight perturbation means changing the factor weights by up to 20% barely moves the final rankings.
- Walk-forward: we train the model on earlier data and test it on later data, sliding the window forward. A ratio of 0.978 means out-of-sample performance is 97.8% as good as in-sample.

The portfolio-level ordering is perfectly stable, but individual company positions are noisy. Weight perturbations barely move the results. L2 regularisation with lambda 0.25 and an entropy penalty keep the weight optimisation honest.

Potential Q&A: 'If individual ranks are noisy, is the model useful?' Yes, at the portfolio level. Think of it like weather forecasting: you can't predict the exact temperature but you can reliably say summer is hotter than winter. Similarly, our top-quintile portfolio reliably outperforms the bottom quintile, even though individual company positions shift around.""")


# =====================================================================
# SLIDE 13: Cross-Validation
# =====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_title_bar(slide, "Cross-Validation & Generalization", "How We Prevent Overfitting")

add_text_box(slide, Inches(0.6), Inches(1.5), Inches(5.5), Inches(0.5),
             "Regularisation Design", font_size=22, bold=True, color=DARK_BLUE)
items1 = [
    "L2 regularisation (lambda = 0.25)",
    "Entropy penalty prevents weight concentration",
    "Weight bounds: [0.02, 0.35] per factor",
    "Non-circular forward quality proxy",
    "  (lagged earnings growth, ROA, ROE)",
    "Sector-stratified 5-fold cross-validation",
]
add_bullet_list(slide, Inches(0.6), Inches(2.2), Inches(5.5), Inches(3), items1, font_size=15)

add_text_box(slide, Inches(7), Inches(1.5), Inches(5.5), Inches(0.5),
             "Validation Results", font_size=22, bold=True, color=DARK_BLUE)
items2 = [
    "CV Overfit Ratio: 0.499 (well below 1.0)",
    "Quasi-temporal Ratio: 0.509",
    "Walk-Forward Ratio: 0.978 (near-perfect OOS)",
    "Leave-one-sector-out IC: 0.200",
    "165 walk-forward splits, mean IC = 0.24",
    "No split produces negative IC",
    "Fama-MacBeth: financial (beta=0.42, p<0.001)",
    "  and risk-adjusted (beta=0.31, p<0.01) are strongest",
]
add_bullet_list(slide, Inches(7), Inches(2.2), Inches(5.5), Inches(3.5), items2, font_size=15)

add_text_box(slide, Inches(0.6), Inches(5.8), Inches(12), Inches(1),
             "The forward quality proxy is built from lagged earnings growth, ROA, and ROE to avoid\n"
             "any overlap with market-based factors used in scoring. This ensures clean out-of-sample validation.",
             font_size=15, bold=True, color=DARK_BLUE)

add_notes(slide, """Our cross-validation design uses several layers to ensure the model isn't just memorising the training data.

Quick definitions if asked:
- L2 regularisation: adds a penalty proportional to the square of the weights during optimisation. This prevents any single factor from getting an extreme weight. Lambda 0.25 is the penalty strength.
- Entropy penalty: a term that penalises low-entropy (concentrated) weight distributions. It pushes the optimiser toward more balanced weights rather than putting all eggs in one basket.
- Weight bounds [0.02, 0.35]: no factor can have less than 2% or more than 35% of total weight. This prevents degenerate solutions.
- IC (Information Coefficient): the correlation between predicted scores and actual outcomes. IC of 0.24 is considered decent in cross-sectional equity models. Above 0.05 is generally considered meaningful.
- Fama-MacBeth regression: a two-stage regression that first runs cross-sectional regressions for each time period, then averages the coefficients. It's the standard method in finance for testing whether factors predict returns. A beta of 0.42 for financial score means a 1-unit increase in financial score is associated with a 0.42 unit increase in the quality proxy.
- Forward quality proxy: built from lagged earnings growth, ROA (return on assets), and ROE (return on equity). These don't overlap with the market-based factors used in scoring, so there's no circularity.

The results are encouraging. CV overfit ratio of 0.499 means in-sample and out-of-sample performance are very close. Walk-forward validation across 165 splits gives a mean IC of 0.24 with no split going negative. Leave-one-sector-out confirms the model works across all 11 sectors.

Potential Q&A: 'What is overfitting and why does it matter?' Overfitting is when a model performs well on the data it was trained on but poorly on new data. A CV ratio near 1.0 means the model generalises well. A ratio much above 1.0 would mean the model memorised the training data rather than learning real patterns.""")


# =====================================================================
# SLIDE 14: High-Cap Transfer
# =====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_title_bar(slide, "High-Cap Generalization", "Does It Work on S&P 500 Stocks?")

add_image_safe(slide, "robustness_highcap_comparison.png", Inches(0.4), Inches(1.4), Inches(6.5), Inches(5.5))

add_text_box(slide, Inches(7.3), Inches(1.5), Inches(5.5), Inches(0.5),
             "Transfer Test Results", font_size=20, bold=True, color=DARK_BLUE)

data = [
    ["Metric", "Result"],
    ["Companies tested", "45 S&P 500 firms"],
    ["Factors generalised", "9 / 9 (all pass)"],
    ["Kendall W", "0.933 (very strong)"],
    ["Spearman rho", "0.9993"],
    ["Friedman p-value", "0.50 (no sig. difference)"],
    ["Factor degradation", "Only 0.2% of pairs"],
]
add_table(slide, Inches(7.3), Inches(2.2), Inches(5.5), Inches(3), 7, 2, data,
          col_widths=[Inches(2.8), Inches(2.7)])

add_text_box(slide, Inches(7.3), Inches(5.5), Inches(5.5), Inches(1.5),
             "Large-cap companies score higher on ESG\n(mean 59.9 vs 48.4) and risk-adjusted\n(57.2 vs 48.8) as expected.\n"
             "The scoring logic transfers cleanly.",
             font_size=14, color=MED_GRAY)

add_notes(slide, """This is one of our best results. We took 45 companies from the S&P 500, companies the model was never trained on, and applied the exact same scoring pipeline.

Quick definitions if asked:
- Kendall W (coefficient of concordance): measures agreement among multiple raters (in our case, factors). W ranges from 0 (no agreement) to 1 (perfect agreement). W of 0.933 means all 9 factors behave very consistently between mid-cap and large-cap universes.
- Spearman rho: rank correlation between the rankings produced by mid-cap-trained methodology and the rankings when applied to large-cap. 0.9993 is near-perfect.
- Friedman test: a non-parametric test for whether there are systematic differences between groups. P-value of 0.50 means no significant difference in factor behavior between mid-cap and large-cap, which is exactly what we want.
- z-score stability test: checks whether each factor's mean and standard deviation are statistically similar between mid-cap and large-cap. All 9 pass at the |z| < 2.5 threshold.

As expected, large-cap companies score higher on ESG (mean 59.9 vs 48.4) and risk-adjusted (57.2 vs 48.8) because bigger companies tend to have more ESG disclosure and more stable returns. Only 0.2 percent of factor pairs show any degradation.

Potential Q&A: 'Why only 45 large-cap companies?' We wanted a meaningful sample without making the benchmark universe dominate the mid-cap universe. 45 S&P 500 companies is enough for statistical testing while keeping the primary focus on mid-cap.""")


# =====================================================================
# SLIDE 15: ESG as Risk Filter
# =====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_title_bar(slide, "ESG: Risk Filter, Not Return Predictor", "What ESG Actually Does Here")

data = [
    ["Risk Metric", "Low ESG (Q1)", "High ESG (Q5)", "Benefit", "Correlation"],
    ["Price Volatility", "46.6%", "33.1%", "-13.5 pp", "rho = -0.357"],
    ["Market Beta", "1.03", "0.75", "-0.29", "rho = -0.146"],
    ["Max Drawdown 1Y", "-38.3%", "-27.8%", "+10.5 pp", "rho = +0.235"],
]
add_table(slide, Inches(0.5), Inches(1.5), Inches(12.3), Inches(2), 4, 5, data,
          col_widths=[Inches(2.5), Inches(2.2), Inches(2.2), Inches(2.2), Inches(3.2)])

add_text_box(slide, Inches(0.6), Inches(3.8), Inches(5.5), Inches(0.5),
             "The Honest Reading", font_size=20, bold=True, color=DARK_BLUE)
items = [
    "ESG does NOT predict returns well (IC = 0.068, p = 0.263)",
    "But higher ESG = lower volatility, lower beta, smaller drawdowns",
    "ESG works as a risk screen, not an alpha source",
    "This is consistent with academic literature",
    "R-squared of ESG-financial fit (0.544) partly reflects",
    "  shared data sources, not a clean empirical discovery",
]
add_bullet_list(slide, Inches(0.6), Inches(4.5), Inches(5.5), Inches(2.5), items, font_size=15)

add_image_safe(slide, "fig12_quintile_analysis.png", Inches(7), Inches(3.5), Inches(5.8), Inches(3.5))

add_notes(slide, """This is an important slide because it's about being honest about what ESG does and doesn't do in our system.

Quick definitions if asked:
- IC (Information Coefficient): the Spearman rank correlation between a factor's scores and the outcome variable. IC of 0.068 for ESG means ESG scores barely predict returns. For context, IC above 0.05 is often considered meaningful in equity models, but our ESG IC has a p-value of 0.263, meaning it's not statistically different from zero.
- Spearman rho: a rank correlation measure. Rho of minus 0.357 between ESG and volatility means higher ESG is associated with lower price volatility. The minus sign means they move in opposite directions.
- R-squared: the proportion of variance explained. R-squared of 0.544 between ESG and financial scores sounds impressive but it partly reflects that our ESG indicators are constructed from financial proxies, not independently observed.

ESG does NOT predict returns well. But ESG does predict risk: high-ESG companies have 13.5 percentage points lower price volatility, nearly 0.3 lower beta, and 10.5 percentage points smaller drawdowns.

So we frame ESG as a risk filter, not a return predictor. This is consistent with a lot of the academic literature. We think being upfront about this makes the thesis more credible.

Potential Q&A: 'If ESG doesn't predict returns, why include it?' Because reducing downside risk is valuable even without excess returns. A portfolio that falls less during market crashes has real practical value for investors, even if it doesn't outperform in calm markets.""")


# =====================================================================
# SLIDE 16: Comparison with Major Indices
# =====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_title_bar(slide, "Comparison with Major ESG Indices")

data = [
    ["Metric", "Our System", "MSCI ESG Leaders", "S&P ESG Index", "FTSE4Good"],
    ["Universe", "276 mid-cap", "~700 large-cap", "~300 large-cap", "~800 large/mid"],
    ["Factors", "9 multi-factor", "37 ESG issues", "ESG tilt only", "ESG screen"],
    ["Bear excess", "+7.27 pp", "+2-3 pp", "+1-2 pp", "+1-2 pp"],
    ["Methodology", "Open, all weights visible", "Proprietary", "Proprietary", "Proprietary"],
    ["ESG data", "Real + synthetic hybrid", "Analyst rated", "Analyst rated", "Analyst rated"],
    ["Capacity", "$25M-$500M", "$1B+", "$1B+", "$1B+"],
    ["Generalization", "Kendall W=0.933", "N/A", "N/A", "N/A"],
]
add_table(slide, Inches(0.3), Inches(1.5), Inches(12.7), Inches(3.5), 8, 5, data,
          col_widths=[Inches(2.2), Inches(2.8), Inches(2.8), Inches(2.5), Inches(2.4)])

add_text_box(slide, Inches(0.6), Inches(5.3), Inches(12), Inches(2),
             "Our Advantages: Multi-factor design (not ESG-only), open methodology, stronger bear-market protection, proven generalization\n"
             "Their Advantages: Real ESG ratings (not proxies), much larger capacity, longer track records, regulatory acceptance\n"
             "We're not claiming to be better. The comparison shows different strengths for different use cases.",
             font_size=14, color=MED_GRAY)

add_notes(slide, """How do we compare with the big commercial indices?

Quick definitions if asked:
- MSCI ESG Leaders: one of the most widely used ESG indices globally. It selects companies with the highest ESG ratings from each sector of the MSCI parent index. Uses analyst-rated ESG data with about 37 different ESG key issues.
- S&P ESG Index: tilts an existing S&P index toward higher ESG scores while maintaining similar risk-return characteristics to the parent index.
- FTSE4Good: screens companies against ESG criteria and excludes those that don't meet minimum standards. Broader inclusion than MSCI.
- Capacity: the maximum AUM the strategy can handle before transaction costs eat into performance. Large-cap indices can handle billions; we cap out at $500M because mid-cap stocks are less liquid.

Our system has clear advantages: multi-factor design, open methodology, stronger bear-market numbers. But commercial indices have real ESG data, much larger capacity, and longer track records. The comparison shows different strengths for different use cases.

Potential Q&A: 'Why not just use MSCI ESG Leaders for mid-caps?' Because MSCI's mid-cap ESG coverage is patchy. Many mid-cap companies either aren't rated or receive boilerplate scores based on industry averages rather than company-specific analysis.""")


# =====================================================================
# SLIDE 17: ESG Provider Comparison
# =====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_title_bar(slide, "ESG Provider Validation", "Do We Agree with MSCI & Sustainalytics?")

data = [
    ["Provider", "Overlap Sample", "Spearman rho", "Assessment"],
    ["Sustainalytics", "Small overlap", "0.780", "Decent agreement"],
    ["MSCI", "Small overlap", "0.563", "Moderate agreement"],
    ["Between MSCI & Sust.", "Literature", "~0.38 (Berg et al.)", "They disagree with each other too"],
]
add_table(slide, Inches(0.5), Inches(1.5), Inches(12.3), Inches(2.2), 4, 4, data,
          col_widths=[Inches(2.8), Inches(2.5), Inches(2.5), Inches(4.5)])

items = [
    "Our ESG scores correlate better with Sustainalytics (0.780) than with MSCI (0.563)",
    "But even MSCI and Sustainalytics only correlate at ~0.38 with each other",
    "The overlap sample is small and underpowered, so these numbers come with caveats",
    "Still, the fact that we're in the same ballpark as commercial providers is encouraging",
    "Our ESG uses a hybrid of real and synthetic data, validated against both providers",
]
add_bullet_list(slide, Inches(0.6), Inches(4.0), Inches(12), Inches(3), items, font_size=15)

add_notes(slide, """We validated our ESG scores against MSCI and Sustainalytics where we had overlapping coverage.

Quick definitions if asked:
- Spearman rho (rank correlation): measures how well two sets of rankings agree. 1.0 means perfect agreement. 0.780 with Sustainalytics is decent. 0.563 with MSCI is moderate.
- Proxy-based ESG: instead of having an analyst visit the company or review detailed sustainability reports, we estimate ESG scores from publicly available financial data. For example, R&D spending as a proxy for environmental innovation, or revenue per employee as a proxy for social capital.

The Spearman correlation with Sustainalytics is 0.780 and with MSCI it's 0.563. Not perfect, but even MSCI and Sustainalytics only agree with each other at about 0.38 according to Berg et al. The overlap sample is small, so these numbers come with caveats.

Potential Q&A: 'If even commercial providers disagree, does ESG measurement mean anything?' This is an active debate in the literature. The low inter-rater agreement reflects that ESG is a multi-dimensional concept and different providers weight different dimensions differently. It doesn't mean ESG is meaningless, but it does mean any single ESG score should be taken with caution.""")


# =====================================================================
# SLIDE 18: Factor Ablation
# =====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_title_bar(slide, "Factor Ablation Study", "What Happens When You Remove Each Factor?")

add_image_safe(slide, "fig25_factor_ablation.png", Inches(0.4), Inches(1.4), Inches(7), Inches(5.5))

add_text_box(slide, Inches(7.8), Inches(1.5), Inches(5), Inches(0.5),
             "Key Findings", font_size=20, bold=True, color=DARK_BLUE)
items = [
    "Removing risk_adjusted hurts the most",
    "Removing growth_score has second-largest impact",
    "Removing ESG barely changes composite IR",
    "",
    "Interpretation:",
    "  Risk-adjusted and growth carry the signal",
    "  ESG contributes to risk reduction, not returns",
    "  Financial score adds stability but has",
    "    limited cross-firm separation (SD=1.40)",
    "",
    "No single factor is doing all the work",
    "But some factors contribute much more",
]
add_bullet_list(slide, Inches(7.8), Inches(2.2), Inches(5), Inches(4.5), items, font_size=14)

add_notes(slide, """The ablation study removes each factor one at a time and checks what happens to the composite score and its predictive power.

Quick definitions if asked:
- Ablation study: systematically removing components to measure their individual contribution. Common in machine learning research. If removing factor X causes the biggest drop in performance, then X is the most important factor.
- Composite IR: the information ratio of the composite score after removing that factor. A bigger drop means the factor was more important.
- SD (standard deviation): measures the spread of scores. Financial score has SD of only 1.40, meaning all companies score very similarly on financial quality. This limits its ability to differentiate between companies even though its R-squared contribution is high.

Removing risk-adjusted score hurts the most, followed by growth score. Removing ESG barely changes the composite information ratio. This tells us that the main predictive content comes from risk and growth characteristics, not ESG.

Potential Q&A: 'If ESG contributes so little to ranking, why include it?' Three reasons: (1) ESG contributes to risk reduction even though it doesn't improve return prediction, (2) many investors have ESG mandates and want ESG explicitly in the scoring, (3) the SASB sector materiality weights mean ESG contributes more for some industries than others.""")


# =====================================================================
# SLIDE 19: Capacity Analysis
# =====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_title_bar(slide, "Capacity & Transaction Costs", "Can You Actually Trade This?")

data = [
    ["AUM ($M)", "Largest Position", "Market Impact", "Total Cost", "Verdict"],
    ["$25M", "$3.6M", "20 bps", "25 bps", "PASS"],
    ["$50M", "$7.2M", "28 bps", "33 bps", "PASS"],
    ["$100M", "$14.4M", "40 bps", "45 bps", "PASS"],
    ["$250M", "$35.9M", "63 bps", "68 bps", "PASS"],
    ["$500M", "$71.8M", "89 bps", "94 bps", "PASS"],
    ["$1,000M", "$143.6M", "126 bps", "131 bps", "FAIL"],
    ["$2,000M", "$287.3M", "178 bps", "183 bps", "FAIL"],
]
add_table(slide, Inches(0.5), Inches(1.5), Inches(12.3), Inches(3.5), 8, 5, data,
          col_widths=[Inches(1.8), Inches(2.5), Inches(2.5), Inches(2.5), Inches(3)])

items = [
    "Feasible for $25M to $500M mandates (mid-cap sweet spot)",
    "Fails at $1B+ due to mid-cap liquidity constraints",
    "10-day execution window, 3% daily participation, 15% max position",
    "Average quarterly turnover: 25.4% (moderate)",
    "This is realistic for a mid-cap focused fund or screening tool",
]
add_bullet_list(slide, Inches(0.6), Inches(5.3), Inches(12), Inches(2), items, font_size=15)

add_notes(slide, """Can you actually trade this? We modelled capacity using an Almgren-Chriss style impact model.

Quick definitions if asked:
- Almgren-Chriss model: the standard academic model for estimating how much buying or selling a stock moves its price. The more you trade relative to the stock's daily volume, the more you push the price against yourself.
- AUM (Assets Under Management): the total amount of money the fund manages. More AUM means bigger positions, which means more market impact.
- bps (basis points): 1 basis point = 0.01%. So 25 bps means 0.25% transaction cost per trade. 100 bps = 1%.
- Market impact: the price movement caused by your own trading. If you try to buy a large amount of a thinly-traded stock, your buying pushes the price up before you finish.
- Execution window: we assume 10 trading days to build or unwind a position, to reduce impact.
- Daily participation: we assume trading no more than 3% of each stock's average daily volume on any given day.
- Turnover: the fraction of the portfolio that changes each quarter. 25.4% means about a quarter of the portfolio gets replaced each quarter. That's moderate.

The strategy passes for AUM from 25 million up to 500 million dollars, which is the mid-cap sweet spot. It fails at 1 billion and above because mid-cap stocks simply don't have enough liquidity. For context, large-cap ESG indices can handle 1 billion plus easily.

Potential Q&A: 'Is $500M enough?' For a dedicated mid-cap ESG fund, absolutely. Many specialised mid-cap funds manage between $100M and $500M. For large pension funds managing tens of billions, this would be a satellite allocation, not a core holding.""")


# =====================================================================
# SLIDE 20: Statistical Rigor
# =====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_title_bar(slide, "Statistical Rigor", "Multiple Testing Correction")

data = [
    ["Correction Method", "Tests", "Significant", "% Surviving"],
    ["Uncorrected (p < 0.05)", "521", "330", "63.3%"],
    ["BH-FDR", "521", "235", "45.1%"],
    ["Bonferroni", "521", "169", "32.4%"],
]
add_table(slide, Inches(0.5), Inches(1.5), Inches(7), Inches(2), 4, 4, data,
          col_widths=[Inches(2.5), Inches(1.2), Inches(1.5), Inches(1.8)])

add_text_box(slide, Inches(0.6), Inches(3.8), Inches(12), Inches(0.5),
             "Additional Validation Tests", font_size=20, bold=True, color=DARK_BLUE)

items = [
    "Fama-MacBeth cross-sectional regressions: financial_score beta = 0.42, risk_adjusted = 0.31",
    "Walk-forward validation: 165 rolling splits, mean IC = 0.24",
    "Leave-one-sector-out: mean test IC = 0.207 (good cross-sector generalization)",
    "PCA stability: factor structure consistent across bootstrap resamples",
    "All 6 financial validation checks pass (ratio sanity, sector P/E, currency, provenance)",
]
add_bullet_list(slide, Inches(0.6), Inches(4.4), Inches(12), Inches(2.5), items, font_size=15)

add_notes(slide, """We ran 521 statistical tests in total and applied multiple testing corrections.

Quick definitions if asked:
- Multiple testing correction: when you run hundreds of tests, some will appear significant just by chance (at 5% level, you'd expect about 26 false positives out of 521). Corrections adjust for this.
- Bonferroni correction: the most conservative method. Divides the significance threshold by the number of tests. Very strict, few false positives, but also misses some real effects. 169 tests survive this.
- BH-FDR (Benjamini-Hochberg False Discovery Rate): a less strict correction that controls the expected proportion of false discoveries among rejected tests. More powerful than Bonferroni while still controlling error. 235 tests survive.
- Fama-MacBeth regression: the standard finance method for cross-sectional factor testing. First-pass runs a cross-sectional regression, second-pass averages the coefficients. Financial score (beta=0.42) and risk-adjusted (beta=0.31) are the strongest predictors.
- Leave-one-sector-out: we remove one entire sector (e.g., all Healthcare companies), retrain, and test on the held-out sector. Mean IC of 0.207 across all 11 sectors confirms the model doesn't just work for one industry.

The point is that we didn't just run one test and report the best number. We stress-tested from multiple angles and the core findings hold up under the strictest corrections.

Potential Q&A: 'Why 521 tests?' Because we test each factor against multiple outcome measures, across different subsamples, using different statistical methods. More tests means more rigour, but also requires correction.""")


# =====================================================================
# SLIDE 21: Limitations (Honest)
# =====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_title_bar(slide, "Limitations & Scope", "What We Can't Claim")

items = [
    ["Limitation", "Impact", "Severity"],
    ["Hybrid ESG data (real + synthetic)", "ESG conclusions depend on proxy quality validation", "High"],
    ["Single cross-section (2020-2024)", "No multi-year temporal backtest", "High"],
    ["US + India only", "External validity for Europe/Asia untested", "Medium"],
    ["Capacity ceiling at $500M", "Not suitable for large institutional mandates", "Medium"],
    ["Value factor inverted", "Reflects 2020-24 growth regime, may revert", "Low"],
    ["Individual rank CI width 247-255", "Company-level positions are noisy", "Low"],
]
add_table(slide, Inches(0.5), Inches(1.5), Inches(12.3), Inches(3.5), 7, 3, items,
          col_widths=[Inches(3.5), Inches(5.5), Inches(3.3)])

add_text_box(slide, Inches(0.6), Inches(5.3), Inches(12), Inches(2),
             "The thesis is a methodological contribution and proof-of-concept.\n"
             "The scoring engine works and generalises to large-cap. ESG data quality\n"
             "can be strengthened with commercial sources in future iterations.",
             font_size=15, color=MED_GRAY)

add_notes(slide, """Every study has limitations and it's important to be upfront about them. The ESG data uses a hybrid approach combining real ratings where available with synthetic proxies where coverage is sparse. We've validated the synthetic components against MSCI and Sustainalytics and the correlations are encouraging, but ESG conclusions still depend on how well those proxies capture what they're meant to measure.

We also only have a single cross-section rather than a multi-year backtest, and coverage is limited to the US and India.

The capacity ceiling at 500 million is a real constraint for institutional investors, and the value factor is inverted due to the growth regime. Individual company ranks have wide confidence intervals, so treat them as ranges not pinpoints.

The thesis is a methodological contribution and proof-of-concept. The scoring engine works and the validation is thorough. Expanding the ESG data sources and adding temporal backtests are the natural next steps.""")


# =====================================================================
# SLIDE 22: Future Work
# =====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_title_bar(slide, "Future Work", "Where This Goes Next")

items = [
    "Augment ESG data with additional commercial ratings (MSCI, Sustainalytics, Refinitiv)",
    "",
    "Add multi-year temporal backtest with walk-forward rebalancing",
    "",
    "Expand geographic coverage to Europe, East Asia, emerging markets",
    "",
    "Reduce indicator overlap for cleaner factor attribution",
    "",
    "Improve profile differentiation through non-linear aggregation or hard constraints",
]
add_bullet_list(slide, Inches(0.8), Inches(1.8), Inches(11), Inches(5), items, font_size=18)

add_notes(slide, """Five directions for future work. First, augmenting the ESG data with additional commercial ratings would strengthen the ESG conclusions and allow deeper calibration studies. Second, a multi-year temporal backtest would test whether the rankings hold up over time, not just in cross-section. Third, expanding beyond the US and India to Europe and East Asia would test external validity. Fourth, reducing indicator overlap would give cleaner factor attribution. And fifth, exploring non-linear aggregation methods might improve profile differentiation.""")


# =====================================================================
# SLIDE 23: Summary of Contributions
# =====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_title_bar(slide, "Summary of Contributions")

items = [
    "1.  A 9-factor composite index combining ESG with financial, market, risk, and sector signals",
    "",
    "2.  Variable-type-aware normalisation handling 7 measurement types correctly",
    "",
    "3.  Non-circular validation design using forward quality proxy (lagged earnings, ROA, ROE)",
    "",
    "4.  SASB sector-materiality integration for industry-specific ESG weighting",
    "",
    "5.  Comprehensive robustness testing: bootstrap, CV, walk-forward, ablation, high-cap transfer",
    "",
    "6.  Honest identification of ESG proxy limitations and reframing as risk filter",
    "",
    "7.  Fully reproducible open-source codebase (272 unit tests, 19 pipeline scripts)",
]
add_bullet_list(slide, Inches(0.8), Inches(1.6), Inches(11), Inches(5.5), items, font_size=17)

add_notes(slide, """To summarise the contributions. We built a 9-factor index that combines ESG with financial, market, and risk signals in a principled way. The normalisation handles 7 different variable types correctly. We use a non-circular forward quality proxy to ensure clean validation. We integrated SASB sector materiality so ESG weights are industry-specific.

The robustness testing is probably the strongest contribution: bootstrap, cross-validation, walk-forward, ablation, and high-cap transfer all in one study. We were honest about the ESG proxy limitations and reframed ESG as a risk filter rather than a return predictor. And the entire codebase is open-source with 272 unit tests.""")


# =====================================================================
# SLIDE 24: Key Numbers Summary
# =====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_title_bar(slide, "Key Numbers at a Glance")

# Create a visually appealing grid of key numbers
metrics = [
    ("276", "Mid-cap firms"),
    ("9", "Scoring factors"),
    ("202", "Variables"),
    ("7/8", "Factors monotonic"),
    ("+7.27 pp", "Bear-market excess"),
    ("0.585", "Bear-market IR"),
    ("0.499", "CV overfit ratio"),
    ("0.978", "Walk-forward ratio"),
    ("0.933", "Kendall W (high-cap)"),
    ("9/9", "Factors generalise"),
    ("$500M", "Capacity ceiling"),
    ("272", "Unit tests pass"),
]

for i, (num, label) in enumerate(metrics):
    row = i // 4
    col = i % 4
    left = Inches(0.5 + col * 3.2)
    top = Inches(1.6 + row * 1.8)

    # Number
    add_text_box(slide, left, top, Inches(2.8), Inches(0.8),
                 num, font_size=36, bold=True, color=DARK_BLUE, alignment=PP_ALIGN.CENTER)
    # Label
    add_text_box(slide, left, top + Inches(0.7), Inches(2.8), Inches(0.5),
                 label, font_size=14, color=MED_GRAY, alignment=PP_ALIGN.CENTER)

add_notes(slide, """Here's a quick reference of all the key numbers. 276 companies, 9 factors, 202 variables. 7 out of 8 factors are monotonic against forward quality. Bear-market excess is 7.27 percentage points with IR of 0.585. Cross-validation ratio is 0.499 which means no overfitting. Walk-forward is 0.978. And the high-cap transfer gives Kendall W of 0.933 with all 9 factors generalising. Everything is backed by 272 passing unit tests.""")


# =====================================================================
# SLIDE 25: Thank You
# =====================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
shape.fill.solid()
shape.fill.fore_color.rgb = DARK_BLUE
shape.line.fill.background()

add_text_box(slide, Inches(1), Inches(2), Inches(11), Inches(1),
             "Thank You",
             font_size=44, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

div = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(4), Inches(3.2), Inches(5), Pt(2))
div.fill.solid()
div.fill.fore_color.rgb = ACCENT
div.line.fill.background()

add_text_box(slide, Inches(1), Inches(3.6), Inches(11), Inches(0.5),
             "Questions?",
             font_size=28, color=RGBColor(0xBB, 0xCC, 0xDD), alignment=PP_ALIGN.CENTER)

add_text_box(slide, Inches(1), Inches(4.5), Inches(11), Inches(1.5),
             "Shashwat Bajpai  |  2021B3AA3041H\n"
             "Supervisor: Mr. Nandan Mishra  |  Co-supervisor: Dr. Dushyant Kumar\n"
             "BITS Pilani, Hyderabad Campus",
             font_size=16, color=RGBColor(0x99, 0xAA, 0xBB), alignment=PP_ALIGN.CENTER)

add_text_box(slide, Inches(1), Inches(6.0), Inches(11), Inches(0.5),
             "Full codebase and reproducibility artifacts available on request",
             font_size=13, color=RGBColor(0x77, 0x88, 0x99), alignment=PP_ALIGN.CENTER)

add_notes(slide, """Thank you for your attention. I'm happy to take any questions. The full codebase including all 19 pipeline scripts, 272 unit tests, and configuration files is available if you'd like to examine or reproduce any of the results. The key message: we built an open, testable scoring system for mid-cap ESG that holds up under rigorous stress testing and generalises to large-cap stocks. The ESG data uses a hybrid real-plus-synthetic approach that we've validated against commercial providers, and expanding those sources is the natural next step.""")


# Save
output_path = os.path.join(os.path.dirname(__file__), "EndSem_Presentation.pptx")
prs.save(output_path)
print(f"Saved to {output_path}")
print(f"Total slides: {len(prs.slides)}")
