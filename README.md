# ESG Hybrid Index

A multi-factor framework integrating ESG metrics with financial factors for portfolio construction and analysis.

## Overview

Constructs a hybrid ESG index that combines environmental, social, and governance scores with traditional financial factors (value, growth, quality, size, momentum) to identify companies with strong sustainability profiles and financial performance.

## Quick Start

`ash
pip install -r requirements.txt
python scripts/run_all.py
`

## Project Structure

- src/ - Core index construction, scoring, similarity analysis
- scripts/ - Pipeline: download → clean → build → validate → report
- 	ests/ - Unit and integration tests
- config/ - Data sources, index parameters, materiality mappings

## Results

Generated reports and visualizations in eports/ and docs/figures/.