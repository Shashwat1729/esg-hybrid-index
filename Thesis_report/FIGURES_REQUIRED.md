# Figures Required for MoBA-DB Thesis Report

This document lists all figures that need to be created and placed in the `Thesis_report/Pictures/` folder.

## Chapter 2: Literature Survey

### Figure 2.1: sparse_attention_patterns.png
**Description:** Show comparison of different sparse attention patterns
- Panel (a): Local window attention - tokens attending to neighbors within fixed window
- Panel (b): Strided attention - tokens attending to every k-th token
- Panel (c): Global attention - special tokens with full attention
- Panel (d): Sink attention - attention concentrated on sink tokens
**Suggested Tool:** Draw.io, PowerPoint, or Python matplotlib

### Figure 2.2: moba_architecture.png
**Description:** Architecture diagram of standard MoBA
- Show sequence divided into fixed blocks
- Illustrate routing mechanism between blocks
- Show intra-block and inter-block attention
**Suggested Tool:** Draw.io or Lucidchart

## Chapter 3: Methodology

### Figure 3.1: token_transfer_mechanism.png
**Description:** Visual explanation of token transfer between blocks
- Show two adjacent blocks (Block_i and Block_{i+1})
- Illustrate learnable transformation matrices (W_left, W_right)
- Show cosine similarity computation
- Arrows indicating token movement direction
**Suggested Tool:** Draw.io, PowerPoint, or TikZ in LaTeX

### Figure 3.2: dynamic_block_example.png
**Description:** Before and after comparison
- Before: Fixed blocks splitting related sentence "Einstein. This groundbreaking..."
- After: Dynamic blocks keeping context together
- Use actual text example with visual block boundaries
**Suggested Tool:** PowerPoint or Inkscape

## Chapter 4: Implementation

### Figure 4.1: system_architecture.png
**Description:** High-level system architecture
- Show three main components:
  1. Token Transfer Module
  2. Block Attention Module
  3. Routing Module
- Show data flow between components
- Integration points with Llama 3.1 8B
**Suggested Tool:** Draw.io or Lucidchart

### Figure 4.2: mobadb_attention_flow.png
**Description:** Data flow diagram for MoBA-DB attention layer
- Flowchart showing:
  1. Input (Q, K, V)
  2. Block Creation
  3. Transfer Score Computation
  4. Boundary Update
  5. Attention Computation
  6. Output
**Suggested Tool:** Draw.io or Flowchart software

## Chapter 5: Experiments and Results

### Figure 5.1: performance_comparison_chart.png
**Description:** Bar chart comparing models
- X-axis: Models (Baseline, MoBA, MoBA-DB)
- Y-axis: F1 Score (%)
- Include both EM and F1 scores as grouped bars
- Add error bars if multiple runs available
**Suggested Tool:** Python matplotlib or seaborn

### Figure 5.2: context_length_performance.png
**Description:** Line graph showing performance vs context length
- X-axis: Context length buckets (Short, Medium, Long)
- Y-axis: F1 Score
- Three lines for Baseline, MoBA, MoBA-DB
- Show performance gap increases with context length
**Suggested Tool:** Python matplotlib

### Figure 5.3: training_curves.png
**Description:** Training and validation loss curves
- X-axis: Training steps
- Y-axis: Loss
- Show curves for all three models
- Include validation performance markers
**Suggested Tool:** Python matplotlib (can extract from training logs)

## Chapter 6: Conclusion

### Figure 6.1: future_vision.png
**Description:** Vision for future extensions
- Show potential enhancements:
  1. Hierarchical blocks (multiple levels)
  2. Multi-modal inputs (text + images)
  3. Cross-block token transfer (non-adjacent)
  4. Integration with other LLMs
**Suggested Tool:** Draw.io or PowerPoint

## Appendix A

### Figure A.1: context_length_distribution.png
**Description:** Histogram of context lengths in SQuAD
- X-axis: Context length (tokens)
- Y-axis: Frequency
- Show distribution concentrated in 500-1500 range
**Suggested Tool:** Python matplotlib (can generate from SQuAD dataset)

### Figure A.2: token_transfer_heatmap.png
**Description:** Heatmap showing token transfer patterns
- X-axis: Block pairs (Block_0-1, Block_1-2, etc.)
- Y-axis: Test samples
- Color intensity: Number of tokens transferred
- Include direction indicators (arrows)
**Suggested Tool:** Python seaborn heatmap

### Figure A.3: attention_patterns.png
**Description:** Attention pattern visualization comparison
- Three panels side by side:
  (a) Full attention - dense matrix
  (b) Fixed block MoBA - block diagonal
  (c) Dynamic MoBA-DB - adaptive block diagonal
- Use attention weight matrices from actual model
**Suggested Tool:** Python matplotlib with imshow

## Optional but Recommended Figures

### complexity_comparison.png
**Description:** Graph showing computational complexity scaling
- X-axis: Sequence length (n)
- Y-axis: Computational cost
- Three curves: O(n²) full attention, O(n·b) MoBA, O(n·b) MoBA-DB
**Suggested Tool:** Python matplotlib

### block_size_distribution.png
**Description:** Distribution of block sizes after dynamic adjustment
- Show histogram of final block sizes
- Compare to initial fixed size
- Demonstrates active adaptation
**Suggested Tool:** Python matplotlib

## Quick Reference - Figure Specifications

All figures should be:
- **Format:** PNG or PDF (PNG preferred for compatibility)
- **Resolution:** 300 DPI minimum for raster images
- **Size:** Width should be reasonable for thesis page (typically 6-8 inches)
- **Colors:** Use colorblind-friendly palette if possible
- **Labels:** Clear axis labels, legends, and titles
- **Font:** Consistent with thesis (Times New Roman or similar)

## Priority Levels

**CRITICAL (Must have):**
- moba_architecture.png
- token_transfer_mechanism.png
- system_architecture.png
- mobadb_attention_flow.png
- performance_comparison_chart.png

**HIGH (Should have):**
- sparse_attention_patterns.png
- dynamic_block_example.png
- context_length_performance.png
- attention_patterns.png

**MEDIUM (Nice to have):**
- training_curves.png
- context_length_distribution.png
- token_transfer_heatmap.png
- future_vision.png

## Notes

- Wherever you see [BLANK_FIGURE] in the compiled PDF, that's where the figure should appear
- LaTeX will automatically handle figure placement and numbering
- All figure files should be saved in: `c:\Users\shbajpa\Downloads\MoBA_DB\Thesis_report\Pictures\`
- File names must match exactly as specified in the LaTeX code
