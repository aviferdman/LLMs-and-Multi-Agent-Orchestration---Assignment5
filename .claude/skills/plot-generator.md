---
name: plot-generator
description: Creates publication-quality plots with matplotlib, seaborn, and plotly. Handles styling, annotations, and export.
allowed-tools: Bash, Read, Write
---

# Plot Generator Skill

Creates publication-quality visualizations with proper styling and annotations.

## Plot Types

- Bar plots with error bars
- Box plots with significance markers
- Line plots with confidence intervals
- Scatter plots with regression lines
- Heatmaps
- Violin plots
- Distribution plots

## Features

- 300 DPI PNG export
- Color-blind friendly palettes
- Statistical annotations (p-values, effect sizes)
- Automatic legend placement
- Grid styling
- Consistent fonts and sizes

## Usage

```python
from src.visualization.plots import (
    plot_accuracy_by_position,
    plot_scaling_curves,
    plot_comparison_bars
)

# Create plot
fig, ax = plot_accuracy_by_position(
    data=results,
    save_path="results/figures/exp1_accuracy.png"
)
```

## Styling Standards

- Figure size: (10, 6) inches
- DPI: 300
- Font: Arial or Helvetica
- Font size: 12pt for labels, 10pt for ticks
- Line width: 2pt
- Color palette: colorblind-friendly

## When to Use

Use this skill for:
- Experiment result visualization
- Statistical analysis plots
- Report figures
- Presentation graphics
