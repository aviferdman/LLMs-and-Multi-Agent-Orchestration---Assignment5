"""
Visualization utilities for Context Windows Research.

This module provides publication-quality plotting functions for experimental results,
including bar charts, line plots, heatmaps, and statistical visualizations.

Classes:
    Visualizer: Main visualization class

Example:
    >>> viz = Visualizer(output_dir="results/figures", dpi=300)
    >>> viz.plot_accuracy_by_position(data, save_path="exp1_accuracy.png")
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from scipy import stats


class Visualizer:
    """
    Publication-quality visualization generator.

    Provides methods for creating charts, plots, and figures with
    consistent styling and formatting.

    Attributes:
        output_dir (Path): Directory for saving figures
        dpi (int): Resolution for saved figures
        style (str): Matplotlib style to use
        color_palette (str): Seaborn color palette
    """

    def __init__(
        self,
        output_dir: str = "results/figures",
        dpi: int = 300,
        style: str = "seaborn-v0_8-paper",
        color_palette: str = "colorblind",
    ):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory for saving figures
            dpi: Resolution for saved figures (300 for publication quality)
            style: Matplotlib style
            color_palette: Seaborn color palette
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dpi = dpi
        self.style = style
        self.color_palette = color_palette

        # Set style
        try:
            plt.style.use(style)
        except:
            logger.warning(f"Style '{style}' not available, using default")

        sns.set_palette(color_palette)

        logger.info(f"Initialized Visualizer: output_dir={output_dir}, dpi={dpi}")

    def plot_accuracy_by_position(
        self,
        data: Dict[str, List[float]],
        title: str = "Accuracy by Position",
        save_path: Optional[str] = None,
        show_stats: bool = True,
    ):
        """
        Create bar chart of accuracy by position with error bars.

        Args:
            data: Dictionary mapping positions to accuracy lists
                 (e.g., {"start": [0.9, 0.85, ...], "middle": [...], "end": [...]})
            title: Plot title
            save_path: Optional path to save figure (relative to output_dir)
            show_stats: Whether to show statistical significance annotations

        Example:
            >>> viz = Visualizer()
            >>> data = {"start": [0.9, 0.92], "middle": [0.7, 0.68], "end": [0.88, 0.90]}
            >>> viz.plot_accuracy_by_position(data)
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Calculate means and confidence intervals
        positions = list(data.keys())
        means = [np.mean(data[pos]) for pos in positions]
        cis = [stats.sem(data[pos]) * 1.96 for pos in positions]  # 95% CI

        # Create bar plot
        bars = ax.bar(positions, means, yerr=cis, capsize=5, alpha=0.7)

        # Add value labels on bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
            )

        # Statistical annotations
        if show_stats and len(positions) >= 2:
            # Add significance stars if appropriate
            # Simple example: compare first vs second position
            if len(positions) >= 2:
                t_stat, p_value = stats.ttest_ind(data[positions[0]], data[positions[1]])
                if p_value < 0.05:
                    y_max = max(means) + max(cis) + 0.05
                    ax.plot([0, 1], [y_max, y_max], "k-", linewidth=1)
                    stars = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
                    ax.text(0.5, y_max, stars, ha="center", va="bottom")

        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_xlabel("Position", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        if save_path:
            full_path = self.output_dir / save_path
            plt.savefig(full_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved figure: {full_path}")

        plt.close()

    def plot_scaling_curve(
        self,
        x: List[float],
        y: List[float],
        xlabel: str = "Context Size",
        ylabel: str = "Accuracy",
        title: str = "Context Size Scaling",
        fit_curve: bool = True,
        save_path: Optional[str] = None,
    ):
        """
        Create scatter plot with fitted curve showing scaling behavior.

        Args:
            x: X-axis values (e.g., context sizes)
            y: Y-axis values (e.g., accuracy scores)
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title
            fit_curve: Whether to fit and plot regression curve
            save_path: Optional path to save figure

        Example:
            >>> viz = Visualizer()
            >>> x = [2, 5, 10, 20, 50]
            >>> y = [0.95, 0.92, 0.88, 0.82, 0.75]
            >>> viz.plot_scaling_curve(x, y)
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Scatter plot
        ax.scatter(x, y, s=100, alpha=0.6, label="Observed")

        # Fit curve
        if fit_curve and len(x) >= 3:
            # Logarithmic fit
            x_arr = np.array(x)
            y_arr = np.array(y)

            # Log transform x
            log_x = np.log(x_arr)
            coeffs = np.polyfit(log_x, y_arr, 1)

            # Generate smooth curve
            x_smooth = np.linspace(min(x), max(x), 100)
            y_smooth = coeffs[0] * np.log(x_smooth) + coeffs[1]

            ax.plot(x_smooth, y_smooth, "r-", linewidth=2, label="Fitted curve")

            # Add equation
            equation = f"y = {coeffs[0]:.3f}*log(x) + {coeffs[1]:.3f}"
            ax.text(
                0.05,
                0.95,
                equation,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            full_path = self.output_dir / save_path
            plt.savefig(full_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved figure: {full_path}")

        plt.close()

    def plot_comparison_bars(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        hue_col: Optional[str] = None,
        title: str = "Comparison",
        save_path: Optional[str] = None,
    ):
        """
        Create grouped bar chart for comparing multiple conditions.

        Args:
            data: DataFrame containing data
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            hue_col: Optional column for grouping
            title: Plot title
            save_path: Optional path to save figure

        Example:
            >>> viz = Visualizer()
            >>> df = pd.DataFrame({
            ...     'method': ['RAG', 'Full', 'RAG', 'Full'],
            ...     'metric': ['accuracy', 'accuracy', 'latency', 'latency'],
            ...     'value': [0.85, 0.90, 0.5, 2.0]
            ... })
            >>> viz.plot_comparison_bars(df, 'metric', 'value', 'method')
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        if hue_col:
            sns.barplot(data=data, x=x_col, y=y_col, hue=hue_col, ax=ax)
        else:
            sns.barplot(data=data, x=x_col, y=y_col, ax=ax)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(x_col.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel(y_col.replace("_", " ").title(), fontsize=12)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        if save_path:
            full_path = self.output_dir / save_path
            plt.savefig(full_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved figure: {full_path}")

        plt.close()

    def plot_heatmap(
        self,
        data: np.ndarray,
        row_labels: List[str],
        col_labels: List[str],
        title: str = "Heatmap",
        cmap: str = "YlOrRd",
        save_path: Optional[str] = None,
    ):
        """
        Create heatmap visualization.

        Args:
            data: 2D array of values
            row_labels: Labels for rows
            col_labels: Labels for columns
            title: Plot title
            cmap: Colormap name
            save_path: Optional path to save figure

        Example:
            >>> viz = Visualizer()
            >>> data = np.random.rand(3, 5)
            >>> viz.plot_heatmap(data, ['A', 'B', 'C'], ['1', '2', '3', '4', '5'])
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        sns.heatmap(
            data,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            xticklabels=col_labels,
            yticklabels=row_labels,
            ax=ax,
            cbar_kws={"label": "Value"},
        )

        ax.set_title(title, fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_path:
            full_path = self.output_dir / save_path
            plt.savefig(full_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved figure: {full_path}")

        plt.close()

    def plot_line_chart(
        self,
        data: Dict[str, List[Tuple[float, float]]],
        xlabel: str = "Step",
        ylabel: str = "Value",
        title: str = "Line Chart",
        save_path: Optional[str] = None,
    ):
        """
        Create line chart with multiple series.

        Args:
            data: Dictionary mapping series names to (x, y) points
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title
            save_path: Optional path to save figure

        Example:
            >>> viz = Visualizer()
            >>> data = {
            ...     'Strategy A': [(1, 0.9), (2, 0.88), (3, 0.85)],
            ...     'Strategy B': [(1, 0.85), (2, 0.84), (3, 0.83)]
            ... }
            >>> viz.plot_line_chart(data)
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        for series_name, points in data.items():
            x_vals = [p[0] for p in points]
            y_vals = [p[1] for p in points]
            ax.plot(x_vals, y_vals, marker="o", label=series_name, linewidth=2)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            full_path = self.output_dir / save_path
            plt.savefig(full_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved figure: {full_path}")

        plt.close()

    def plot_box_plots(
        self,
        data: Dict[str, List[float]],
        title: str = "Box Plot",
        ylabel: str = "Value",
        save_path: Optional[str] = None,
    ):
        """
        Create box plots for distribution comparison.

        Args:
            data: Dictionary mapping categories to value lists
            title: Plot title
            ylabel: Y-axis label
            save_path: Optional path to save figure

        Example:
            >>> viz = Visualizer()
            >>> data = {'A': [1, 2, 3, 4, 5], 'B': [2, 3, 4, 5, 6]}
            >>> viz.plot_box_plots(data)
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        positions = list(data.keys())
        values = [data[pos] for pos in positions]

        bp = ax.boxplot(values, labels=positions, patch_artist=True)

        # Color boxes
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
            patch.set_alpha(0.7)

        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        if save_path:
            full_path = self.output_dir / save_path
            plt.savefig(full_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved figure: {full_path}")

        plt.close()

    def __repr__(self) -> str:
        """String representation of Visualizer."""
        return f"Visualizer(output_dir={self.output_dir}, dpi={self.dpi})"


# Standalone plotting functions for specific experiments

def plot_rag_comparison(
    query_results: List[Any],
    save_path: Optional[str] = None,
    dpi: int = 300
) -> None:
    """
    Plot comparison between RAG and Full Context approaches.

    Args:
        query_results: List of QueryResult objects
        save_path: Optional path to save figure
        dpi: Resolution for saved figure
    """
    # Prepare data
    rag_results = [r for r in query_results if r.approach == "RAG"]
    full_results = [r for r in query_results if r.approach == "FullContext"]

    # Calculate metrics for each approach
    metrics = {}

    # RAG metrics
    if rag_results:
        rag_factual = [r for r in rag_results if r.ground_truth is not None]
        metrics['RAG'] = {
            'accuracy': np.mean([r.is_correct for r in rag_factual]) if rag_factual else 0,
            'latency': np.mean([r.latency for r in rag_results]),
            'latency_std': np.std([r.latency for r in rag_results])
        }

    # Full Context metrics
    if full_results:
        full_factual = [r for r in full_results if r.ground_truth is not None]
        metrics['FullContext'] = {
            'accuracy': np.mean([r.is_correct for r in full_factual]) if full_factual else 0,
            'latency': np.mean([r.latency for r in full_results]),
            'latency_std': np.std([r.latency for r in full_results])
        }

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Accuracy comparison
    approaches = list(metrics.keys())
    accuracies = [metrics[app]['accuracy'] for app in approaches]

    axes[0].bar(approaches, accuracies, alpha=0.7, color=['#1f77b4', '#ff7f0e'])
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Accuracy: RAG vs Full Context', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (app, acc) in enumerate(zip(approaches, accuracies)):
        axes[0].text(i, acc + 0.02, f'{acc:.3f}', ha='center', va='bottom', fontsize=10)

    # Plot 2: Latency comparison
    latencies = [metrics[app]['latency'] for app in approaches]
    latency_stds = [metrics[app]['latency_std'] for app in approaches]

    axes[1].bar(approaches, latencies, yerr=latency_stds, capsize=5, alpha=0.7,
                color=['#1f77b4', '#ff7f0e'])
    axes[1].set_ylabel('Latency (seconds)', fontsize=12)
    axes[1].set_title('Latency: RAG vs Full Context', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (app, lat) in enumerate(zip(approaches, latencies)):
        axes[1].text(i, lat + latency_stds[i] + 0.1, f'{lat:.2f}s',
                     ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved RAG comparison plot: {save_path}")

    plt.close()


def plot_top_k_analysis(
    query_results: List[Any],
    save_path: Optional[str] = None,
    dpi: int = 300
) -> None:
    """
    Plot analysis of different top_k values for RAG.

    Args:
        query_results: List of QueryResult objects
        save_path: Optional path to save figure
        dpi: Resolution for saved figure
    """
    # Filter RAG results only
    rag_results = [r for r in query_results if r.approach == "RAG"]

    if not rag_results:
        logger.warning("No RAG results found for top_k analysis")
        return

    # Group by top_k
    top_k_values = sorted(set(r.top_k for r in rag_results if r.top_k is not None))

    if len(top_k_values) == 0:
        logger.warning("No top_k values found in RAG results")
        return

    # Calculate metrics for each top_k
    metrics_by_k = {}
    for k in top_k_values:
        k_results = [r for r in rag_results if r.top_k == k]
        k_factual = [r for r in k_results if r.ground_truth is not None]

        metrics_by_k[k] = {
            'accuracy': np.mean([r.is_correct for r in k_factual]) if k_factual else 0,
            'accuracy_std': np.std([r.is_correct for r in k_factual]) if k_factual else 0,
            'latency': np.mean([r.latency for r in k_results]),
            'latency_std': np.std([r.latency for r in k_results])
        }

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Accuracy vs top_k
    accuracies = [metrics_by_k[k]['accuracy'] for k in top_k_values]
    accuracy_stds = [metrics_by_k[k]['accuracy_std'] for k in top_k_values]

    axes[0].plot(top_k_values, accuracies, marker='o', linewidth=2, markersize=8)
    axes[0].fill_between(
        top_k_values,
        np.array(accuracies) - np.array(accuracy_stds),
        np.array(accuracies) + np.array(accuracy_stds),
        alpha=0.3
    )
    axes[0].set_xlabel('Top-K Value', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Accuracy vs Top-K', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(alpha=0.3)

    # Plot 2: Latency vs top_k
    latencies = [metrics_by_k[k]['latency'] for k in top_k_values]
    latency_stds = [metrics_by_k[k]['latency_std'] for k in top_k_values]

    axes[1].plot(top_k_values, latencies, marker='s', linewidth=2, markersize=8,
                 color='#ff7f0e')
    axes[1].fill_between(
        top_k_values,
        np.array(latencies) - np.array(latency_stds),
        np.array(latencies) + np.array(latency_stds),
        alpha=0.3,
        color='#ff7f0e'
    )
    axes[1].set_xlabel('Top-K Value', fontsize=12)
    axes[1].set_ylabel('Latency (seconds)', fontsize=12)
    axes[1].set_title('Latency vs Top-K', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved top-k analysis plot: {save_path}")

    plt.close()
