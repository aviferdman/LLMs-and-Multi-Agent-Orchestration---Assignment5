"""
Generate Visualizations for Experiment 4: Context Engineering Strategies

Creates publication-ready plots comparing different context management strategies.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

def load_results():
    """Load Experiment 4 results."""
    results_file = Path('results/raw/experiment_4_ollama_latest.json')
    
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    return data

def plot_strategy_comparison(results, output_dir):
    """Compare accuracy across strategies."""
    summary = results.get('summary_statistics', {})
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    strategies = ['SELECT', 'COMPRESS', 'WRITE', 'HYBRID']
    accuracies = [
        summary.get('select', {}).get('mean_accuracy', 0),
        summary.get('compress', {}).get('mean_accuracy', 0),
        summary.get('write', {}).get('mean_accuracy', 0),
        summary.get('hybrid', {}).get('mean_accuracy', 0)
    ]
    stds = [
        summary.get('select', {}).get('std_accuracy', 0),
        summary.get('compress', {}).get('std_accuracy', 0),
        summary.get('write', {}).get('std_accuracy', 0),
        summary.get('hybrid', {}).get('std_accuracy', 0)
    ]
    
    # Colors
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    # Patterns for bars with same height (SELECT and HYBRID both 10%)
    patterns = ['///', '', '\\\\\\', 'xxx']
    
    # Bar plot with patterns
    x_pos = np.arange(len(strategies))
    bars = ax.bar(x_pos, accuracies, yerr=stds, capsize=5,
                  color=colors, alpha=0.7, edgecolor='black', linewidth=2.5)
    
    # Add patterns to distinguish bars
    for bar, pattern in zip(bars, patterns):
        bar.set_hatch(pattern)
    
    ax.set_xlabel('Context Engineering Strategy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Experiment 4: Context Engineering Strategy Comparison\n(SELECT & HYBRID both at 10%)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(strategies, fontsize=11)
    ax.set_ylim(0, 0.30)  # Reduced max to make small bars more visible
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add horizontal line at 10% to highlight SELECT and HYBRID
    ax.axhline(y=0.10, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='10% baseline')
    
    # Add value labels with background for visibility
    for i, (bar, val) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        # Position label above bar or inside if too tall
        y_offset = 5 if height < 0.25 else -15
        va = 'bottom' if height < 0.25 else 'top'
        
        ax.annotate(f'{val:.1%}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, y_offset), textcoords="offset points",
                   ha='center', va=va, fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    output_file = output_dir / 'exp4_strategy_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def plot_latency_by_strategy(results, output_dir):
    """Compare latency across strategies."""
    summary = results.get('summary_statistics', {})
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    strategies = ['SELECT', 'COMPRESS', 'WRITE', 'HYBRID']
    latencies = [
        summary.get('select', {}).get('mean_latency', 0),
        summary.get('compress', {}).get('mean_latency', 0),
        summary.get('write', {}).get('mean_latency', 0),
        summary.get('hybrid', {}).get('mean_latency', 0)
    ]
    stds = [
        summary.get('select', {}).get('std_latency', 0),
        summary.get('compress', {}).get('std_latency', 0),
        summary.get('write', {}).get('std_latency', 0),
        summary.get('hybrid', {}).get('std_latency', 0)
    ]
    
    # Colors
    colors = ['#9b59b6', '#1abc9c', '#e67e22', '#34495e']
    
    # Bar plot
    x_pos = np.arange(len(strategies))
    bars = ax.bar(x_pos, latencies, yerr=stds, capsize=5,
                  color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    ax.set_xlabel('Context Engineering Strategy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latency (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Experiment 4: Query Latency by Strategy', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(strategies)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, latencies):
        height = bar.get_height()
        ax.annotate(f'{val:.2f}s',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 5), textcoords="offset points",
                   ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / 'exp4_latency_by_strategy.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def plot_context_size_by_strategy(results, output_dir):
    """Compare context sizes across strategies."""
    summary = results.get('summary_statistics', {})
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    strategies = ['SELECT', 'COMPRESS', 'WRITE', 'HYBRID']
    context_sizes = [
        summary.get('select', {}).get('mean_context_size', 0),
        summary.get('compress', {}).get('mean_context_size', 0),
        summary.get('write', {}).get('mean_context_size', 0),
        summary.get('hybrid', {}).get('mean_context_size', 0)
    ]
    stds = [
        summary.get('select', {}).get('std_context_size', 0),
        summary.get('compress', {}).get('std_context_size', 0),
        summary.get('write', {}).get('std_context_size', 0),
        summary.get('hybrid', {}).get('std_context_size', 0)
    ]
    
    # Colors
    colors = ['#16a085', '#c0392b', '#2980b9', '#8e44ad']
    
    # Bar plot
    x_pos = np.arange(len(strategies))
    bars = ax.bar(x_pos, context_sizes, yerr=stds, capsize=5,
                  color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    ax.set_xlabel('Context Engineering Strategy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Context Size (tokens)', fontsize=12, fontweight='bold')
    ax.set_title('Experiment 4: Context Size by Strategy', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(strategies)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, context_sizes):
        height = bar.get_height()
        ax.annotate(f'{val:.1f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 5), textcoords="offset points",
                   ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / 'exp4_context_size_by_strategy.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def plot_accuracy_vs_context_size(results, output_dir):
    """Plot accuracy vs context size trade-off."""
    summary = results.get('summary_statistics', {})
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Extract data
    strategies = ['SELECT', 'COMPRESS', 'WRITE', 'HYBRID']
    accuracies = [
        summary.get('select', {}).get('mean_accuracy', 0),
        summary.get('compress', {}).get('mean_accuracy', 0),
        summary.get('write', {}).get('mean_accuracy', 0),
        summary.get('hybrid', {}).get('mean_accuracy', 0)
    ]
    context_sizes = [
        summary.get('select', {}).get('mean_context_size', 0),
        summary.get('compress', {}).get('mean_context_size', 0),
        summary.get('write', {}).get('mean_context_size', 0),
        summary.get('hybrid', {}).get('mean_context_size', 0)
    ]
    
    # Apply jitter to separate overlapping points
    # SELECT (31.0, 0.1) and HYBRID (31.0, 0.1) overlap completely
    # COMPRESS (42.5, 0.233) and WRITE (42.5, 0.2) have same x-coordinate
    jitter_amounts = [
        (-0.8, -0.005),  # SELECT: shift left and down
        (0.5, 0.005),     # COMPRESS: shift right and up
        (0.5, -0.005),    # WRITE: shift right and down
        (0.8, 0.005)      # HYBRID: shift right and up
    ]
    
    context_sizes_jittered = [ctx + jitter[0] for ctx, jitter in zip(context_sizes, jitter_amounts)]
    accuracies_jittered = [acc + jitter[1] for acc, jitter in zip(accuracies, jitter_amounts)]
    
    # Colors and markers - more distinct
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    markers = ['o', 's', '^', 'D']
    sizes = [300, 250, 250, 300]  # Make SELECT and HYBRID larger
    
    # Scatter plot with jittered positions
    for strategy, acc, acc_jit, ctx, ctx_jit, color, marker, size in zip(
        strategies, accuracies, accuracies_jittered, context_sizes, context_sizes_jittered, colors, markers, sizes
    ):
        # Plot with jittered position
        ax.scatter(ctx_jit, acc_jit, s=size, color=color, marker=marker,
                  edgecolor='black', linewidth=2.5, label=strategy, alpha=0.85, zorder=3)
        
        # Draw a subtle line from jittered position to actual position
        if abs(ctx - ctx_jit) > 0.1 or abs(acc - acc_jit) > 0.001:
            ax.plot([ctx_jit, ctx], [acc_jit, acc], 'k--', alpha=0.2, linewidth=1, zorder=1)
            # Mark actual position with small cross
            ax.scatter(ctx, acc, s=30, color='gray', marker='x', alpha=0.4, zorder=2)
    
    ax.set_xlabel('Average Context Size (tokens)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Experiment 4: Accuracy vs Context Size Trade-off\n(Points offset for visibility; dashed lines show actual positions)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    
    # Annotate points with actual values
    for strategy, acc, ctx, acc_jit, ctx_jit in zip(strategies, accuracies, context_sizes, accuracies_jittered, context_sizes_jittered):
        # Determine annotation position based on jitter direction
        if strategy == 'SELECT':
            xytext = (-30, 10)
            ha = 'right'
        elif strategy == 'HYBRID':
            xytext = (30, 10)
            ha = 'left'
        elif strategy == 'COMPRESS':
            xytext = (0, 15)
            ha = 'center'
        else:  # WRITE
            xytext = (0, -20)
            ha = 'center'
        
        ax.annotate(f'{strategy}\n({ctx:.1f}, {acc:.1%})', 
                   (ctx_jit, acc_jit),
                   textcoords="offset points", 
                   xytext=xytext,
                   ha=ha, 
                   fontsize=9, 
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray', linewidth=1))
    
    # Add note about overlapping points
    ax.text(0.02, 0.98, 
            'Note: SELECT & HYBRID have identical values (31.0, 10%)\n' +
            'COMPRESS & WRITE share same context size (42.5)\n' +
            'Points are slightly offset for visibility',
            transform=ax.transAxes, 
            fontsize=9, 
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8, edgecolor='orange', linewidth=1.5))
    
    plt.tight_layout()
    output_file = output_dir / 'exp4_accuracy_vs_context_size.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def main():
    """Generate all Experiment 4 visualizations."""
    print("=" * 80)
    print("GENERATING EXPERIMENT 4 VISUALIZATIONS")
    print("=" * 80)
    
    # Load results
    print("\nLoading results...")
    results = load_results()
    print(f"✓ Loaded results")
    
    # Create output directory
    output_dir = Path('results/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_strategy_comparison(results, output_dir)
    plot_latency_by_strategy(results, output_dir)
    plot_context_size_by_strategy(results, output_dir)
    plot_accuracy_vs_context_size(results, output_dir)
    
    print("\n" + "=" * 80)
    print("✅ ALL EXPERIMENT 4 VISUALIZATIONS GENERATED!")
    print("=" * 80)
    print(f"Saved to: {output_dir}")

if __name__ == '__main__':
    main()
