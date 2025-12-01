"""
Generate Visualizations for Experiment 3: RAG vs Full Context

Creates publication-ready plots comparing RAG and Full Context approaches.
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
    """Load Experiment 3 results."""
    results_file = Path('results/raw/experiment_3_ollama_latest.json')
    
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    return data

def plot_accuracy_comparison(results, output_dir):
    """Compare accuracy between RAG and Full Context."""
    metrics = results.get('aggregate_metrics', {})
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data from by_top_k and by_approach
    by_top_k = metrics.get('by_top_k', {})
    by_approach = metrics.get('by_approach', {})
    
    methods = ['RAG\n(top-k=1)', 'RAG\n(top-k=3)', 'RAG\n(top-k=5)', 'Full Context']
    accuracies = [
        by_top_k.get('k=1', {}).get('accuracy', 0),
        by_top_k.get('k=3', {}).get('accuracy', 0),
        by_top_k.get('k=5', {}).get('accuracy', 0),
        by_approach.get('FullContext', {}).get('accuracy', 0)
    ]
    # No std available in this format, use zeros
    stds = [0, 0, 0, 0]
    
    # Colors
    colors = ['#3498db', '#2980b9', '#1f618d', '#e74c3c']
    
    # Bar plot
    x_pos = np.arange(len(methods))
    bars = ax.bar(x_pos, accuracies, yerr=stds, capsize=5,
                  color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    ax.set_xlabel('Retrieval Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Experiment 3: RAG vs Full Context Accuracy Comparison', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{val:.1%}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 5), textcoords="offset points",
                   ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / 'exp3_accuracy_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def plot_latency_comparison(results, output_dir):
    """Compare latency between RAG and Full Context."""
    metrics = results.get('aggregate_metrics', {})
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    by_top_k = metrics.get('by_top_k', {})
    by_approach = metrics.get('by_approach', {})
    
    methods = ['RAG\n(top-k=1)', 'RAG\n(top-k=3)', 'RAG\n(top-k=5)', 'Full Context']
    latencies = [
        by_top_k.get('k=1', {}).get('mean_latency', 0),
        by_top_k.get('k=3', {}).get('mean_latency', 0),
        by_top_k.get('k=5', {}).get('mean_latency', 0),
        by_approach.get('FullContext', {}).get('mean_latency', 0)
    ]
    stds = [
        by_top_k.get('k=1', {}).get('std_latency', 0),
        by_top_k.get('k=3', {}).get('std_latency', 0),
        by_top_k.get('k=5', {}).get('std_latency', 0),
        by_approach.get('FullContext', {}).get('std_latency', 0)
    ]
    
    # Colors
    colors = ['#2ecc71', '#27ae60', '#1e8449', '#f39c12']
    
    # Bar plot
    x_pos = np.arange(len(methods))
    bars = ax.bar(x_pos, latencies, yerr=stds, capsize=5,
                  color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    ax.set_xlabel('Retrieval Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latency (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Experiment 3: Query Latency Comparison', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, latencies):
        height = bar.get_height()
        ax.annotate(f'{val:.1f}s',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 5), textcoords="offset points",
                   ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / 'exp3_latency_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def plot_accuracy_vs_latency(results, output_dir):
    """Plot accuracy vs latency trade-off."""
    metrics = results.get('aggregate_metrics', {})
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    by_top_k = metrics.get('by_top_k', {})
    by_approach = metrics.get('by_approach', {})
    
    methods = ['RAG (k=1)', 'RAG (k=3)', 'RAG (k=5)', 'Full Context']
    accuracies = [
        by_top_k.get('k=1', {}).get('accuracy', 0),
        by_top_k.get('k=3', {}).get('accuracy', 0),
        by_top_k.get('k=5', {}).get('accuracy', 0),
        by_approach.get('FullContext', {}).get('accuracy', 0)
    ]
    latencies = [
        by_top_k.get('k=1', {}).get('mean_latency', 0),
        by_top_k.get('k=3', {}).get('mean_latency', 0),
        by_top_k.get('k=5', {}).get('mean_latency', 0),
        by_approach.get('FullContext', {}).get('mean_latency', 0)
    ]
    
    # Colors and markers
    colors = ['#3498db', '#2980b9', '#1f618d', '#e74c3c']
    markers = ['o', 's', '^', 'D']
    
    # Scatter plot with connecting line
    for i, (method, acc, lat, color, marker) in enumerate(zip(methods, accuracies, latencies, colors, markers)):
        ax.scatter(lat, acc, s=200, color=color, marker=marker, 
                  edgecolor='black', linewidth=2, label=method, alpha=0.8, zorder=3)
    
    # Connect RAG points with line
    ax.plot(latencies[:3], accuracies[:3], 'b--', alpha=0.3, linewidth=2, zorder=1)
    
    ax.set_xlabel('Latency (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Experiment 3: Accuracy vs Latency Trade-off', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    
    # Annotate points
    for method, acc, lat in zip(methods, accuracies, latencies):
        ax.annotate(f'{acc:.1%}', (lat, acc), 
                   textcoords="offset points", xytext=(0, -20),
                   ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / 'exp3_accuracy_vs_latency.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def main():
    """Generate all Experiment 3 visualizations."""
    print("=" * 80)
    print("GENERATING EXPERIMENT 3 VISUALIZATIONS")
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
    plot_accuracy_comparison(results, output_dir)
    plot_latency_comparison(results, output_dir)
    plot_accuracy_vs_latency(results, output_dir)
    
    print("\n" + "=" * 80)
    print("✅ ALL EXPERIMENT 3 VISUALIZATIONS GENERATED!")
    print("=" * 80)
    print(f"Saved to: {output_dir}")

if __name__ == '__main__':
    main()
