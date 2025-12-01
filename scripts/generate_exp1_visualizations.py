"""
Generate Visualizations for Experiment 1: Baseline Context Window Performance

Creates publication-ready plots showing:
1. Accuracy vs Context Size (line plot with confidence intervals)
2. Latency vs Context Size (line plot)
3. Accuracy Distribution by Size (box plot)
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
    """Load Experiment 1 results."""
    results_file = Path('results/raw/experiment_1_ollama_latest.json')
    
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    return data

def extract_metrics_by_position(results):
    """Extract metrics grouped by fact position."""
    trials = results.get('trials', [])
    
    # Group by position
    by_position = {}
    for trial in trials:
        position = trial['position']
        if position not in by_position:
            by_position[position] = {
                'accuracies': [],
                'latencies': [],
                'tokens': []
            }
        
        by_position[position]['accuracies'].append(1.0 if trial['correct'] else 0.0)
        by_position[position]['latencies'].append(trial['latency'])
        by_position[position]['tokens'].append(trial['tokens'])
    
    return by_position

def plot_accuracy_by_position(by_position, output_dir):
    """Plot accuracy by fact position."""
    positions = ['start', 'middle', 'end']
    mean_acc = [np.mean(by_position[p]['accuracies']) for p in positions]
    std_acc = [np.std(by_position[p]['accuracies']) for p in positions]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar plot with error bars
    x_pos = np.arange(len(positions))
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    bars = ax.bar(x_pos, mean_acc, yerr=std_acc, capsize=5,
                  color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    ax.set_xlabel('Fact Position in Context', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Experiment 1: "Lost in the Middle" - Accuracy by Position (Ollama llama2:latest)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([p.capitalize() for p in positions])
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, mean_acc):
        height = bar.get_height()
        ax.annotate(f'{val:.1%}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 5), textcoords="offset points",
                   ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / 'exp1_accuracy_by_position.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def plot_latency_by_position(by_position, output_dir):
    """Plot latency by fact position."""
    positions = ['start', 'middle', 'end']
    mean_lat = [np.mean(by_position[p]['latencies']) for p in positions]
    std_lat = [np.std(by_position[p]['latencies']) for p in positions]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar plot with error bars
    x_pos = np.arange(len(positions))
    colors = ['#3498db', '#9b59b6', '#e67e22']
    bars = ax.bar(x_pos, mean_lat, yerr=std_lat, capsize=5,
                  color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    ax.set_xlabel('Fact Position in Context', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latency (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Experiment 1: Query Latency by Position', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([p.capitalize() for p in positions])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, mean_lat):
        height = bar.get_height()
        ax.annotate(f'{val:.2f}s',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 5), textcoords="offset points",
                   ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / 'exp1_latency_by_position.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def plot_latency_distribution(by_position, output_dir):
    """Plot latency distribution as violin plots."""
    positions = ['start', 'middle', 'end']
    data_to_plot = [by_position[p]['latencies'] for p in positions]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Violin plot
    parts = ax.violinplot(data_to_plot, positions=[0, 1, 2],
                          showmeans=True, showmedians=True, widths=0.6)
    
    # Color violins
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    # Style means and medians
    parts['cmeans'].set_color('blue')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_color('darkred')
    parts['cmedians'].set_linewidth(2)
    
    ax.set_xlabel('Fact Position in Context', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latency (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Experiment 1: Latency Distribution by Position', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([p.capitalize() for p in positions])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = output_dir / 'exp1_latency_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

def main():
    """Generate all Experiment 1 visualizations."""
    print("=" * 80)
    print("GENERATING EXPERIMENT 1 VISUALIZATIONS")
    print("=" * 80)
    
    # Load results
    print("\nLoading results...")
    results = load_results()
    print(f"✓ Loaded {len(results.get('trials', []))} trials")
    
    # Extract metrics
    print("\nExtracting metrics by position...")
    by_position = extract_metrics_by_position(results)
    positions = sorted(by_position.keys())
    print(f"✓ Found {len(positions)} positions: {positions}")
    
    # Create output directory
    output_dir = Path('results/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_accuracy_by_position(by_position, output_dir)
    plot_latency_by_position(by_position, output_dir)
    plot_latency_distribution(by_position, output_dir)
    
    print("\n" + "=" * 80)
    print("✅ ALL EXPERIMENT 1 VISUALIZATIONS GENERATED!")
    print("=" * 80)
    print(f"Saved to: {output_dir}")

if __name__ == '__main__':
    main()
