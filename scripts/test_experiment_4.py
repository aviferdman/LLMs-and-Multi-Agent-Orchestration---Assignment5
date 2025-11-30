"""
Test Script for Experiment 4: Context Engineering Strategies

This script runs Experiment 4 to test different context management strategies
across multi-turn conversations.

Usage:
    python scripts/test_experiment_4.py
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import directly to avoid __init__.py import errors from other experiments
import importlib.util
spec = importlib.util.spec_from_file_location(
    "experiment_4",
    Path(__file__).parent.parent / 'src' / 'experiments' / 'experiment_4.py'
)
experiment_4_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment_4_module)
Experiment4 = experiment_4_module.Experiment4

from loguru import logger


def setup_logging():
    """Configure logging for the test."""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "logs/experiment_4_test.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG"
    )


def create_test_config():
    """
    Create test configuration for Experiment 4.

    Returns:
        Dictionary with experiment configuration
    """
    config = {
        'model': 'llama2:13b',
        'num_steps': 10,
        'max_tokens': 2000,
        'num_runs': 3,
        'scenario_type': 'sequential',
        'output_dir': 'results/experiment_4/',
        'random_seed': 42
    }
    return config


def save_results(results, analysis, output_dir):
    """
    Save experiment results to files.

    Args:
        results: Results from experiment run
        analysis: Analysis results
        output_dir: Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save raw results
    results_file = output_path / f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

    # Convert results to JSON-serializable format
    json_results = {
        'summary_statistics': results['summary_statistics'],
        'analysis': analysis,
        'timestamp': datetime.now().isoformat()
    }

    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    logger.info(f"Results saved to {results_file}")

    # Save summary report
    report_file = output_path / 'experiment_4_summary.txt'
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EXPERIMENT 4: Context Engineering Strategies - Test Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("STRATEGY PERFORMANCE SUMMARY\n")
        f.write("-" * 80 + "\n\n")

        for strategy_name, stats in results['summary_statistics'].items():
            f.write(f"{strategy_name.upper()} Strategy:\n")
            f.write(f"  Mean Accuracy: {stats['mean_accuracy']:.3f} ± {stats['std_accuracy']:.3f}\n")
            f.write(f"  Mean Context Size: {stats['mean_context_size']:.1f} ± {stats['std_context_size']:.1f} tokens\n")
            f.write(f"  Mean Latency: {stats['mean_latency']:.3f} ± {stats['std_latency']:.3f} seconds\n")
            f.write(f"  Total Compressions: {stats['total_compressions']}\n")
            f.write(f"  Recommendation: {analysis['recommendations'].get(strategy_name, 'N/A')}\n")
            f.write("\n")

        f.write("STRATEGY RANKINGS\n")
        f.write("-" * 80 + "\n\n")

        for strategy, rank in sorted(analysis['strategy_rankings'].items(), key=lambda x: x[1]):
            f.write(f"  Rank {rank}: {strategy.upper()}\n")

        f.write(f"\nBest Strategy: {analysis['best_strategy'].upper()}\n")

    logger.info(f"Summary report saved to {report_file}")


def print_summary(results, analysis):
    """
    Print summary of results to console.

    Args:
        results: Results from experiment run
        analysis: Analysis results
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: Context Engineering Strategies - Test Results")
    print("=" * 80 + "\n")

    print("STRATEGY PERFORMANCE:")
    print("-" * 80)

    for strategy_name, stats in results['summary_statistics'].items():
        print(f"\n{strategy_name.upper()}:")
        print(f"  Accuracy: {stats['mean_accuracy']:.3f} ± {stats['std_accuracy']:.3f}")
        print(f"  Context Size: {stats['mean_context_size']:.1f} ± {stats['std_context_size']:.1f} tokens")
        print(f"  Latency: {stats['mean_latency']:.3f} ± {stats['std_latency']:.3f} seconds")
        print(f"  Compressions: {stats['total_compressions']}")
        print(f"  Recommendation: {analysis['recommendations'].get(strategy_name, 'N/A')}")

    print("\n" + "-" * 80)
    print("RANKINGS:")
    print("-" * 80)

    for strategy, rank in sorted(analysis['strategy_rankings'].items(), key=lambda x: x[1]):
        print(f"  #{rank}: {strategy.upper()}")

    print(f"\nBest Strategy: {analysis['best_strategy'].upper()}")
    print("=" * 80 + "\n")


def main():
    """Main test function."""
    setup_logging()

    logger.info("=" * 80)
    logger.info("Starting Experiment 4 Test: Context Engineering Strategies")
    logger.info("=" * 80)

    try:
        # Create configuration
        logger.info("Creating test configuration...")
        config = create_test_config()
        logger.info(f"Configuration: {config}")

        # Initialize experiment
        logger.info("Initializing Experiment 4...")
        experiment = Experiment4(config)
        logger.success("Experiment 4 initialized successfully")

        # Run experiment
        logger.info("Running experiment with all strategies...")
        logger.info("This may take a few minutes...")
        results = experiment.run()
        logger.success("Experiment run completed")

        # Analyze results
        logger.info("Analyzing results...")
        analysis = experiment.analyze(results)
        logger.success("Analysis completed")

        # Print summary
        print_summary(results, analysis)

        # Save results
        logger.info("Saving results...")
        save_results(results, analysis, config['output_dir'])
        logger.success("Results saved successfully")

        logger.success("=" * 80)
        logger.success("Experiment 4 Test Completed Successfully")
        logger.success("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Error during experiment: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
