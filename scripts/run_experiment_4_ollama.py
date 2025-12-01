"""
Run Experiment 4: Context Engineering Strategies with Real Ollama LLM

This script executes Experiment 4 with real Ollama to test different
context management strategies (SELECT, COMPRESS, WRITE, HYBRID) across
multi-turn conversations.

Expected Runtime: ~30-40 minutes (120-200 queries)
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments.experiment_4 import Experiment4
from loguru import logger

def setup_logging():
    """Configure logging for experiment execution."""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    # Create logs directory
    log_dir = Path(__file__).parent.parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    logger.add(
        log_dir / 'experiment_4_ollama_execution.log',
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG"
    )

def save_results(results, analysis, output_dir):
    """Save experiment results to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        """Convert numpy types to native Python types."""
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj
    
    # Prepare results for JSON
    json_results = {
        'summary_statistics': convert_to_native(results['summary_statistics']),
        'analysis': convert_to_native(analysis),
        'timestamp': datetime.now().isoformat(),
        'config': {
            'model': 'llama2:latest',
            'num_steps': 10,
            'num_runs': 3,
            'max_tokens': 2000,
            'scenario_type': 'sequential'
        },
        'metadata': {
            'total_queries': len(results.get('raw_metrics', [])),
            'strategies_tested': list(results['summary_statistics'].keys())
        }
    }
    
    # Save raw results
    raw_file = output_path / 'raw' / f'experiment_4_ollama_results_{timestamp}.json'
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(raw_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Raw results saved to {raw_file}")
    
    # Save as latest
    latest_file = output_path / 'raw' / 'experiment_4_ollama_latest.json'
    with open(latest_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Latest results saved to {latest_file}")
    
    # Save processed analysis
    analysis_file = output_path / 'processed' / f'experiment_4_ollama_analysis_{timestamp}.json'
    analysis_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(analysis_file, 'w') as f:
        json.dump({
            'summary_statistics': convert_to_native(results['summary_statistics']),
            'analysis': convert_to_native(analysis),
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    logger.info(f"Analysis saved to {analysis_file}")
    
    return raw_file, latest_file, analysis_file

def print_summary(results, analysis):
    """Print concise summary to console."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: Context Engineering Strategies - Ollama Results")
    print("=" * 80 + "\n")
    
    print("STRATEGY PERFORMANCE SUMMARY:")
    print("-" * 80)
    
    # Sort by ranking
    for strategy, rank in sorted(analysis['strategy_rankings'].items(), key=lambda x: x[1]):
        stats = results['summary_statistics'][strategy]
        print(f"\n#{rank}. {strategy.upper()}:")
        print(f"  Accuracy:     {stats['mean_accuracy']:.3f} ± {stats['std_accuracy']:.3f}")
        print(f"  Context Size: {stats['mean_context_size']:.1f} ± {stats['std_context_size']:.1f} tokens")
        print(f"  Latency:      {stats['mean_latency']:.3f} ± {stats['std_latency']:.3f} seconds")
        print(f"  Compressions: {stats['total_compressions']}")
        print(f"  Best For:     {analysis['recommendations'].get(strategy, 'N/A')}")
    
    print("\n" + "-" * 80)
    print(f"🏆 BEST OVERALL STRATEGY: {analysis['best_strategy'].upper()}")
    print("=" * 80 + "\n")

def main():
    """Run Experiment 4 with real Ollama LLM."""
    setup_logging()
    
    logger.info("=" * 80)
    logger.info("EXPERIMENT 4: Context Engineering Strategies - OLLAMA EXECUTION")
    logger.info("=" * 80)
    
    try:
        # Configuration
        config = {
            'model': 'llama2:latest',
            'num_steps': 10,
            'max_tokens': 2000,
            'num_runs': 3,  # 3 runs for statistical validity
            'output_dir': 'results/'
        }
        
        logger.info("Configuration:")
        logger.info(f"  Model: {config['model']}")
        logger.info(f"  Steps per run: {config['num_steps']}")
        logger.info(f"  Runs per strategy: {config['num_runs']}")
        logger.info(f"  Strategies: SELECT, COMPRESS, WRITE, HYBRID")
        
        total_queries = 4 * config['num_steps'] * config['num_runs']
        logger.info(f"  Expected queries: ~{total_queries}")
        logger.info(f"  Estimated time: ~30-40 minutes")
        
        # Initialize experiment with real Ollama
        logger.info("\nInitializing Experiment 4 with Ollama...")
        experiment = Experiment4(config)
        logger.success("✓ Experiment 4 initialized with real Ollama LLM")
        
        # Run experiment
        logger.info("\n" + "=" * 80)
        logger.info("STARTING EXPERIMENT EXECUTION")
        logger.info("=" * 80)
        logger.info("\nTesting context management strategies:")
        logger.info("  1. SELECT: RAG-based retrieval of relevant context")
        logger.info("  2. COMPRESS: Summarization when context exceeds limit")
        logger.info("  3. WRITE: External memory with key fact extraction")
        logger.info("  4. HYBRID: Combined SELECT + COMPRESS approach")
        logger.info("\nScenario: Sequential (weather monitoring over 10 days)")
        logger.info("This will take approximately 30-40 minutes...\n")
        
        results = experiment.run()
        
        logger.success("\n" + "=" * 80)
        logger.success("✓ EXPERIMENT EXECUTION COMPLETED")
        logger.success("=" * 80)
        
        # Analyze results
        logger.info("\nPerforming statistical analysis...")
        analysis = experiment.analyze(results)
        logger.success("✓ Analysis completed")
        
        # Print summary
        print_summary(results, analysis)
        
        # Save results
        logger.info("Saving results...")
        raw_file, latest_file, analysis_file = save_results(
            results, analysis, config['output_dir']
        )
        
        logger.success("\n" + "=" * 80)
        logger.success("✅ EXPERIMENT 4 COMPLETED SUCCESSFULLY!")
        logger.success("=" * 80)
        logger.info(f"\nResults saved to:")
        logger.info(f"  Raw: {raw_file}")
        logger.info(f"  Latest: {latest_file}")
        logger.info(f"  Analysis: {analysis_file}")
        
        logger.info(f"\n📊 Key Finding: {analysis['best_strategy'].upper()} strategy performed best")
        logger.info(f"   with {results['summary_statistics'][analysis['best_strategy']]['mean_accuracy']:.1%} accuracy")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Error during experiment execution: {e}")
        logger.exception("Full traceback:")
        return 1

if __name__ == "__main__":
    sys.exit(main())
