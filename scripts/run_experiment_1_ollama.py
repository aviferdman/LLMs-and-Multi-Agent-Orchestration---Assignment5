"""
Script to run Experiment 1: Lost in the Middle with Real Ollama LLM

This script executes Experiment 1 using the real Ollama interface
to collect production data for the research project.
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.llm_interface import OllamaInterface
from src.experiments.experiment_1 import Experiment1
from loguru import logger

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def main():
    """Run Experiment 1 with real Ollama LLM."""

    logger.info("="*70)
    logger.info("EXPERIMENT 1 EXECUTION: Lost in the Middle (PRODUCTION RUN)")
    logger.info("="*70)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load configuration
    config = Config()

    # Get experiment configuration
    exp1_config = config.get("experiment_1", {})
    
    # Allow override via command line or use default
    num_trials = int(sys.argv[1]) if len(sys.argv) > 1 else exp1_config.get("num_trials_per_position", 30)
    model = sys.argv[2] if len(sys.argv) > 2 else "llama2:latest"
    
    exp1_config["num_trials_per_position"] = num_trials

    logger.info(f"\nConfiguration:")
    logger.info(f"  - Model: {model}")
    logger.info(f"  - Positions to test: {exp1_config.get('fact_positions', ['start', 'middle', 'end'])}")
    logger.info(f"  - Trials per position: {num_trials}")
    logger.info(f"  - Total trials: {len(exp1_config.get('fact_positions', ['start', 'middle', 'end'])) * num_trials}")
    
    estimated_time = num_trials * len(exp1_config.get('fact_positions', ['start', 'middle', 'end'])) * 8  # ~8 seconds per query
    logger.info(f"  - Estimated time: ~{estimated_time/60:.1f} minutes")

    # Create real Ollama LLM interface
    logger.info(f"\nInitializing Ollama interface...")
    try:
        llm = OllamaInterface(
            model=model,
            timeout=300,  # 5 minute timeout for slower queries
            max_retries=3
        )
        logger.success("✓ Ollama interface initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Ollama: {e}")
        logger.error("Please ensure Ollama is running: ollama serve")
        sys.exit(1)

    # Initialize experiment with real Ollama
    logger.info("Initializing Experiment 1...")
    experiment = Experiment1(config, llm=llm)

    # Run experiment
    logger.info("\n" + "="*70)
    logger.info("STARTING EXPERIMENT EXECUTION")
    logger.info("="*70 + "\n")

    start_time = time.time()
    
    try:
        results = experiment.run()
        elapsed_time = time.time() - start_time
        
        logger.success(f"\n✓ Experiment completed in {elapsed_time/60:.1f} minutes")
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise

    # Save results with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path(config.get("general", {}).get("output_dir", "results"))
    raw_results_dir = results_dir / "raw"
    raw_results_dir.mkdir(parents=True, exist_ok=True)

    # Save raw results
    results_file = raw_results_dir / f"experiment_1_ollama_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.success(f"\n✓ Raw results saved to: {results_file}")

    # Also save as latest
    latest_file = raw_results_dir / "experiment_1_ollama_latest.json"
    with open(latest_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.success(f"✓ Latest results saved to: {latest_file}")

    # Save summary
    summary = results["summary"]
    summary_file = results_dir / "processed" / f"experiment_1_ollama_summary_{timestamp}.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.success(f"✓ Summary saved to: {summary_file}")

    # Perform statistical analysis
    logger.info("\n" + "="*70)
    logger.info("STATISTICAL ANALYSIS")
    logger.info("="*70)

    analysis_results = experiment.analyze(results)

    # Save analysis (convert numpy types first)
    analysis_file = results_dir / "processed" / f"experiment_1_ollama_analysis_{timestamp}.json"
    analysis_results_converted = convert_numpy_types(analysis_results)
    with open(analysis_file, "w") as f:
        json.dump(analysis_results_converted, f, indent=2)
    logger.success(f"✓ Analysis saved to: {analysis_file}")

    # Print key findings
    logger.info("\n" + "="*70)
    logger.info("KEY FINDINGS")
    logger.info("="*70)

    for position in ["start", "middle", "end"]:
        if position in summary:
            stats = summary[position]
            logger.info(f"\n{position.upper()} Position:")
            logger.info(f"  Accuracy: {stats['accuracy']['mean']:.1%} ± {stats['accuracy']['std']:.3f}")
            logger.info(f"  Latency: {stats['latency']['mean']:.3f}s ± {stats['latency']['std']:.3f}s")
            logger.info(f"  Semantic Similarity: {stats['semantic_similarity']['mean']:.3f}")

    if "anova" in analysis_results:
        logger.info(f"\nANOVA Results:")
        logger.info(f"  F-statistic: {analysis_results['anova']['f_statistic']:.3f}")
        logger.info(f"  p-value: {analysis_results['anova']['p_value']:.6f}")
        logger.info(f"  Significant: {analysis_results['anova']['significant']}")

    # Check for "Lost in the Middle" effect
    if "start" in summary and "middle" in summary:
        start_acc = summary["start"]["accuracy"]["mean"]
        middle_acc = summary["middle"]["accuracy"]["mean"]
        effect_size = start_acc - middle_acc
        
        logger.info(f"\n'Lost in the Middle' Effect:")
        logger.info(f"  Start accuracy: {start_acc:.1%}")
        logger.info(f"  Middle accuracy: {middle_acc:.1%}")
        logger.info(f"  Effect size: {effect_size:.1%}")
        
        if effect_size > 0.05:  # 5% threshold
            logger.warning(f"  ⚠️ Significant degradation in middle position detected!")
        else:
            logger.info(f"  ✓ No significant 'Lost in the Middle' effect observed")

    logger.info("\n" + "="*70)
    logger.success("EXPERIMENT 1 EXECUTION COMPLETE!")
    logger.info("="*70)
    logger.info(f"\nResults saved to: {results_dir}")
    logger.info(f"Total execution time: {elapsed_time/60:.1f} minutes")

    return results, analysis_results

if __name__ == "__main__":
    try:
        results, analysis = main()
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️ Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\n❌ Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
