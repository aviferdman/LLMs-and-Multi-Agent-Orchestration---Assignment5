"""
Script to run Experiment 2: Context Size Impact with Real Ollama LLM

This script executes Experiment 2 using the real Ollama interface
to test how accuracy and latency scale with context size.
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
from src.experiments.experiment_2 import Experiment2
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
    """Run Experiment 2 with real Ollama LLM."""

    logger.info("="*70)
    logger.info("EXPERIMENT 2 EXECUTION: Context Size Impact (PRODUCTION RUN)")
    logger.info("="*70)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load configuration
    config = Config()

    # Get experiment configuration
    exp2_config = config.get("experiment_2", {})
    
    # Allow override via command line
    num_runs = int(sys.argv[1]) if len(sys.argv) > 1 else exp2_config.get("num_runs_per_size", 10)
    model = sys.argv[2] if len(sys.argv) > 2 else "llama2:latest"
    
    # Get context sizes to test
    context_sizes = exp2_config.get("context_sizes", [2, 5, 10, 20, 50])
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  - Model: {model}")
    logger.info(f"  - Context sizes: {context_sizes}")
    logger.info(f"  - Runs per size: {num_runs}")
    logger.info(f"  - Total trials: {len(context_sizes) * num_runs}")
    
    # Estimate time (larger contexts take longer)
    avg_time_per_trial = 12  # seconds (conservative estimate)
    total_trials = len(context_sizes) * num_runs
    estimated_time = (total_trials * avg_time_per_trial) / 60
    logger.info(f"  - Estimated time: ~{estimated_time:.1f} minutes")

    # Create real Ollama LLM interface
    logger.info(f"\nInitializing Ollama interface...")
    try:
        llm = OllamaInterface(
            model=model,
            timeout=300,
            max_retries=3
        )
        logger.success("✓ Ollama interface initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Ollama: {e}")
        logger.error("Please ensure Ollama is running: ollama serve")
        sys.exit(1)

    # Initialize experiment with real Ollama
    logger.info("Initializing Experiment 2...")
    experiment = Experiment2(config, llm=llm)

    # Run experiment
    logger.info("\n" + "="*70)
    logger.info("STARTING EXPERIMENT EXECUTION")
    logger.info("="*70 + "\n")

    start_time = time.time()
    
    try:
        # Run context size sweep
        results = experiment.run_context_size_sweep(
            context_sizes=context_sizes,
            num_runs_per_size=num_runs
        )
        
        elapsed_time = time.time() - start_time
        logger.success(f"\n✓ Experiment completed in {elapsed_time/60:.1f} minutes")
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise

    # Perform analysis
    logger.info("\n" + "="*70)
    logger.info("ANALYZING RESULTS")
    logger.info("="*70)
    
    analysis = experiment.analyze_results(results)

    # Generate visualizations
    logger.info("\n" + "="*70)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("="*70)
    
    try:
        experiment.generate_visualizations(results, analysis)
        logger.success("✓ Visualizations generated")
    except Exception as e:
        logger.warning(f"Visualization generation failed: {e}")

    # Save results with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path(config.get("general", {}).get("output_dir", "results"))
    
    # Save raw results
    raw_results_dir = results_dir / "raw"
    raw_results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = raw_results_dir / f"experiment_2_ollama_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(convert_numpy_types(results), f, indent=2)
    logger.success(f"\n✓ Raw results saved to: {results_file}")
    
    # Also save as latest
    latest_file = raw_results_dir / "experiment_2_ollama_latest.json"
    with open(latest_file, "w") as f:
        json.dump(convert_numpy_types(results), f, indent=2)
    logger.success(f"✓ Latest results saved to: {latest_file}")

    # Save analysis
    processed_dir = results_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    analysis_file = processed_dir / f"experiment_2_ollama_analysis_{timestamp}.json"
    with open(analysis_file, "w") as f:
        json.dump(convert_numpy_types(analysis), f, indent=2)
    logger.success(f"✓ Analysis saved to: {analysis_file}")

    # Print key findings
    logger.info("\n" + "="*70)
    logger.info("KEY FINDINGS")
    logger.info("="*70)
    
    # Summary by context size
    import pandas as pd
    df = pd.DataFrame(results)
    
    logger.info("\nAccuracy by Context Size:")
    for size in context_sizes:
        size_data = df[df["num_documents"] == size]
        mean_acc = size_data["accuracy"].mean()
        std_acc = size_data["accuracy"].std()
        mean_lat = size_data["latency"].mean()
        logger.info(f"  {size:2d} docs: Accuracy={mean_acc:.1%} ± {std_acc:.3f}, Latency={mean_lat:.2f}s")
    
    # Correlation results
    if "correlation_accuracy_size" in analysis:
        corr = analysis["correlation_accuracy_size"]
        logger.info(f"\nAccuracy vs Context Size:")
        logger.info(f"  Correlation: r={corr.get('correlation', 0):.3f}")
        logger.info(f"  p-value: {corr.get('p_value', 1):.6f}")
        logger.info(f"  Significant: {corr.get('significant', False)}")
    
    if "correlation_latency_size" in analysis:
        corr_lat = analysis["correlation_latency_size"]
        logger.info(f"\nLatency vs Context Size:")
        logger.info(f"  Correlation: r={corr_lat.get('correlation', 0):.3f}")
        logger.info(f"  p-value: {corr_lat.get('p_value', 1):.6f}")

    logger.info("\n" + "="*70)
    logger.success("EXPERIMENT 2 EXECUTION COMPLETE!")
    logger.info("="*70)
    logger.info(f"\nResults saved to: {results_dir}")
    logger.info(f"Total execution time: {elapsed_time/60:.1f} minutes")

    return results, analysis

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
