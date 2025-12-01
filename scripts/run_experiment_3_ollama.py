"""
Experiment 3 Ollama Runner

Simplified runner for Experiment 3: RAG Impact Analysis
Compares RAG vs Full Context approaches using Ollama.
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.config import ConfigManager
from src.llm_interface import OllamaInterface
from src.experiments.experiment_3 import Experiment3, CorpusManager, QueryGenerator
from src.metrics import MetricsCalculator
from src.visualization import Visualizer

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def main():
    """Run simplified Experiment 3 with Ollama."""
    
    logger.info("=" * 70)
    logger.info("EXPERIMENT 3 EXECUTION: RAG vs Full Context (SIMPLIFIED)")
    logger.info("=" * 70)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize configuration
    config_manager = ConfigManager(config_dir="config")
    exp_config = config_manager.get_experiment_config(3)
    general_config = config_manager.get_general_config()
    
    # Use smaller corpus and fewer queries for faster execution
    corpus_size = 10  # Reduced from 30
    queries_per_type = 5  # Reduced from 10
    top_k_values = [1, 3, 5]  # Reduced from [1, 2, 3, 5, 7, 10]
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  - Model: {general_config.get('ollama_model', 'llama2:latest')}")
    logger.info(f"  - Corpus size: {corpus_size} documents")
    logger.info(f"  - Queries per type: {queries_per_type}")
    logger.info(f"  - Top-k values: {top_k_values}")
    logger.info(f"  - Total expected queries: ~{queries_per_type * 2 * (len(top_k_values) + 1)}")
    logger.info(f"  - Estimated time: ~15-20 minutes")
    
    # Initialize Ollama interface
    logger.info(f"\nInitializing Ollama interface...")
    model = general_config.get("ollama_model", "llama2:latest")
    llm = OllamaInterface(model=model)
    
    try:
        # Verify connection
        if hasattr(llm, '_verify_connection'):
            llm._verify_connection()
        logger.success("✓ Ollama interface initialized successfully")
    except Exception as e:
        logger.error(f"Failed to connect to Ollama: {e}")
        logger.error("Please ensure Ollama is running: ollama serve")
        return
    
    # Initialize experiment components
    logger.info(f"\nInitializing Experiment 3...")
    exp = Experiment3(
        config_path="config",
        llm_interface=llm
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("STARTING EXPERIMENT EXECUTION")
    logger.info("=" * 70)
    
    start_time = time.time()
    
    try:
        # Generate corpus
        logger.info("\nStep 1: Generating corpus...")
        corpus_manager = CorpusManager(min_size=corpus_size)
        documents, facts = corpus_manager.load_or_generate_corpus()
        logger.success(f"✓ Generated {len(documents)} documents with {len(facts)} facts")
        
        # Generate queries
        logger.info("\nStep 2: Generating queries...")
        query_generator = QueryGenerator(
            random_seed=general_config.get("random_seed", 42)
        )
        queries = query_generator.generate_queries(
            facts=facts,
            num_factual=queries_per_type,
            num_analytical=queries_per_type
        )
        logger.success(f"✓ Generated {len(queries)} queries")
        
        # Run experiments
        all_results = []
        
        # Test RAG with different top_k values
        for top_k in top_k_values:
            logger.info(f"\n--- Testing RAG with top_k={top_k} ---")
            results = exp._run_rag_queries(
                documents=documents,
                queries=queries,
                top_k=top_k
            )
            all_results.extend(results)
            logger.success(f"✓ Completed {len(results)} RAG queries (top_k={top_k})")
        
        # Test Full Context
        logger.info(f"\n--- Testing Full Context approach ---")
        full_results = exp._run_full_context_queries(
            documents=documents,
            queries=queries
        )
        all_results.extend(full_results)
        logger.success(f"✓ Completed {len(full_results)} Full Context queries")
        
        # Calculate metrics
        logger.info("\n" + "=" * 70)
        logger.info("ANALYZING RESULTS")
        logger.info("=" * 70)
        
        aggregate_metrics = exp._calculate_aggregate_metrics(all_results)
        statistical_tests = exp._perform_statistical_tests(all_results)
        
        # Save results
        output_dir = Path("results/raw")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dict = {
            "experiment_name": exp_config['name'],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "corpus_size": corpus_size,
                "queries_per_type": queries_per_type,
                "top_k_values": top_k_values,
                "model": model
            },
            "query_results": [
                {
                    "query": r.query,
                    "query_type": r.query_type,
                    "approach": r.approach,
                    "top_k": r.top_k,
                    "answer": r.answer,
                    "ground_truth": r.ground_truth,
                    "latency": r.latency,
                    "tokens": r.tokens,
                    "is_correct": r.is_correct,
                    "retrieved_chunks": r.retrieved_chunks,
                    "similarity_score": r.similarity_score
                }
                for r in all_results
            ],
            "aggregate_metrics": aggregate_metrics,
            "statistical_tests": statistical_tests
        }
        
        # Convert numpy types to native Python types
        results_dict = convert_numpy_types(results_dict)
        
        # Save timestamped results
        results_file = output_dir / f"experiment_3_ollama_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        logger.success(f"\n✓ Raw results saved to: {results_file}")
        
        # Save latest results
        latest_file = output_dir / "experiment_3_ollama_latest.json"
        with open(latest_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        logger.success(f"✓ Latest results saved to: {latest_file}")
        
        # Save analysis
        analysis_dir = Path("results/processed")
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        analysis_file = analysis_dir / f"experiment_3_ollama_analysis_{timestamp}.json"
        analysis_dict = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "aggregate_metrics": convert_numpy_types(aggregate_metrics),
            "statistical_tests": convert_numpy_types(statistical_tests)
        }
        with open(analysis_file, 'w') as f:
            json.dump(analysis_dict, f, indent=2)
        logger.success(f"✓ Analysis saved to: {analysis_file}")
        
        # Generate visualizations
        logger.info("\n" + "=" * 70)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("=" * 70)
        
        figures_dir = Path("results/figures")
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Note: Visualizations would be generated here
        # For now, we'll skip them to save time
        logger.info("Visualization generation skipped for this simplified run")
        
        # Print summary
        elapsed_time = time.time() - start_time
        logger.info("\n" + "=" * 70)
        logger.info("KEY FINDINGS")
        logger.info("=" * 70)
        
        logger.info(f"\nTotal queries: {len(all_results)}")
        logger.info(f"\nBy Approach:")
        for approach, metrics in aggregate_metrics.get('by_approach', {}).items():
            logger.info(f"  {approach}:")
            logger.info(f"    Count: {metrics.get('count', 0)}")
            logger.info(f"    Accuracy: {metrics.get('accuracy', 0):.3f}")
            logger.info(f"    Mean Latency: {metrics.get('mean_latency', 0):.3f}s")
        
        logger.info(f"\nBy Top-k:")
        for top_k_label, metrics in aggregate_metrics.get('by_top_k', {}).items():
            logger.info(f"  {top_k_label}:")
            logger.info(f"    Accuracy: {metrics.get('accuracy', 0):.3f}")
            logger.info(f"    Mean Latency: {metrics.get('mean_latency', 0):.3f}s")
        
        logger.info("\n" + "=" * 70)
        logger.success("EXPERIMENT 3 EXECUTION COMPLETE!")
        logger.info("=" * 70)
        logger.info(f"\nResults saved to: results/")
        logger.info(f"Total execution time: {elapsed_time/60:.1f} minutes")
        
    except Exception as e:
        logger.error(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
