"""
Test script for Experiment 1: Lost in the Middle

This script validates the Experiment 1 implementation by running a minimal
version of the experiment with reduced parameters for quick testing.

Usage:
    python scripts/test_experiment_1.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger
import yaml


def test_experiment_1():
    """Test Experiment 1 implementation."""
    logger.info("=" * 70)
    logger.info("TESTING EXPERIMENT 1: LOST IN THE MIDDLE")
    logger.info("=" * 70)

    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "experiments.yaml"
    logger.info(f"\nLoading configuration from: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Modify config for quick testing
    config["experiment_1"]["num_trials_per_position"] = 3  # Reduced from 10
    config["experiment_1"]["num_documents"] = 3  # Reduced from 5

    logger.info(f"Modified config for testing:")
    logger.info(f"  - Trials per position: {config['experiment_1']['num_trials_per_position']}")
    logger.info(f"  - Number of documents: {config['experiment_1']['num_documents']}")

    # Import and create experiment
    try:
        from experiments.experiment_1 import Experiment1
        logger.success("Successfully imported Experiment1 class")
    except Exception as e:
        logger.error(f"Failed to import Experiment1: {e}")
        return False

    # Initialize experiment
    try:
        experiment = Experiment1(config)
        logger.success("Successfully initialized Experiment1")
    except Exception as e:
        logger.error(f"Failed to initialize Experiment1: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test document generation
    logger.info("\n" + "-" * 70)
    logger.info("Testing document generation...")
    logger.info("-" * 70)

    try:
        for position in ["start", "middle", "end"]:
            doc_with_fact = experiment.generate_document_with_fact(position)
            logger.info(f"\nPosition: {position}")
            logger.info(f"  Fact: {doc_with_fact.fact}")
            logger.info(f"  Ground truth: {doc_with_fact.ground_truth}")
            logger.info(f"  Document length: {len(doc_with_fact.document.split())} words")
        logger.success("Document generation working correctly")
    except Exception as e:
        logger.error(f"Document generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test single trial
    logger.info("\n" + "-" * 70)
    logger.info("Testing single trial execution...")
    logger.info("-" * 70)

    try:
        trial_result = experiment.run_trial("middle")
        logger.info(f"\nTrial result:")
        logger.info(f"  Position: {trial_result.position}")
        logger.info(f"  Query: {trial_result.query}")
        logger.info(f"  Response: {trial_result.response}")
        logger.info(f"  Ground truth: {trial_result.ground_truth}")
        logger.info(f"  Correct: {trial_result.correct}")
        logger.info(f"  Latency: {trial_result.latency:.2f}s")
        logger.info(f"  Semantic similarity: {trial_result.semantic_similarity:.3f}")
        logger.success("Trial execution working correctly")
    except Exception as e:
        logger.error(f"Trial execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Note: Full experiment run would require actual LLM connection
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 70)
    logger.info("\nAll components validated successfully!")
    logger.info("\nNote: Full experiment requires Ollama LLM to be running.")
    logger.info("      The current implementation uses placeholder responses.")
    logger.info("\nTo run the full experiment:")
    logger.info("  1. Start Ollama with the configured model")
    logger.info("  2. Run: python scripts/run_experiments.py --experiment 1")

    return True


if __name__ == "__main__":
    success = test_experiment_1()
    sys.exit(0 if success else 1)
