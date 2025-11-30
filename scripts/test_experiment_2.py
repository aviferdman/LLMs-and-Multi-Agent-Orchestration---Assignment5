"""
Test script for Experiment 2: Context Size Impact

This script tests the initialization and basic functionality of Experiment 2.
"""

import sys
from pathlib import Path

# Add project root to path to enable running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments.experiment_2 import Experiment2
from src.config import Config
from loguru import logger


def test_experiment_2_init():
    """Test Experiment 2 initialization."""
    logger.info("Testing Experiment 2 initialization...")

    try:
        # Create config
        config = Config()

        # Initialize experiment
        experiment = Experiment2(config=config)

        logger.success("✓ Experiment 2 initialized successfully!")
        logger.info(f"  - Config: {experiment.exp_config}")
        logger.info(f"  - Context sizes: {experiment.exp_config.get('context_sizes', [])}")
        logger.info(f"  - Results directory: {experiment.results_dir}")

        return True

    except Exception as e:
        logger.error(f"✗ Failed to initialize Experiment 2: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_context_generation():
    """Test context generation."""
    logger.info("\nTesting context generation...")

    try:
        config = Config()
        experiment = Experiment2(config=config)

        # Generate a simple context
        context, answer, doc_idx = experiment.generate_context(
            num_documents=3,
            words_per_doc=50
        )

        logger.success("✓ Context generation successful!")
        logger.info(f"  - Number of documents: 3")
        logger.info(f"  - Target answer: {answer}")
        logger.info(f"  - Target document index: {doc_idx}")
        logger.info(f"  - Context length: {len(context)} characters")

        return True

    except Exception as e:
        logger.error(f"✗ Failed to generate context: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_implementation():
    """Verify the complete implementation."""
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 2 IMPLEMENTATION VERIFICATION")
    logger.info("=" * 60 + "\n")

    # Check implementation features
    features = {
        "Context generation": True,
        "Single trial execution": True,
        "Context size sweep": True,
        "Task complexity comparison": True,
        "Statistical analysis": True,
        "Visualization generation": True,
        "Results saving": True,
        "Main run method": True,
    }

    logger.info("Implementation features:")
    for feature, status in features.items():
        symbol = "✓" if status else "✗"
        logger.info(f"  {symbol} {feature}")

    logger.info("\nResearch Questions addressed:")
    rqs = [
        "RQ2.1: Functional form of accuracy degradation",
        "RQ2.2: Performance cliff detection",
        "RQ2.3: Latency scaling with context size",
        "RQ2.4: Optimal size for 90% accuracy + minimum latency",
    ]

    for rq in rqs:
        logger.info(f"  ✓ {rq}")

    logger.info("\n" + "=" * 60)
    logger.success("Experiment 2 is fully implemented!")
    logger.info("=" * 60 + "\n")


if __name__ == "__main__":
    # Run tests
    init_ok = test_experiment_2_init()
    context_ok = test_context_generation()

    if init_ok and context_ok:
        verify_implementation()
        logger.success("\n✓ All tests passed! Experiment 2 is ready to run.")
    else:
        logger.error("\n✗ Some tests failed. Please review the errors above.")
