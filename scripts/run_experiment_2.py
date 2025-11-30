"""
Run Experiment 2: Context Size Impact

This script executes Experiment 2 with either a real Ollama instance or a mock LLM
for testing purposes.
"""

import sys
from pathlib import Path
import time
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments.experiment_2 import Experiment2
from src.config import Config
from src.llm_interface import LLMInterface, Response
from loguru import logger


class MockLLMInterface(LLMInterface):
    """
    Mock LLM interface for testing purposes.

    Simulates realistic LLM behavior with degrading accuracy as context size increases.
    """

    def __init__(self, model: str = "mock-llm"):
        """Initialize mock LLM."""
        self.model = model
        self.call_count = 0
        logger.info(f"Initialized MockLLMInterface with model: {model}")

        # Simulate realistic names for responses
        self.names = ["David Cohen", "Sarah Williams", "Michael Brown",
                     "Emma Davis", "James Wilson", "Lisa Anderson"]

    def query(self, context: str, query: str, **kwargs) -> Response:
        """
        Simulate LLM query with realistic behavior.

        Accuracy degrades with context size following research patterns.
        """
        self.call_count += 1
        start_time = time.time()

        # Extract target answer from context
        target_answer = None
        for name in self.names:
            if name in context:
                target_answer = name
                break

        # Count documents (rough estimate)
        num_docs = context.count("Document")

        # Simulate accuracy degradation based on context size
        # Following "Lost in the Middle" research: accuracy drops with more context
        base_accuracy = 0.95
        degradation_factor = 0.08  # 8% drop per doubling of context

        # Logarithmic degradation
        if num_docs > 2:
            accuracy = base_accuracy - (degradation_factor * np.log2(num_docs / 2))
        else:
            accuracy = base_accuracy

        # Add some randomness
        accuracy += np.random.uniform(-0.05, 0.05)
        accuracy = max(0.3, min(1.0, accuracy))  # Clamp between 30% and 100%

        # Decide if response is correct
        is_correct = np.random.random() < accuracy

        if is_correct and target_answer:
            response_text = f"The CEO's name is {target_answer}."
        elif target_answer and np.random.random() < 0.3:
            # Sometimes give a plausible wrong answer
            wrong_names = [n for n in self.names if n != target_answer]
            wrong_answer = np.random.choice(wrong_names)
            response_text = f"The CEO's name is {wrong_answer}."
        else:
            # Give a vague or "I don't know" response
            responses = [
                "I couldn't find that information in the provided context.",
                "The information about the CEO is not clear from the documents.",
                "Based on the documents, I'm not certain about the CEO's name.",
            ]
            response_text = np.random.choice(responses)

        # Simulate latency based on context size
        # Quadratic scaling: O(n^2) for transformer attention
        base_latency = 0.5  # base 500ms
        latency_per_doc = 0.1  # 100ms per document
        quadratic_factor = 0.01  # quadratic term

        simulated_latency = (base_latency +
                           latency_per_doc * num_docs +
                           quadratic_factor * (num_docs ** 2))

        # Add some randomness to latency
        simulated_latency += np.random.uniform(-0.1, 0.2)
        simulated_latency = max(0.1, simulated_latency)

        # Actual sleep time (much shorter for testing)
        time.sleep(0.05)  # Just 50ms to simulate processing

        actual_latency = time.time() - start_time

        # Use simulated latency for realistic results
        latency = simulated_latency

        # Token count (rough estimate)
        response_tokens = len(response_text.split()) * 1.3

        logger.debug(
            f"Mock LLM query #{self.call_count}: "
            f"docs={num_docs}, accuracy={accuracy:.2f}, latency={latency:.2f}s"
        )

        return Response(
            text=response_text,
            latency=latency,
            tokens=int(response_tokens),
            confidence=accuracy,
            metadata={
                "model": self.model,
                "num_documents": num_docs,
                "simulated_accuracy": accuracy,
                "is_mock": True,
            }
        )

    def embed(self, text: str) -> np.ndarray:
        """Generate mock embedding."""
        # Simple hash-based embedding
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(384)  # 384-dim embedding

    def count_tokens(self, text: str) -> int:
        """Count tokens (simple word-based estimate)."""
        return int(len(text.split()) * 1.3)


def main():
    """Run Experiment 2."""
    logger.info("=" * 70)
    logger.info("EXPERIMENT 2: Context Size Impact - EXECUTION")
    logger.info("=" * 70)

    # Load configuration
    config = Config()

    # Check if Ollama is available
    use_mock = True  # Set to False if you have Ollama running

    if use_mock:
        logger.warning("Using MockLLMInterface for testing (no Ollama required)")
        logger.info("Results will be simulated based on research patterns")
        llm = MockLLMInterface()
    else:
        logger.info("Using real OllamaInterface")
        from src.llm_interface import OllamaInterface
        llm = OllamaInterface(
            model=config.get("general.ollama_model", "llama2:13b")
        )

    # Initialize experiment with custom LLM
    logger.info("\nInitializing Experiment 2...")
    experiment = Experiment2(config=config, llm=llm)

    # Run experiment
    logger.info("\nStarting experiment execution...")
    logger.info("This will test context sizes: [2, 5, 10, 20, 50] documents")
    logger.info("Number of runs per size: 5")
    logger.info("")

    start_time = time.time()

    # Run main experiment (without task complexity to save time)
    results = experiment.run(include_task_complexity=False)

    total_time = time.time() - start_time

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT COMPLETED")
    logger.info("=" * 70)
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    logger.info(f"Total trials: {results['summary']['num_trials']}")
    logger.info(f"Context sizes tested: {results['summary']['context_sizes']}")

    # Print key findings
    logger.info("\n" + "=" * 70)
    logger.info("KEY FINDINGS")
    logger.info("=" * 70)
    for key, value in results['summary']['key_findings'].items():
        logger.info(f"{key}: {value}")

    logger.info("\n" + "=" * 70)
    logger.success("Experiment 2 execution completed successfully!")
    logger.info("=" * 70)
    logger.info("\nResults saved to:")
    logger.info(f"  - results/exp2_raw_results.json")
    logger.info(f"  - results/exp2_results.csv")
    logger.info(f"  - results/exp2_analysis.json")
    logger.info(f"  - results/exp2_summary.csv")
    logger.info("\nVisualizations saved to results/figures/:")
    logger.info(f"  - exp2_accuracy_vs_size.png")
    logger.info(f"  - exp2_latency_vs_size.png")
    logger.info(f"  - exp2_accuracy_distribution.png")

    return results


if __name__ == "__main__":
    np.random.seed(42)  # For reproducible mock results
    results = main()
