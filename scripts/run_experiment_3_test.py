"""
Test Execution Script for Experiment 3: RAG Impact Analysis

This script runs Experiment 3 with a mock LLM interface to demonstrate
functionality without requiring a running Ollama instance.
"""

import sys
import time
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments.experiment_3 import Experiment3
from src.llm_interface import LLMInterface, Response
from loguru import logger


class MockLLMInterface(LLMInterface):
    """
    Mock LLM interface for testing purposes.

    Generates simulated responses based on query content.
    """

    def __init__(self, model: str = "mock-llm"):
        """Initialize mock interface."""
        self.model = model
        self.query_count = 0
        logger.info(f"MockLLMInterface initialized with model: {model}")

    def query(self, context: str, query: str, **kwargs) -> Response:
        """
        Generate mock response based on query.

        For factual queries, attempts to extract answer from context.
        For analytical queries, generates generic analysis.
        """
        self.query_count += 1
        start_time = time.time()

        # Simulate processing time
        time.sleep(0.1)

        # Simple mock answer generation
        if "what was mentioned" in query.lower() or "what is" in query.lower():
            # Factual query - try to extract from context
            words = context.split()
            # Look for percentage patterns or numbers
            for i, word in enumerate(words):
                if '%' in word or word.endswith('%'):
                    answer = f"The value mentioned is {word}"
                    break
                elif word.isdigit() and i < len(words) - 1 and 'students' in words[i+1]:
                    answer = f"The number is {word} students"
                    break
            else:
                answer = "The specific metric showed improvement in the sector."
        else:
            # Analytical query
            answer = "Analysis shows positive trends with increasing adoption and efficiency gains across the sector. Multiple factors contribute to growth including technological advancement and market expansion."

        latency = time.time() - start_time
        tokens = len(answer.split())

        return Response(
            text=answer,
            latency=latency,
            tokens=tokens,
            confidence=0.85,
            metadata={"model": self.model, "query_number": self.query_count}
        )

    def embed(self, text: str) -> Any:
        """Mock embedding (not used in this test)."""
        import numpy as np
        return np.random.rand(384)

    def count_tokens(self, text: str) -> int:
        """Simple token counting."""
        return len(text) // 4


def main():
    """Run Experiment 3 with mock LLM."""
    print("=" * 80)
    print("EXPERIMENT 3: RAG IMPACT ANALYSIS - TEST EXECUTION")
    print("=" * 80)
    print()
    print("Using Mock LLM Interface for demonstration purposes")
    print("This simulates the full experiment workflow without requiring Ollama")
    print()

    # Create output directory
    output_dir = Path("results/experiment_3_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize experiment with mock LLM
    print("Initializing Experiment 3...")
    mock_llm = MockLLMInterface(model="mock-llm-v1")

    try:
        # Pass config directory path (ConfigManager expects directory, not file)
        config_dir = Path(__file__).parent.parent / "config"

        # ConfigManager is initialized with directory in Experiment3.__init__
        # So we need to pass the config directory, not the YAML file
        # But Experiment3 expects config_path parameter for ConfigManager
        # Let's just use the default by passing the directory
        exp = Experiment3(
            config_path=str(config_dir),
            llm_interface=mock_llm
        )

        print(f"[OK] Configuration loaded")
        print(f"[OK] Experiment initialized: {exp.config['name']}")
        print()

        # Run the experiment
        print("Running experiment...")
        print("-" * 80)

        results = exp.run()

        print("-" * 80)
        print()
        print("[OK] Experiment execution completed")
        print()

        # Save results
        print("Saving results...")
        exp.save_results(results, output_dir=str(output_dir))
        print(f"[OK] Results saved to: {output_dir}")
        print()

        # Print summary
        print("=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        print(f"Total queries processed: {results.aggregate_metrics['total_queries']}")
        print()

        print("Results by Approach:")
        for approach, metrics in results.aggregate_metrics['by_approach'].items():
            print(f"\n  {approach}:")
            print(f"    Count:          {metrics['count']}")
            print(f"    Accuracy:       {metrics['accuracy']:.3f}")
            print(f"    Mean Latency:   {metrics['mean_latency']:.3f}s")
            print(f"    Mean Tokens:    {metrics['mean_tokens']:.1f}")

        print("\n" + "-" * 80)
        print("\nResults by Query Type:")
        for qtype, metrics in results.aggregate_metrics['by_query_type'].items():
            print(f"\n  {qtype.capitalize()}:")
            print(f"    Count:          {metrics['count']}")
            print(f"    Accuracy:       {metrics['accuracy']:.3f}")
            print(f"    Mean Latency:   {metrics['mean_latency']:.3f}s")

        print("\n" + "-" * 80)
        print("\nRAG Performance by Top-K:")
        for top_k, metrics in results.aggregate_metrics['by_top_k'].items():
            print(f"\n  {top_k}:")
            print(f"    Accuracy:       {metrics['accuracy']:.3f}")
            print(f"    Mean Latency:   {metrics['mean_latency']:.3f}s")

        print("\n" + "=" * 80)
        print("\nStatistical Tests:")
        for test_name, test_result in results.statistical_tests.items():
            print(f"\n  {test_name}:")
            if 'p_value' in test_result:
                print(f"    p-value:        {test_result['p_value']:.4f}")
                print(f"    Significant:    {test_result.get('significant', 'N/A')}")

        print("\n" + "=" * 80)
        print("\n[OK] Experiment 3 test execution completed successfully!")
        print(f"\nResults and visualizations available in: {output_dir.absolute()}")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\nX Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
