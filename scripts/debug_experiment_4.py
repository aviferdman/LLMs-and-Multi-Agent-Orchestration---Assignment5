"""
Debug script for Experiment 4 to see what's happening with responses.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import experiment_4 module
import importlib.util
spec = importlib.util.spec_from_file_location(
    "experiment_4",
    Path(__file__).parent.parent / 'src' / 'experiments' / 'experiment_4.py'
)
experiment_4_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment_4_module)
Experiment4 = experiment_4_module.Experiment4
SelectStrategy = experiment_4_module.SelectStrategy

from scripts.run_experiment_4 import MockLLMForExperiment4

def debug_single_query():
    """Test a single query to see what's happening."""

    # Create mock LLM
    llm = MockLLMForExperiment4()

    # Create a simple scenario
    context = "Day 1: Temperature 22C, weather sunny."
    query = "What was the temperature on Day 1?"
    ground_truth = "22C"

    print("=" * 80)
    print("DEBUG: Testing single query")
    print("=" * 80)
    print(f"Context: {context}")
    print(f"Query: {query}")
    print(f"Expected ground truth: {ground_truth}")
    print()

    # Get response from mock LLM
    response = llm.query(context, query)

    print(f"Mock LLM Response: {response.text}")
    print(f"Response (stripped/lower): '{response.text.strip().lower()}'")
    print(f"Ground truth (stripped/lower): '{ground_truth.strip().lower()}'")
    print(f"Exact match: {response.text.strip().lower() == ground_truth.strip().lower()}")
    print()

    # Test SELECT strategy
    print("-" * 80)
    print("Testing SELECT Strategy Context Management")
    print("-" * 80)

    config = {'model': 'mock-llm', 'num_steps': 10, 'max_tokens': 2000, 'num_runs': 1}
    exp = Experiment4(config)
    exp.llm = llm

    # Update strategy LLM
    strategy = SelectStrategy(llm, max_tokens=2000, top_k=5)

    # Add observation and manage context
    observation = "Day 1: Temperature 22C, weather sunny."
    managed_context, metrics = strategy.manage_context(observation, query)

    print(f"Managed context: {managed_context}")
    print(f"Context metrics: {metrics}")
    print()

    # Query with managed context
    response = llm.query(managed_context, query)
    print(f"Response with managed context: {response.text}")
    print()

    # Now test a few steps
    print("=" * 80)
    print("Testing Multi-Step Scenario")
    print("=" * 80)

    strategy.reset()

    steps = [
        ("Day 1: Temperature 22C, weather sunny.", "What was the temperature on Day 1?", "22C"),
        ("Day 2: Temperature 24C, weather sunny.", "What was the temperature on Day 2?", "24C"),
        ("Day 3: Temperature 26C, weather cloudy.", "What was the temperature on Day 3?", "26C"),
    ]

    for i, (obs, q, gt) in enumerate(steps, 1):
        print(f"\nStep {i}:")
        print(f"  Observation: {obs}")
        print(f"  Query: {q}")
        print(f"  Ground truth: {gt}")

        managed_ctx, _ = strategy.manage_context(obs, q)
        print(f"  Managed context length: {len(managed_ctx)} chars")

        resp = llm.query(managed_ctx, q)
        print(f"  Response: {resp.text}")
        print(f"  Match: {resp.text.strip().lower() == gt.strip().lower()}")

if __name__ == "__main__":
    debug_single_query()
