"""
Script to run Experiment 1: Lost in the Middle

This script executes Experiment 1 with a mock LLM interface that simulates
realistic position bias patterns for testing and demonstration purposes.
"""

import sys
import time
import json
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.llm_interface import LLMInterface, Response
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


class MockLLMInterface(LLMInterface):
    """
    Mock LLM that simulates position bias for testing.

    Accuracy pattern (simulating "Lost in the Middle" effect):
    - Start position: 90% accuracy
    - Middle position: 65% accuracy (degraded)
    - End position: 85% accuracy (slight recency effect)
    """

    def __init__(self, model: str = "mock-llm"):
        self.model = model
        self.call_count = 0

        # Position-based accuracy probabilities
        self.accuracy_rates = {
            "start": 0.90,
            "middle": 0.65,  # Lower accuracy in middle (Lost in the Middle)
            "end": 0.85
        }

        logger.info(f"MockLLMInterface initialized with position bias:")
        logger.info(f"  Start: {self.accuracy_rates['start']:.1%}")
        logger.info(f"  Middle: {self.accuracy_rates['middle']:.1%}")
        logger.info(f"  End: {self.accuracy_rates['end']:.1%}")

    def query(self, context: str, query: str, **kwargs) -> Response:
        """
        Simulate LLM query with position-based accuracy.
        Extracts CEO name from context to determine ground truth.
        """
        self.call_count += 1

        # Extract ground truth from context by finding "The CEO of the company is X."
        ground_truth = "Unknown"
        if "The CEO of the company is " in context:
            # Find the sentence with CEO information
            start_idx = context.find("The CEO of the company is ")
            end_idx = context.find(".", start_idx)
            if end_idx != -1:
                ceo_sentence = context[start_idx:end_idx+1]
                # Extract the name (text between "is" and ".")
                name_start = ceo_sentence.find(" is ") + 4
                name_end = ceo_sentence.find(".", name_start)
                ground_truth = ceo_sentence[name_start:name_end].strip()

        # Determine position based on where the CEO fact appears in the context
        # Approximate position based on character location
        if "The CEO of the company is " in context:
            fact_position = context.find("The CEO of the company is ")
            context_length = len(context)
            relative_position = fact_position / context_length

            if relative_position < 0.3:
                position = "start"
            elif relative_position > 0.7:
                position = "end"
            else:
                position = "middle"
        else:
            position = "middle"  # Default

        # Simulate latency (slightly higher for longer contexts)
        base_latency = 0.5
        context_penalty = len(context) / 10000  # Small penalty for longer contexts
        latency = base_latency + context_penalty + np.random.normal(0, 0.1)
        latency = max(0.2, latency)  # Minimum 0.2s

        # Simulate thinking time
        time.sleep(0.01)  # Small delay for realism

        # Determine if this response should be correct based on position
        accuracy_rate = self.accuracy_rates.get(position, 0.70)
        is_correct = np.random.random() < accuracy_rate

        # Generate response
        if is_correct:
            # Correct response
            response_text = f"The answer is {ground_truth}."
        else:
            # Incorrect response (simulate common errors)
            wrong_answers = [
                "I cannot find this information in the provided context.",
                "The documents do not contain this information.",
                f"Based on the context, I believe the answer might be John Smith.",  # Wrong name
                "I'm not certain, but it could be Sarah Johnson.",  # Wrong name
            ]
            response_text = np.random.choice(wrong_answers)

        # Estimate token count
        tokens = len(response_text.split())

        return Response(
            text=response_text,
            latency=latency,
            tokens=tokens,
            metadata={
                "model": self.model,
                "call_count": self.call_count,
                "position": position,
                "simulated_correct": is_correct,
                "extracted_truth": ground_truth
            }
        )

    def embed(self, text: str) -> np.ndarray:
        """Mock embedding (not used in Experiment 1)."""
        return np.random.randn(384)

    def count_tokens(self, text: str) -> int:
        """Simple token count estimation."""
        return len(text.split())


def main():
    """Run Experiment 1 with mock LLM."""

    logger.info("="*70)
    logger.info("EXPERIMENT 1 EXECUTION: Lost in the Middle")
    logger.info("="*70)

    # Load configuration
    config = Config()

    # Override some settings for faster testing
    exp1_config = config.get("experiment_1", {})
    exp1_config["num_trials_per_position"] = 15  # 15 trials per position

    logger.info(f"\nConfiguration:")
    logger.info(f"  - Positions to test: {exp1_config.get('fact_positions', ['start', 'middle', 'end'])}")
    logger.info(f"  - Trials per position: {exp1_config['num_trials_per_position']}")
    logger.info(f"  - Total trials: {len(exp1_config.get('fact_positions', ['start', 'middle', 'end'])) * exp1_config['num_trials_per_position']}")

    # Create mock LLM
    mock_llm = MockLLMInterface()

    # Initialize experiment with mock LLM
    experiment = Experiment1(config, llm=mock_llm)

    # Run experiment
    logger.info("\n" + "="*70)
    logger.info("STARTING EXPERIMENT EXECUTION")
    logger.info("="*70 + "\n")

    results = experiment.run()

    # Save results
    results_dir = Path(config.get("general", {}).get("output_dir", "results"))
    raw_results_dir = results_dir / "raw"
    raw_results_dir.mkdir(parents=True, exist_ok=True)

    results_file = raw_results_dir / "experiment_1_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.success(f"\n✓ Results saved to: {results_file}")

    # Save summary
    summary = results["summary"]
    summary_file = results_dir / "processed" / "experiment_1_summary.json"
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
    analysis_file = results_dir / "processed" / "experiment_1_analysis.json"
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
            logger.info(f"  Accuracy: {stats['accuracy_mean']:.1%} ± {stats['accuracy_std']:.3f}")
            logger.info(f"  Latency: {stats['latency_mean']:.3f}s ± {stats['latency_std']:.3f}s")
            logger.info(f"  Semantic Similarity: {stats['semantic_similarity_mean']:.3f}")

    if "anova" in analysis_results:
        logger.info(f"\nANOVA Results:")
        logger.info(f"  F-statistic: {analysis_results['anova']['f_statistic']:.3f}")
        logger.info(f"  p-value: {analysis_results['anova']['p_value']:.6f}")
        logger.info(f"  Significant: {analysis_results['anova']['significant']}")

    logger.info("\n" + "="*70)
    logger.success("EXPERIMENT 1 EXECUTION COMPLETE!")
    logger.info("="*70)

    return results, analysis_results


if __name__ == "__main__":
    results, analysis = main()
