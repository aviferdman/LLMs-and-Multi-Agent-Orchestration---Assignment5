"""
Experiment 1: Lost in the Middle

This module implements the "Lost in the Middle" experiment, testing whether
LLMs exhibit position bias when retrieving information from long contexts.

The experiment embeds facts at different positions (start, middle, end) within
documents and measures retrieval accuracy, demonstrating whether middle-positioned
information is harder to retrieve.

Classes:
    Experiment1: Main experimental logic
    DocumentWithFact: Container for document with embedded fact
    TrialResult: Results from a single trial

Research Questions:
    RQ1.1: Is there statistically significant accuracy degradation for
           middle-positioned facts?

Example:
    >>> from experiments.experiment_1 import Experiment1
    >>> exp = Experiment1(config)
    >>> results = exp.run()
    >>> exp.analyze(results)
    >>> exp.visualize(results)
"""

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from loguru import logger

from ..data_generation import DocumentGenerator, FactEmbedder, FactGenerator
from ..llm_interface import LLMInterface, OllamaInterface, Response
from ..metrics import MetricsCalculator
from ..statistics import anova_test, t_test, confidence_interval


@dataclass
class DocumentWithFact:
    """
    Container for a document with an embedded fact.

    Attributes:
        document: The full document text with embedded fact
        fact: The embedded fact
        position: Position where fact was embedded ('start', 'middle', 'end')
        ground_truth: The expected answer
        metadata: Additional metadata
    """
    document: str
    fact: str
    position: str
    ground_truth: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TrialResult:
    """
    Results from a single trial.

    Attributes:
        position: Fact position in document
        query: Query asked
        response: LLM response
        ground_truth: Expected answer
        correct: Whether response was correct
        latency: Response time in seconds
        tokens: Number of tokens in response
        semantic_similarity: Similarity score between response and ground truth
        metadata: Additional trial metadata
    """
    position: str
    query: str
    response: str
    ground_truth: str
    correct: bool
    latency: float
    tokens: int
    semantic_similarity: float
    metadata: Optional[Dict[str, Any]] = None


class Experiment1:
    """
    Implementation of the "Lost in the Middle" experiment.

    Tests whether fact position (start, middle, end) affects retrieval
    accuracy in LLM responses.

    Attributes:
        config: Experiment configuration dictionary
        llm: LLM interface for querying
        metrics_calculator: Metrics calculation utility
        doc_generator: Document generation utility
        fact_embedder: Fact embedding utility
        fact_generator: Fact generation utility
        results_dir: Directory for saving results

    Methods:
        run(): Execute full experiment
        analyze(): Perform statistical analysis
        visualize(): Create result plots
        save_results(): Save results to disk
    """

    def __init__(self, config: Dict[str, Any], llm: Optional[LLMInterface] = None):
        """
        Initialize Experiment 1.

        Args:
            config: Configuration dictionary containing experiment parameters
            llm: Optional LLM interface (creates default if not provided)
        """
        self.config = config

        # Extract experiment-specific config
        exp_config = config.get("experiment_1", {})
        general_config = config.get("general", {})

        # Configuration parameters
        self.num_documents = exp_config.get("num_documents", 5)
        self.words_per_document = exp_config.get("words_per_document", 200)
        self.word_variance = exp_config.get("word_variance", 20)
        self.fact_positions = exp_config.get("fact_positions", ["start", "middle", "end"])
        self.num_trials_per_position = exp_config.get("num_trials_per_position", 10)
        self.query_template = exp_config.get("query_template", "What is the CEO's name?")
        self.random_seed = general_config.get("random_seed", 42)

        # Initialize components
        self.llm = llm or OllamaInterface(model=general_config.get("ollama_model", "llama2:13b"))
        self.metrics_calculator = MetricsCalculator()
        self.doc_generator = DocumentGenerator(random_seed=self.random_seed)
        self.fact_embedder = FactEmbedder()
        self.fact_generator = FactGenerator(random_seed=self.random_seed)

        # Setup results directory
        self.results_dir = Path(general_config.get("output_dir", "results")) / "raw"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized Experiment 1: {exp_config.get('name', 'Lost in the Middle')}")
        logger.info(f"  - Positions: {self.fact_positions}")
        logger.info(f"  - Trials per position: {self.num_trials_per_position}")
        logger.info(f"  - Total trials: {len(self.fact_positions) * self.num_trials_per_position}")

    def generate_document_with_fact(
        self,
        position: str,
        fact_type: str = "ceo_name"
    ) -> DocumentWithFact:
        """
        Generate a document with an embedded fact at specified position.

        Args:
            position: Position to embed fact ('start', 'middle', 'end')
            fact_type: Type of fact to generate

        Returns:
            DocumentWithFact object
        """
        # Generate base documents
        documents = self.doc_generator.generate_documents(
            num_documents=self.num_documents,
            num_words=self.words_per_document,
            word_variance=self.word_variance
        )

        # Generate fact and extract ground truth
        fact = self.fact_generator.generate_fact(fact_type)
        ground_truth = self._extract_answer_from_fact(fact, fact_type)

        # Embed fact in one of the documents (middle document for consistency)
        target_doc_idx = len(documents) // 2
        documents[target_doc_idx] = self.fact_embedder.embed_fact(
            documents[target_doc_idx],
            fact,
            position
        )

        # Concatenate all documents
        full_document = "\n\n".join([
            f"Document {i+1}:\n{doc}"
            for i, doc in enumerate(documents)
        ])

        return DocumentWithFact(
            document=full_document,
            fact=fact,
            position=position,
            ground_truth=ground_truth,
            metadata={
                "num_documents": self.num_documents,
                "target_document": target_doc_idx + 1,
                "fact_type": fact_type,
                "total_words": len(full_document.split())
            }
        )

    def _extract_answer_from_fact(self, fact: str, fact_type: str) -> str:
        """
        Extract the expected answer from a fact.

        Args:
            fact: The fact string
            fact_type: Type of fact

        Returns:
            Expected answer string
        """
        if fact_type == "ceo_name":
            # Extract name from "The CEO of the company is {name}."
            if "is" in fact:
                parts = fact.split("is")
                if len(parts) >= 2:
                    name = parts[-1].strip().rstrip(".")
                    return name
        elif fact_type == "revenue":
            # Extract amount from "The Q3 revenue was ${amount}M."
            if "$" in fact:
                parts = fact.split("$")
                if len(parts) >= 2:
                    amount = parts[-1].strip().rstrip("M.").strip()
                    return f"${amount}M"
        elif fact_type == "location":
            # Extract city from "The headquarters is located in {city}."
            if "in" in fact:
                parts = fact.split("in")
                if len(parts) >= 2:
                    city = parts[-1].strip().rstrip(".")
                    return city

        # Fallback: return the whole fact
        logger.warning(f"Could not extract answer from fact: {fact}")
        return fact

    def run_trial(
        self,
        position: str,
        fact_type: str = "ceo_name"
    ) -> TrialResult:
        """
        Run a single trial of the experiment.

        Args:
            position: Position to test ('start', 'middle', 'end')
            fact_type: Type of fact to test

        Returns:
            TrialResult object
        """
        # Generate document with fact
        doc_with_fact = self.generate_document_with_fact(position, fact_type)

        # Query LLM
        logger.debug(f"Querying LLM for position '{position}'")
        response = self.llm.query(
            context=doc_with_fact.document,
            query=self.query_template
        )

        # Evaluate response
        correct = self._check_correctness(
            response.text,
            doc_with_fact.ground_truth
        )

        semantic_sim = self.metrics_calculator.semantic_similarity(
            response.text,
            doc_with_fact.ground_truth
        )

        result = TrialResult(
            position=position,
            query=self.query_template,
            response=response.text,
            ground_truth=doc_with_fact.ground_truth,
            correct=correct,
            latency=response.latency,
            tokens=response.tokens,
            semantic_similarity=semantic_sim,
            metadata={
                "fact": doc_with_fact.fact,
                "fact_type": fact_type,
                "document_length": len(doc_with_fact.document.split()),
                "timestamp": datetime.now().isoformat()
            }
        )

        logger.debug(f"Trial result: position={position}, correct={correct}, "
                    f"similarity={semantic_sim:.3f}, latency={response.latency:.2f}s")

        return result

    def _check_correctness(self, response: str, ground_truth: str) -> bool:
        """
        Check if response contains the correct answer.

        Args:
            response: LLM response text
            ground_truth: Expected answer

        Returns:
            True if correct, False otherwise
        """
        # Normalize both strings
        response_lower = response.lower().strip()
        truth_lower = ground_truth.lower().strip()

        # Check for exact match or substring match
        return truth_lower in response_lower

    def run(self) -> Dict[str, Any]:
        """
        Execute the full experiment.

        Runs multiple trials for each position and collects all results.

        Returns:
            Dictionary containing all results and metadata
        """
        logger.info("=" * 60)
        logger.info("Starting Experiment 1: Lost in the Middle")
        logger.info("=" * 60)

        start_time = time.time()
        all_trials = []

        # Run trials for each position
        for position in self.fact_positions:
            logger.info(f"\nTesting position: {position.upper()}")
            logger.info(f"Running {self.num_trials_per_position} trials...")

            for trial_num in range(self.num_trials_per_position):
                logger.info(f"  Trial {trial_num + 1}/{self.num_trials_per_position}")

                trial_result = self.run_trial(position)
                all_trials.append(trial_result)

        # Aggregate results
        total_time = time.time() - start_time

        results = {
            "experiment": "experiment_1",
            "name": "Lost in the Middle",
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": total_time,
            "configuration": {
                "num_documents": self.num_documents,
                "words_per_document": self.words_per_document,
                "fact_positions": self.fact_positions,
                "num_trials_per_position": self.num_trials_per_position,
                "random_seed": self.random_seed
            },
            "trials": [asdict(trial) for trial in all_trials],
            "summary": self._calculate_summary(all_trials)
        }

        logger.info("\n" + "=" * 60)
        logger.info(f"Experiment 1 completed in {total_time:.2f} seconds")
        logger.info("=" * 60)

        # Log summary statistics
        self._log_summary(results["summary"])

        return results

    def _calculate_summary(self, trials: List[TrialResult]) -> Dict[str, Any]:
        """
        Calculate summary statistics from trial results.

        Args:
            trials: List of trial results

        Returns:
            Dictionary with summary statistics
        """
        summary = {}

        # Group by position
        for position in self.fact_positions:
            position_trials = [t for t in trials if t.position == position]

            if not position_trials:
                continue

            accuracies = [1.0 if t.correct else 0.0 for t in position_trials]
            latencies = [t.latency for t in position_trials]
            semantic_sims = [t.semantic_similarity for t in position_trials]

            summary[position] = {
                "num_trials": len(position_trials),
                "accuracy": {
                    "mean": np.mean(accuracies),
                    "std": np.std(accuracies),
                    "values": accuracies
                },
                "latency": {
                    "mean": np.mean(latencies),
                    "std": np.std(latencies),
                    "min": np.min(latencies),
                    "max": np.max(latencies)
                },
                "semantic_similarity": {
                    "mean": np.mean(semantic_sims),
                    "std": np.std(semantic_sims)
                }
            }

        return summary

    def _log_summary(self, summary: Dict[str, Any]) -> None:
        """
        Log summary statistics to console.

        Args:
            summary: Summary statistics dictionary
        """
        logger.info("\nSUMMARY STATISTICS:")
        logger.info("-" * 60)

        for position in self.fact_positions:
            if position not in summary:
                continue

            stats = summary[position]
            acc = stats["accuracy"]["mean"]
            lat = stats["latency"]["mean"]
            sem = stats["semantic_similarity"]["mean"]

            logger.info(f"{position.upper():>10}: Accuracy={acc:.2%}, "
                       f"Latency={lat:.2f}s, Similarity={sem:.3f}")

    def analyze(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform statistical analysis on results.

        Includes:
        - One-way ANOVA for position effect
        - Post-hoc pairwise t-tests
        - Effect sizes (Cohen's d, eta-squared)
        - Confidence intervals

        Args:
            results: Results dictionary from run()

        Returns:
            Dictionary with statistical analysis results
        """
        logger.info("\n" + "=" * 60)
        logger.info("STATISTICAL ANALYSIS")
        logger.info("=" * 60)

        # Extract accuracy data by position
        accuracy_by_position = {}
        for position in self.fact_positions:
            if position in results["summary"]:
                accuracy_by_position[position] = np.array(
                    results["summary"][position]["accuracy"]["values"]
                )

        # Prepare groups for ANOVA
        groups = [accuracy_by_position[pos] for pos in self.fact_positions
                  if pos in accuracy_by_position]

        analysis = {}

        # 1. One-way ANOVA
        logger.info("\n1. One-Way ANOVA")
        logger.info("-" * 40)
        anova_results = anova_test(groups)
        analysis["anova"] = anova_results

        logger.info(f"F-statistic: {anova_results['f_statistic']:.4f}")
        logger.info(f"p-value: {anova_results['p_value']:.4f}")
        logger.info(f"Eta-squared (��): {anova_results['eta_squared']:.4f}")
        logger.info(f"Significant: {anova_results['significant']}")

        # 2. Post-hoc pairwise comparisons
        logger.info("\n2. Post-hoc Pairwise Comparisons")
        logger.info("-" * 40)
        pairwise_results = {}

        positions = list(accuracy_by_position.keys())
        for i, pos1 in enumerate(positions):
            for pos2 in positions[i+1:]:
                comparison_key = f"{pos1}_vs_{pos2}"

                test_result = t_test(
                    accuracy_by_position[pos1],
                    accuracy_by_position[pos2],
                    paired=False
                )

                pairwise_results[comparison_key] = test_result

                logger.info(f"\n{pos1.upper()} vs {pos2.upper()}:")
                logger.info(f"  t-statistic: {test_result['t_statistic']:.4f}")
                logger.info(f"  p-value: {test_result['p_value']:.4f}")
                logger.info(f"  Cohen's d: {test_result['cohens_d']:.4f}")
                logger.info(f"  Significant: {test_result['significant']}")

        analysis["pairwise"] = pairwise_results

        # 3. Confidence Intervals
        logger.info("\n3. 95% Confidence Intervals")
        logger.info("-" * 40)
        confidence_intervals = {}

        for position, data in accuracy_by_position.items():
            ci = confidence_interval(data, confidence=0.95)
            confidence_intervals[position] = {
                "lower": ci[0],
                "upper": ci[1],
                "mean": np.mean(data)
            }

            logger.info(f"{position.upper():>10}: {np.mean(data):.3f} "
                       f"[{ci[0]:.3f}, {ci[1]:.3f}]")

        analysis["confidence_intervals"] = confidence_intervals

        # 4. Effect size interpretation
        logger.info("\n4. Effect Size Interpretation")
        logger.info("-" * 40)
        logger.info(f"Eta-squared: {anova_results['eta_squared']:.4f}")

        eta_sq = anova_results['eta_squared']
        if eta_sq < 0.01:
            interpretation = "negligible"
        elif eta_sq < 0.06:
            interpretation = "small"
        elif eta_sq < 0.14:
            interpretation = "medium"
        else:
            interpretation = "large"

        logger.info(f"Interpretation: {interpretation} effect")
        analysis["effect_size_interpretation"] = interpretation

        return analysis

    def save_results(
        self,
        results: Dict[str, Any],
        analysis: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save results to disk.

        Args:
            results: Results dictionary from run()
            analysis: Optional analysis results from analyze()

        Returns:
            Path to saved results file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exp1_results_{timestamp}.json"
        filepath = self.results_dir / filename

        output = {
            "results": results,
            "analysis": analysis
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"\nResults saved to: {filepath}")
        return filepath

    def visualize(self, results: Dict[str, Any]) -> None:
        """
        Create visualizations of results.

        Note: Visualization implementation will be added by visualization module.
        This method provides a placeholder for integration.

        Args:
            results: Results dictionary from run()
        """
        logger.info("\nVisualization will be implemented by visualization module")
        logger.info("Expected visualizations:")
        logger.info("  - Bar chart: Mean accuracy by position with 95% CI")
        logger.info("  - Box plots: Accuracy distributions by position")
        logger.info("  - Statistical annotations with p-values")

        # TODO: Integrate with visualization module when available
        pass


def main():
    """
    Main function for standalone execution.

    Example usage:
        python -m src.experiments.experiment_1
    """
    import yaml

    # Load configuration
    config_path = Path("config/experiments.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create and run experiment
    experiment = Experiment1(config)
    results = experiment.run()
    analysis = experiment.analyze(results)
    experiment.save_results(results, analysis)
    experiment.visualize(results)


if __name__ == "__main__":
    main()
