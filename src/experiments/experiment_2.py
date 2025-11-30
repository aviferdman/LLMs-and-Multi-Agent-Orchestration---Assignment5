"""
Experiment 2: Context Size Impact

Tests how accuracy degrades as context size increases.
Measures latency scaling and identifies optimal context size ranges.

Research Questions:
- RQ2.1: What is the functional form of accuracy degradation?
- RQ2.2: Is there a "cliff" where performance drops sharply?
- RQ2.3: How does latency scale with context size?
- RQ2.4: What is optimal size for 90% accuracy + minimum latency?

Based on PRD Section 4.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ..config import Config
from ..data_generation import DocumentGenerator, FactEmbedder, FactGenerator
from ..llm_interface import LLMInterface, OllamaInterface
from ..metrics import MetricsCalculator
from ..statistics import (
    anova_test,
    cohen_d,
    confidence_interval,
    correlation,
    regression_linear,
)
from ..visualization import Visualizer


class Experiment2:
    """
    Context Size Impact Experiment.

    Tests accuracy degradation and latency scaling as context size grows.

    Attributes:
        config: Configuration object
        llm: LLM interface
        metrics_calculator: Metrics calculator
        visualizer: Visualization generator
        doc_generator: Document generator
        fact_embedder: Fact embedder
        fact_generator: Fact generator
        results_dir: Directory for saving results
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        llm: Optional[LLMInterface] = None,
    ):
        """
        Initialize Experiment 2.

        Args:
            config: Configuration object (creates default if None)
            llm: LLM interface (creates OllamaInterface if None)
        """
        self.config = config or Config()
        self.llm = llm or OllamaInterface(
            model=self.config.get("general.ollama_model", "llama2:13b")
        )

        # Initialize components
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = Visualizer(
            output_dir=str(Path(self.config.get("general.output_dir", "results")) / "figures")
        )

        # Get experiment-specific config
        self.exp_config = self.config.get_experiment_config(2)

        # Random seed for reproducibility
        random_seed = self.config.get("general.random_seed", 42)
        np.random.seed(random_seed)

        # Initialize generators
        self.doc_generator = DocumentGenerator(random_seed=random_seed)
        self.fact_embedder = FactEmbedder()
        self.fact_generator = FactGenerator(random_seed=random_seed)

        # Results directory
        self.results_dir = Path(self.config.get("general.output_dir", "results"))
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Experiment 2 initialized")

    def generate_context(
        self,
        num_documents: int,
        words_per_doc: int = 200,
        target_doc_index: Optional[int] = None,
    ) -> Tuple[str, str, int]:
        """
        Generate context with multiple documents.

        Args:
            num_documents: Number of documents to generate
            words_per_doc: Words per document
            target_doc_index: Which document contains the target fact (random if None)

        Returns:
            Tuple of (context_string, target_answer, target_doc_index)
        """
        # Generate documents
        documents = self.doc_generator.generate_documents(
            num_documents=num_documents,
            num_words=words_per_doc,
            word_variance=20,
        )

        # Choose random document to embed fact
        if target_doc_index is None:
            target_doc_index = np.random.randint(0, num_documents)

        # Generate fact and embed it
        fact = self.fact_generator.generate_fact("ceo_name")
        # Extract answer from fact
        target_answer = fact.split("is ")[-1].rstrip(".")

        # Embed fact in middle of target document
        documents[target_doc_index] = self.fact_embedder.embed_fact(
            documents[target_doc_index], fact, position="middle"
        )

        # Combine all documents
        context = "\n\n".join(
            [f"Document {i+1}:\n{doc}" for i, doc in enumerate(documents)]
        )

        return context, target_answer, target_doc_index

    def run_single_trial(
        self,
        num_documents: int,
        query: str = "What is the CEO's name?",
        task_type: str = "simple",
    ) -> Dict[str, Any]:
        """
        Run a single trial with specified number of documents.

        Args:
            num_documents: Number of documents in context
            query: Query to ask
            task_type: 'simple' or 'complex'

        Returns:
            Dictionary with trial results
        """
        # Generate context
        context, target_answer, target_doc_index = self.generate_context(
            num_documents=num_documents
        )

        # Count tokens
        context_tokens = self.llm.count_tokens(context)
        query_tokens = self.llm.count_tokens(query)
        total_tokens = context_tokens + query_tokens

        # Query LLM
        logger.debug(
            f"Querying LLM with {num_documents} docs, {total_tokens} tokens"
        )
        response = self.llm.query(context=context, query=query)

        # Evaluate response
        is_correct = target_answer.lower() in response.text.lower()
        semantic_sim = self.metrics_calculator.semantic_similarity(
            response.text, target_answer
        )

        result = {
            "num_documents": num_documents,
            "context_tokens": context_tokens,
            "total_tokens": total_tokens,
            "target_doc_index": target_doc_index,
            "query": query,
            "target_answer": target_answer,
            "response": response.text,
            "is_correct": is_correct,
            "accuracy": 1.0 if is_correct else 0.0,
            "semantic_similarity": semantic_sim,
            "latency": response.latency,
            "response_tokens": response.tokens,
            "task_type": task_type,
            "timestamp": datetime.now().isoformat(),
        }

        logger.debug(
            f"Trial: {num_documents} docs, accuracy={result['accuracy']}, "
            f"latency={result['latency']:.2f}s"
        )

        return result

    def run_context_size_sweep(
        self,
        context_sizes: Optional[List[int]] = None,
        num_runs_per_size: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Run experiment across multiple context sizes.

        Args:
            context_sizes: List of context sizes to test
            num_runs_per_size: Number of trials per size

        Returns:
            List of all trial results
        """
        if context_sizes is None:
            context_sizes = self.exp_config.get(
                "context_sizes", [2, 5, 10, 20, 50]
            )

        logger.info(
            f"Starting context size sweep: sizes={context_sizes}, "
            f"runs_per_size={num_runs_per_size}"
        )

        all_results = []

        for size in context_sizes:
            logger.info(f"Testing context size: {size} documents")

            for run in range(num_runs_per_size):
                try:
                    result = self.run_single_trial(num_documents=size)
                    all_results.append(result)

                    logger.info(
                        f"  Run {run+1}/{num_runs_per_size}: "
                        f"accuracy={result['accuracy']:.2f}, "
                        f"latency={result['latency']:.2f}s"
                    )

                except Exception as e:
                    logger.error(f"Error in trial (size={size}, run={run}): {e}")
                    continue

        logger.success(
            f"Context size sweep completed: {len(all_results)} trials"
        )
        return all_results

    def run_task_complexity_comparison(
        self,
        context_sizes: Optional[List[int]] = None,
        num_runs_per_condition: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Compare simple vs complex tasks across context sizes.

        Args:
            context_sizes: List of context sizes to test
            num_runs_per_condition: Runs per (size, task_type) pair

        Returns:
            List of all trial results
        """
        if context_sizes is None:
            context_sizes = self.exp_config.get("context_sizes", [2, 5, 10, 20])

        logger.info("Starting task complexity comparison")

        all_results = []

        # Simple task
        simple_query = "What is the CEO's name?"

        # Complex task
        complex_query = (
            "Based on all the information provided, what is the CEO's name "
            "and how does this information relate to the overall context?"
        )

        for size in context_sizes:
            logger.info(f"Testing size {size} - simple task")
            for run in range(num_runs_per_condition):
                try:
                    result = self.run_single_trial(
                        num_documents=size,
                        query=simple_query,
                        task_type="simple",
                    )
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Error in simple task: {e}")

            logger.info(f"Testing size {size} - complex task")
            for run in range(num_runs_per_condition):
                try:
                    result = self.run_single_trial(
                        num_documents=size,
                        query=complex_query,
                        task_type="complex",
                    )
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Error in complex task: {e}")

        logger.success(f"Task complexity comparison completed")
        return all_results

    def analyze_results(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Perform statistical analysis on results.

        Args:
            results: List of trial results

        Returns:
            Dictionary with analysis results
        """
        logger.info("Analyzing results...")

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Group by context size
        size_groups = df.groupby("num_documents")

        # Calculate summary statistics
        summary_stats = {
            "mean_accuracy": size_groups["accuracy"].mean().to_dict(),
            "std_accuracy": size_groups["accuracy"].std().to_dict(),
            "mean_latency": size_groups["latency"].mean().to_dict(),
            "std_latency": size_groups["latency"].std().to_dict(),
            "mean_tokens": size_groups["total_tokens"].mean().to_dict(),
        }

        # Confidence intervals for accuracy
        ci_accuracy = {}
        for size, group in size_groups:
            if len(group) > 1:
                ci_lower, ci_upper = confidence_interval(
                    group["accuracy"].values, confidence=0.95
                )
                ci_accuracy[size] = {
                    "lower": ci_lower,
                    "upper": ci_upper,
                    "mean": group["accuracy"].mean(),
                }

        # Correlation: accuracy vs context size
        corr_acc_size = correlation(
            df["num_documents"].values,
            df["accuracy"].values,
            method="pearson",
        )

        # Correlation: latency vs context size
        corr_lat_size = correlation(
            df["num_documents"].values,
            df["latency"].values,
            method="pearson",
        )

        # Linear regression: accuracy ~ log(size)
        log_sizes = np.log(df["num_documents"].values)
        reg_acc = regression_linear(log_sizes, df["accuracy"].values)

        # Linear regression: latency ~ size (check for quadratic)
        reg_lat_linear = regression_linear(
            df["num_documents"].values, df["latency"].values
        )

        # Quadratic regression for latency
        size_arr = df["num_documents"].values
        lat_arr = df["latency"].values
        poly_coeffs = np.polyfit(size_arr, lat_arr, 2)
        reg_lat_quadratic = {
            "a": poly_coeffs[0],
            "b": poly_coeffs[1],
            "c": poly_coeffs[2],
            "equation": f"{poly_coeffs[0]:.4f}*x^2 + {poly_coeffs[1]:.4f}*x + {poly_coeffs[2]:.4f}",
        }

        analysis = {
            "summary_statistics": summary_stats,
            "confidence_intervals": ci_accuracy,
            "correlation_accuracy_size": corr_acc_size,
            "correlation_latency_size": corr_lat_size,
            "regression_accuracy": reg_acc,
            "regression_latency_linear": reg_lat_linear,
            "regression_latency_quadratic": reg_lat_quadratic,
            "num_trials": len(df),
            "context_sizes_tested": sorted(df["num_documents"].unique().tolist()),
        }

        logger.success("Analysis completed")
        return analysis

    def generate_visualizations(
        self,
        results: List[Dict[str, Any]],
        analysis: Dict[str, Any],
    ):
        """
        Generate all visualizations for Experiment 2.

        Args:
            results: List of trial results
            analysis: Analysis results
        """
        logger.info("Generating visualizations...")

        df = pd.DataFrame(results)

        # 1. Accuracy vs Context Size (with fitted curve)
        sizes = sorted(df["num_documents"].unique())
        mean_acc = [df[df["num_documents"] == s]["accuracy"].mean() for s in sizes]

        self.visualizer.plot_scaling_curve(
            x=sizes,
            y=mean_acc,
            xlabel="Number of Documents",
            ylabel="Accuracy",
            title="Accuracy vs Context Size",
            fit_curve=True,
            save_path="exp2_accuracy_vs_size.png",
        )

        # 2. Latency vs Context Size
        mean_lat = [df[df["num_documents"] == s]["latency"].mean() for s in sizes]

        self.visualizer.plot_scaling_curve(
            x=sizes,
            y=mean_lat,
            xlabel="Number of Documents",
            ylabel="Latency (seconds)",
            title="Latency vs Context Size",
            fit_curve=True,
            save_path="exp2_latency_vs_size.png",
        )

        # 3. Box plots of accuracy distribution by size
        accuracy_by_size = {
            str(size): df[df["num_documents"] == size]["accuracy"].tolist()
            for size in sizes
        }

        self.visualizer.plot_box_plots(
            data=accuracy_by_size,
            title="Accuracy Distribution by Context Size",
            ylabel="Accuracy",
            save_path="exp2_accuracy_distribution.png",
        )

        # 4. Task complexity comparison (if available)
        if "task_type" in df.columns and len(df["task_type"].unique()) > 1:
            self.visualizer.plot_comparison_bars(
                data=df,
                x_col="num_documents",
                y_col="accuracy",
                hue_col="task_type",
                title="Accuracy: Simple vs Complex Tasks",
                save_path="exp2_task_complexity.png",
            )

        logger.success("Visualizations generated")

    def save_results(
        self,
        results: List[Dict[str, Any]],
        analysis: Dict[str, Any],
        filename_prefix: str = "exp2",
    ):
        """
        Save results to JSON and CSV files.

        Args:
            results: List of trial results
            analysis: Analysis results
            filename_prefix: Prefix for output files
        """
        logger.info("Saving results...")

        # Save raw results to JSON
        raw_file = self.results_dir / f"{filename_prefix}_raw_results.json"
        with open(raw_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved raw results: {raw_file}")

        # Save analysis to JSON
        analysis_file = self.results_dir / f"{filename_prefix}_analysis.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"Saved analysis: {analysis_file}")

        # Save results to CSV
        df = pd.DataFrame(results)
        csv_file = self.results_dir / f"{filename_prefix}_results.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"Saved results CSV: {csv_file}")

        # Save summary statistics to CSV
        summary_df = pd.DataFrame(analysis["summary_statistics"])
        summary_file = self.results_dir / f"{filename_prefix}_summary.csv"
        summary_df.to_csv(summary_file)
        logger.info(f"Saved summary: {summary_file}")

        logger.success("Results saved")

    def run(
        self,
        include_task_complexity: bool = False,
    ) -> Dict[str, Any]:
        """
        Run complete Experiment 2.

        Args:
            include_task_complexity: Whether to include task complexity comparison

        Returns:
            Dictionary with all results and analysis
        """
        logger.info("=" * 60)
        logger.info("Starting Experiment 2: Context Size Impact")
        logger.info("=" * 60)

        start_time = time.time()

        # Get configuration
        context_sizes = self.exp_config.get("context_sizes", [2, 5, 10, 20, 50])
        num_runs = self.config.get("general.num_runs", 3)

        # Run main experiment
        logger.info("Phase 1: Context size sweep")
        results = self.run_context_size_sweep(
            context_sizes=context_sizes,
            num_runs_per_size=num_runs,
        )

        # Optional: Task complexity comparison
        if include_task_complexity:
            logger.info("Phase 2: Task complexity comparison")
            complexity_results = self.run_task_complexity_comparison(
                context_sizes=context_sizes[:4],  # Use smaller sizes
                num_runs_per_condition=num_runs,
            )
            results.extend(complexity_results)

        # Analyze results
        analysis = self.analyze_results(results)

        # Generate visualizations
        self.generate_visualizations(results, analysis)

        # Save results
        self.save_results(results, analysis)

        # Calculate total duration
        duration = time.time() - start_time

        # Summary
        summary = {
            "experiment": "Experiment 2: Context Size Impact",
            "num_trials": len(results),
            "context_sizes": context_sizes,
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat(),
            "key_findings": self._generate_key_findings(analysis),
        }

        logger.info("=" * 60)
        logger.info("Experiment 2 Summary:")
        logger.info(f"  Total trials: {len(results)}")
        logger.info(f"  Duration: {duration:.2f} seconds")
        logger.info(f"  Context sizes: {context_sizes}")
        logger.info("=" * 60)

        return {
            "summary": summary,
            "results": results,
            "analysis": analysis,
        }

    def _generate_key_findings(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate key findings from analysis.

        Args:
            analysis: Analysis results

        Returns:
            Dictionary of key findings
        """
        findings = {}

        # Accuracy degradation
        corr = analysis.get("correlation_accuracy_size", {})
        findings["accuracy_correlation"] = (
            f"Accuracy vs context size correlation: r={corr.get('correlation', 0):.3f}, "
            f"p={corr.get('p_value', 1):.4f}"
        )

        # Regression equation
        reg = analysis.get("regression_accuracy", {})
        findings["accuracy_model"] = (
            f"Accuracy = {reg.get('slope', 0):.3f}*log(size) + {reg.get('intercept', 0):.3f}, "
            f"R^2={reg.get('r_squared', 0):.3f}"
        )

        # Latency scaling
        lat_reg = analysis.get("regression_latency_quadratic", {})
        findings["latency_scaling"] = (
            f"Latency scales quadratically: {lat_reg.get('equation', 'N/A')}"
        )

        return findings


def main():
    """Main function for running Experiment 2."""
    # Load configuration
    config = Config()

    # Initialize experiment
    experiment = Experiment2(config=config)

    # Run experiment
    results = experiment.run(include_task_complexity=True)

    logger.success("Experiment 2 completed successfully!")

    return results


if __name__ == "__main__":
    main()
