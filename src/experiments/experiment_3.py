"""
Experiment 3: RAG Impact Analysis

Compares Retrieval-Augmented Generation (RAG) versus Full Context approaches
to understand trade-offs between retrieval precision and context completeness.

Research Questions:
1. How does RAG performance compare to full context for different query types?
2. What is the optimal top_k for retrieval?
3. How does latency scale with RAG vs full context?
4. What are the accuracy trade-offs?

Classes:
    Experiment3: Main experiment implementation
    QueryGenerator: Generate test queries
    CorpusManager: Manage document corpus

Example:
    >>> exp = Experiment3(config_path="config/experiments.yaml")
    >>> results = exp.run()
    >>> exp.save_results(results, "results/experiment_3/")
"""

import json
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from loguru import logger

from ..config import ConfigManager
from ..llm_interface import LLMInterface, OllamaInterface, Response
from ..rag_pipeline import RAGPipeline
from ..metrics import MetricsCalculator
from ..data_generation import DocumentGenerator
from ..visualization import plot_rag_comparison, plot_top_k_analysis
from ..statistics import StatisticalAnalyzer


@dataclass
class QueryResult:
    """Result from a single query."""
    query: str
    query_type: str
    approach: str  # "RAG" or "FullContext"
    top_k: Optional[int]  # Only for RAG
    answer: str
    ground_truth: Optional[str]
    latency: float
    tokens: int
    is_correct: bool
    retrieved_chunks: Optional[int] = None
    similarity_score: Optional[float] = None


@dataclass
class ExperimentResults:
    """Results from Experiment 3."""
    experiment_name: str
    timestamp: str
    config: Dict[str, Any]
    query_results: List[QueryResult]
    aggregate_metrics: Dict[str, Any]
    statistical_tests: Dict[str, Any]


class CorpusManager:
    """
    Manage document corpus for RAG experiments.

    Handles loading existing corpus or generating synthetic documents
    with embedded facts for testing.
    """

    def __init__(self, corpus_path: Optional[Path] = None, min_size: int = 20):
        """
        Initialize corpus manager.

        Args:
            corpus_path: Path to existing corpus
            min_size: Minimum corpus size
        """
        self.corpus_path = corpus_path
        self.min_size = min_size
        self.documents = []
        self.facts = []

        logger.info(f"CorpusManager initialized: min_size={min_size}")

    def load_or_generate_corpus(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Load corpus from path or generate synthetic corpus.

        Returns:
            Tuple of (documents, facts) where facts contain ground truth
        """
        if self.corpus_path and self.corpus_path.exists():
            logger.info(f"Loading corpus from {self.corpus_path}")
            documents, facts = self._load_corpus()
        else:
            logger.info("Generating synthetic corpus")
            documents, facts = self._generate_corpus()

        if len(documents) < self.min_size:
            logger.warning(f"Corpus size {len(documents)} < min_size {self.min_size}")
            # Generate additional documents
            additional_docs, additional_facts = self._generate_corpus(
                num_docs=self.min_size - len(documents)
            )
            documents.extend(additional_docs)
            facts.extend(additional_facts)

        self.documents = documents
        self.facts = facts

        logger.info(f"Corpus ready: {len(documents)} documents, {len(facts)} facts")
        return documents, facts

    def _load_corpus(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Load corpus from files.

        Returns:
            Tuple of (documents, facts)
        """
        documents = []
        facts = []

        # Try to load .txt files
        if self.corpus_path.is_dir():
            for file_path in self.corpus_path.glob("*.txt"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    documents.append(f.read())

        # Try to load facts.json
        facts_file = self.corpus_path / "facts.json" if self.corpus_path.is_dir() else None
        if facts_file and facts_file.exists():
            with open(facts_file, 'r', encoding='utf-8') as f:
                facts = json.load(f)

        return documents, facts

    def _generate_corpus(self, num_docs: int = 30) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Generate synthetic corpus with embedded facts.

        Args:
            num_docs: Number of documents to generate

        Returns:
            Tuple of (documents, facts)
        """
        generator = DocumentGenerator(random_seed=42)
        documents = []
        facts = []

        # Generate diverse documents
        topics = [
            "technology", "healthcare", "finance", "education", "environment",
            "energy", "transportation", "agriculture", "manufacturing", "retail"
        ]

        for i in range(num_docs):
            topic = topics[i % len(topics)]

            # Generate base document
            doc = generator.generate_document(num_words=random.randint(300, 600))

            # Embed factual information
            fact_value = self._generate_fact(topic, i)
            fact_sentence = f"In the {topic} sector, {fact_value['statement']}"

            # Insert fact at random position
            words = doc.split()
            insert_pos = random.randint(len(words) // 4, 3 * len(words) // 4)
            words.insert(insert_pos, fact_sentence)
            doc_with_fact = " ".join(words)

            documents.append(doc_with_fact)
            facts.append({
                "doc_id": i,
                "topic": topic,
                "fact": fact_value,
                "position": insert_pos
            })

        logger.info(f"Generated {num_docs} synthetic documents")
        return documents, facts

    def _generate_fact(self, topic: str, idx: int) -> Dict[str, Any]:
        """Generate a fact for a specific topic."""
        fact_types = {
            "technology": {
                "statement": f"the adoption rate increased by {random.randint(20, 80)}% in 2024",
                "answer": f"{random.randint(20, 80)}%"
            },
            "healthcare": {
                "statement": f"patient satisfaction scores reached {random.randint(70, 95)}%",
                "answer": f"{random.randint(70, 95)}%"
            },
            "finance": {
                "statement": f"the quarterly growth was {random.randint(5, 25)}%",
                "answer": f"{random.randint(5, 25)}%"
            },
            "education": {
                "statement": f"enrollment increased by {random.randint(10, 40)} students",
                "answer": f"{random.randint(10, 40)} students"
            },
            "environment": {
                "statement": f"emissions decreased by {random.randint(15, 50)}%",
                "answer": f"{random.randint(15, 50)}%"
            }
        }

        return fact_types.get(topic, {
            "statement": f"the key metric improved significantly",
            "answer": "significantly"
        })


class QueryGenerator:
    """
    Generate test queries for RAG evaluation.

    Creates factual and analytical queries based on corpus content.
    """

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize query generator.

        Args:
            random_seed: Random seed for reproducibility
        """
        if random_seed:
            random.seed(random_seed)

        logger.info("QueryGenerator initialized")

    def generate_queries(
        self,
        facts: List[Dict[str, Any]],
        num_factual: int = 10,
        num_analytical: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generate test queries.

        Args:
            facts: List of facts from corpus
            num_factual: Number of factual queries
            num_analytical: Number of analytical queries

        Returns:
            List of query dictionaries
        """
        queries = []

        # Generate factual queries
        for i in range(min(num_factual, len(facts))):
            fact = random.choice(facts)
            queries.append({
                "query": f"What was mentioned about the {fact['topic']} sector?",
                "type": "factual",
                "ground_truth": fact['fact']['answer'],
                "doc_id": fact['doc_id']
            })

        # Generate analytical queries
        topics = list(set(f['topic'] for f in facts))
        for i in range(min(num_analytical, len(topics))):
            topic = random.choice(topics)
            queries.append({
                "query": f"Analyze the trends and developments in the {topic} sector.",
                "type": "analytical",
                "ground_truth": None,  # No ground truth for analytical
                "doc_id": None
            })

        logger.info(f"Generated {len(queries)} queries")
        return queries


class Experiment3:
    """
    RAG Impact Analysis Experiment.

    Compares RAG vs Full Context approaches across different configurations
    to understand retrieval trade-offs.
    """

    def __init__(
        self,
        config_path: str = "config",
        llm_interface: Optional[LLMInterface] = None
    ):
        """
        Initialize Experiment 3.

        Args:
            config_path: Path to configuration directory (default: "config")
            llm_interface: Optional LLM interface (creates default if None)
        """
        self.config_manager = ConfigManager(config_dir=config_path)
        self.config = self.config_manager.get_experiment_config(3)
        self.general_config = self.config_manager.get_general_config()

        # Initialize LLM interface
        if llm_interface:
            self.llm = llm_interface
        else:
            model = self.general_config.get("ollama_model", "llama2:13b")
            self.llm = OllamaInterface(model=model)

        # Initialize metrics and statistics
        self.metrics_calc = MetricsCalculator()
        self.stats_analyzer = StatisticalAnalyzer()

        # Set random seed
        random_seed = self.general_config.get("random_seed", 42)
        random.seed(random_seed)
        np.random.seed(random_seed)

        logger.info(f"Experiment3 initialized: {self.config['name']}")

    def run(self) -> ExperimentResults:
        """
        Run the complete RAG impact experiment.

        Returns:
            ExperimentResults object with all results
        """
        logger.info("=" * 80)
        logger.info(f"Starting {self.config['name']}")
        logger.info("=" * 80)

        # Step 1: Load or generate corpus
        corpus_manager = CorpusManager(
            corpus_path=Path(self.config.get("corpus_path", "data/corpora/hebrew_corpus/")),
            min_size=self.config.get("min_corpus_size", 20)
        )
        documents, facts = corpus_manager.load_or_generate_corpus()

        # Step 2: Generate queries
        query_generator = QueryGenerator(
            random_seed=self.general_config.get("random_seed", 42)
        )

        query_config = self.config.get("query_types", {})
        num_factual = query_config.get("factual", {}).get("num_queries", 10)
        num_analytical = query_config.get("analytical", {}).get("num_queries", 10)

        queries = query_generator.generate_queries(
            facts=facts,
            num_factual=num_factual,
            num_analytical=num_analytical
        )

        # Step 3: Run RAG experiments with different top_k values
        query_results = []
        top_k_values = self.config.get("top_k_values", [1, 2, 3, 5, 7, 10])

        for top_k in top_k_values:
            logger.info(f"\n--- Testing RAG with top_k={top_k} ---")
            results = self._run_rag_queries(
                documents=documents,
                queries=queries,
                top_k=top_k
            )
            query_results.extend(results)

        # Step 4: Run Full Context experiments
        logger.info(f"\n--- Testing Full Context approach ---")
        full_context_results = self._run_full_context_queries(
            documents=documents,
            queries=queries
        )
        query_results.extend(full_context_results)

        # Step 5: Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(query_results)

        # Step 6: Perform statistical tests
        statistical_tests = self._perform_statistical_tests(query_results)

        # Create results object
        results = ExperimentResults(
            experiment_name=self.config['name'],
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            config=self.config,
            query_results=query_results,
            aggregate_metrics=aggregate_metrics,
            statistical_tests=statistical_tests
        )

        logger.info("=" * 80)
        logger.info(f"Experiment 3 completed: {len(query_results)} queries processed")
        logger.info("=" * 80)

        return results

    def _run_rag_queries(
        self,
        documents: List[str],
        queries: List[Dict[str, Any]],
        top_k: int
    ) -> List[QueryResult]:
        """
        Run queries using RAG approach.

        Args:
            documents: Corpus documents
            queries: List of queries to run
            top_k: Number of documents to retrieve

        Returns:
            List of QueryResult objects
        """
        # Initialize RAG pipeline
        rag = RAGPipeline(
            chunk_size=self.config.get("chunk_size", 500),
            chunk_overlap=self.config.get("chunk_overlap", 50),
            top_k=top_k,
            embedding_model=self.config.get("embedding_model", "nomic-embed-text")
        )

        # Index documents
        rag.index_documents(documents)

        results = []
        for query_dict in queries:
            query = query_dict["query"]
            query_type = query_dict["type"]
            ground_truth = query_dict.get("ground_truth")

            # Retrieve and query
            start_time = time.time()
            retrieved_chunks = rag.retrieve(query, k=top_k)
            context = "\n\n".join(chunk["text"] for chunk in retrieved_chunks)

            # Query LLM with retrieved context
            response = self.llm.query(context=context, query=query)
            latency = time.time() - start_time

            # Evaluate correctness
            is_correct = False
            similarity_score = 0.0

            if ground_truth:
                # For factual queries, check if answer contains ground truth
                is_correct = ground_truth.lower() in response.text.lower()
                similarity_score = self.metrics_calc.semantic_similarity(
                    response.text, ground_truth
                )

            results.append(QueryResult(
                query=query,
                query_type=query_type,
                approach="RAG",
                top_k=top_k,
                answer=response.text,
                ground_truth=ground_truth,
                latency=latency,
                tokens=response.tokens,
                is_correct=is_correct,
                retrieved_chunks=len(retrieved_chunks),
                similarity_score=similarity_score
            ))

            logger.debug(f"RAG query (top_k={top_k}): {query[:50]}... -> {is_correct}")

        logger.info(f"Completed {len(results)} RAG queries with top_k={top_k}")
        return results

    def _run_full_context_queries(
        self,
        documents: List[str],
        queries: List[Dict[str, Any]]
    ) -> List[QueryResult]:
        """
        Run queries using Full Context approach.

        Args:
            documents: Corpus documents
            queries: List of queries to run

        Returns:
            List of QueryResult objects
        """
        # Combine all documents into full context
        full_context = "\n\n".join(documents)

        results = []
        for query_dict in queries:
            query = query_dict["query"]
            query_type = query_dict["type"]
            ground_truth = query_dict.get("ground_truth")

            # Query LLM with full context
            start_time = time.time()
            response = self.llm.query(context=full_context, query=query)
            latency = time.time() - start_time

            # Evaluate correctness
            is_correct = False
            similarity_score = 0.0

            if ground_truth:
                is_correct = ground_truth.lower() in response.text.lower()
                similarity_score = self.metrics_calc.semantic_similarity(
                    response.text, ground_truth
                )

            results.append(QueryResult(
                query=query,
                query_type=query_type,
                approach="FullContext",
                top_k=None,
                answer=response.text,
                ground_truth=ground_truth,
                latency=latency,
                tokens=response.tokens,
                is_correct=is_correct,
                retrieved_chunks=len(documents),
                similarity_score=similarity_score
            ))

            logger.debug(f"Full context query: {query[:50]}... -> {is_correct}")

        logger.info(f"Completed {len(results)} Full Context queries")
        return results

    def _calculate_aggregate_metrics(
        self,
        query_results: List[QueryResult]
    ) -> Dict[str, Any]:
        """
        Calculate aggregate metrics across all queries.

        Args:
            query_results: List of all query results

        Returns:
            Dictionary of aggregate metrics
        """
        metrics = {
            "total_queries": len(query_results),
            "by_approach": {},
            "by_query_type": {},
            "by_top_k": {}
        }

        # Group by approach
        for approach in ["RAG", "FullContext"]:
            approach_results = [r for r in query_results if r.approach == approach]
            if approach_results:
                metrics["by_approach"][approach] = self._calculate_group_metrics(
                    approach_results
                )

        # Group by query type
        for query_type in ["factual", "analytical"]:
            type_results = [r for r in query_results if r.query_type == query_type]
            if type_results:
                metrics["by_query_type"][query_type] = self._calculate_group_metrics(
                    type_results
                )

        # Group by top_k (for RAG only)
        rag_results = [r for r in query_results if r.approach == "RAG"]
        top_k_values = sorted(set(r.top_k for r in rag_results if r.top_k))

        for top_k in top_k_values:
            top_k_results = [r for r in rag_results if r.top_k == top_k]
            if top_k_results:
                metrics["by_top_k"][f"k={top_k}"] = self._calculate_group_metrics(
                    top_k_results
                )

        return metrics

    def _calculate_group_metrics(
        self,
        results: List[QueryResult]
    ) -> Dict[str, float]:
        """Calculate metrics for a group of results."""
        if not results:
            return {}

        # Filter results with ground truth for accuracy calculation
        factual_results = [r for r in results if r.ground_truth is not None]

        return {
            "count": len(results),
            "accuracy": np.mean([r.is_correct for r in factual_results]) if factual_results else 0.0,
            "mean_latency": np.mean([r.latency for r in results]),
            "std_latency": np.std([r.latency for r in results]),
            "mean_tokens": np.mean([r.tokens for r in results]),
            "mean_similarity": np.mean([r.similarity_score for r in factual_results]) if factual_results else 0.0
        }

    def _perform_statistical_tests(
        self,
        query_results: List[QueryResult]
    ) -> Dict[str, Any]:
        """
        Perform statistical tests on results.

        Args:
            query_results: List of all query results

        Returns:
            Dictionary of statistical test results
        """
        tests = {}

        # Compare RAG vs Full Context latency
        rag_latencies = [r.latency for r in query_results if r.approach == "RAG"]
        full_latencies = [r.latency for r in query_results if r.approach == "FullContext"]

        if rag_latencies and full_latencies:
            tests["latency_rag_vs_full"] = self.stats_analyzer.t_test(
                rag_latencies,
                full_latencies,
                alternative="two-sided"
            )

        # Compare accuracies across top_k values
        rag_results = [r for r in query_results if r.approach == "RAG" and r.ground_truth]
        top_k_values = sorted(set(r.top_k for r in rag_results if r.top_k))

        if len(top_k_values) > 1:
            accuracy_groups = []
            for top_k in top_k_values:
                top_k_results = [r for r in rag_results if r.top_k == top_k]
                accuracies = [float(r.is_correct) for r in top_k_results]
                if accuracies:
                    accuracy_groups.append(accuracies)

            if len(accuracy_groups) > 1:
                tests["accuracy_across_top_k"] = self.stats_analyzer.anova(
                    accuracy_groups,
                    group_names=[f"k={k}" for k in top_k_values]
                )

        return tests

    def save_results(
        self,
        results: ExperimentResults,
        output_dir: str = "results/experiment_3/"
    ) -> None:
        """
        Save experiment results to disk.

        Args:
            results: ExperimentResults object
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save JSON results
        results_dict = {
            "experiment_name": results.experiment_name,
            "timestamp": results.timestamp,
            "config": results.config,
            "query_results": [asdict(r) for r in results.query_results],
            "aggregate_metrics": results.aggregate_metrics,
            "statistical_tests": results.statistical_tests
        }

        # Custom JSON encoder to handle numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.bool_):
                    return bool(obj)
                return super().default(obj)

        json_path = output_path / f"results_{results.timestamp.replace(':', '-').replace(' ', '_')}.json"
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=2, cls=NumpyEncoder)

        logger.info(f"Results saved to {json_path}")

        # Generate visualizations
        self._generate_visualizations(results, output_path)

    def _generate_visualizations(
        self,
        results: ExperimentResults,
        output_path: Path
    ) -> None:
        """
        Generate visualizations for experiment results.

        Args:
            results: ExperimentResults object
            output_path: Directory to save plots
        """
        try:
            # Plot RAG vs Full Context comparison
            plot_rag_comparison(
                results.query_results,
                save_path=str(output_path / "rag_vs_full_comparison.png")
            )

            # Plot top_k analysis
            plot_top_k_analysis(
                results.query_results,
                save_path=str(output_path / "top_k_analysis.png")
            )

            logger.info("Visualizations generated successfully")
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")


def main():
    """Main entry point for running Experiment 3."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Experiment 3: RAG Impact")
    parser.add_argument(
        "--config",
        default="config",
        help="Path to configuration directory"
    )
    parser.add_argument(
        "--output",
        default="results/experiment_3/",
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Run experiment
    exp = Experiment3(config_path=args.config)
    results = exp.run()
    exp.save_results(results, output_dir=args.output)

    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT 3 SUMMARY")
    print("=" * 80)
    print(f"Total queries: {results.aggregate_metrics['total_queries']}")
    print("\nBy Approach:")
    for approach, metrics in results.aggregate_metrics['by_approach'].items():
        print(f"  {approach}:")
        print(f"    Accuracy: {metrics['accuracy']:.3f}")
        print(f"    Mean Latency: {metrics['mean_latency']:.3f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
