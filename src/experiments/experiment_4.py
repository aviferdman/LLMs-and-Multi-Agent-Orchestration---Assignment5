"""
Experiment 4: Context Engineering Strategies

This experiment evaluates different strategies for managing growing context
in multi-turn conversations. Tests SELECT, COMPRESS, WRITE, and HYBRID strategies
across 10-step interactions.

Classes:
    ContextStrategy: Abstract base class for context management strategies
    SelectStrategy: RAG-based retrieval of relevant past context
    CompressStrategy: Summarization-based context compression
    WriteStrategy: External memory with key fact extraction
    HybridStrategy: Combination of SELECT and COMPRESS
    Experiment4: Main experiment class

Mathematical Framework:
    Context Growth (Unmanaged): |C_t| = |C_0| + sum(|O_i|) for i in 1..t
    SELECT: |C_t| = k * L_avg + |O_t| (constant space O(k))
    COMPRESS: |C_t| bounded by M with compression ratio r
    WRITE: |C_t| = |S_t| + |O_t| where S_t = extracted facts

Example:
    >>> from experiments.experiment_4 import Experiment4
    >>> exp = Experiment4(config)
    >>> results = exp.run()
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time
import sys
from pathlib import Path

import numpy as np
from loguru import logger

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from llm_interface import LLMInterface, OllamaInterface, Response
from metrics import MetricsCalculator


@dataclass
class ContextMetrics:
    """
    Metrics for context management at each step.

    Attributes:
        step: Step number (1-10)
        context_size: Number of tokens in context
        latency: Response time in seconds
        memory_usage: RAM usage in MB
        accuracy: Correctness of answer (0 or 1)
        was_compressed: Whether compression was applied
        retrieval_time: Time spent on retrieval/selection (if applicable)
    """
    step: int
    context_size: int
    latency: float
    memory_usage: float
    accuracy: float
    was_compressed: bool = False
    retrieval_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepData:
    """
    Data for a single conversation step.

    Attributes:
        action_type: Type of action (retrieval, synthesis, reasoning, comparison)
        context: Current context content
        query: Question to ask
        ground_truth: Expected answer
        observation: New information from this step
    """
    action_type: str
    context: str
    query: str
    ground_truth: str
    observation: str


class ContextStrategy(ABC):
    """
    Abstract base class for context management strategies.

    All strategies must implement manage_context method.
    """

    def __init__(self, llm: LLMInterface, max_tokens: int = 2000):
        """
        Initialize strategy.

        Args:
            llm: LLM interface for queries
            max_tokens: Maximum context size in tokens
        """
        self.llm = llm
        self.max_tokens = max_tokens
        self.history: List[str] = []

    @abstractmethod
    def manage_context(
        self,
        new_observation: str,
        query: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Manage context and prepare for query.

        Args:
            new_observation: New information to add
            query: Query to answer

        Returns:
            Tuple of (managed_context, metrics_dict)
        """
        pass

    def reset(self):
        """Reset strategy state for new experiment run."""
        self.history = []


class SelectStrategy(ContextStrategy):
    """
    SELECT strategy: Keep most relevant context using RAG-based retrieval.

    Maintains constant space O(k) by selecting top-k relevant past observations.

    Attributes:
        top_k: Number of relevant items to retrieve
        embeddings: Simple embeddings for similarity search
    """

    def __init__(self, llm: LLMInterface, max_tokens: int = 2000, top_k: int = 5):
        """
        Initialize SELECT strategy.

        Args:
            llm: LLM interface
            max_tokens: Maximum context size
            top_k: Number of relevant items to retrieve
        """
        super().__init__(llm, max_tokens)
        self.top_k = top_k
        self.embeddings: List[Tuple[str, np.ndarray]] = []

    def _embed_text(self, text: str) -> np.ndarray:
        """
        Simple embedding using word occurrence (placeholder).

        Args:
            text: Text to embed

        Returns:
            Simple embedding vector
        """
        # Simple word-based embedding (placeholder for actual embeddings)
        words = text.lower().split()
        # Create a simple bag-of-words representation
        vocab = set()
        for hist, _ in self.embeddings:
            vocab.update(hist.lower().split())
        vocab.update(words)

        vocab_list = sorted(list(vocab))
        embedding = np.zeros(max(len(vocab_list), 100))

        for i, word in enumerate(vocab_list):
            if word in words:
                embedding[i] = 1.0

        return embedding

    def _similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(vec1, vec2) / (norm1 * norm2)

    def manage_context(
        self,
        new_observation: str,
        query: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Select top-k relevant observations for context.

        Args:
            new_observation: New observation to add
            query: Query to answer

        Returns:
            Tuple of (selected_context, metrics)
        """
        start_time = time.time()

        # Add new observation to history
        if new_observation:
            obs_embedding = self._embed_text(new_observation)
            self.embeddings.append((new_observation, obs_embedding))
            self.history.append(new_observation)

        # Retrieve most relevant observations
        query_embedding = self._embed_text(query)

        # Calculate similarities
        similarities = [
            (obs, self._similarity(query_embedding, emb))
            for obs, emb in self.embeddings
        ]

        # Sort by similarity and take top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        relevant_obs = [obs for obs, _ in similarities[:self.top_k]]

        # Construct context
        selected_context = "\n\n".join(relevant_obs)

        retrieval_time = time.time() - start_time
        context_size = self.llm.count_tokens(selected_context)

        return selected_context, {
            'context_size': context_size,
            'retrieval_time': retrieval_time,
            'num_retrieved': len(relevant_obs),
            'was_compressed': False
        }

    def reset(self):
        """Reset strategy state."""
        super().reset()
        self.embeddings = []


class CompressStrategy(ContextStrategy):
    """
    COMPRESS strategy: Summarize context when it exceeds limit.

    Maintains bounded space O(M) through periodic summarization.

    Attributes:
        compression_ratio: Target compression ratio (default 0.5)
    """

    def __init__(
        self,
        llm: LLMInterface,
        max_tokens: int = 2000,
        compression_ratio: float = 0.5
    ):
        """
        Initialize COMPRESS strategy.

        Args:
            llm: LLM interface
            max_tokens: Maximum context size before compression
            compression_ratio: Target ratio after compression
        """
        super().__init__(llm, max_tokens)
        self.compression_ratio = compression_ratio
        self.full_history = ""

    def _summarize(self, text: str, target_tokens: int) -> str:
        """
        Summarize text to target token count.

        Args:
            text: Text to summarize
            target_tokens: Target token count

        Returns:
            Summarized text
        """
        # Simple summarization: extract key sentences
        # In production, would use LLM summarization
        sentences = text.split('.')

        # Keep sentences until target reached
        summary_parts = []
        current_tokens = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            tokens = self.llm.count_tokens(sentence)
            if current_tokens + tokens <= target_tokens:
                summary_parts.append(sentence)
                current_tokens += tokens
            else:
                break

        summary = '. '.join(summary_parts)
        if summary and not summary.endswith('.'):
            summary += '.'

        return summary

    def manage_context(
        self,
        new_observation: str,
        query: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Add observation and compress if needed.

        Args:
            new_observation: New observation to add
            query: Query to answer

        Returns:
            Tuple of (managed_context, metrics)
        """
        # Add new observation
        if new_observation:
            if self.full_history:
                self.full_history += "\n\n" + new_observation
            else:
                self.full_history = new_observation
            self.history.append(new_observation)

        current_size = self.llm.count_tokens(self.full_history)
        was_compressed = False

        # Compress if exceeds limit
        if current_size > self.max_tokens:
            target_tokens = int(self.max_tokens * self.compression_ratio)
            self.full_history = self._summarize(self.full_history, target_tokens)
            was_compressed = True
            current_size = self.llm.count_tokens(self.full_history)

        return self.full_history, {
            'context_size': current_size,
            'was_compressed': was_compressed,
            'retrieval_time': 0.0
        }

    def reset(self):
        """Reset strategy state."""
        super().reset()
        self.full_history = ""


class WriteStrategy(ContextStrategy):
    """
    WRITE strategy: Extract and store key facts in external memory.

    Maintains space O(F) where F = number of extracted facts.

    Attributes:
        scratchpad: Dictionary storing extracted facts
    """

    def __init__(self, llm: LLMInterface, max_tokens: int = 2000):
        """
        Initialize WRITE strategy.

        Args:
            llm: LLM interface
            max_tokens: Maximum context size
        """
        super().__init__(llm, max_tokens)
        self.scratchpad: Dict[str, str] = {}
        self.fact_counter = 0

    def _extract_key_facts(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract key facts from text.

        Args:
            text: Text to extract facts from

        Returns:
            List of (key, fact) tuples
        """
        # Simple fact extraction: split into sentences and label
        # In production, would use NLP/LLM extraction
        facts = []
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        for sentence in sentences:
            if len(sentence) > 10:  # Filter very short sentences
                key = f"fact_{self.fact_counter}"
                facts.append((key, sentence))
                self.fact_counter += 1

        return facts

    def _retrieve_relevant_facts(self, query: str, k: int = 10) -> List[str]:
        """
        Retrieve relevant facts from scratchpad.

        Args:
            query: Query to match against
            k: Number of facts to retrieve

        Returns:
            List of relevant facts
        """
        query_words = set(query.lower().split())

        # Score facts by word overlap
        scored_facts = []
        for key, fact in self.scratchpad.items():
            fact_words = set(fact.lower().split())
            overlap = len(query_words & fact_words)
            if overlap > 0:
                scored_facts.append((fact, overlap))

        # Sort by relevance
        scored_facts.sort(key=lambda x: x[1], reverse=True)

        return [fact for fact, _ in scored_facts[:k]]

    def manage_context(
        self,
        new_observation: str,
        query: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Extract facts and retrieve relevant ones for query.

        Args:
            new_observation: New observation to process
            query: Query to answer

        Returns:
            Tuple of (relevant_facts_context, metrics)
        """
        start_time = time.time()

        # Extract and store facts from new observation
        if new_observation:
            facts = self._extract_key_facts(new_observation)
            for key, fact in facts:
                self.scratchpad[key] = fact
            self.history.append(new_observation)

        # Retrieve relevant facts
        relevant_facts = self._retrieve_relevant_facts(query, k=10)
        context = "\n".join(relevant_facts)

        retrieval_time = time.time() - start_time
        context_size = self.llm.count_tokens(context)

        return context, {
            'context_size': context_size,
            'retrieval_time': retrieval_time,
            'scratchpad_size': len(self.scratchpad),
            'was_compressed': False
        }

    def reset(self):
        """Reset strategy state."""
        super().reset()
        self.scratchpad = {}
        self.fact_counter = 0


class HybridStrategy(ContextStrategy):
    """
    HYBRID strategy: Combine SELECT and COMPRESS.

    First selects relevant context, then compresses if needed.
    """

    def __init__(
        self,
        llm: LLMInterface,
        max_tokens: int = 2000,
        top_k: int = 5,
        compression_ratio: float = 0.5
    ):
        """
        Initialize HYBRID strategy.

        Args:
            llm: LLM interface
            max_tokens: Maximum context size
            top_k: Number of items to select
            compression_ratio: Compression ratio if needed
        """
        super().__init__(llm, max_tokens)
        self.select_strategy = SelectStrategy(llm, max_tokens, top_k)
        self.compress_strategy = CompressStrategy(llm, max_tokens, compression_ratio)

    def manage_context(
        self,
        new_observation: str,
        query: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Apply SELECT then COMPRESS if needed.

        Args:
            new_observation: New observation
            query: Query to answer

        Returns:
            Tuple of (managed_context, metrics)
        """
        # Phase 1: SELECT relevant context
        selected_context, select_metrics = self.select_strategy.manage_context(
            new_observation, query
        )

        # Phase 2: COMPRESS if still too large
        context_size = self.llm.count_tokens(selected_context)
        was_compressed = False

        if context_size > self.max_tokens:
            target_tokens = int(self.max_tokens * self.compress_strategy.compression_ratio)
            selected_context = self.compress_strategy._summarize(
                selected_context, target_tokens
            )
            was_compressed = True
            context_size = self.llm.count_tokens(selected_context)

        return selected_context, {
            'context_size': context_size,
            'retrieval_time': select_metrics['retrieval_time'],
            'was_compressed': was_compressed,
            'num_retrieved': select_metrics.get('num_retrieved', 0)
        }

    def reset(self):
        """Reset both underlying strategies."""
        self.select_strategy.reset()
        self.compress_strategy.reset()


class Experiment4:
    """
    Experiment 4: Context Engineering Strategies.

    Evaluates SELECT, COMPRESS, WRITE, and HYBRID strategies across
    multi-turn conversations with 10 steps.

    Attributes:
        config: Experiment configuration
        llm: LLM interface
        metrics_calc: Metrics calculator
        strategies: Dictionary of strategy instances
        num_steps: Number of conversation steps
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Experiment 4.

        Args:
            config: Configuration dictionary with experiment settings
        """
        self.config = config

        # Initialize LLM
        model = config.get('model', 'llama2:13b')
        self.llm = OllamaInterface(model=model)

        # Initialize metrics calculator
        self.metrics_calc = MetricsCalculator()

        # Get experiment parameters
        self.num_steps = config.get('num_steps', 10)
        self.max_tokens = config.get('max_tokens', 2000)
        self.num_runs = config.get('num_runs', 3)

        # Initialize strategies
        self.strategies = self._initialize_strategies()

        logger.info(f"Initialized Experiment4 with {len(self.strategies)} strategies")

    def _initialize_strategies(self) -> Dict[str, ContextStrategy]:
        """
        Initialize all context management strategies.

        Returns:
            Dictionary mapping strategy names to instances
        """
        strategies = {
            'select': SelectStrategy(self.llm, self.max_tokens, top_k=5),
            'compress': CompressStrategy(
                self.llm, self.max_tokens, compression_ratio=0.5
            ),
            'write': WriteStrategy(self.llm, self.max_tokens),
            'hybrid': HybridStrategy(
                self.llm, self.max_tokens, top_k=5, compression_ratio=0.5
            )
        }

        return strategies

    def _generate_scenario(self, scenario_type: str = "sequential") -> List[StepData]:
        """
        Generate multi-step conversation scenario.

        Args:
            scenario_type: Type of scenario (sequential, reasoning, adversarial)

        Returns:
            List of StepData for each step
        """
        if scenario_type == "sequential":
            return self._generate_sequential_scenario()
        elif scenario_type == "reasoning":
            return self._generate_reasoning_scenario()
        elif scenario_type == "adversarial":
            return self._generate_adversarial_scenario()
        else:
            return self._generate_sequential_scenario()

    def _generate_sequential_scenario(self) -> List[StepData]:
        """
        Generate sequential data collection scenario (e.g., weather monitoring).

        Returns:
            List of StepData for 10 steps
        """
        steps = []

        # Weather monitoring over 10 days
        temperatures = [22, 24, 26, 25, 23, 21, 20, 19, 21, 23]
        conditions = ["sunny", "sunny", "cloudy", "rainy", "cloudy",
                     "rainy", "rainy", "cloudy", "sunny", "sunny"]

        for i in range(self.num_steps):
            day = i + 1
            temp = temperatures[i]
            condition = conditions[i]

            observation = f"Day {day}: Temperature {temp}C, weather {condition}."

            # Different action types
            if i < 3:
                # Retrieval
                query = f"What was the temperature on Day {day}?"
                ground_truth = f"{temp}�C"
                action_type = "retrieval"
            elif i < 6:
                # Synthesis
                query = f"What has been the average temperature so far?"
                avg_temp = sum(temperatures[:day]) / day
                ground_truth = f"{avg_temp:.1f}�C"
                action_type = "synthesis"
            elif i < 8:
                # Comparison
                if day > 3:
                    query = f"How does today's temperature compare to Day {day-3}?"
                    diff = temp - temperatures[day-4]
                    ground_truth = f"{abs(diff)}�C {'higher' if diff > 0 else 'lower'}"
                else:
                    query = f"What was the temperature on Day {day}?"
                    ground_truth = f"{temp}�C"
                action_type = "comparison"
            else:
                # Reasoning
                query = "What is the temperature trend over the past 3 days?"
                recent_temps = temperatures[day-3:day]
                if recent_temps == sorted(recent_temps):
                    ground_truth = "increasing"
                elif recent_temps == sorted(recent_temps, reverse=True):
                    ground_truth = "decreasing"
                else:
                    ground_truth = "mixed"
                action_type = "reasoning"

            steps.append(StepData(
                action_type=action_type,
                context="",  # Will be managed by strategy
                query=query,
                ground_truth=ground_truth,
                observation=observation
            ))

        return steps

    def _generate_reasoning_scenario(self) -> List[StepData]:
        """
        Generate multi-step reasoning scenario (e.g., trip planning).

        Returns:
            List of StepData for 10 steps
        """
        steps = []

        # Trip planning with evolving constraints
        observations = [
            "Destination decided: Paris, France.",
            "Budget set: $2000 total for the trip.",
            "Dates: June 15-22 (7 days).",
            "Hotel booked: $150 per night, downtown location.",
            "Flight cost: $600 round trip.",
            "Daily food budget: $50 per day.",
            "Museum pass: $120 for all museums.",
            "Transportation: Metro pass $30 for week.",
            "Shopping budget: $200 for souvenirs.",
            "Emergency fund: $100 set aside."
        ]

        queries = [
            "What is the destination?",
            "What is the total budget?",
            "How many days is the trip?",
            "What is the total hotel cost?",
            "What is the remaining budget after flight and hotel?",
            "What is the total food cost?",
            "Are we within budget after hotel, flight, food, and museum pass?",
            "What is the total transportation cost?",
            "What is the remaining budget for shopping?",
            "What is the final remaining budget?"
        ]

        ground_truths = [
            "Paris, France",
            "$2000",
            "7 days",
            "$1050",
            "$350",
            "$350",
            "No, exceeded by $120",
            "$30",
            "$200 allocated",
            "$0 (budget fully allocated)"
        ]

        for i in range(min(self.num_steps, len(observations))):
            steps.append(StepData(
                action_type="reasoning",
                context="",
                query=queries[i],
                ground_truth=ground_truths[i],
                observation=observations[i]
            ))

        return steps

    def _generate_adversarial_scenario(self) -> List[StepData]:
        """
        Generate adversarial scenario with contradictions.

        Returns:
            List of StepData for 10 steps
        """
        steps = []

        observations = [
            "The CEO announced Q1 revenue of $10M.",
            "The CFO confirmed strong financial performance.",
            "Q1 profit margin was 20%.",
            "New product launch exceeded expectations.",
            "Internal memo: Q1 revenue actually $8M (correction).",  # Contradiction
            "Customer satisfaction increased 15%.",
            "Market share grew by 3%.",
            "Press release states Q1 revenue $10M.",  # Contradiction
            "Audit report: Revenue $8M is correct.",  # Resolution
            "Board approved expansion plans."
        ]

        queries = [
            "What was Q1 revenue?",
            "Was financial performance good?",
            "What was the profit margin?",
            "How did the product launch go?",
            "What is the correct Q1 revenue?",
            "Did customer satisfaction improve?",
            "What happened to market share?",
            "Is there conflicting information about revenue?",
            "What is the final confirmed Q1 revenue?",
            "What did the board decide?"
        ]

        ground_truths = [
            "$10M",
            "Yes, strong",
            "20%",
            "Exceeded expectations",
            "$8M (corrected)",
            "Yes, 15%",
            "Grew 3%",
            "Yes, conflicting",
            "$8M (confirmed)",
            "Approved expansion"
        ]

        for i in range(min(self.num_steps, len(observations))):
            steps.append(StepData(
                action_type="reasoning" if i >= 4 else "retrieval",
                context="",
                query=queries[i],
                ground_truth=ground_truths[i],
                observation=observations[i]
            ))

        return steps

    def _evaluate_answer(self, answer: str, ground_truth: str) -> float:
        """
        Evaluate answer correctness.

        Args:
            answer: LLM-generated answer
            ground_truth: Correct answer

        Returns:
            Accuracy score (0 or 1, or semantic similarity)
        """
        # Simple exact match
        if answer.strip().lower() == ground_truth.strip().lower():
            return 1.0

        # Semantic similarity
        similarity = self.metrics_calc.semantic_similarity(answer, ground_truth)

        # Threshold for acceptance
        return 1.0 if similarity > 0.7 else 0.0

    def _run_strategy(
        self,
        strategy_name: str,
        strategy: ContextStrategy,
        scenario: List[StepData]
    ) -> List[ContextMetrics]:
        """
        Run a single strategy on a scenario.

        Args:
            strategy_name: Name of strategy
            strategy: Strategy instance
            scenario: List of conversation steps

        Returns:
            List of metrics for each step
        """
        logger.info(f"Running {strategy_name} strategy")

        strategy.reset()
        metrics_list = []

        for step_num, step_data in enumerate(scenario, 1):
            # Manage context and get managed version
            managed_context, context_metrics = strategy.manage_context(
                step_data.observation,
                step_data.query
            )

            # Query LLM
            start_time = time.time()
            response = self.llm.query(managed_context, step_data.query)
            latency = time.time() - start_time

            # Evaluate answer
            accuracy = self._evaluate_answer(response.text, step_data.ground_truth)

            # Record metrics
            metrics = ContextMetrics(
                step=step_num,
                context_size=context_metrics['context_size'],
                latency=latency,
                memory_usage=0.0,  # Placeholder for actual memory measurement
                accuracy=accuracy,
                was_compressed=context_metrics.get('was_compressed', False),
                retrieval_time=context_metrics.get('retrieval_time', 0.0),
                metadata={
                    'action_type': step_data.action_type,
                    'strategy': strategy_name,
                    **context_metrics
                }
            )

            metrics_list.append(metrics)

            logger.debug(
                f"{strategy_name} - Step {step_num}: "
                f"Accuracy={accuracy:.2f}, "
                f"Context={context_metrics['context_size']} tokens"
            )

        return metrics_list

    def run(self) -> Dict[str, Any]:
        """
        Run Experiment 4: Test all strategies across scenarios.

        Returns:
            Dictionary containing:
                - results_by_strategy: Metrics for each strategy
                - summary_statistics: Aggregated statistics
                - visualizations: Plot data
        """
        logger.info("Starting Experiment 4: Context Engineering Strategies")

        results = {
            'results_by_strategy': {},
            'summary_statistics': {},
            'raw_metrics': []
        }

        # Generate scenario
        scenario = self._generate_scenario("sequential")

        # Run each strategy
        for strategy_name, strategy in self.strategies.items():
            logger.info(f"Testing {strategy_name} strategy")

            strategy_metrics = []

            # Multiple runs for statistical validity
            for run in range(self.num_runs):
                logger.info(f"  Run {run + 1}/{self.num_runs}")

                run_metrics = self._run_strategy(
                    strategy_name,
                    strategy,
                    scenario
                )

                strategy_metrics.extend(run_metrics)
                results['raw_metrics'].extend(run_metrics)

            # Store results
            results['results_by_strategy'][strategy_name] = strategy_metrics

            # Calculate summary statistics
            accuracies = [m.accuracy for m in strategy_metrics]
            context_sizes = [m.context_size for m in strategy_metrics]
            latencies = [m.latency for m in strategy_metrics]

            results['summary_statistics'][strategy_name] = {
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'mean_context_size': np.mean(context_sizes),
                'std_context_size': np.std(context_sizes),
                'mean_latency': np.mean(latencies),
                'std_latency': np.std(latencies),
                'total_compressions': sum(1 for m in strategy_metrics if m.was_compressed)
            }

            logger.info(
                f"{strategy_name}: "
                f"Accuracy={np.mean(accuracies):.3f}+/-{np.std(accuracies):.3f}, "
                f"Context={np.mean(context_sizes):.1f}+/-{np.std(context_sizes):.1f} tokens"
            )

        logger.success("Experiment 4 completed successfully")

        return results

    def analyze(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform statistical analysis on results.

        Args:
            results: Results from run()

        Returns:
            Dictionary with statistical analyses
        """
        logger.info("Analyzing Experiment 4 results")

        analysis = {
            'strategy_rankings': {},
            'best_strategy': None,
            'recommendations': {}
        }

        # Rank strategies by accuracy
        strategy_scores = []
        for strategy_name, stats in results['summary_statistics'].items():
            score = stats['mean_accuracy']
            strategy_scores.append((strategy_name, score))

        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        analysis['strategy_rankings'] = {
            name: rank + 1
            for rank, (name, _) in enumerate(strategy_scores)
        }

        analysis['best_strategy'] = strategy_scores[0][0]

        # Generate recommendations
        for strategy_name in results['summary_statistics'].keys():
            stats = results['summary_statistics'][strategy_name]

            recommendation = []

            if stats['mean_accuracy'] > 0.8:
                recommendation.append("High accuracy")
            if stats['mean_context_size'] < 1000:
                recommendation.append("Memory efficient")
            if stats['mean_latency'] < 2.0:
                recommendation.append("Fast response")

            if not recommendation:
                recommendation.append("Consider alternatives")

            analysis['recommendations'][strategy_name] = ", ".join(recommendation)

        logger.success("Analysis completed")

        return analysis
