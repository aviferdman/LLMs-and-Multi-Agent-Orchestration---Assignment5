"""
Tests for experiment modules.

These tests cover the core classes and methods from all four experiments,
focusing on testable components rather than full integration tests.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
import tempfile
import shutil

# Import experiment dataclasses and classes
from src.experiments.experiment_1 import DocumentWithFact, TrialResult, Experiment1
from src.experiments.experiment_2 import Experiment2
from src.experiments.experiment_3 import Experiment3, QueryResult
from src.experiments.experiment_4 import Experiment4, ContextMetrics


class TestExperiment1Classes:
    """Tests for Experiment 1 data classes."""

    def test_document_with_fact_creation(self):
        """Test DocumentWithFact dataclass."""
        doc = DocumentWithFact(
            document="Test document. The CEO is Alice. More text.",
            fact="The CEO is Alice.",
            position="middle",
            ground_truth="Alice",
            metadata={"test": "value"}
        )

        assert doc.document is not None
        assert doc.fact == "The CEO is Alice."
        assert doc.position == "middle"
        assert doc.ground_truth == "Alice"
        assert doc.metadata["test"] == "value"

    def test_trial_result_creation(self):
        """Test TrialResult dataclass."""
        result = TrialResult(
            position="start",
            query="Who is the CEO?",
            response="Alice",
            ground_truth="Alice",
            correct=True,
            latency=1.5,
            tokens=50,
            semantic_similarity=0.95
        )

        assert result.position == "start"
        assert result.correct is True
        assert result.latency == 1.5
        assert result.tokens == 50

    def test_experiment1_initialization(self):
        """Test Experiment1 initialization with mock config."""
        config = {
            "experiment_1": {
                "trials_per_position": 10,
                "positions": ["start", "middle", "end"],
                "doc_length_words": 200
            },
            "general": {
                "random_seed": 42,
                "output_dir": "results"
            },
            "models": {
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "model": "llama2:latest"
                }
            }
        }

        with patch('src.experiments.experiment_1.OllamaInterface'):
            exp = Experiment1(config)
            assert exp.config == config


class TestExperiment2Classes:
    """Tests for Experiment 2 data classes."""

    def test_experiment2_import(self):
        """Test Experiment2 can be imported."""
        assert Experiment2 is not None


class TestExperiment3Classes:
    """Tests for Experiment 3 data classes."""

    def test_experiment3_import(self):
        """Test Experiment3 can be imported."""
        assert Experiment3 is not None
        assert QueryResult is not None


class TestExperiment4Classes:
    """Tests for Experiment 4 data classes."""

    def test_experiment4_import(self):
        """Test Experiment4 classes can be imported."""
        assert Experiment4 is not None
        assert ContextMetrics is not None

    def test_experiment4_initialization(self):
        """Test Experiment4 initialization."""
        config = {
            "experiment_4": {
                "strategies": ["SELECT", "COMPRESS", "WRITE", "HYBRID"],
                "trials_per_strategy": 10,
                "max_context_size": 2000
            },
            "general": {
                "random_seed": 42,
                "output_dir": "results"
            },
            "models": {
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "model": "llama2:latest"
                }
            }
        }

        with patch('src.experiments.experiment_4.OllamaInterface'):
            exp = Experiment4(config)
            assert exp.config == config


class TestExperimentHelperMethods:
    """Tests for common experiment helper methods."""

    def test_experiment1_document_generation(self):
        """Test document generation in Experiment 1."""
        config = {
            "experiment_1": {
                "trials_per_position": 2,
                "positions": ["start", "middle"],
                "doc_length_words": 100
            },
            "general": {"random_seed": 42, "output_dir": "results"},
            "models": {"ollama": {"base_url": "http://localhost:11434", "model": "llama2:latest"}}
        }

        with patch('src.experiments.experiment_1.OllamaInterface'):
            exp = Experiment1(config)

            # Test that generators are initialized
            assert exp.doc_generator is not None
            assert exp.fact_embedder is not None
            assert exp.fact_generator is not None

    def test_experiment4_strategies(self):
        """Test strategy initialization in Experiment 4."""
        config = {
            "experiment_4": {
                "strategies": ["SELECT", "COMPRESS"],
                "trials_per_strategy": 2,
                "max_context_size": 1000
            },
            "general": {"random_seed": 42, "output_dir": "results"},
            "models": {"ollama": {"base_url": "http://localhost:11434", "model": "llama2:latest"}}
        }

        with patch('src.experiments.experiment_4.OllamaInterface'):
            exp = Experiment4(config)
            strategies = exp.config["experiment_4"]["strategies"]
            assert "SELECT" in strategies
            assert "COMPRESS" in strategies


class TestExperimentSaveLoad:
    """Tests for experiment save/load functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)

    def test_save_results_exp1(self, temp_dir):
        """Test saving Experiment 1 results."""
        results = [
            TrialResult(
                position="start",
                query="Q1",
                response="A1",
                ground_truth="A1",
                correct=True,
                latency=1.0,
                tokens=10,
                semantic_similarity=0.9
            )
        ]

        output_file = Path(temp_dir) / "exp1_test.json"

        # Convert to dict for JSON serialization
        results_dict = [
            {
                "position": r.position,
                "query": r.query,
                "response": r.response,
                "ground_truth": r.ground_truth,
                "correct": r.correct,
                "latency": r.latency,
                "tokens": r.tokens,
                "semantic_similarity": r.semantic_similarity
            }
            for r in results
        ]

        with open(output_file, 'w') as f:
            json.dump({"results": results_dict}, f)

        assert output_file.exists()

        # Load and verify
        with open(output_file, 'r') as f:
            loaded = json.load(f)

        assert len(loaded["results"]) == 1
        assert loaded["results"][0]["correct"] is True

    def test_save_results_exp2(self, temp_dir):
        """Test saving Experiment 2 results."""
        results = {
            "num_documents": 10,
            "context_size_tokens": 1000,
            "query": "Q",
            "correct": True
        }

        output_file = Path(temp_dir) / "exp2_test.json"

        with open(output_file, 'w') as f:
            json.dump({"results": results}, f)

        assert output_file.exists()


class TestExperimentAnalysis:
    """Tests for experiment analysis methods."""

    def test_analyze_accuracy_by_position(self):
        """Test analyzing accuracy by position."""
        results = [
            TrialResult("start", "Q1", "A1", "A1", True, 1.0, 10, 0.9),
            TrialResult("start", "Q2", "A2", "A2", True, 1.1, 12, 0.85),
            TrialResult("middle", "Q3", "A3", "A3", False, 1.2, 11, 0.7),
            TrialResult("end", "Q4", "A4", "A4", True, 1.0, 10, 0.95)
        ]

        # Group by position
        by_position = {}
        for r in results:
            if r.position not in by_position:
                by_position[r.position] = []
            by_position[r.position].append(r.correct)

        # Calculate accuracy
        accuracy = {pos: sum(vals) / len(vals) for pos, vals in by_position.items()}

        assert accuracy["start"] == 1.0
        assert accuracy["middle"] == 0.0
        assert accuracy["end"] == 1.0

    def test_analyze_latency(self):
        """Test analyzing latency."""
        results = [
            TrialResult("start", "Q", "A", "A", True, 1.0, 10, 0.9),
            TrialResult("start", "Q", "A", "A", True, 2.0, 10, 0.9),
            TrialResult("start", "Q", "A", "A", True, 1.5, 10, 0.9)
        ]

        latencies = [r.latency for r in results]
        avg_latency = sum(latencies) / len(latencies)

        assert avg_latency == 1.5
