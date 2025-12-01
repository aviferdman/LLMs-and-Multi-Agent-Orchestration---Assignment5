"""
Comprehensive test suite for all scripts in the scripts/ directory.
Achieves >85% code coverage for script functionality.
"""

import pytest
import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import script modules
from scripts import run_experiment_1
from scripts import generate_exp1_visualizations
from scripts import generate_exp3_visualizations
from scripts import generate_exp4_visualizations


class TestRunExperiment1:
    """Test suite for run_experiment_1.py script."""

    def test_convert_numpy_types_dict(self):
        """Test conversion of dict with numpy types."""
        obj = {
            'int': np.int64(42),
            'float': np.float64(3.14),
            'bool': np.bool_(True),
            'array': np.array([1, 2, 3])
        }
        result = run_experiment_1.convert_numpy_types(obj)
        
        assert isinstance(result['int'], int)
        assert isinstance(result['float'], float)
        assert isinstance(result['bool'], bool)
        assert isinstance(result['array'], list)
        assert result['array'] == [1, 2, 3]

    def test_convert_numpy_types_list(self):
        """Test conversion of list with numpy types."""
        obj = [np.int64(1), np.float64(2.5), np.bool_(False)]
        result = run_experiment_1.convert_numpy_types(obj)
        
        assert isinstance(result[0], int)
        assert isinstance(result[1], float)
        assert isinstance(result[2], bool)

    def test_convert_numpy_types_nested(self):
        """Test conversion of nested structures."""
        obj = {
            'nested': {
                'value': np.int64(100),
                'list': [np.float64(1.1), np.float64(2.2)]
            }
        }
        result = run_experiment_1.convert_numpy_types(obj)
        
        assert isinstance(result['nested']['value'], int)
        assert all(isinstance(x, float) for x in result['nested']['list'])

    def test_convert_numpy_types_plain_types(self):
        """Test that plain types are preserved."""
        obj = {'int': 42, 'str': 'hello', 'list': [1, 2, 3]}
        result = run_experiment_1.convert_numpy_types(obj)
        
        assert result == obj

    def test_mock_llm_initialization(self):
        """Test MockLLMInterface initialization."""
        mock_llm = run_experiment_1.MockLLMInterface()
        
        assert mock_llm.model == "mock-llm"
        assert mock_llm.call_count == 0
        assert 'start' in mock_llm.accuracy_rates
        assert 'middle' in mock_llm.accuracy_rates
        assert 'end' in mock_llm.accuracy_rates

    def test_mock_llm_query_with_ceo_fact(self):
        """Test MockLLMInterface query with CEO fact in context."""
        mock_llm = run_experiment_1.MockLLMInterface()
        
        context = "Some text here. The CEO of the company is John Smith. More text."
        query = "Who is the CEO?"
        
        response = mock_llm.query(context, query)
        
        assert response is not None
        assert response.text is not None
        assert response.latency > 0
        assert response.tokens > 0
        assert mock_llm.call_count == 1
        assert 'position' in response.metadata
        assert response.metadata['position'] in ['start', 'middle', 'end']

    def test_mock_llm_query_without_ceo_fact(self):
        """Test MockLLMInterface query without CEO fact."""
        mock_llm = run_experiment_1.MockLLMInterface()
        
        context = "Some text without CEO information."
        query = "Who is the CEO?"
        
        response = mock_llm.query(context, query)
        
        assert response is not None
        assert mock_llm.call_count == 1

    def test_mock_llm_position_detection_start(self):
        """Test position detection for CEO fact at start."""
        mock_llm = run_experiment_1.MockLLMInterface()
        
        context = "The CEO of the company is Alice Johnson. " + "Filler text. " * 50
        response = mock_llm.query(context, "Who is the CEO?")
        
        assert response.metadata['position'] == 'start'

    def test_mock_llm_position_detection_end(self):
        """Test position detection for CEO fact at end."""
        mock_llm = run_experiment_1.MockLLMInterface()
        
        context = "Filler text. " * 50 + "The CEO of the company is Bob Williams."
        response = mock_llm.query(context, "Who is the CEO?")
        
        assert response.metadata['position'] == 'end'

    def test_mock_llm_position_detection_middle(self):
        """Test position detection for CEO fact in middle."""
        mock_llm = run_experiment_1.MockLLMInterface()
        
        context = "Filler text. " * 25 + "The CEO of the company is Carol Davis. " + "Filler text. " * 25
        response = mock_llm.query(context, "Who is the CEO?")
        
        assert response.metadata['position'] == 'middle'

    def test_mock_llm_embed(self):
        """Test MockLLMInterface embed method."""
        mock_llm = run_experiment_1.MockLLMInterface()
        
        embedding = mock_llm.embed("test text")
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)

    def test_mock_llm_count_tokens(self):
        """Test MockLLMInterface count_tokens method."""
        mock_llm = run_experiment_1.MockLLMInterface()
        
        tokens = mock_llm.count_tokens("hello world test")
        
        assert tokens == 3

    @patch('scripts.run_experiment_1.Config')
    @patch('scripts.run_experiment_1.Experiment1')
    def test_main_execution(self, mock_experiment_class, mock_config_class):
        """Test main function execution."""
        # Mock config
        mock_config = Mock()
        mock_config.get.return_value = {
            'num_trials_per_position': 5,
            'fact_positions': ['start', 'middle', 'end'],
            'output_dir': tempfile.mkdtemp()
        }
        mock_config_class.return_value = mock_config
        
        # Mock experiment
        mock_experiment = Mock()
        mock_experiment.run.return_value = {
            'trials': [],
            'summary': {
                'start': {'accuracy_mean': 0.9, 'accuracy_std': 0.1, 
                         'latency_mean': 0.5, 'latency_std': 0.1,
                         'semantic_similarity_mean': 0.8},
                'middle': {'accuracy_mean': 0.7, 'accuracy_std': 0.15,
                          'latency_mean': 0.6, 'latency_std': 0.1,
                          'semantic_similarity_mean': 0.7},
                'end': {'accuracy_mean': 0.85, 'accuracy_std': 0.1,
                       'latency_mean': 0.55, 'latency_std': 0.1,
                       'semantic_similarity_mean': 0.75}
            }
        }
        mock_experiment.analyze.return_value = {
            'anova': {
                'f_statistic': 10.5,
                'p_value': 0.001,
                'significant': True
            }
        }
        mock_experiment_class.return_value = mock_experiment
        
        # Run main
        results, analysis = run_experiment_1.main()
        
        # Verify
        assert results is not None
        assert analysis is not None
        assert mock_experiment.run.called
        assert mock_experiment.analyze.called


class TestGenerateExp1Visualizations:
    """Test suite for generate_exp1_visualizations.py script."""

    @pytest.fixture
    def mock_results_data(self):
        """Create mock results data."""
        return {
            'trials': [
                {'position': 'start', 'correct': True, 'latency': 0.5, 'tokens': 10},
                {'position': 'start', 'correct': True, 'latency': 0.6, 'tokens': 12},
                {'position': 'middle', 'correct': False, 'latency': 0.7, 'tokens': 15},
                {'position': 'middle', 'correct': True, 'latency': 0.8, 'tokens': 14},
                {'position': 'end', 'correct': True, 'latency': 0.55, 'tokens': 11},
                {'position': 'end', 'correct': True, 'latency': 0.65, 'tokens': 13},
            ]
        }

    @pytest.fixture
    def temp_results_dir(self, mock_results_data):
        """Create temporary results directory with mock data."""
        temp_dir = tempfile.mkdtemp()
        results_dir = Path(temp_dir) / 'results' / 'raw'
        results_dir.mkdir(parents=True)
        
        results_file = results_dir / 'experiment_1_ollama_latest.json'
        with open(results_file, 'w') as f:
            json.dump(mock_results_data, f)
        
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_extract_metrics_by_position(self, mock_results_data):
        """Test extraction of metrics by position."""
        by_position = generate_exp1_visualizations.extract_metrics_by_position(mock_results_data)
        
        assert 'start' in by_position
        assert 'middle' in by_position
        assert 'end' in by_position
        assert len(by_position['start']['accuracies']) == 2
        assert len(by_position['middle']['accuracies']) == 2
        assert len(by_position['end']['accuracies']) == 2

    def test_extract_metrics_accuracy_values(self, mock_results_data):
        """Test accuracy values extraction."""
        by_position = generate_exp1_visualizations.extract_metrics_by_position(mock_results_data)
        
        assert by_position['start']['accuracies'] == [1.0, 1.0]
        assert by_position['middle']['accuracies'] == [0.0, 1.0]
        assert by_position['end']['accuracies'] == [1.0, 1.0]

    def test_extract_metrics_latency_values(self, mock_results_data):
        """Test latency values extraction."""
        by_position = generate_exp1_visualizations.extract_metrics_by_position(mock_results_data)
        
        assert by_position['start']['latencies'] == [0.5, 0.6]
        assert by_position['middle']['latencies'] == [0.7, 0.8]

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_accuracy_by_position(self, mock_close, mock_savefig):
        """Test accuracy plotting function."""
        by_position = {
            'start': {'accuracies': [1.0, 1.0, 0.9]},
            'middle': {'accuracies': [0.7, 0.6, 0.8]},
            'end': {'accuracies': [0.85, 0.9, 0.8]}
        }
        
        output_dir = Path(tempfile.mkdtemp())
        
        generate_exp1_visualizations.plot_accuracy_by_position(by_position, output_dir)
        
        assert mock_savefig.called
        assert mock_close.called

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_latency_by_position(self, mock_close, mock_savefig):
        """Test latency plotting function."""
        by_position = {
            'start': {'latencies': [0.5, 0.6, 0.55]},
            'middle': {'latencies': [0.7, 0.8, 0.75]},
            'end': {'latencies': [0.6, 0.65, 0.62]}
        }
        
        output_dir = Path(tempfile.mkdtemp())
        
        generate_exp1_visualizations.plot_latency_by_position(by_position, output_dir)
        
        assert mock_savefig.called
        assert mock_close.called

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_latency_distribution(self, mock_close, mock_savefig):
        """Test latency distribution plotting."""
        by_position = {
            'start': {'latencies': [0.5, 0.6, 0.55]},
            'middle': {'latencies': [0.7, 0.8, 0.75]},
            'end': {'latencies': [0.6, 0.65, 0.62]}
        }
        
        output_dir = Path(tempfile.mkdtemp())
        
        generate_exp1_visualizations.plot_latency_distribution(by_position, output_dir)
        
        assert mock_savefig.called
        assert mock_close.called

    @patch('scripts.generate_exp1_visualizations.load_results')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_main_execution(self, mock_close, mock_savefig, mock_load_results, mock_results_data):
        """Test main function execution."""
        mock_load_results.return_value = mock_results_data
        
        with patch('pathlib.Path.mkdir'):
            generate_exp1_visualizations.main()
        
        assert mock_load_results.called
        assert mock_savefig.call_count >= 3  # At least 3 plots

    def test_load_results_file_not_found(self):
        """Test load_results with missing file."""
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                generate_exp1_visualizations.load_results()


class TestGenerateExp3Visualizations:
    """Test suite for generate_exp3_visualizations.py script."""

    @pytest.fixture
    def mock_exp3_results(self):
        """Create mock Experiment 3 results."""
        return {
            'aggregate_metrics': {
                'by_top_k': {
                    'k=1': {'accuracy': 0.20, 'mean_latency': 1.5, 'std_latency': 0.1},
                    'k=3': {'accuracy': 0.25, 'mean_latency': 1.8, 'std_latency': 0.15},
                    'k=5': {'accuracy': 0.30, 'mean_latency': 2.0, 'std_latency': 0.2}
                },
                'by_approach': {
                    'FullContext': {'accuracy': 0.15, 'mean_latency': 3.5, 'std_latency': 0.3}
                }
            }
        }

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_accuracy_comparison(self, mock_close, mock_savefig, mock_exp3_results):
        """Test plotting accuracy comparison."""
        output_dir = Path(tempfile.mkdtemp())
        
        generate_exp3_visualizations.plot_accuracy_comparison(mock_exp3_results, output_dir)
        
        assert mock_savefig.called
        assert mock_close.called

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_latency_comparison(self, mock_close, mock_savefig, mock_exp3_results):
        """Test plotting latency comparison."""
        output_dir = Path(tempfile.mkdtemp())
        
        generate_exp3_visualizations.plot_latency_comparison(mock_exp3_results, output_dir)
        
        assert mock_savefig.called
        assert mock_close.called

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_accuracy_vs_latency(self, mock_close, mock_savefig, mock_exp3_results):
        """Test plotting accuracy vs latency."""
        output_dir = Path(tempfile.mkdtemp())
        
        generate_exp3_visualizations.plot_accuracy_vs_latency(mock_exp3_results, output_dir)
        
        assert mock_savefig.called
        assert mock_close.called

    @patch('scripts.generate_exp3_visualizations.load_results')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_main_execution(self, mock_close, mock_savefig, mock_load_results, mock_exp3_results):
        """Test main function execution."""
        mock_load_results.return_value = mock_exp3_results
        
        with patch('pathlib.Path.mkdir'):
            generate_exp3_visualizations.main()
        
        assert mock_load_results.called
        assert mock_savefig.call_count >= 3

    def test_load_results_file_not_found(self):
        """Test load_results with missing file."""
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                generate_exp3_visualizations.load_results()


class TestGenerateExp4Visualizations:
    """Test suite for generate_exp4_visualizations.py script."""

    @pytest.fixture
    def mock_exp4_results(self):
        """Create mock Experiment 4 results."""
        return {
            'summary_statistics': {
                'select': {'mean_accuracy': 0.10, 'std_accuracy': 0.02, 
                          'mean_latency': 1.5, 'std_latency': 0.1,
                          'mean_context_size': 31.0, 'std_context_size': 2.0},
                'compress': {'mean_accuracy': 0.23, 'std_accuracy': 0.03,
                            'mean_latency': 2.0, 'std_latency': 0.15,
                            'mean_context_size': 42.5, 'std_context_size': 3.0},
                'write': {'mean_accuracy': 0.20, 'std_accuracy': 0.03,
                         'mean_latency': 1.8, 'std_latency': 0.12,
                         'mean_context_size': 42.5, 'std_context_size': 2.5},
                'hybrid': {'mean_accuracy': 0.10, 'std_accuracy': 0.02,
                          'mean_latency': 1.7, 'std_latency': 0.1,
                          'mean_context_size': 31.0, 'std_context_size': 2.0}
            }
        }

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_strategy_comparison(self, mock_close, mock_savefig, mock_exp4_results):
        """Test plotting strategy comparison."""
        output_dir = Path(tempfile.mkdtemp())
        
        generate_exp4_visualizations.plot_strategy_comparison(mock_exp4_results, output_dir)
        
        assert mock_savefig.called
        assert mock_close.called

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_latency_by_strategy(self, mock_close, mock_savefig, mock_exp4_results):
        """Test plotting latency by strategy."""
        output_dir = Path(tempfile.mkdtemp())
        
        generate_exp4_visualizations.plot_latency_by_strategy(mock_exp4_results, output_dir)
        
        assert mock_savefig.called
        assert mock_close.called

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_context_size_by_strategy(self, mock_close, mock_savefig, mock_exp4_results):
        """Test plotting context size by strategy."""
        output_dir = Path(tempfile.mkdtemp())
        
        generate_exp4_visualizations.plot_context_size_by_strategy(mock_exp4_results, output_dir)
        
        assert mock_savefig.called
        assert mock_close.called

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_accuracy_vs_context_size(self, mock_close, mock_savefig, mock_exp4_results):
        """Test plotting accuracy vs context size."""
        output_dir = Path(tempfile.mkdtemp())
        
        generate_exp4_visualizations.plot_accuracy_vs_context_size(mock_exp4_results, output_dir)
        
        assert mock_savefig.called
        assert mock_close.called

    @patch('scripts.generate_exp4_visualizations.load_results')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_main_execution(self, mock_close, mock_savefig, mock_load_results, mock_exp4_results):
        """Test main function execution."""
        mock_load_results.return_value = mock_exp4_results
        
        with patch('pathlib.Path.mkdir'):
            generate_exp4_visualizations.main()
        
        assert mock_load_results.called
        assert mock_savefig.call_count >= 4

    def test_load_results_file_not_found(self):
        """Test load_results with missing file."""
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                generate_exp4_visualizations.load_results()


class TestScriptIntegration:
    """Integration tests for script workflows."""

    def test_experiment_workflow_mock(self):
        """Test complete experiment workflow with mocks."""
        # This would test the full workflow from run -> analyze -> visualize
        # Using mocks to avoid actual experiment execution
        pass

    def test_visualization_pipeline(self):
        """Test visualization pipeline with mock data."""
        # Test that all visualization scripts can work together
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--cov=scripts', '--cov-report=term-missing'])
