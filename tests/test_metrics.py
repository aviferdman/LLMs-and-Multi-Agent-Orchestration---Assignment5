"""
Tests for metrics evaluation module
"""

import pytest
from src.metrics import MetricsCalculator

class TestMetricsCalculator:
    """Test suite for MetricsCalculator class"""
    
    @pytest.fixture
    def calculator(self):
        """Create MetricsCalculator instance for testing"""
        return MetricsCalculator()
    
    def test_accuracy_perfect_match(self, calculator):
        """Test accuracy with perfect match"""
        predictions = ["Paris", "London", "Berlin"]
        ground_truth = ["Paris", "London", "Berlin"]
        result = calculator.accuracy(predictions, ground_truth)
        assert result == 1.0
    
    def test_accuracy_partial_match(self, calculator):
        """Test accuracy with partial match"""
        predictions = ["Paris", "London", "Berlin"]
        ground_truth = ["Paris", "Madrid", "Berlin"]
        result = calculator.accuracy(predictions, ground_truth)
        assert result == 2/3
    
    def test_accuracy_no_match(self, calculator):
        """Test accuracy with no match"""
        predictions = ["Paris", "London", "Berlin"]
        ground_truth = ["Madrid", "Rome", "Vienna"]
        result = calculator.accuracy(predictions, ground_truth)
        assert result == 0.0
    
    def test_accuracy_empty_lists(self, calculator):
        """Test accuracy with empty lists"""
        result = calculator.accuracy([], [])
        # Empty lists return 0.0 (no data to evaluate)
        assert result == 0.0
    
    def test_semantic_similarity_identical(self, calculator):
        """Test semantic similarity with identical strings"""
        result = calculator.semantic_similarity("test", "test")
        assert result == 1.0
    
    def test_semantic_similarity_different(self, calculator):
        """Test semantic similarity with different strings"""
        result = calculator.semantic_similarity("cat", "dog")
        assert 0.0 <= result <= 1.0
    
    def test_semantic_similarity_similar(self, calculator):
        """Test semantic similarity with similar strings"""
        result = calculator.semantic_similarity("hello world", "hello")
        assert 0.0 <= result <= 1.0
    
    def test_precision_at_k(self, calculator):
        """Test Precision@k calculation"""
        retrieved = ['a', 'b', 'c', 'd', 'e']
        relevant = ['a', 'c', 'f']
        precision = calculator.precision_at_k(retrieved, relevant, k=3)
        assert precision == 2/3  # 'a' and 'c' in top 3
    
    def test_recall_at_k(self, calculator):
        """Test Recall@k calculation"""
        retrieved = ['a', 'b', 'c']
        relevant = ['a', 'c', 'd', 'e']
        recall = calculator.recall_at_k(retrieved, relevant, k=3)
        assert recall == 2/4  # Found 2 out of 4 relevant
    
    def test_mrr_calculation(self, calculator):
        """Test Mean Reciprocal Rank"""
        retrieved_lists = [['a', 'b', 'c'], ['d', 'e', 'f']]
        relevant_lists = [['b'], ['d']]
        mrr = calculator.mrr(retrieved_lists, relevant_lists)
        assert mrr == (1/2 + 1/1) / 2  # Average of 1/2 and 1/1
    
    def test_ndcg_at_k(self, calculator):
        """Test nDCG@k calculation"""
        retrieved = ['a', 'b', 'c', 'd']
        relevance_scores = [3, 2, 1, 0]
        ndcg = calculator.ndcg_at_k(retrieved, relevance_scores, k=3)
        assert 0.0 <= ndcg <= 1.0
    
    def test_precision_at_k_edge_cases(self, calculator):
        """Test Precision@k with edge cases"""
        assert calculator.precision_at_k([], [], k=5) == 0.0
        assert calculator.precision_at_k(['a'], ['b'], k=1) == 0.0
    
    def test_recall_at_k_edge_cases(self, calculator):
        """Test Recall@k with edge cases"""
        assert calculator.recall_at_k(['a'], [], k=1) == 0.0
        assert calculator.recall_at_k([], ['a'], k=1) == 0.0
