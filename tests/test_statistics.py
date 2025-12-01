"""
Tests for statistical analysis module
"""

import pytest
import numpy as np
from src.statistics import StatisticalAnalyzer, t_test, anova_test, cohen_d, correlation

class TestStatisticalAnalyzer:
    """Test suite for StatisticalAnalyzer class"""
    
    @pytest.fixture
    def analyzer(self):
        """Create StatisticalAnalyzer instance for testing"""
        return StatisticalAnalyzer(alpha=0.05)
    
    def test_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer is not None
        assert analyzer.alpha == 0.05
    
    def test_t_test(self, analyzer):
        """Test t-test method"""
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([6, 7, 8, 9, 10])
        result = analyzer.t_test(group1, group2)
        assert 't_statistic' in result
        assert 'p_value' in result
        assert isinstance(result['p_value'], float)
    
    def test_anova(self, analyzer):
        """Test ANOVA method"""
        groups = [
            np.array([1, 2, 3, 4, 5]),
            np.array([6, 7, 8, 9, 10]),
            np.array([11, 12, 13, 14, 15])
        ]
        result = analyzer.anova(groups)
        assert 'f_statistic' in result
        assert 'p_value' in result
        assert isinstance(result['p_value'], float)
    
    def test_correlation(self, analyzer):
        """Test correlation method"""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        result = analyzer.correlation(x, y)
        assert 'correlation' in result
        assert 'p_value' in result
        assert abs(result['correlation'] - 1.0) < 0.01
    
    def test_regression(self, analyzer):
        """Test linear regression"""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        result = analyzer.regression(x, y)
        assert 'slope' in result
        assert 'intercept' in result
        assert 'r_squared' in result
        assert abs(result['slope'] - 2.0) < 0.01

class TestStatisticalFunctions:
    """Test suite for standalone statistical functions"""
    
    def test_t_test_function(self):
        """Test t_test function"""
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([6, 7, 8, 9, 10])
        result = t_test(group1, group2)
        # Function returns dict
        assert isinstance(result, dict)
        assert 'p_value' in result
        assert result['p_value'] < 0.05
    
    def test_anova_function(self):
        """Test anova_test function"""
        groups = [
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array([7, 8, 9])
        ]
        result = anova_test(groups)
        # Function returns dict
        assert isinstance(result, dict)
        assert 'p_value' in result
    
    def test_cohen_d_function(self):
        """Test cohen_d function"""
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([6, 7, 8, 9, 10])
        d = cohen_d(group1, group2)
        assert isinstance(d, float)
        assert abs(d) > 0
    
    def test_correlation_function(self):
        """Test correlation function"""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        result = correlation(x, y)
        # Function returns dict
        assert isinstance(result, dict)
        assert 'correlation' in result
        assert abs(result['correlation'] - 1.0) < 0.01
