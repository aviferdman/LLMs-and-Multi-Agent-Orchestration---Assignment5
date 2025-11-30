"""
Test suite for Context Windows Research Framework.

This package contains comprehensive tests for all modules:
- Unit tests for individual functions and classes
- Integration tests for end-to-end workflows
- Mock tests for LLM API interactions

Test categories (use pytest markers):
    @pytest.mark.unit: Fast unit tests
    @pytest.mark.integration: Integration tests
    @pytest.mark.llm: Tests requiring LLM API
    @pytest.mark.slow: Long-running tests

Run tests:
    pytest tests/              # All tests
    pytest tests/ -m unit      # Unit tests only
    pytest tests/ --cov=src    # With coverage
"""

__version__ = "1.0.0"
