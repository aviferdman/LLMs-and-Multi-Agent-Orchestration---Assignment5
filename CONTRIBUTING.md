# Contributing to LLM Context Windows Research

Thank you for your interest in contributing to this research project! This document provides guidelines for contributing code, documentation, experiments, and analyses.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Standards](#documentation-standards)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Research Contributions](#research-contributions)

## Code of Conduct

This project adheres to academic research ethics and open science principles:

- **Reproducibility**: All contributions must be reproducible with documented dependencies and random seeds
- **Transparency**: Methods, data, and analyses must be clearly documented
- **Attribution**: Properly cite sources and acknowledge contributions
- **Rigor**: Maintain high standards for testing and statistical validity
- **Respect**: Treat all contributors with respect and professionalism

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Ollama installed and running (for LLM experiments)
- Git for version control
- Basic understanding of statistical analysis and NLP

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd llm-context-windows-research

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development tools

# Install in development mode
pip install -e .
```

### Verify Installation

```bash
# Run tests
pytest tests/ -v

# Check coverage
pytest tests/ --cov=src --cov-report=term-missing

# Verify Ollama connection
python test_ollama_connection.py
```

## Development Setup

### Development Tools

We use the following tools for development:

- **pytest**: Testing framework (v9.0.0+)
- **pytest-cov**: Coverage reporting (v4.1.0+)
- **black**: Code formatting (optional)
- **flake8**: Linting (optional)
- **mypy**: Type checking (optional)

### Project Structure

```
.
├── src/                    # Core library code
│   ├── config.py          # Configuration management
│   ├── llm_interface.py   # LLM integration
│   ├── rag_pipeline.py    # RAG implementation
│   ├── metrics.py         # Evaluation metrics
│   ├── statistics.py      # Statistical analysis
│   ├── visualization.py   # Plotting utilities
│   └── data_generation.py # Test data generation
├── tests/                  # Test suite (181 tests)
├── scripts/                # Experiment runner scripts
├── config/                 # YAML configuration files
├── docs/                   # Documentation
└── results/                # Experimental results
```

## How to Contribute

### Types of Contributions

We welcome the following types of contributions:

1. **Bug Fixes**: Fix existing issues or bugs
2. **New Features**: Add new functionality or experiments
3. **Documentation**: Improve or expand documentation
4. **Tests**: Add tests to increase coverage
5. **Research**: Contribute new experiments or analyses
6. **Visualizations**: Improve or add visualizations

### Reporting Bugs

When reporting bugs, please include:

1. **Description**: Clear description of the bug
2. **Steps to Reproduce**: Minimal steps to reproduce the issue
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**: Python version, OS, Ollama version
6. **Logs**: Relevant error messages or logs

Example:

```markdown
**Bug Description**: LLM query fails with timeout error

**Steps to Reproduce**:
1. Run experiment 1 with 20 documents
2. Observe timeout after 30 seconds

**Expected**: Query completes successfully
**Actual**: RuntimeError: Query timeout after 3 attempts

**Environment**:
- Python 3.11.9
- Windows 11
- Ollama 0.1.0
- Model: llama2:latest

**Logs**:
```
ERROR | Query timeout after 3 attempts
```
```

### Suggesting Features

When suggesting features, please include:

1. **Use Case**: What problem does this solve?
2. **Proposed Solution**: How should it work?
3. **Alternatives**: Other approaches considered
4. **Impact**: Who benefits and how?

## Testing Guidelines

### Test Coverage Requirements

- **Minimum Coverage**: 85% per module
- **Overall Target**: >90%
- **Test Types**: Unit tests, integration tests, mocking

### Writing Tests

All new code must include tests:

```python
# tests/test_new_feature.py
import pytest
from src.new_feature import NewClass

class TestNewClass:
    """Test suite for NewClass"""

    @pytest.fixture
    def instance(self):
        """Create instance for testing"""
        return NewClass()

    def test_initialization(self, instance):
        """Test proper initialization"""
        assert instance is not None

    def test_method_with_valid_input(self, instance):
        """Test method with valid input"""
        result = instance.method("valid")
        assert result == expected_value

    def test_method_with_invalid_input(self, instance):
        """Test method handles invalid input"""
        with pytest.raises(ValueError):
            instance.method("invalid")
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_llm_interface.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_llm_interface.py::TestOllamaInterface::test_query_success -v
```

### Test Best Practices

1. **Descriptive Names**: Use clear, descriptive test names
2. **One Assertion**: Test one thing per test function
3. **Fixtures**: Use pytest fixtures for setup
4. **Mocking**: Mock external dependencies (APIs, file I/O)
5. **Edge Cases**: Test boundary conditions and error cases
6. **Documentation**: Include docstrings explaining what is tested

## Documentation Standards

### Code Documentation

```python
def function_name(param1: str, param2: int) -> dict:
    """
    Brief description of function purpose.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When validation fails
        RuntimeError: When operation fails

    Example:
        >>> result = function_name("test", 42)
        >>> print(result['status'])
        'success'
    """
    pass
```

### Markdown Documentation

- Use clear headings and structure
- Include code examples
- Add links to related documentation
- Keep language clear and concise
- Use proper formatting for code blocks

### Updating Documentation

When adding features, update:

1. **README.md**: If it affects usage or setup
2. **ARCHITECTURE.md**: If it changes system design
3. **API.md**: If it adds/modifies public APIs
4. **USER_GUIDE.md**: If it affects user workflows
5. **CHANGELOG.md**: For all user-facing changes

## Commit Messages

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **test**: Adding or modifying tests
- **refactor**: Code refactoring
- **perf**: Performance improvements
- **style**: Code style changes (formatting)
- **chore**: Build/tooling changes

### Examples

```
feat(rag): Add semantic chunking with overlap

Implement overlapping chunks for better context preservation.
Adds chunk_overlap parameter to RAGPipeline.

Closes #42
```

```
fix(llm): Handle connection timeout with retry

Add exponential backoff for connection timeouts.
Fixes issue where queries fail on slow connections.

Fixes #38
```

```
test(metrics): Increase coverage to 95%

Add tests for edge cases in precision/recall calculation.
Tests empty lists, zero values, and invalid inputs.
```

## Pull Request Process

### Before Submitting

1. **Tests Pass**: Ensure all tests pass
   ```bash
   pytest tests/ -v
   ```

2. **Coverage**: Maintain >85% coverage
   ```bash
   pytest tests/ --cov=src --cov-report=term-missing
   ```

3. **Documentation**: Update relevant docs

4. **Commit Messages**: Follow commit message format

5. **Clean History**: Squash/rebase if needed

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Test coverage
- [ ] Research/experiment

## Testing
- [ ] All tests pass
- [ ] Coverage maintained/increased
- [ ] Manual testing completed

## Documentation
- [ ] Code comments added/updated
- [ ] Documentation files updated
- [ ] CHANGELOG.md updated

## Related Issues
Closes #<issue-number>
```

### Review Process

1. **Automated Checks**: CI/CD runs tests and coverage
2. **Code Review**: At least one maintainer reviews
3. **Feedback**: Address review comments
4. **Approval**: Maintainer approves changes
5. **Merge**: Squash and merge to main

## Research Contributions

### Adding New Experiments

To contribute a new experiment:

1. **Design**: Document research question and methodology
2. **Implementation**: Create experiment script in `scripts/`
3. **Configuration**: Add config to `config/experiments.yaml`
4. **Testing**: Write tests for experiment components
5. **Execution**: Run experiment with real LLM
6. **Analysis**: Generate results and visualizations
7. **Documentation**: Write experiment documentation

### Experiment Template

```python
# scripts/run_experiment_5_new.py
"""
Experiment 5: <Research Question>

Investigates: <What this experiment tests>
"""

from src.config import Config
from src.llm_interface import OllamaInterface
from src.metrics import MetricsCalculator

def run_experiment():
    """Run Experiment 5"""
    config = Config()
    exp_config = config.get_experiment_config(5)

    # Setup
    llm = OllamaInterface(model=exp_config['model'])

    # Execute trials
    results = []
    for trial in range(exp_config['num_trials']):
        # Run trial
        result = execute_trial(trial)
        results.append(result)

    # Analyze and save
    save_results(results)

if __name__ == "__main__":
    run_experiment()
```

### Statistical Analysis Standards

All research contributions must include:

1. **Hypothesis**: Clear null and alternative hypotheses
2. **Statistical Tests**: Appropriate tests (t-test, ANOVA, etc.)
3. **P-values**: Report significance levels
4. **Effect Sizes**: Include Cohen's d or similar
5. **Confidence Intervals**: 95% CI for key findings
6. **Assumptions**: Verify and document test assumptions

### Visualization Standards

All visualizations must:

1. **Resolution**: 300 DPI for publication quality
2. **Style**: Use consistent color scheme and fonts
3. **Labels**: Clear axis labels and titles
4. **Legend**: Include legend when needed
5. **Annotations**: Add statistical annotations
6. **Format**: Save as PNG with transparent background

## Questions?

If you have questions about contributing:

1. Check existing [documentation](docs/)
2. Search [existing issues](issues)
3. Open a new issue with your question
4. Contact maintainers

## Thank You!

Your contributions help advance open science and reproducible research in NLP and LLM evaluation. We appreciate your time and effort!
