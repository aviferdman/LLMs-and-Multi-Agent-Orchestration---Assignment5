# Context Windows Research Framework

A comprehensive graduate-level research framework for analyzing Large Language Model context window behavior.

## Overview

This project investigates fundamental questions about LLM context windows through four rigorous experiments:

1. Lost in the Middle: Analyzing position effects on information retrieval
2. Context Size Impact: Measuring accuracy degradation with growing context
3. RAG Impact: Comparing RAG vs full context approaches
4. Context Engineering Strategies: Evaluating management strategies for multi-turn conversations

## Quick Start

### Prerequisites

- Python 3.9 or higher
- Ollama installed and running (for local LLM)
- 8GB+ RAM recommended

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd LLMs-and-Multi-Agent-Orchestration---Assignment5

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Copy environment template
cp .env.example .env

# Edit .env with your configuration
nano .env
```

### Running Experiments

```bash
# Run all experiments
python scripts/run_experiments.py

# Run specific experiment
python scripts/run_experiments.py --experiment 1

# Run with custom configuration
python scripts/run_experiments.py --config config/experiments.yaml
```

## Project Structure

```
.
├── src/                     # Source code
│   ├── config.py           # Configuration management
│   ├── llm_interface.py    # LLM abstraction layer
│   ├── metrics.py          # Evaluation metrics
│   ├── statistics.py       # Statistical analysis
│   ├── visualization.py    # Plotting functions
│   ├── data_generation.py  # Synthetic data creation
│   ├── rag_pipeline.py     # RAG implementation
│   └── experiments/        # Experiment modules
├── data/                   # Data files
├── results/                # Experimental outputs
├── tests/                  # Test suite
├── config/                 # Configuration files
├── scripts/                # Utility scripts
├── docs/                   # Documentation
└── notebooks/              # Analysis notebooks
```

## Configuration

Edit config/experiments.yaml to customize experiment parameters.

## Documentation

See the docs/ directory for detailed documentation.

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Requirements

See requirements.txt for full dependencies.

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub.
