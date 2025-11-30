"""
Experimental modules for Context Windows Research.

This package contains implementations of all four core experiments:
- Experiment 1: Lost in the Middle
- Experiment 2: Context Size Impact
- Experiment 3: RAG Impact Analysis
- Experiment 4: Context Engineering Strategies

Each experiment module provides:
- Experimental design implementation
- Data collection procedures
- Statistical analysis
- Visualization generation
- Result reporting

Example:
    >>> from experiments import Experiment1
    >>> config = load_config('config/experiments.yaml')
    >>> exp = Experiment1(config)
    >>> results = exp.run()
    >>> exp.analyze(results)
    >>> exp.visualize(results)
"""

from .experiment_1 import Experiment1
from .experiment_2 import Experiment2
from .experiment_3 import Experiment3
from .experiment_4 import Experiment4

__all__ = [
    "Experiment1",
    "Experiment2",
    "Experiment3",
    "Experiment4",
]
