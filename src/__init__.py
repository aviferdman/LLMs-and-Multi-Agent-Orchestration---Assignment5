"""
Context Windows Research Framework

A comprehensive toolkit for analyzing LLM context window behavior.

This package provides infrastructure for conducting graduate-level research
on context window management in Large Language Models, including:
- Lost in the Middle phenomenon analysis
- Context size scaling laws
- RAG (Retrieval-Augmented Generation) impact studies
- Context engineering strategy evaluation

Modules:
    config: Configuration management
    llm_interface: LLM abstraction layer
    data_generation: Synthetic data creation
    metrics: Evaluation metrics
    statistics: Statistical analysis tools
    visualization: Publication-quality plotting
    rag_pipeline: RAG implementation
    experiments: Experimental modules

Author: Research Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Research Team"

# Export main interfaces
from .config import Config
from .llm_interface import LLMInterface, OllamaInterface
from .metrics import MetricsCalculator
from .visualization import Visualizer, plot_rag_comparison, plot_top_k_analysis
from .statistics import StatisticalAnalyzer

__all__ = [
    "Config",
    "LLMInterface",
    "OllamaInterface",
    "MetricsCalculator",
    "Visualizer",
    "StatisticalAnalyzer",
    "plot_rag_comparison",
    "plot_top_k_analysis",
]
