"""
Setup configuration for Context Windows Research project.

This package provides a comprehensive framework for analyzing LLM context window behavior.
"""

from setuptools import setup, find_packages

setup(
    name="context-windows-research",
    version="1.0.0",
    description="Graduate research on LLM context window behavior",
    author="Research Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "ollama>=0.1.0",
        "chromadb>=0.4.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.60.0",
        "loguru>=0.7.0",
        "python-dotenv>=1.0.0",
        "sentence-transformers>=2.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "black",
            "flake8",
            "mypy",
            "isort",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
