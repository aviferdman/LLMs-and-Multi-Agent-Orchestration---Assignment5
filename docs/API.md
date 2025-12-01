# API Reference Documentation

**Project**: Context Windows in Practice - Research Framework  
**Version**: 1.0  
**Date**: December 1, 2025  
**Status**: Complete

---

## Table of Contents

1. [Overview](#overview)
2. [LLM Interface API](#llm-interface-api)
3. [RAG Pipeline API](#rag-pipeline-api)
4. [Metrics API](#metrics-api)
5. [Statistics API](#statistics-api)
6. [Visualization API](#visualization-api)
7. [Configuration API](#configuration-api)
8. [Usage Examples](#usage-examples)

---

## Overview

### Purpose

This document provides comprehensive API reference for all public interfaces in the Context Windows Research Framework.

### Installation

```bash
# Clone repository
git clone <repository-url>
cd LLMs-and-Multi-Agent-Orchestration---Assignment5

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Quick Start

```python
from src.llm_interface import OllamaInterface
from src.rag_pipeline import RAGPipeline, ChunkingStrategy

# Initialize LLM
llm = OllamaInterface(model="llama2:latest")

# Create RAG pipeline
rag = RAGPipeline(
    llm_interface=llm,
    chunking_strategy=ChunkingStrategy.SEMANTIC,
    chunk_size=512
)

# Add documents
rag.add_document("doc1", "Your document content...")

# Query
response = rag.query("What is X?", top_k=3)
print(response.text)
```

---

## LLM Interface API

### OllamaInterface

**Class**: `src.llm_interface.OllamaInterface`

Primary interface for interacting with Ollama LLM backend.

#### Constructor

```python
OllamaInterface(
    model: str = "llama2:latest",
    base_url: str = "http://localhost:11434",
    timeout: int = 300,
    max_retries: int = 3,
    retry_delay: float = 2.0
)
```

**Parameters:**
- `model` (str): Ollama model name (e.g., "llama2:latest", "llama3:latest")
- `base_url` (str): Ollama server URL
- `timeout` (int): Request timeout in seconds
- `max_retries` (int): Maximum retry attempts on failure
- `retry_delay` (float): Seconds between retries

**Raises:**
- `ConnectionError`: If Ollama server is not accessible

**Example:**
```python
from src.llm_interface import OllamaInterface

llm = OllamaInterface(
    model="llama2:13b",
    timeout=600,
    max_retries=5
)
```

#### query()

Query the LLM with context and question.

```python
def query(
    self,
    context: str,
    query: str,
    **kwargs
) -> Response
```

**Parameters:**
- `context` (str): Contextual information to provide
- `query` (str): Question to ask
- `**kwargs`: Additional options:
  - `temperature` (float): 0.0-1.0, default 0.1
  - `top_p` (float): 0.0-1.0, default 0.9
  - `top_k` (int): default 40
  - `num_ctx` (int): Context window size, default 2048

**Returns:**
- `Response`: Object containing:
  - `text` (str): Generated response
  - `latency` (float): Time in seconds
  - `tokens` (int): Number of tokens
  - `metadata` (dict): Additional info

**Raises:**
- `RuntimeError`: If all retry attempts fail

**Example:**
```python
response = llm.query(
    context="The capital of France is Paris.",
    query="What is the capital of France?",
    temperature=0.1,
    top_p=0.9
)

print(f"Answer: {response.text}")
print(f"Latency: {response.latency:.2f}s")
print(f"Tokens: {response.tokens}")
```

#### embed()

Generate embedding vector for text.

```python
def embed(self, text: str) -> np.ndarray
```

**Parameters:**
- `text` (str): Text to embed

**Returns:**
- `np.ndarray`: Embedding vector

**Example:**
```python
embedding = llm.embed("Sample text")
print(f"Embedding shape: {embedding.shape}")
```

#### count_tokens()

Count approximate tokens in text.

```python
def count_tokens(self, text: str) -> int
```

**Parameters:**
- `text` (str): Text to count

**Returns:**
- `int`: Approximate token count

**Example:**
```python
text = "This is a sample sentence."
tokens = llm.count_tokens(text)
print(f"Tokens: {tokens}")
```

### Response

**Dataclass**: `src.llm_interface.Response`

Structured response from LLM queries.

**Attributes:**
- `text` (str): Generated text
- `latency` (float): Generation time in seconds
- `tokens` (int): Number of tokens
- `confidence` (float, optional): Model confidence score
- `metadata` (dict, optional): Provider-specific data

**Example:**
```python
response = llm.query(context, query)
print(f"Response: {response.text}")
print(f"Took {response.latency:.2f}s for {response.tokens} tokens")
```

---

## RAG Pipeline API

### RAGPipeline

**Class**: `src.rag_pipeline.RAGPipeline`

Retrieval-Augmented Generation pipeline with multiple chunking strategies.

#### Constructor

```python
RAGPipeline(
    llm_interface: LLMInterface,
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
    chunk_size: int = 512,
    chunk_overlap: int = 50
)
```

**Parameters:**
- `llm_interface` (LLMInterface): LLM instance for queries and embeddings
- `chunking_strategy` (ChunkingStrategy): Chunking method (FIXED, SEMANTIC, SLIDING)
- `chunk_size` (int): Target chunk size in characters
- `chunk_overlap` (int): Overlap between chunks

**Example:**
```python
from src.rag_pipeline import RAGPipeline, ChunkingStrategy

rag = RAGPipeline(
    llm_interface=llm,
    chunking_strategy=ChunkingStrategy.SEMANTIC,
    chunk_size=512,
    chunk_overlap=50
)
```

#### add_document()

Add document to the pipeline.

```python
def add_document(
    self,
    document_id: str,
    content: str
) -> None
```

**Parameters:**
- `document_id` (str): Unique identifier
- `content` (str): Document text

**Example:**
```python
rag.add_document("doc1", "Your document content here...")
rag.add_document("doc2", "Another document...")
```

#### query()

Query the pipeline with retrieval.

```python
def query(
    self,
    question: str,
    top_k: int = 3
) -> Response
```

**Parameters:**
- `question` (str): User query
- `top_k` (int): Number of chunks to retrieve

**Returns:**
- `Response`: LLM response with retrieved context

**Example:**
```python
response = rag.query("What is machine learning?", top_k=5)
print(response.text)
```

#### chunk_document()

Chunk document using configured strategy.

```python
def chunk_document(
    self,
    content: str
) -> List[str]
```

**Parameters:**
- `content` (str): Document text

**Returns:**
- `List[str]`: List of text chunks

**Example:**
```python
chunks = rag.chunk_document("Long document text...")
print(f"Created {len(chunks)} chunks")
```

### ChunkingStrategy

**Enum**: `src.rag_pipeline.ChunkingStrategy`

Available chunking strategies.

**Values:**
- `FIXED`: Simple character-based splitting
- `SEMANTIC`: Sentence-aware splitting with overlap
- `SLIDING`: Overlapping sliding windows

**Example:**
```python
from src.rag_pipeline import ChunkingStrategy

strategy = ChunkingStrategy.SEMANTIC
```

---

## Metrics API

### MetricsEvaluator

**Class**: `src.metrics.MetricsEvaluator`

Static methods for evaluating model performance.

#### exact_match_accuracy()

Calculate binary exact match accuracy.

```python
@staticmethod
def exact_match_accuracy(
    predicted: str,
    ground_truth: str
) -> float
```

**Parameters:**
- `predicted` (str): Model prediction
- `ground_truth` (str): Correct answer

**Returns:**
- `float`: 1.0 if exact match, 0.0 otherwise

**Example:**
```python
from src.metrics import MetricsEvaluator

accuracy = MetricsEvaluator.exact_match_accuracy(
    predicted="Paris",
    ground_truth="Paris"
)
print(f"Accuracy: {accuracy}")  # 1.0
```

#### semantic_similarity()

Calculate embedding-based similarity.

```python
@staticmethod
def semantic_similarity(
    predicted: str,
    ground_truth: str,
    llm_interface: LLMInterface
) -> float
```

**Parameters:**
- `predicted` (str): Model prediction
- `ground_truth` (str): Correct answer  
- `llm_interface` (LLMInterface): LLM for embeddings

**Returns:**
- `float`: Cosine similarity score (0.0 to 1.0)

**Example:**
```python
similarity = MetricsEvaluator.semantic_similarity(
    predicted="The capital is Paris",
    ground_truth="Paris is the capital",
    llm_interface=llm
)
print(f"Similarity: {similarity:.3f}")
```

#### measure_latency()

Measure function execution time.

```python
@staticmethod
def measure_latency(func: Callable) -> Tuple[Any, float]
```

**Parameters:**
- `func` (Callable): Function to measure

**Returns:**
- `Tuple[Any, float]`: (result, latency_seconds)

**Example:**
```python
def slow_function():
    time.sleep(1)
    return "result"

result, latency = MetricsEvaluator.measure_latency(slow_function)
print(f"Took {latency:.2f}s")
```

---

## Statistics API

### StatisticalAnalyzer

**Class**: `src.statistics.StatisticalAnalyzer`

Statistical analysis methods.

#### anova_one_way()

Perform one-way ANOVA.

```python
@staticmethod
def anova_one_way(
    *groups: List[float]
) -> Dict[str, float]
```

**Parameters:**
- `*groups`: Variable number of data groups

**Returns:**
- `dict`: Contains `f_statistic`, `p_value`, `eta_squared`

**Example:**
```python
from src.statistics import StatisticalAnalyzer

group1 = [1.0, 1.0, 1.0]
group2 = [0.5, 0.6, 0.7]
group3 = [0.3, 0.4, 0.5]

results = StatisticalAnalyzer.anova_one_way(group1, group2, group3)
print(f"F-statistic: {results['f_statistic']:.3f}")
print(f"p-value: {results['p_value']:.4f}")
print(f"Effect size (η²): {results['eta_squared']:.3f}")
```

#### correlation_analysis()

Compute Pearson and Spearman correlations.

```python
@staticmethod
def correlation_analysis(
    x: List[float],
    y: List[float]
) -> Dict[str, float]
```

**Parameters:**
- `x` (List[float]): First variable
- `y` (List[float]): Second variable

**Returns:**
- `dict`: Contains `pearson_r`, `pearson_p`, `spearman_r`, `spearman_p`

**Example:**
```python
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

results = StatisticalAnalyzer.correlation_analysis(x, y)
print(f"Pearson r: {results['pearson_r']:.3f}")
print(f"p-value: {results['pearson_p']:.4f}")
```

#### effect_size_cohens_d()

Calculate Cohen's d effect size.

```python
@staticmethod
def effect_size_cohens_d(
    group1: List[float],
    group2: List[float]
) -> float
```

**Parameters:**
- `group1` (List[float]): First group
- `group2` (List[float]): Second group

**Returns:**
- `float`: Cohen's d value

**Example:**
```python
group1 = [1.0, 1.0, 1.0]
group2 = [0.5, 0.6, 0.7]

d = StatisticalAnalyzer.effect_size_cohens_d(group1, group2)
print(f"Cohen's d: {d:.3f}")
```

---

## Visualization API

### VisualizationGenerator

**Class**: `src.visualization.VisualizationGenerator`

Generate publication-quality plots.

#### plot_bar_chart()

Create bar chart with error bars.

```python
@staticmethod
def plot_bar_chart(
    data: List[float],
    labels: List[str],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: str,
    errors: Optional[List[float]] = None
) -> None
```

**Parameters:**
- `data` (List[float]): Bar heights
- `labels` (List[str]): Bar labels
- `title` (str): Plot title
- `xlabel` (str): X-axis label
- `ylabel` (str): Y-axis label
- `output_path` (str): Save path
- `errors` (List[float], optional): Error bar values

**Example:**
```python
from src.visualization import VisualizationGenerator

VisualizationGenerator.plot_bar_chart(
    data=[0.85, 0.78, 0.81],
    labels=["SEMANTIC", "FIXED", "SLIDING"],
    title="Chunking Strategy Comparison",
    xlabel="Strategy",
    ylabel="Accuracy",
    output_path="results/figures/strategies.png",
    errors=[0.05, 0.08, 0.06]
)
```

#### plot_scatter()

Create scatter plot with optional regression line.

```python
@staticmethod
def plot_scatter(
    x: List[float],
    y: List[float],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: str,
    add_regression: bool = False
) -> None
```

**Parameters:**
- `x` (List[float]): X coordinates
- `y` (List[float]): Y coordinates
- `title` (str): Plot title
- `xlabel` (str): X-axis label
- `ylabel` (str): Y-axis label
- `output_path` (str): Save path
- `add_regression` (bool): Add regression line

**Example:**
```python
VisualizationGenerator.plot_scatter(
    x=[2, 5, 10, 20, 50],
    y=[1.0, 1.0, 0.4, 0.3, 0.2],
    title="Accuracy vs Context Size",
    xlabel="Context Size (documents)",
    ylabel="Accuracy",
    output_path="results/figures/scaling.png",
    add_regression=True
)
```

---

## Configuration API

### Config

**Class**: `src.config.Config`

Singleton configuration manager.

#### get_instance()

Get singleton instance.

```python
@classmethod
def get_instance(cls) -> 'Config'
```

**Returns:**
- `Config`: Singleton instance

**Example:**
```python
from src.config import Config

config = Config.get_instance()
```

#### get_experiment_config()

Get configuration for specific experiment.

```python
def get_experiment_config(
    self,
    experiment_id: str
) -> Dict[str, Any]
```

**Parameters:**
- `experiment_id` (str): Experiment identifier (e.g., "experiment_1")

**Returns:**
- `dict`: Experiment configuration

**Example:**
```python
config = Config.get_instance()
exp_config = config.get_experiment_config("experiment_1")
print(exp_config["num_runs"])
```

---

## Usage Examples

### Complete RAG Pipeline Example

```python
from src.llm_interface import OllamaInterface
from src.rag_pipeline import RAGPipeline, ChunkingStrategy
from src.metrics import MetricsEvaluator

# 1. Initialize LLM
llm = OllamaInterface(
    model="llama2:latest",
    timeout=300,
    max_retries=3
)

# 2. Create RAG pipeline
rag = RAGPipeline(
    llm_interface=llm,
    chunking_strategy=ChunkingStrategy.SEMANTIC,
    chunk_size=512,
    chunk_overlap=50
)

# 3. Add documents
rag.add_document("doc1", "Machine learning is a subset of AI...")
rag.add_document("doc2", "Deep learning uses neural networks...")

# 4. Query
response = rag.query(
    question="What is machine learning?",
    top_k=3
)

# 5. Evaluate
ground_truth = "Machine learning is a subset of AI"
accuracy = MetricsEvaluator.exact_match_accuracy(
    predicted=response.text,
    ground_truth=ground_truth
)

print(f"Response: {response.text}")
print(f"Latency: {response.latency:.2f}s")
print(f"Accuracy: {accuracy}")
```

### Experiment Execution Example

```python
from src.llm_interface import OllamaInterface
from src.metrics import MetricsEvaluator
from src.statistics import StatisticalAnalyzer
import json

# Initialize
llm = OllamaInterface(model="llama2:latest")
results = []

# Run experiment
positions = ["START", "MIDDLE", "END"]
for position in positions:
    for run in range(10):
        # Create context
        context = create_context(position)
        
        # Query
        response = llm.query(context, "What is X?")
        
        # Evaluate
        accuracy = MetricsEvaluator.exact_match_accuracy(
            response.text,
            "X is Y"
        )
        
        # Store result
        results.append({
            "position": position,
            "run": run,
            "accuracy": accuracy,
            "latency": response.latency
        })

# Analyze
groups = {
    "START": [r["accuracy"] for r in results if r["position"] == "START"],
    "MIDDLE": [r["accuracy"] for r in results if r["position"] == "MIDDLE"],
    "END": [r["accuracy"] for r in results if r["position"] == "END"]
}

anova = StatisticalAnalyzer.anova_one_way(
    groups["START"],
    groups["MIDDLE"],
    groups["END"]
)

print(f"F-statistic: {anova['f_statistic']:.3f}")
print(f"p-value: {anova['p_value']:.4f}")

# Save
with open("results/raw/experiment_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Batch Processing Example

```python
from concurrent.futures import ThreadPoolExecutor
from src.llm_interface import OllamaInterface

llm = OllamaInterface(model="llama2:latest")

# Prepare queries
queries = [
    ("Context 1", "Question 1"),
    ("Context 2", "Question 2"),
    ("Context 3", "Question 3"),
]

# Parallel execution
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [
        executor.submit(llm.query, ctx, q)
        for ctx, q in queries
    ]
    
    responses = [f.result() for f in futures]

# Process results
for i, response in enumerate(responses):
    print(f"Query {i+1}: {response.text}")
    print(f"Latency: {response.latency:.2f}s\n")
```

---

## Error Handling

### Common Exceptions

**ConnectionError**: Ollama server not accessible
```python
try:
    llm = OllamaInterface()
except ConnectionError as e:
    print(f"Error: {e}")
    print("Make sure Ollama is running: ollama serve")
```

**RuntimeError**: Query failed after retries
```python
try:
    response = llm.query(context, query)
except RuntimeError as e:
    print(f"Query failed: {e}")
    # Log error and continue
```

**Timeout**: Request exceeded timeout
```python
try:
    response = llm.query(context, query, timeout=60)
except Exception as e:
    print(f"Timeout: {e}")
    # Retry with longer timeout
    response = llm.query(context, query, timeout=300)
```

---

## Best Practices

### 1. Resource Management

```python
# Use context managers when available
with open("results.json", "w") as f:
    json.dump(results, f)

# Clean up large objects
del large_embedding_cache
import gc; gc.collect()
```

### 2. Error Handling

```python
# Always handle potential failures
try:
    response = llm.query(context, query)
except Exception as e:
    logger.error(f"Query failed: {e}")
    # Implement fallback logic
    response = fallback_response()
```

### 3. Logging

```python
from loguru import logger

logger.add("experiments.log", rotation="100 MB")
logger.info(f"Starting experiment with model: {model}")
```

### 4. Configuration

```python
# Use configuration files instead of hardcoding
config = Config.get_instance()
model = config.get_model_config("ollama")["default_model"]
```

---

## Version Compatibility

**Python**: 3.10+
**Dependencies**:
- numpy >= 1.21.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- requests >= 2.26.0
- loguru >= 0.5.3

---

**Document Version**: 1.0  
**Last Updated**: December 1, 2025  
**Maintained By**: Development Team  
**Status**: Complete

*End of API Reference Documentation*
