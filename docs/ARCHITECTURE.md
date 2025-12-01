# System Architecture Documentation

**Project**: Context Windows in Practice - Empirical Research Framework  
**Version**: 1.0  
**Date**: December 1, 2025  
**Status**: Complete

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Data Flow](#data-flow)
5. [Module Specifications](#module-specifications)
6. [Experiment Pipeline](#experiment-pipeline)
7. [Configuration Management](#configuration-management)
8. [Storage and Persistence](#storage-and-persistence)
9. [Deployment Architecture](#deployment-architecture)

---

## Overview

### Purpose

This document describes the architectural design of the Context Windows Research Framework, a modular system for conducting controlled experiments on LLM context window behavior.

### Design Principles

1. **Modularity**: Clear separation of concerns with reusable components
2. **Extensibility**: Easy to add new experiments, LLM providers, or chunking strategies
3. **Reproducibility**: Deterministic execution with comprehensive logging
4. **Scalability**: Efficient resource usage for large-scale experiments
5. **Testability**: Each component independently testable

### Technology Stack

| Layer | Technologies |
|-------|-------------|
| **Language** | Python 3.10+ |
| **LLM Interface** | Ollama API, REST |
| **Data Processing** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Statistics** | SciPy, statsmodels |
| **Configuration** | YAML |
| **Logging** | Loguru |
| **Testing** | pytest |

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Experiment Scripts                          │
│  (run_experiment_1_ollama.py, run_experiment_2_ollama.py, ...)  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Core Framework (src/)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐   │
│  │   Experiment │  │     RAG      │  │   LLM Interface     │   │
│  │   Modules    │  │   Pipeline   │  │   (Ollama/OpenAI)   │   │
│  └──────────────┘  └──────────────┘  └─────────────────────┘   │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐   │
│  │    Metrics   │  │   Chunking   │  │   Visualization     │   │
│  │  Evaluation  │  │  Strategies  │  │     Generator       │   │
│  └──────────────┘  └──────────────┘  └─────────────────────┘   │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐   │
│  │  Statistics  │  │  Data Gen    │  │   Configuration     │   │
│  └──────────────┘  └──────────────┘  └─────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    External Services                             │
│  ┌──────────────────┐       ┌─────────────────────────────┐    │
│  │  Ollama Server   │       │    File System Storage      │    │
│  │  (LLM Backend)   │       │  (Results/Logs/Config)      │    │
│  └──────────────────┘       └─────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Component Interaction Diagram

```
┌─────────────┐
│  Experiment │
│   Script    │
└──────┬──────┘
       │
       ├─────► ┌──────────────┐      ┌────────────┐
       │       │  Config      │◄─────┤   YAML     │
       │       │  Manager     │      │   Files    │
       │       └──────────────┘      └────────────┘
       │
       ├─────► ┌──────────────┐      ┌────────────┐
       │       │  Data        │◄─────┤  Document  │
       │       │  Generator   │      │  Templates │
       │       └──────────────┘      └────────────┘
       │
       ├─────► ┌──────────────┐      ┌────────────┐
       │       │  RAG         │◄─────┤  Chunking  │
       │       │  Pipeline    │      │  Strategies│
       │       └──────┬───────┘      └────────────┘
       │              │
       │              ▼
       │       ┌──────────────┐      ┌────────────┐
       │       │  LLM         │◄─────┤  Ollama    │
       │       │  Interface   │      │  Server    │
       │       └──────┬───────┘      └────────────┘
       │              │
       │              ▼
       ├─────► ┌──────────────┐
       │       │  Metrics     │
       │       │  Evaluator   │
       │       └──────┬───────┘
       │              │
       │              ▼
       └─────► ┌──────────────┐      ┌────────────┐
               │  Statistics  │─────►│  Results   │
               │  Analyzer    │      │  Storage   │
               └──────────────┘      └────────────┘
                      │
                      ▼
               ┌──────────────┐      ┌────────────┐
               │Visualization │─────►│  PNG/PDF   │
               │  Generator   │      │  Figures   │
               └──────────────┘      └────────────┘
```

---

## Core Components

### 1. LLM Interface Layer

**Module**: `src/llm_interface.py`

**Purpose**: Abstract interface for multiple LLM providers

**Key Classes**:

```python
LLMInterface (ABC)
├── query(context, query) → Response
├── embed(text) → np.ndarray
└── count_tokens(text) → int

OllamaInterface(LLMInterface)
├── __init__(model, base_url, timeout, max_retries)
├── query() → Response (with retry logic)
├── embed() → np.ndarray (with fallback)
├── count_tokens() → int (heuristic)
└── _verify_connection() → bool
```

**Features**:
- **Retry Logic**: Automatic retry with exponential backoff
- **Timeout Handling**: Configurable timeout with graceful failure
- **Connection Verification**: Health check on initialization
- **Embedding Support**: Vector generation for RAG
- **Token Counting**: Approximate tokenization

**Configuration**:
```python
llm = OllamaInterface(
    model="llama2:latest",
    base_url="http://localhost:11434",
    timeout=300,  # 5 minutes
    max_retries=3,
    retry_delay=2.0
)
```

### 2. RAG Pipeline

**Module**: `src/rag_pipeline.py`

**Purpose**: Retrieval-Augmented Generation with multiple chunking strategies

**Key Classes**:

```python
ChunkingStrategy (Enum)
├── FIXED
├── SEMANTIC
└── SLIDING

RAGPipeline
├── __init__(llm_interface, chunking_strategy, chunk_size)
├── add_document(document_id, content)
├── chunk_document(content) → List[str]
├── query(question, top_k) → Response
└── retrieve_relevant_chunks(query, top_k) → List[Tuple]
```

**Chunking Strategies**:

1. **FIXED**: Simple character-based splitting
   ```python
   chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
   ```

2. **SEMANTIC**: Sentence-aware splitting with overlap
   ```python
   - Split on sentence boundaries
   - Maintain context with overlapping chunks
   - Preserve semantic coherence
   ```

3. **SLIDING**: Overlapping windows
   ```python
   chunks = [text[i:i+chunk_size] 
             for i in range(0, len(text), step_size)]
   ```

**Retrieval**:
- **Method**: Cosine similarity on embeddings
- **Top-K Selection**: Configurable number of chunks
- **Context Assembly**: Concatenation with separators

### 3. Experiment Framework

**Module**: `src/experiments/`

**Structure**:
```
experiments/
├── experiment_1.py  # Position Effect
├── experiment_2.py  # Context Size Scaling
├── experiment_3.py  # Chunking Strategies
└── experiment_4.py  # Context Management
```

**Base Experiment Class**:

```python
class BaseExperiment(ABC):
    def __init__(self, config: Dict):
        self.config = config
        self.llm = self._initialize_llm()
        self.results = []
    
    @abstractmethod
    def run(self) → Dict:
        """Execute experiment and return results"""
        pass
    
    def save_results(self, output_path: str):
        """Persist results to JSON"""
        pass
    
    def generate_visualizations(self, output_dir: str):
        """Create plots from results"""
        pass
```

### 4. Metrics and Evaluation

**Module**: `src/metrics.py`

**Metrics Computed**:

```python
class MetricsEvaluator:
    @staticmethod
    def exact_match_accuracy(predicted, ground_truth) → float:
        """Binary exact match"""
        return 1.0 if predicted.strip() == ground_truth.strip() else 0.0
    
    @staticmethod
    def semantic_similarity(pred, truth, llm) → float:
        """Embedding-based similarity"""
        emb1 = llm.embed(pred)
        emb2 = llm.embed(truth)
        return cosine_similarity(emb1, emb2)
    
    @staticmethod
    def measure_latency(func) → Tuple[Any, float]:
        """Execution time measurement"""
        start = time.time()
        result = func()
        latency = time.time() - start
        return result, latency
    
    @staticmethod
    def count_context_tokens(context, llm) → int:
        """Token count in context"""
        return llm.count_tokens(context)
```

### 5. Statistical Analysis

**Module**: `src/statistics.py`

**Capabilities**:

```python
class StatisticalAnalyzer:
    @staticmethod
    def anova_one_way(groups) → Dict:
        """One-way ANOVA with post-hoc tests"""
        f_stat, p_value = f_oneway(*groups)
        effect_size = eta_squared(groups)
        return {"f": f_stat, "p": p_value, "eta2": effect_size}
    
    @staticmethod
    def correlation_analysis(x, y) → Dict:
        """Pearson and Spearman correlation"""
        pearson_r, p_pearson = pearsonr(x, y)
        spearman_r, p_spearman = spearmanr(x, y)
        return {...}
    
    @staticmethod
    def linear_regression(x, y) → Dict:
        """Simple linear regression"""
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        return {...}
    
    @staticmethod
    def effect_size_cohens_d(group1, group2) → float:
        """Cohen's d effect size"""
        return (mean(group1) - mean(group2)) / pooled_std(group1, group2)
```

### 6. Visualization System

**Module**: `src/visualization.py`

**Plot Types**:

```python
class VisualizationGenerator:
    @staticmethod
    def plot_bar_chart(data, labels, title, output_path):
        """Bar chart with error bars"""
    
    @staticmethod
    def plot_line_chart(x, y, title, xlabel, ylabel, output_path):
        """Line plot with confidence intervals"""
    
    @staticmethod
    def plot_scatter(x, y, labels, title, output_path):
        """Scatter plot with regression line"""
    
    @staticmethod
    def plot_heatmap(matrix, labels, title, output_path):
        """Correlation or confusion heatmap"""
    
    @staticmethod
    def plot_distribution(data, title, output_path):
        """Histogram with KDE overlay"""
```

**Styling**:
- **Theme**: Seaborn whitegrid
- **Resolution**: 300 DPI (publication quality)
- **Color Palette**: Color-blind friendly
- **Fonts**: 10-14pt, bold titles

### 7. Data Generation

**Module**: `src/data_generation.py`

**Purpose**: Generate synthetic documents and ground truth

**Functions**:

```python
def generate_document(
    topic: str,
    length: int,
    style: str = "academic"
) → str:
    """Generate synthetic document"""
    
def create_qa_pairs(
    document: str,
    num_pairs: int
) → List[Tuple[str, str]]:
    """Extract Q&A pairs from document"""
    
def inject_noise(
    document: str,
    noise_ratio: float
) → str:
    """Add irrelevant content"""
    
def generate_corpus(
    num_docs: int,
    topics: List[str]
) → Dict[str, str]:
    """Create document corpus"""
```

---

## Data Flow

### Experiment Execution Flow

```
1. Configuration Loading
   ├── Read YAML config files
   ├── Initialize parameters
   └── Set random seeds

2. Data Preparation
   ├── Generate or load documents
   ├── Create Q&A pairs
   └── Prepare ground truth

3. Experiment Loop
   For each condition:
   ├── Configure RAG pipeline
   ├── Process context
   ├── Query LLM
   ├── Measure metrics
   └── Store results

4. Analysis
   ├── Statistical tests
   ├── Effect size calculation
   └── Result summarization

5. Visualization
   ├── Generate plots
   ├── Save figures
   └── Create reports

6. Persistence
   ├── Save raw results (JSON)
   ├── Save processed data (CSV)
   └── Save visualizations (PNG)
```

### Data Transformation Pipeline

```
Raw Documents
     │
     ▼
┌────────────┐
│  Chunking  │
└─────┬──────┘
      │
      ▼
┌────────────┐
│ Embedding  │
└─────┬──────┘
      │
      ▼
┌────────────┐
│  Indexing  │
└─────┬──────┘
      │
      ▼
┌────────────┐
│  Retrieval │
└─────┬──────┘
      │
      ▼
┌────────────┐
│   Query    │
│    LLM     │
└─────┬──────┘
      │
      ▼
┌────────────┐
│ Evaluation │
└─────┬──────┘
      │
      ▼
Results Storage
```

---

## Module Specifications

### Configuration Module

**File**: `src/config.py`

**Purpose**: Centralized configuration management

```python
class Config:
    """Configuration manager using Singleton pattern"""
    
    _instance = None
    
    def __init__(self):
        self.experiments = self._load_yaml("config/experiments.yaml")
        self.models = self._load_yaml("config/models.yaml")
        self.paths = self._load_yaml("config/paths.yaml")
    
    @classmethod
    def get_instance(cls) → 'Config':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get_experiment_config(self, exp_id: str) → Dict:
        """Get configuration for specific experiment"""
        return self.experiments.get(exp_id, {})
```

**Configuration Files**:

1. **experiments.yaml**:
```yaml
experiment_1:
  name: "Position Effect"
  positions: ["START", "MIDDLE", "END"]
  num_runs: 10
  num_documents: 5
  
experiment_2:
  name: "Context Size Scaling"
  sizes: [2, 5, 10, 20, 50]
  num_runs: 10
```

2. **models.yaml**:
```yaml
ollama:
  base_url: "http://localhost:11434"
  default_model: "llama2:latest"
  timeout: 300
  max_retries: 3
```

3. **paths.yaml**:
```yaml
data:
  corpora: "data/corpora"
  ground_truth: "data/ground_truth"
  
results:
  raw: "results/raw"
  processed: "results/processed"
  figures: "results/figures"
```

---

## Experiment Pipeline

### Experiment 1: Position Effect

**Pipeline**:
```
1. Generate 5 documents
2. Create single Q&A pair
3. For each position (START, MIDDLE, END):
   - Insert answer-containing doc at position
   - Create context with all 5 docs
   - Query LLM
   - Record accuracy and latency
4. Repeat for 10 runs (30 queries total)
5. Statistical analysis (ANOVA)
6. Visualization (bar charts)
```

### Experiment 2: Context Size Scaling

**Pipeline**:
```
1. Generate 50 documents
2. For each size (2, 5, 10, 20, 50 docs):
   - Select random subset
   - Inject answer in random position
   - Query LLM
   - Record accuracy, latency, tokens
3. Repeat for 10 runs per size (50 queries total)
4. Correlation and regression analysis
5. Visualization (line plots, scatter)
```

### Experiment 3: Chunking Strategies

**Pipeline**:
```
1. Load document corpus
2. For each strategy (FIXED, SEMANTIC, SLIDING):
   For each chunk size (128, 256, 512, 1024):
     - Configure RAG pipeline
     - Process documents
     - Query with retrieval
     - Record accuracy and latency
3. Two-way ANOVA (strategy × size)
4. Visualization (heatmaps, grouped bars)
```

### Experiment 4: Context Management

**Pipeline**:
```
1. For each strategy (SELECT, COMPRESS, WRITE, HYBRID):
   For 10 conversation turns:
     - Apply context management
     - Query LLM
     - Update context
     - Record accuracy, tokens, latency
2. Repeat for 3 runs (120 queries total)
3. Repeated measures ANOVA
4. Visualization (trajectory plots)
```

---

## Configuration Management

### Environment Variables

```bash
# .env file
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2:latest
OLLAMA_TIMEOUT=300

LOG_LEVEL=INFO
LOG_FILE=logs/experiments.log

RANDOM_SEED=42
NUM_WORKERS=4
```

### Runtime Configuration

**Dynamic Override**:
```python
# Command-line override
python run_experiment_1_ollama.py \
    --model llama2:13b \
    --num-runs 20 \
    --output results/custom
```

**Programmatic Override**:
```python
config = Config.get_instance()
config.override("experiment_1.num_runs", 20)
```

---

## Storage and Persistence

### Directory Structure

```
project/
├── config/               # Configuration files
│   ├── experiments.yaml
│   ├── models.yaml
│   └── paths.yaml
│
├── data/
│   ├── corpora/         # Document collections
│   ├── ground_truth/    # Q&A pairs
│   └── processed/       # Intermediate data
│
├── results/
│   ├── raw/             # Raw JSON results
│   │   ├── experiment_1_ollama_latest.json
│   │   ├── experiment_2_ollama_latest.json
│   │   └── ...
│   ├── processed/       # Analyzed data (CSV)
│   └── figures/         # Visualizations (PNG)
│
├── logs/                # Execution logs
│   ├── experiments.log
│   └── debug.log
│
└── src/                 # Source code
    ├── __init__.py
    ├── llm_interface.py
    ├── rag_pipeline.py
    └── ...
```

### Result Schema

**JSON Format** (`results/raw/experiment_*.json`):

```json
{
  "experiment_id": "experiment_2",
  "timestamp": "2025-12-01T14:30:28.534204",
  "config": {
    "model": "llama2:latest",
    "num_runs": 10
  },
  "results": [
    {
      "run_id": 1,
      "condition": "size_10",
      "accuracy": 0.4,
      "latency": 16.44,
      "context_tokens": 2702,
      "query": "What is X?",
      "response": "X is...",
      "ground_truth": "X is..."
    }
  ],
  "summary_statistics": {
    "mean_accuracy": 0.58,
    "std_accuracy": 0.38
  }
}
```

---

## Deployment Architecture

### Local Development Setup

```
Developer Machine
├── Python 3.10+ Environment
├── Ollama Server (Local)
│   └── llama2:latest model
├── Project Dependencies
│   ├── numpy, pandas
│   ├── matplotlib, seaborn
│   ├── scipy, statsmodels
│   └── loguru, pyyaml
└── Storage
    ├── Code (Git)
    ├── Results (Local FS)
    └── Logs (Local FS)
```

### Production Deployment

```
Cloud Infrastructure (Optional)
├── Compute Instance
│   ├── GPU-enabled (T4/A100)
│   ├── 16GB+ RAM
│   └── 100GB+ Storage
│
├── Ollama Server
│   ├── Model Cache
│   └── API Endpoint
│
├── Application
│   ├── Experiment Scripts
│   ├── Analysis Pipeline
│   └── Visualization Generator
│
└── Storage
    ├── Object Store (S3/Azure Blob)
    │   ├── Raw Results
    │   └── Visualizations
    └── Database (Optional)
        └── Experiment Metadata
```

### Scaling Considerations

**Horizontal Scaling**:
- Multiple Ollama instances behind load balancer
- Parallel experiment execution (multiprocessing)
- Distributed computation (Ray, Dask)

**Vertical Scaling**:
- Larger GPU for faster inference
- More RAM for larger contexts
- SSD for faster I/O

---

## Security and Reliability

### Error Handling

**Layered Approach**:

1. **LLM Interface**: Retry logic, timeouts, fallbacks
2. **RAG Pipeline**: Graceful degradation, empty result handling
3. **Experiments**: Transaction-like saves, partial result recovery
4. **Logging**: Comprehensive error tracking

**Example**:
```python
try:
    response = llm.query(context, query)
except Timeout Exception:
    logger.error("Query timeout, retrying...")
    response = llm.query(context, query, timeout=600)
except ConnectionError:
    logger.critical("Ollama server unreachable")
    raise
```

### Logging Strategy

**Log Levels**:
- **DEBUG**: Detailed execution traces
- **INFO**: High-level progress
- **WARNING**: Recoverable errors
- **ERROR**: Failed operations
- **CRITICAL**: System failures

**Log Rotation**:
```python
logger.add(
    "logs/experiments.log",
    rotation="100 MB",
    retention="30 days",
    compression="zip"
)
```

---

## Performance Optimization

### Caching

**LLM Response Cache**:
```python
@lru_cache(maxsize=1000)
def query_cached(context_hash, query_hash):
    return llm.query(context, query)
```

**Embedding Cache**:
```python
embedding_cache = {}
def get_embedding(text):
    key = hash(text)
    if key not in embedding_cache:
        embedding_cache[key] = llm.embed(text)
    return embedding_cache[key]
```

### Batch Processing

**Parallel Queries** (when independent):
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(llm.query, ctx, q)
        for ctx, q in zip(contexts, queries)
    ]
    results = [f.result() for f in futures]
```

### Memory Management

**Chunked Processing**:
```python
def process_large_corpus(corpus, batch_size=100):
    for i in range(0, len(corpus), batch_size):
        batch = corpus[i:i+batch_size]
        process_batch(batch)
        gc.collect()  # Force garbage collection
```

---

## Testing Architecture

### Test Suite Overview

**Total Tests**: 181 (all passing)
**Overall Coverage**: 93.69%
**Target**: >85% per module (achieved)

### Test Structure

**Actual Structure**:
```
tests/
├── test_config.py              # 22 tests - Configuration management
├── test_data_generation.py     # 21 tests - Document and fact generation
├── test_experiments.py         # 13 tests - Experiment dataclasses
├── test_llm_interface.py       # 29 tests - LLM interface & retry logic
├── test_metrics.py             # 13 tests - Accuracy, precision, recall
├── test_rag_pipeline.py        # 33 tests - RAG pipeline & vector store
├── test_statistics.py          # 9 tests - Statistical analysis
└── test_visualization.py       # 41 tests - All visualization functions
```

### Coverage by Module

| Module | Coverage | Statements | Missing | Status |
|--------|----------|------------|---------|--------|
| data_generation.py | 100.00% | 59 | 0 | ✓ |
| llm_interface.py | 99.03% | 103 | 1 | ✓ |
| visualization.py | 99.01% | 202 | 2 | ✓ |
| config.py | 90.36% | 83 | 8 | ✓ |
| metrics.py | 89.29% | 56 | 6 | ✓ |
| statistics.py | 87.34% | 79 | 10 | ✓ |
| rag_pipeline.py | 85.37% | 123 | 18 | ✓ |
| **Overall** | **93.69%** | **713** | **45** | **✓** |

### Test Configuration

**pytest.ini**:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --strict-markers
```

**.coveragerc**:
```ini
[run]
source = src
omit = */experiments/*, */tests/*, */__pycache__/*

[report]
exclude_lines = pragma: no cover, def __repr__
precision = 2
```

### Test Categories

**Unit Tests** (168 tests):
- Mock external dependencies (Ollama API, file I/O)
- Test individual functions and methods
- Cover edge cases and error handling
- Validate input/output contracts

**Integration Tests** (13 tests):
- Test experiment dataclasses
- Validate end-to-end workflows
- Verify file save/load operations

### Example Test Patterns

**Mocking External APIs**:
```python
@patch('requests.post')
def test_query_with_retry(mock_post, ollama_interface):
    mock_post.side_effect = requests.Timeout("Timeout")
    with pytest.raises(RuntimeError, match="Query timeout after"):
        ollama_interface.query("Context", "Question")
    assert mock_post.call_count == ollama_interface.max_retries
```

**Testing RAG Pipeline**:
```python
def test_rag_retrieval():
    pipeline = RAGPipeline(chunk_size=10, top_k=3)
    documents = ["Doc 1 content", "Doc 2 content"]
    pipeline.index_documents(documents)

    results = pipeline.retrieve("query")
    assert len(results) <= 3
    assert all('score' in r for r in results)
```

**Visualization Testing**:
```python
def test_plot_generation(temp_output_dir):
    viz = Visualizer(output_dir=temp_output_dir)
    data = {"accuracy": [0.9, 0.85, 0.8]}

    viz.plot_line_chart(data, save_path="test.png")
    assert (Path(temp_output_dir) / "test.png").exists()
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html --cov-config=.coveragerc

# Run specific test file
pytest tests/test_llm_interface.py -v

# Run specific test
pytest tests/test_llm_interface.py::TestOllamaInterface::test_query_with_retry -v

# View coverage report
start htmlcov/index.html  # Windows
open htmlcov/index.html   # Mac/Linux
```

### Integration Tests

**End-to-End**:
```python
def test_experiment_1_pipeline():
    # Run full experiment
    results = run_experiment_1(num_runs=2)
    
    # Verify results
    assert len(results) == 6  # 3 positions × 2 runs
    assert all(r["accuracy"] in [0.0, 1.0] for r in results)
    
    # Verify outputs
    assert os.path.exists("results/raw/experiment_1_latest.json")
    assert os.path.exists("results/figures/exp1_accuracy_by_position.png")
```

---

## Future Enhancements

### Planned Improvements

1. **Multi-Model Support**: GPT-4, Claude, Gemini integration
2. **Real-Time Dashboard**: Web UI for experiment monitoring
3. **Distributed Execution**: Kubernetes deployment for scale
4. **Advanced RAG**: Hybrid search, reranking, query expansion
5. **A/B Testing Framework**: Compare strategies systematically

### Extension Points

**Custom Chunking Strategy**:
```python
class CustomChunkingStrategy(ChunkingStrategy):
    def chunk(self, text: str) → List[str]:
        # Implement custom logic
        return chunks
```

**Custom Metrics**:
```python
class F1ScoreMetric(Metric):
    def evaluate(self, pred: str, truth: str) → float:
        # Calculate F1 score
        return f1_score
```

---

## Appendix

### Glossary

- **RAG**: Retrieval-Augmented Generation
- **Chunking**: Splitting documents into smaller units
- **Embedding**: Vector representation of text
- **Context Window**: Maximum input length for LLM
- **Top-K**: Selecting top K relevant chunks
- **ANOVA**: Analysis of Variance (statistical test)

### References

1. Ollama API Documentation: https://ollama.ai/docs
2. RAG Best Practices: https://www.pinecone.io/learn/rag/
3. Statistical Testing in Python: SciPy documentation
4. Matplotlib Gallery: https://matplotlib.org/stable/gallery/

---

**Document Version**: 1.0  
**Last Updated**: December 1, 2025  
**Maintained By**: Architecture Team  
**Status**: Complete

*End of Architecture Documentation*
