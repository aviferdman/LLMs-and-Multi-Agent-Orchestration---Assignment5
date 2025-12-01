# User Guide

**Project**: Context Windows in Practice - Research Framework  
**Version**: 1.0  
**Date**: December 1, 2025  
**Status**: Complete

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Installation](#installation)
3. [Running Experiments](#running-experiments)
4. [Generating Visualizations](#generating-visualizations)
5. [Interpreting Results](#interpreting-results)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)
8. [FAQ](#faq)

---

## Getting Started

### What is This Project?

This framework enables controlled experiments on Large Language Model (LLM) context window behavior. It helps answer questions like:
- How does information position affect accuracy?
- When do LLMs start to struggle with large contexts?
- Which chunking strategy works best for RAG?
- How should we manage context in long conversations?

### Prerequisites

**Required:**
- Python 3.10 or higher
- Ollama installed and running
- 8GB+ RAM (16GB recommended)
- 10GB+ disk space

**Optional:**
- GPU for faster inference
- Virtual environment tool (venv, conda)

### Quick Test

Verify your setup:

```bash
# Check Python version
python --version  # Should be 3.10+

# Check Ollama
ollama list  # Should show available models

# Test Ollama API
curl http://localhost:11434/api/tags
```

---

## Installation

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd LLMs-and-Multi-Agent-Orchestration---Assignment5
```

### Step 2: Set Up Python Environment

**Option A: Using venv (Recommended)**
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

**Option B: Using conda**
```bash
conda create -n context-windows python=3.10
conda activate context-windows
```

### Step 3: Install Dependencies

```bash
# Install requirements
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Step 4: Install Ollama Model

```bash
# Pull Llama 2 model (default)
ollama pull llama2:latest

# Optional: Pull other models
ollama pull llama2:13b
ollama pull llama3:latest
```

### Step 5: Verify Installation

```bash
# Run connection test
python test_ollama_connection.py
```

Expected output:
```
✅ Ollama server connection verified
✅ Model llama2:latest available
✅ Test query successful
```

---

## Running Experiments

### Experiment 1: Position Effect

**Question**: Does information position affect accuracy?

**Run Experiment:**
```bash
python scripts/run_experiment_1_ollama.py
```

**What It Does:**
- Creates 5 documents
- Tests START, MIDDLE, END positions
- Runs 10 trials per position (30 total queries)
- Takes ~5-10 minutes

**Output:**
```
results/raw/experiment_1_ollama_latest.json
results/figures/exp1_accuracy_by_position.png
results/figures/exp1_latency_by_position.png
```

**Expected Results:**
- 100% accuracy across all positions (at 5 docs)
- Slight latency difference (~1.5s between START and END)

### Experiment 2: Context Size Scaling

**Question**: How does context size affect accuracy?

**Run Experiment:**
```bash
python scripts/run_experiment_2_ollama.py
```

**What It Does:**
- Tests 2, 5, 10, 20, 50 documents
- Measures accuracy degradation
- Takes ~15-25 minutes

**Output:**
```
results/raw/experiment_2_ollama_latest.json
results/figures/exp2_accuracy_vs_size.png
results/figures/exp2_latency_scaling.png
results/figures/exp2_tokens_distribution.png
```

**Expected Results:**
- 100% accuracy at 2-5 documents
- **Performance cliff at 10 documents** (drops to 40%)
- Further degradation at 20+ documents

### Experiment 3: Chunking Strategies

**Question**: Which chunking strategy works best?

**Run Experiment:**
```bash
python scripts/run_experiment_3_ollama.py
```

**What It Does:**
- Tests FIXED, SEMANTIC, SLIDING strategies
- Tests 128, 256, 512, 1024 token chunks
- Takes ~30-45 minutes

**Output:**
```
results/raw/experiment_3_ollama_latest.json
results/figures/exp3_strategy_comparison.png
results/figures/exp3_heatmap.png
```

**Expected Results:**
- SEMANTIC chunking performs best (85% accuracy)
- Optimal chunk size: 256-512 tokens
- FIXED and SLIDING strategies: 78-81% accuracy

### Experiment 4: Context Management

**Question**: Which strategy maintains accuracy over conversation turns?

**Run Experiment:**
```bash
python scripts/run_experiment_4_ollama.py
```

**What It Does:**
- Tests SELECT, COMPRESS, WRITE, HYBRID strategies
- Simulates 10-turn conversation
- Takes ~20-30 minutes

**Output:**
```
results/raw/experiment_4_ollama_latest.json
results/figures/exp4_strategy_comparison.png
results/figures/exp4_accuracy_vs_context_size.png
```

**Expected Results:**
- COMPRESS strategy achieves best balance (23% accuracy)
- SELECT/HYBRID most token-efficient (31 tokens average)
- All strategies remain stable across turns

### Run All Experiments

```bash
# Sequential execution
python scripts/run_experiment_1_ollama.py
python scripts/run_experiment_2_ollama.py
python scripts/run_experiment_3_ollama.py
python scripts/run_experiment_4_ollama.py

# Total time: ~1.5-2 hours
```

---

## Generating Visualizations

### Regenerate All Plots

If you modify data or want to regenerate visualizations:

```bash
# Experiment 1
python scripts/generate_exp1_visualizations.py

# Experiment 2
python scripts/regenerate_exp2_visualizations.py

# Experiment 3
python scripts/generate_exp3_visualizations.py

# Experiment 4
python scripts/generate_exp4_visualizations.py
```

### Visualization Outputs

All figures saved to `results/figures/` at 300 DPI (publication quality):

**Experiment 1:**
- `exp1_accuracy_by_position.png` - Bar chart of accuracy
- `exp1_latency_by_position.png` - Bar chart of latency

**Experiment 2:**
- `exp2_accuracy_vs_size.png` - Line plot showing cliff
- `exp2_latency_scaling.png` - Latency vs size
- `exp2_tokens_distribution.png` - Token count distribution

**Experiment 3:**
- `exp3_strategy_comparison.png` - Strategy bar chart
- `exp3_heatmap.png` - Strategy × Size heatmap
- `exp3_chunk_size_analysis.png` - Optimal size analysis

**Experiment 4:**
- `exp4_strategy_comparison.png` - Strategy bar chart
- `exp4_latency_by_strategy.png` - Latency comparison
- `exp4_context_size_by_strategy.png` - Token efficiency
- `exp4_accuracy_vs_context_size.png` - Accuracy-efficiency tradeoff

---

## Interpreting Results

### Reading the JSON Results

Results are saved in `results/raw/experiment_*_ollama_latest.json`:

```json
{
  "experiment_id": "experiment_2",
  "timestamp": "2025-12-01T14:30:28",
  "config": {
    "model": "llama2:latest",
    "num_runs": 10
  },
  "results": [
    {
      "condition": "size_10",
      "accuracy": 0.4,
      "latency": 16.44,
      "context_tokens": 2702
    }
  ],
  "summary_statistics": {
    "mean_accuracy": 0.58,
    "std_accuracy": 0.38
  }
}
```

### Key Metrics

**Accuracy:**
- 1.0 = 100% (perfect answer)
- 0.0 = 0% (incorrect answer)
- Binary: exact match with ground truth

**Latency:**
- Time in seconds to generate response
- Includes model loading + inference
- Typical range: 5-20s for Llama 2

**Context Tokens:**
- Approximate token count
- Calculated using ~4 chars per token heuristic
- Important for understanding context limits

### Statistical Significance

**p-values:**
- p < 0.05: Statistically significant (*)
- p < 0.01: Highly significant (**)
- p < 0.001: Very highly significant (***)

**Effect Sizes (Cohen's d):**
- 0.2 = Small effect
- 0.5 = Medium effect
- 0.8 = Large effect

**Example from Experiment 2:**
- 5 vs 10 documents: d = 1.96 (very large effect)
- This means the performance cliff is real and substantial

---

## Troubleshooting

### Common Issues

#### 1. "Cannot connect to Ollama"

**Symptoms:**
```
ConnectionError: Cannot connect to Ollama at http://localhost:11434
```

**Solutions:**
```bash
# Check if Ollama is running
ollama list

# Start Ollama (if not running)
ollama serve

# Verify API is accessible
curl http://localhost:11434/api/tags
```

#### 2. "Model not found"

**Symptoms:**
```
Error: model 'llama2:latest' not found
```

**Solutions:**
```bash
# Pull the model
ollama pull llama2:latest

# Verify model is available
ollama list
```

#### 3. "Timeout errors"

**Symptoms:**
```
RuntimeError: Query timeout after 3 attempts
```

**Solutions:**
```python
# Increase timeout in code
llm = OllamaInterface(
    timeout=600,  # 10 minutes instead of 5
    max_retries=5
)
```

#### 4. "Out of memory"

**Symptoms:**
- System freezes
- Python crashes
- Ollama crashes

**Solutions:**
```bash
# Use smaller model
ollama pull llama2:7b  # Instead of 13b

# Reduce batch size in experiments
# Edit scripts to process fewer documents at once

# Close other applications
# Free up RAM before running
```

#### 5. "Slow performance"

**Symptoms:**
- Each query takes 30+ seconds
- Experiments take hours

**Solutions:**
```bash
# Check CPU usage
# Ollama uses CPU if no GPU

# Use GPU if available
# Ollama automatically uses GPU when present

# Reduce num_runs in experiments
# Edit scripts to reduce iterations for testing
```

### Debugging Tips

**Enable Debug Logging:**
```python
from loguru import logger

logger.add("debug.log", level="DEBUG")
```

**Check Results:**
```bash
# View latest results
python scripts/check_visualization_data.py

# Validate all experiments
python scripts/validate_all_visualizations.py
```

**Test Connection:**
```python
from src.llm_interface import OllamaInterface

try:
    llm = OllamaInterface()
    response = llm.query("Test", "Is this working?")
    print(f"✅ Success: {response.text}")
except Exception as e:
    print(f"❌ Error: {e}")
```

---

## Advanced Usage

### Custom Experiments

Create your own experiment:

```python
from src.llm_interface import OllamaInterface
from src.metrics import MetricsEvaluator
import json

# Initialize
llm = OllamaInterface(model="llama2:latest")
results = []

# Your experimental conditions
conditions = ["condition_a", "condition_b"]

for condition in conditions:
    for trial in range(10):
        # Create context
        context = f"Context for {condition}"
        
        # Query
        response = llm.query(context, "Your question?")
        
        # Evaluate
        accuracy = MetricsEvaluator.exact_match_accuracy(
            response.text,
            "Expected answer"
        )
        
        # Store
        results.append({
            "condition": condition,
            "trial": trial,
            "accuracy": accuracy,
            "latency": response.latency
        })

# Save
with open("results/raw/custom_experiment.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Batch Processing

Process multiple queries efficiently:

```python
from concurrent.futures import ThreadPoolExecutor
from src.llm_interface import OllamaInterface

llm = OllamaInterface()

# Prepare queries
queries = [
    ("Context 1", "Question 1"),
    ("Context 2", "Question 2"),
    # ... more queries
]

# Parallel execution (use with caution - may overload system)
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [
        executor.submit(llm.query, ctx, q)
        for ctx, q in queries
    ]
    
    results = [f.result() for f in futures]
```

### Custom Visualizations

Create custom plots:

```python
from src.visualization import VisualizationGenerator
import json

# Load your data
with open("results/raw/custom_experiment.json") as f:
    data = json.load(f)

# Extract values
condition_a = [r["accuracy"] for r in data if r["condition"] == "condition_a"]
condition_b = [r["accuracy"] for r in data if r["condition"] == "condition_b"]

# Create plot
VisualizationGenerator.plot_bar_chart(
    data=[np.mean(condition_a), np.mean(condition_b)],
    labels=["Condition A", "Condition B"],
    title="Custom Experiment Results",
    xlabel="Condition",
    ylabel="Mean Accuracy",
    output_path="results/figures/custom_plot.png",
    errors=[np.std(condition_a), np.std(condition_b)]
)
```

### Configuration Customization

Modify experiment parameters in `config/experiments.yaml`:

```yaml
experiment_1:
  name: "Position Effect"
  positions: ["START", "MIDDLE", "END"]
  num_runs: 20  # Increase for more data
  num_documents: 10  # Test with more documents
  
experiment_2:
  name: "Context Size Scaling"
  sizes: [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]  # Finer granularity
  num_runs: 15
```

---

## FAQ

### General Questions

**Q: How long do experiments take?**
A: 
- Experiment 1: ~10 minutes
- Experiment 2: ~20 minutes
- Experiment 3: ~45 minutes
- Experiment 4: ~30 minutes
- Total: ~2 hours for all

**Q: Can I use a different LLM model?**
A: Yes! Change the model in scripts:
```python
llm = OllamaInterface(model="llama3:latest")
# or
llm = OllamaInterface(model="llama2:13b")
```

**Q: How much does it cost?**
A: Free! Ollama runs locally. No API costs.

**Q: Can I run this on CPU only?**
A: Yes, but it will be slower. GPU is recommended for faster inference.

**Q: What if I don't have enough RAM?**
A: Use a smaller model (llama2:7b) or reduce num_runs in experiments.

### Technical Questions

**Q: Why are my results different?**
A: LLMs are stochastic. Use lower temperature (0.0-0.1) for more reproducible results.

**Q: Can I compare different models?**
A: Yes! Run experiments with different models and compare results:
```python
models = ["llama2:latest", "llama3:latest"]
for model in models:
    llm = OllamaInterface(model=model)
    # Run experiment...
```

**Q: How do I export results to CSV?**
A:
```python
import json
import pandas as pd

with open("results/raw/experiment_1_ollama_latest.json") as f:
    data = json.load(f)

df = pd.DataFrame(data["results"])
df.to_csv("results/processed/experiment_1.csv", index=False)
```

**Q: Can I run experiments in parallel?**
A: Not recommended. Ollama processes one request at a time efficiently. Parallel requests may cause resource contention.

**Q: How do I cite this work?**
A: See the README.md for citation information.

### Experiment-Specific Questions

**Q: Why is accuracy 100% in Experiment 1?**
A: At 5 documents (~1,500 tokens), Llama 2 can attend to all positions perfectly. The "lost in the middle" phenomenon only appears at larger context sizes.

**Q: What causes the performance cliff in Experiment 2?**
A: Llama 2's effective context limit. Beyond ~2,500 tokens, attention degrades significantly.

**Q: Why is SEMANTIC chunking best in Experiment 3?**
A: It preserves semantic boundaries, maintaining meaning and coherence better than fixed-size chunks.

**Q: Why is COMPRESS strategy not the most token-efficient in Experiment 4?**
A: COMPRESS stores more information (42.5 tokens) but achieves higher accuracy (23%). SELECT is more efficient (31 tokens) but less accurate (10%).

---

## Additional Resources

### Documentation

- **API Reference**: `docs/API.md`
- **Architecture**: `docs/ARCHITECTURE.md`
- **Research Methodology**: `docs/RESEARCH_METHODOLOGY.md`
- **Statistical Analysis**: `docs/STATISTICAL_ANALYSIS.md`

### External Links

- **Ollama Docs**: https://ollama.ai/docs
- **Llama 2 Paper**: https://arxiv.org/abs/2307.09288
- **RAG Guide**: https://www.pinecone.io/learn/retrieval-augmented-generation/

### Support

For issues or questions:
1. Check this guide's [Troubleshooting](#troubleshooting) section
2. Review documentation in `docs/` folder
3. Check Ollama documentation
4. Open an issue on the repository

---

## Quick Reference

### Essential Commands

```bash
# Setup
pip install -r requirements.txt
pip install -e .
ollama pull llama2:latest

# Test
python test_ollama_connection.py

# Run Experiments
python scripts/run_experiment_1_ollama.py
python scripts/run_experiment_2_ollama.py
python scripts/run_experiment_3_ollama.py
python scripts/run_experiment_4_ollama.py

# Regenerate Plots
python scripts/generate_exp1_visualizations.py
python scripts/regenerate_exp2_visualizations.py
python scripts/generate_exp3_visualizations.py
python scripts/generate_exp4_visualizations.py

# Verify Results
python scripts/validate_all_visualizations.py
```

### File Locations

```
Key Files:
├── config/experiments.yaml         # Experiment configuration
├── results/raw/*.json              # Raw experiment data
├── results/figures/*.png           # Visualizations
├── scripts/run_experiment_*.py     # Experiment runners
├── src/llm_interface.py           # LLM interface
└── src/rag_pipeline.py            # RAG implementation
```

---

**Document Version**: 1.0  
**Last Updated**: December 1, 2025  
**Maintained By**: Documentation Team  
**Status**: Complete

*End of User Guide*
