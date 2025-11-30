# Experiment 2: Context Size Impact

## Overview

Experiment 2 investigates how accuracy and latency scale as context window size increases. This experiment measures performance degradation and identifies optimal context size ranges for balancing accuracy and efficiency.

## Research Questions

The experiment addresses four key research questions:

- **RQ2.1**: What is the functional form of accuracy degradation as context size increases?
- **RQ2.2**: Is there a "performance cliff" where accuracy drops sharply?
- **RQ2.3**: How does latency scale with context size?
- **RQ2.4**: What is the optimal context size for achieving 90% accuracy with minimum latency?

## Implementation

### Core Components

The experiment is implemented in `src/experiments/experiment_2.py` with the following main components:

1. **Context Generation** (`generate_context`)
   - Generates multiple documents with varying sizes
   - Embeds target facts within documents
   - Returns context string, target answer, and document index

2. **Single Trial Execution** (`run_single_trial`)
   - Tests LLM with a specific number of documents
   - Measures accuracy and latency
   - Tracks token counts and semantic similarity

3. **Context Size Sweep** (`run_context_size_sweep`)
   - Tests multiple context sizes: [2, 5, 10, 20, 50] documents
   - Runs multiple trials per size (configurable, default: 3-5)
   - Collects comprehensive performance metrics

4. **Task Complexity Comparison** (`run_task_complexity_comparison`)
   - Compares simple vs. complex query performance
   - Simple task: "What is the CEO's name?"
   - Complex task: "Based on all the information provided, what is the CEO's name and how does this information relate to the overall context?"

### Statistical Analysis

The `analyze_results` method performs:

- **Summary Statistics**: Mean and standard deviation for accuracy and latency by context size
- **Confidence Intervals**: 95% CI for accuracy at each context size
- **Correlation Analysis**:
  - Pearson correlation between accuracy and context size
  - Pearson correlation between latency and context size
- **Regression Analysis**:
  - Linear regression: accuracy ~ log(context_size)
  - Linear regression: latency ~ context_size
  - Quadratic regression: latency ~ context_size^2

### Visualizations

The experiment generates publication-quality figures (300 DPI):

1. **Accuracy vs. Context Size** - Scatter plot with fitted logarithmic curve
2. **Latency vs. Context Size** - Scaling curve showing quadratic relationship
3. **Accuracy Distribution** - Box plots by context size
4. **Task Complexity Comparison** - Grouped bar chart (if enabled)

## Configuration

Experiment parameters are defined in `config/experiments.yaml`:

```yaml
experiment_2:
  name: "Context Size Impact"
  enabled: true
  context_sizes: [2, 5, 10, 20, 50]
  document_length: 200
  task_types: ["simple", "complex"]
  num_trials_per_size: 5
```

## Running the Experiment

### Basic Usage

```python
from src.experiments.experiment_2 import Experiment2
from src.config import Config

# Initialize
config = Config()
experiment = Experiment2(config=config)

# Run experiment
results = experiment.run(include_task_complexity=False)
```

### With Task Complexity Analysis

```python
# Run with task complexity comparison
results = experiment.run(include_task_complexity=True)
```

### Command Line

```bash
# Run directly
python -m src.experiments.experiment_2

# Or use the test script
python scripts/test_experiment_2.py
```

## Output Files

The experiment produces the following outputs in the `results/` directory:

### Data Files
- `exp2_raw_results.json` - Complete trial-level data
- `exp2_results.csv` - Tabular format for analysis
- `exp2_analysis.json` - Statistical analysis results
- `exp2_summary.csv` - Aggregated statistics by context size

### Visualizations (in `results/figures/`)
- `exp2_accuracy_vs_size.png` - Accuracy degradation curve
- `exp2_latency_vs_size.png` - Latency scaling curve
- `exp2_accuracy_distribution.png` - Accuracy box plots
- `exp2_task_complexity.png` - Task comparison (if enabled)

## Results Interpretation

### Key Findings Format

The experiment generates key findings including:

```json
{
  "accuracy_correlation": "Accuracy vs context size correlation: r=-0.XXX, p=0.XXXX",
  "accuracy_model": "Accuracy = -0.XXX*log(size) + 0.XXX, R^2=0.XXX",
  "latency_scaling": "Latency scales quadratically: 0.XXXX*x^2 + 0.XXXX*x + 0.XXXX"
}
```

### Expected Patterns

Based on research literature, we expect:

1. **Accuracy**: Logarithmic decay with increasing context size
2. **Latency**: Quadratic scaling (O(n²) for attention mechanism)
3. **Optimal Range**: Typically 5-15 documents for balanced performance
4. **Performance Cliff**: Possible sharp drop beyond model's effective context window

## Dependencies

- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `scipy` - Statistical analysis
- `matplotlib` - Visualization
- `seaborn` - Statistical plotting
- `loguru` - Logging

## Testing

Run the test suite to verify implementation:

```bash
python scripts/test_experiment_2.py
```

Expected output:
```
✓ Experiment 2 initialized successfully!
✓ Context generation successful!
✓ All tests passed! Experiment 2 is ready to run.
```

## Implementation Status

✅ **COMPLETE** - All features implemented and tested

- [x] Context generation with variable sizes
- [x] Single trial execution with metrics collection
- [x] Context size sweep across multiple sizes
- [x] Task complexity comparison
- [x] Comprehensive statistical analysis
- [x] Publication-quality visualizations
- [x] Results persistence (JSON/CSV)
- [x] Complete documentation

## Related Experiments

- **Experiment 1**: Lost in the Middle - Tests position bias within fixed context
- **Experiment 3**: RAG Impact - Compares retrieval vs. full context approaches
- **Experiment 4**: Context Engineering - Tests management strategies for growing context

## References

- Liu et al. (2023). "Lost in the Middle: How Language Models Use Long Contexts"
- Anthropic (2023). Claude 2 Technical Documentation
- Vaswani et al. (2017). "Attention Is All You Need" - Quadratic complexity of attention

## Contact

For questions or issues related to Experiment 2, please refer to the main project README.
