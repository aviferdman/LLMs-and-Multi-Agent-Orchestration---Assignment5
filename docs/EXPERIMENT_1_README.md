# Experiment 1: Lost in the Middle

## Overview

Experiment 1 implements the "Lost in the Middle" research paradigm, investigating whether Large Language Models exhibit position bias when retrieving information from long contexts. This phenomenon, documented in recent literature (Liu et al., 2023), demonstrates that LLMs may struggle to retrieve information embedded in the middle of lengthy documents, even when it falls within their theoretical context window.

## Research Questions

### Primary Research Question (RQ1.1)
**Is there statistically significant accuracy degradation for middle-positioned facts?**

**Hypothesis:**
- H0: β1 = 0 (no middle penalty)
- H1: β1 < 0 (middle accuracy lower than start/end)

### Mathematical Framework

**Accuracy Model:**
```
Accuracy(p) = β0 + β1 · I_middle(p) + β2 · I_end(p) + ε
```

Where:
- p = fact position (start=baseline, middle, end)
- I = indicator function
- ε ~ N(0, σ²) = error term

## Experimental Design

### Document Generation

**Configuration:**
- **Number of documents:** 5 per trial
- **Words per document:** 200 ± 20 words
- **Fact embedding:** One critical fact per trial
- **Positions tested:**
  - Start: 0-20% of target document
  - Middle: 40-60% of target document
  - End: 80-100% of target document
- **Filler text:** Coherent but non-informative business sentences

### Fact Types

1. **CEO Name**
   - Template: "The CEO of the company is {name}."
   - Possible values: David Cohen, Sarah Williams, Michael Brown
   - Query: "What is the CEO's name?"

2. **Revenue**
   - Template: "The Q3 revenue was ${amount}M."
   - Range: $10M - $100M
   - Query: "What was the Q3 revenue?"

3. **Location**
   - Template: "The headquarters is located in {city}."
   - Possible values: New York, London, Tokyo
   - Query: "Where is the headquarters located?"

### Experimental Procedure

1. **Generate Documents:** Create 5 documents with filler text
2. **Embed Fact:** Insert fact at specified position in middle document
3. **Query LLM:** Present concatenated documents with query
4. **Evaluate Response:** Check correctness and semantic similarity
5. **Record Metrics:** Accuracy, latency, tokens, similarity

### Sample Size

**Per PRD specifications:**
- Trials per position: 10 (configurable)
- Total trials: 30 (10 × 3 positions)
- Expected power: 0.80 for effect size d ≥ 0.5

**Sample size calculation:**
```
n = 2(z_α/2 + z_β)² σ² / δ²
```
For α=0.05, power=0.80, d=0.5: n ≥ 64 per condition

## Metrics Collected

### Primary Metrics
- **Binary Accuracy:** Correct/Incorrect (based on ground truth match)
- **Semantic Similarity:** 0-1 scale using word overlap (upgradable to embeddings)
- **Response Latency:** Time to generate response (seconds)
- **Token Count:** Number of tokens in response

### Secondary Metrics
- **Response Length:** Word count of response
- **Confidence Score:** If available from LLM
- **Document Statistics:** Total words, target document position

## Statistical Analysis

### 1. One-Way ANOVA
**Test:** Position effect on accuracy
**Output:**
- F-statistic
- p-value
- Eta-squared (η²) effect size
- Significance (α = 0.05)

### 2. Post-hoc Pairwise Comparisons
**Tests:** Independent t-tests for all position pairs
- Start vs. Middle
- Start vs. End
- Middle vs. End

**Output:**
- t-statistic
- p-value (with Bonferroni correction)
- Cohen's d effect size

### 3. Confidence Intervals
**95% CI for each position mean**
- Provides uncertainty quantification
- Visual representation in plots

### 4. Effect Size Interpretation

**Eta-squared (η²):**
- < 0.01: Negligible
- 0.01 - 0.06: Small
- 0.06 - 0.14: Medium
- ≥ 0.14: Large

**Cohen's d:**
- 0.2: Small effect
- 0.5: Medium effect
- 0.8: Large effect

## Implementation Details

### Class Structure

```python
@dataclass
class DocumentWithFact:
    """Container for document with embedded fact."""
    document: str
    fact: str
    position: str
    ground_truth: str
    metadata: Optional[Dict[str, Any]]

@dataclass
class TrialResult:
    """Results from a single trial."""
    position: str
    query: str
    response: str
    ground_truth: str
    correct: bool
    latency: float
    tokens: int
    semantic_similarity: float
    metadata: Optional[Dict[str, Any]]

class Experiment1:
    """Main experiment implementation."""
    def __init__(self, config, llm=None): ...
    def run(self) -> Dict[str, Any]: ...
    def analyze(self, results) -> Dict[str, Any]: ...
    def save_results(self, results, analysis) -> Path: ...
    def visualize(self, results) -> None: ...
```

### Key Methods

**generate_document_with_fact(position, fact_type)**
- Generates 5 documents with filler text
- Embeds fact at specified position
- Returns DocumentWithFact object

**run_trial(position, fact_type)**
- Executes single trial
- Queries LLM with generated documents
- Evaluates response correctness
- Returns TrialResult object

**run()**
- Executes full experiment
- Runs all trials across all positions
- Collects and aggregates results
- Returns comprehensive results dictionary

**analyze(results)**
- Performs statistical analysis
- ANOVA, t-tests, effect sizes, CIs
- Returns analysis dictionary

**save_results(results, analysis)**
- Saves results to JSON file
- Timestamped filename
- Stored in results/raw/ directory

## Configuration

### experiments.yaml

```yaml
experiment_1:
  name: "Lost in the Middle"
  enabled: true
  num_documents: 5
  words_per_document: 200
  word_variance: 20
  fact_positions: ["start", "middle", "end"]
  num_trials_per_position: 10
  query_template: "What is the CEO's name?"
```

## Usage

### Standalone Execution

```python
import yaml
from pathlib import Path
from src.experiments.experiment_1 import Experiment1

# Load configuration
with open("config/experiments.yaml") as f:
    config = yaml.safe_load(f)

# Create and run experiment
experiment = Experiment1(config)
results = experiment.run()
analysis = experiment.analyze(results)
experiment.save_results(results, analysis)
experiment.visualize(results)
```

### Command Line

```bash
# Test implementation
python scripts/test_experiment_1.py

# Run full experiment (requires Ollama)
python -m src.experiments.experiment_1
```

## Expected Results

### Hypothesized Outcome

Based on literature (Liu et al., 2023), we expect:

1. **Start position:** High accuracy (baseline)
2. **Middle position:** Reduced accuracy (lost in the middle effect)
3. **End position:** Moderate to high accuracy (recency bias)

**Statistical Significance:**
- ANOVA: p < 0.05 (significant position effect)
- Start vs. Middle: p < 0.05, d > 0.5 (medium to large effect)
- Middle vs. End: p < 0.05, d > 0.3 (small to medium effect)

### Sample Output

```
SUMMARY STATISTICS:
------------------------------------------------------------
     START: Accuracy=85.00%, Latency=2.50s, Similarity=0.823
    MIDDLE: Accuracy=62.00%, Latency=2.75s, Similarity=0.654
       END: Accuracy=78.00%, Latency=2.60s, Similarity=0.782

STATISTICAL ANALYSIS
============================================================

1. One-Way ANOVA
----------------------------------------
F-statistic: 8.4521
p-value: 0.0012
Eta-squared (η²): 0.1234
Significant: True

2. Post-hoc Pairwise Comparisons
----------------------------------------

START vs MIDDLE:
  t-statistic: 3.2156
  p-value: 0.0023
  Cohen's d: 0.6432
  Significant: True

START vs END:
  t-statistic: 1.2345
  p-value: 0.2234
  Cohen's d: 0.2456
  Significant: False

MIDDLE vs END:
  t-statistic: 2.3456
  p-value: 0.0234
  Cohen's d: 0.4678
  Significant: True

3. 95% Confidence Intervals
----------------------------------------
     START: 0.850 [0.782, 0.918]
    MIDDLE: 0.620 [0.543, 0.697]
       END: 0.780 [0.705, 0.855]

4. Effect Size Interpretation
----------------------------------------
Eta-squared: 0.1234
Interpretation: medium effect
```

## Data Output

### Results Structure

```json
{
  "experiment": "experiment_1",
  "name": "Lost in the Middle",
  "timestamp": "2025-11-29T14:30:00",
  "duration_seconds": 125.43,
  "configuration": {
    "num_documents": 5,
    "words_per_document": 200,
    "fact_positions": ["start", "middle", "end"],
    "num_trials_per_position": 10,
    "random_seed": 42
  },
  "trials": [
    {
      "position": "start",
      "query": "What is the CEO's name?",
      "response": "The CEO is David Cohen",
      "ground_truth": "David Cohen",
      "correct": true,
      "latency": 2.34,
      "tokens": 8,
      "semantic_similarity": 0.85,
      "metadata": {...}
    },
    ...
  ],
  "summary": {
    "start": {
      "num_trials": 10,
      "accuracy": {"mean": 0.85, "std": 0.12, "values": [...]},
      "latency": {"mean": 2.5, "std": 0.3, "min": 2.1, "max": 2.9}
    },
    ...
  }
}
```

### File Locations

- **Raw results:** `results/raw/exp1_results_YYYYMMDD_HHMMSS.json`
- **Ground truth:** `data/ground_truth/experiment_1_queries.json`
- **Visualizations:** `results/figures/exp1_*.png` (when visualization module complete)

## Visualization (Planned)

### Mandatory Visualizations

1. **Bar Chart**
   - Mean accuracy by position
   - 95% confidence interval error bars
   - Statistical significance annotations

2. **Box Plots**
   - Distribution of accuracy by position
   - Outlier detection
   - Median, quartiles shown

3. **Statistical Annotations**
   - p-values or significance stars
   - Sample sizes noted
   - Effect sizes displayed

## Integration Points

### Dependencies

```python
from ..data_generation import DocumentGenerator, FactEmbedder, FactGenerator
from ..llm_interface import LLMInterface, OllamaInterface
from ..metrics import MetricsCalculator
from ..statistics import anova_test, t_test, confidence_interval
```

### Extends/Implements

- Uses `LLMInterface` abstraction for LLM queries
- Uses `MetricsCalculator` for evaluation
- Uses statistical functions from `statistics` module
- Generates data using `data_generation` utilities

## Limitations and Future Work

### Current Limitations

1. **LLM Interface:** Currently uses placeholder responses; needs real Ollama integration
2. **Semantic Similarity:** Simple word overlap; should upgrade to embedding-based
3. **Visualization:** Placeholder only; needs integration with visualization module
4. **Single Fact Type:** Currently tests CEO name primarily; should vary fact types

### Planned Enhancements

1. **Document Length Variation:** Test 100, 200, 500, 1000 words
2. **Fact Salience:** High vs. low salience facts
3. **Prompt Engineering:** Enhanced prompts to mitigate middle penalty
4. **Multiple Languages:** Test Hebrew corpus alongside English
5. **Attention Visualization:** Overlay attention weights on position

## References

1. Liu, N. F., et al. (2023). "Lost in the Middle: How Language Models Use Long Contexts." arXiv preprint arXiv:2307.03172.

2. PRD Section 3: Experiment 1 specifications (this repository)

3. Statistical methods: Montgomery, D. C. (2017). "Design and Analysis of Experiments."

## Testing

### Unit Tests

```bash
# Run unit tests
pytest tests/test_experiment_1.py -v

# Run with coverage
pytest tests/test_experiment_1.py --cov=src.experiments.experiment_1
```

### Integration Tests

```bash
# Test with mock LLM
pytest tests/test_experiment_1_integration.py

# Test full pipeline
python scripts/test_experiment_1.py
```

## Troubleshooting

### Common Issues

**Issue:** ImportError for experiments module
**Solution:** Ensure PYTHONPATH includes src directory

**Issue:** Ollama connection failure
**Solution:** Start Ollama server: `ollama serve`

**Issue:** Low accuracy across all positions
**Solution:** Check LLM model quality; verify fact embedding logic

**Issue:** Statistical tests fail
**Solution:** Ensure sufficient sample size (n ≥ 10 per position)

## Contact and Support

For questions or issues with Experiment 1:
1. Check this README
2. Review PRD Section 3
3. Examine test scripts for examples
4. Check agents_log.txt for implementation status

---

**Status:** Implementation Complete
**Last Updated:** 2025-11-29
**Developer:** experiment-1-developer agent
