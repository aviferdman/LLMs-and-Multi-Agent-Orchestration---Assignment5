# Research Methodology

**Project**: Context Windows in Practice - Empirical Study
**Date**: December 1, 2025
**Status**: Complete Documentation

---

## Table of Contents

1. [Research Overview](#research-overview)
2. [Research Questions](#research-questions)
3. [Hypotheses](#hypotheses)
4. [Experimental Design](#experimental-design)
5. [Variables and Controls](#variables-and-controls)
6. [Data Collection Methods](#data-collection-methods)
7. [Validity and Reliability](#validity-and-reliability)
8. [Ethical Considerations](#ethical-considerations)

---

## Research Overview

### Objective

This research investigates how Large Language Models (LLMs) handle varying context window sizes and content organization, with focus on practical implications for Retrieval-Augmented Generation (RAG) systems and context engineering strategies.

### Research Scope

**Primary Focus:**
- Context size impact on LLM performance
- Information position effects ("lost in the middle")
- RAG chunking strategies effectiveness
- Context engineering approaches comparison

**Model Tested:**
- Primary: `llama2:latest` (Llama 2 7B/13B)
- Rationale: Open-source, locally deployable, representative of modern LLMs

**Metrics:**
- Accuracy: Correctness of LLM responses
- Latency: Response time per query
- Context size: Token count in prompts
- Memory efficiency: Context management overhead

### Research Philosophy

This study employs an **empirical, quantitative approach** with:
- Controlled experiments
- Statistical hypothesis testing
- Reproducible methodology
- Open-source implementation

---

## Research Questions

### Experiment 1: Lost in the Middle

**RQ1.1**: Does the position of relevant information within the context window affect LLM accuracy?

**RQ1.2**: If position matters, which positions (START, MIDDLE, END) perform best?

**RQ1.3**: How does position affect query latency?

**Rationale**: Prior work (Liu et al., 2023) showed GPT models struggle with middle-position information. We test if this generalizes to Llama 2.

### Experiment 2: Context Size Impact

**RQ2.1**: How does context window size affect LLM accuracy?

**RQ2.2**: Is there a critical threshold where performance degrades?

**RQ2.3**: What is the relationship between context size and query latency?

**RQ2.4**: Does accuracy degrade linearly, logarithmically, or exponentially with context size?

**Rationale**: Understanding performance-size tradeoffs guides RAG system design.

### Experiment 3: RAG Chunking Strategies

**RQ3.1**: Which chunking strategy (FIXED, SEMANTIC, SLIDING) yields highest accuracy?

**RQ3.2**: How do chunking strategies affect retrieval latency?

**RQ3.3**: What is the optimal chunk size for each strategy?

**Rationale**: RAG systems require chunking large documents; optimal strategy is unknown.

### Experiment 4: Context Engineering

**RQ4.1**: Which context management strategy (SELECT, COMPRESS, WRITE, HYBRID) maintains accuracy over multiple conversation turns?

**RQ4.2**: What are the latency and memory efficiency tradeoffs?

**RQ4.3**: Do optimal strategies vary by use case (sequential, reasoning, adversarial)?

**Rationale**: Long conversations require context management to avoid window overflow.

---

## Hypotheses

### Experiment 1 Hypotheses

**H1.1** (Primary): Information position significantly affects LLM accuracy
- **Null (H0)**: Position has no effect on accuracy
- **Alternative (H1)**: Position affects accuracy (two-tailed)

**H1.2** (Directional): END position yields highest accuracy, MIDDLE lowest
- Based on: Recency bias and attention mechanisms

**H1.3** (Latency): Position does not significantly affect latency
- Rationale: Transformer attention is O(n²) regardless of position

### Experiment 2 Hypotheses

**H2.1** (Primary): Larger context size decreases accuracy
- **Null (H0)**: Context size has no effect on accuracy
- **Alternative (H1)**: Negative correlation exists (one-tailed)

**H2.2** (Threshold): Performance cliff exists at specific document count
- Expected: Between 5-15 documents based on literature

**H2.3** (Latency): Latency increases quadratically with context size
- Based on: O(n²) attention complexity

**H2.4** (Functional Form): Accuracy decays logarithmically with size
- Model: `Accuracy = α - β*log(context_size)`

### Experiment 3 Hypotheses

**H3.1** (Primary): SEMANTIC chunking outperforms FIXED
- Rationale: Preserves semantic coherence

**H3.2** (Latency): SEMANTIC has higher preprocessing time but better retrieval
- Tradeoff between chunking cost and retrieval quality

**H3.3** (Chunk Size): Optimal size is 256-512 tokens
- Based on: Semantic unit size and model capacity

### Experiment 4 Hypotheses

**H4.1** (Primary): HYBRID strategy achieves best accuracy-efficiency balance
- Combines benefits of SELECT and COMPRESS

**H4.2** (Memory): SELECT and HYBRID most memory efficient
- Constant space O(k) vs linear O(n)

**H4.3** (Use Case Dependency): Optimal strategy varies by scenario type
- Sequential → WRITE, Reasoning → COMPRESS, Adversarial → SELECT

---

## Experimental Design

### Overall Framework

**Design Type**: Within-subjects repeated measures
- Same model tested across all conditions
- Controls for model-specific variations
- Higher statistical power

**Replication**: 3 runs per condition minimum
- Ensures reliability
- Enables statistical testing
- Accounts for stochastic variation

**Randomization**: 
- Document order randomized (Exp 1, 2)
- Question selection randomized
- Seed controlled for reproducibility

### Experiment 1 Design

**Independent Variables:**
- Position: {START, MIDDLE, END}
- Number of documents: 5 (constant)

**Dependent Variables:**
- Accuracy (binary: correct/incorrect)
- Latency (seconds)

**Design**: 3 × 30 = 90 queries
- 3 positions × 30 trials per position
- Duration: ~13-15 minutes

**Procedure:**
1. Generate 5 synthetic documents with facts
2. Create ground truth Q&A pairs
3. For each position:
   - Place relevant document at START/MIDDLE/END
   - Fill remaining positions with distractors
   - Query LLM
   - Measure accuracy and latency
4. Repeat 30 times per position

**Controls:**
- Same documents used across positions
- Same questions
- Same distractor documents
- Same model parameters (temperature=0 for consistency)

### Experiment 2 Design

**Independent Variable:**
- Context size: {2, 5, 10, 20, 50} documents

**Dependent Variables:**
- Accuracy
- Latency
- Token count

**Design**: 5 × 10 = 50 queries
- 5 context sizes × 10 trials per size
- Duration: ~20-25 minutes

**Procedure:**
1. Generate document corpus
2. For each context size N:
   - Sample N documents (1 relevant + N-1 distractors)
   - Place relevant document randomly
   - Query LLM
   - Measure metrics
3. Repeat 10 times per size

**Controls:**
- Same document pool
- Same questions
- Random relevant document position (controlled variable)
- Consistent sampling methodology

### Experiment 3 Design

**Independent Variables:**
- Chunking strategy: {FIXED, SEMANTIC, SLIDING}
- Chunk size: {128, 256, 512, 1024} tokens

**Dependent Variables:**
- Retrieval accuracy
- Query latency
- Chunk quality score

**Design**: 3 × 4 × 10 = 120 queries
- 3 strategies × 4 chunk sizes × 10 trials
- Duration: ~30-40 minutes

**Procedure:**
1. Prepare source documents
2. For each strategy-size combination:
   - Chunk documents
   - Build vector index
   - Retrieve relevant chunks
   - Query LLM with retrieved context
   - Measure accuracy and latency
3. Repeat 10 times per combination

**Controls:**
- Same source documents
- Same retrieval algorithm (cosine similarity)
- Same number of retrieved chunks (k=3)
- Same embedding model

### Experiment 4 Design

**Independent Variables:**
- Strategy: {SELECT, COMPRESS, WRITE, HYBRID}
- Scenario type: {SEQUENTIAL, REASONING, ADVERSARIAL}

**Dependent Variables:**
- Accuracy over time
- Context size growth
- Latency per turn
- Memory usage

**Design**: 4 × 3 × 10 = 120 step executions
- 4 strategies × 3 scenarios × 10 steps per scenario
- 3 runs per combination
- Duration: ~25-35 minutes

**Procedure:**
1. For each strategy-scenario combination:
   - Initialize conversation with empty context
   - For each of 10 turns:
     - Apply context management strategy
     - Query LLM
     - Record metrics
     - Add observation to history
   - Analyze trajectory
2. Repeat 3 times for reliability

**Controls:**
- Same conversation scenarios
- Same queries per turn
- Same max_tokens limit (2000)
- Consistent strategy implementations

---

## Variables and Controls

### Independent Variables (Manipulated)

| Experiment | Variable | Levels | Type |
|------------|----------|--------|------|
| Exp 1 | Position | START, MIDDLE, END | Categorical |
| Exp 2 | Context Size | 2, 5, 10, 20, 50 docs | Ordinal |
| Exp 3 | Chunking Strategy | FIXED, SEMANTIC, SLIDING | Categorical |
| Exp 3 | Chunk Size | 128, 256, 512, 1024 tokens | Ordinal |
| Exp 4 | Management Strategy | SELECT, COMPRESS, WRITE, HYBRID | Categorical |
| Exp 4 | Scenario Type | SEQUENTIAL, REASONING, ADVERSARIAL | Categorical |

### Dependent Variables (Measured)

**Primary Metrics:**
- **Accuracy**: Binary (correct/incorrect) or continuous (0-1 similarity score)
- **Latency**: Query response time in seconds
- **Context Size**: Token count in prompt

**Secondary Metrics:**
- **Memory Usage**: Peak memory consumption
- **Compression Ratio**: For COMPRESS strategy
- **Retrieval Time**: For SELECT strategy
- **Token Efficiency**: Tokens per correct answer

### Controlled Variables

**Model Parameters:**
- Temperature: 0.0 (deterministic for consistency)
- Max tokens: 2000 (unless testing context limits)
- Top-p: 1.0
- Model version: Fixed (llama2:latest)

**Data Generation:**
- Random seed: 42 (reproducibility)
- Document length: ~200 words (constant)
- Fact density: 2-3 facts per document
- Distractor relevance: Low (semantic distance > 0.7)

**Infrastructure:**
- Same hardware (Ollama local server)
- Same Python environment
- No concurrent processes
- Stable network conditions

### Extraneous Variables

**Addressed:**
- **Time of day**: All experiments run consecutively
- **Model warmup**: First 3 queries discarded
- **Cache effects**: Controlled via query variation
- **Prompt formatting**: Consistent template across all experiments

**Acknowledged Limitations:**
- **Model stochasticity**: Mitigated via temperature=0 and replication
- **Hardware variations**: Single machine, but specs documented
- **Corpus bias**: Synthetic data may not represent real-world distribution

---

## Data Collection Methods

### Automated Logging

**Implementation:**
```python
class MetricsCollector:
    def record_query(self, query, response, metadata):
        return {
            'timestamp': time.time(),
            'query': query,
            'response': response,
            'ground_truth': metadata['ground_truth'],
            'accuracy': self._evaluate_accuracy(response, metadata),
            'latency': metadata['latency'],
            'context_size': metadata['context_size'],
            'model': metadata['model']
        }
```

**Storage Format:**
- Raw results: JSON (structured, machine-readable)
- Processed results: JSON (aggregated statistics)
- Logs: Text (human-readable, debugging)
- Visualizations: PNG (300 DPI, publication quality)

### Accuracy Evaluation

**Method 1: Exact Match**
```python
def exact_match(response, ground_truth):
    return response.strip().lower() == ground_truth.strip().lower()
```

**Method 2: Semantic Similarity**
```python
def semantic_similarity(response, ground_truth):
    emb1 = embed(response)
    emb2 = embed(ground_truth)
    return cosine_similarity(emb1, emb2)
```

**Threshold**: similarity >= 0.8 considered correct

**Rationale**: Handles paraphrasing and minor variations

### Latency Measurement

```python
import time

start = time.perf_counter()
response = llm.query(prompt)
latency = time.perf_counter() - start
```

**Precision**: Microseconds (sufficient for ~1-100s queries)

**Exclusions**: Model loading time (measured once), network latency (local server)

### Context Size Calculation

```python
from tiktoken import encoding_for_model

def count_tokens(text, model="gpt-3.5-turbo"):
    enc = encoding_for_model(model)
    return len(enc.encode(text))
```

**Tokenizer**: tiktoken (OpenAI's tokenizer, compatible with Llama)

**Includes**: System prompt + user context + query

---

## Validity and Reliability

### Internal Validity

**Threats Addressed:**

1. **History**: All experiments run in controlled timeframe
2. **Maturation**: Model state fixed (no fine-tuning during experiments)
3. **Testing**: Different queries prevent learning effects
4. **Instrumentation**: Consistent measurement tools
5. **Selection**: Within-subjects design (same model)
6. **Mortality**: No dropout (automated)

**Design Features:**
- Randomization of conditions
- Counterbalancing where applicable
- Controlled variables (temperature, seed)
- Multiple replications

### External Validity

**Generalizability Considerations:**

**Strengths:**
- Tests realistic RAG scenarios
- Uses open-source model (reproducible)
- Varied experiment types (broad coverage)
- Public dataset potential

**Limitations:**
- Single model family (Llama 2)
- Synthetic data (may not reflect real documents)
- English only
- Limited domain coverage

**Mitigation:**
- Document model specifications
- Provide code for replication with other models
- Design experiments generalizable to other LLMs

### Construct Validity

**Accuracy as Performance Measure:**
- Face validity: Direct measure of correctness
- Content validity: Covers factual recall, reasoning, synthesis
- Criterion validity: Aligns with downstream task performance

**Latency as Efficiency Measure:**
- Operationalization: Response time
- Relevance: Critical for production systems
- Measurement: Precise timing

### Reliability

**Inter-rater Reliability**: N/A (automated evaluation)

**Test-retest Reliability:**
- Replication (3 runs per condition)
- Consistency checking
- Statistical analysis of variance

**Internal Consistency:**
- Multiple queries per condition
- Cronbach's alpha calculated where applicable

**Measurement Reliability:**
- Automated metrics (eliminates human error)
- Validated evaluation functions
- Logged raw data for verification

---

## Ethical Considerations

### Research Ethics

**Transparency:**
- Open-source code (public GitHub repository)
- Reproducible methodology (detailed documentation)
- Raw data available (JSON results)

**Honesty:**
- Report all results (including negative findings)
- Acknowledge limitations
- No p-hacking or selective reporting

**Attribution:**
- Cite prior work (Liu et al., 2023 for "Lost in the Middle")
- Credit tools and libraries
- Follow academic integrity standards

### Data Ethics

**Privacy:**
- Synthetic data only (no personal information)
- No proprietary data
- Publicly shareable results

**Bias Considerations:**
- Acknowledge dataset limitations
- Note model biases (Llama 2 known biases)
- Discuss generalizability constraints

**Environmental:**
- Local inference (no cloud costs)
- Energy consumption documented
- Model size considerations

### Responsible AI

**Misuse Prevention:**
- Results used for research only
- No adversarial applications
- Focus on improving systems

**Beneficial Use:**
- Improve RAG system design
- Guide context window usage
- Inform model selection decisions

---

## Statistical Analysis Plan

(See [STATISTICAL_ANALYSIS.md](STATISTICAL_ANALYSIS.md) for complete details)

### Planned Tests

**Experiment 1:**
- One-way ANOVA (position effect)
- Post-hoc Tukey HSD (pairwise comparisons)
- Effect size (η²)

**Experiment 2:**
- Pearson correlation (accuracy vs size)
- Regression analysis (functional form)
- Threshold detection (piecewise regression)

**Experiment 3:**
- Two-way ANOVA (strategy × size)
- Multiple comparisons correction (Bonferroni)
- Interaction effects

**Experiment 4:**
- Repeated measures ANOVA (strategy × time)
- Mixed-effects models (account for nesting)
- Within-subjects comparisons

### Significance Level

**Alpha**: 0.05 (conventional threshold)

**Multiple Comparisons**: Bonferroni or Holm-Bonferroni correction applied

**Power Analysis**: Post-hoc power calculated for all tests

---

## Implementation Details

### Technology Stack

**Language**: Python 3.10+

**Key Libraries:**
- `ollama`: LLM interface
- `numpy`: Numerical computations
- `scipy`: Statistical tests
- `matplotlib/seaborn`: Visualization
- `sentence-transformers`: Embeddings (Exp 3)
- `tiktoken`: Tokenization

**Infrastructure:**
- Ollama server (local)
- 16GB RAM minimum
- GPU optional (faster inference)

### Code Organization

```
src/
├── experiments/       # Experiment implementations
├── llm_interface.py   # Model abstraction
├── metrics.py         # Evaluation functions
├── statistics.py      # Statistical analysis
├── visualization.py   # Plotting
└── data_generation.py # Synthetic data

scripts/
├── run_experiment_1_ollama.py
├── run_experiment_2_ollama.py
├── run_experiment_3_ollama.py
└── run_experiment_4_ollama.py

docs/
├── RESEARCH_METHODOLOGY.md     # This file
├── STATISTICAL_ANALYSIS.md     # Analysis details
├── ARCHITECTURE.md             # System design
└── EXPERIMENT_*_RESULTS.md     # Per-experiment findings
```

### Reproducibility Checklist

- [x] Random seed fixed (42)
- [x] Dependencies specified (requirements.txt)
- [x] Model version documented (llama2:latest)
- [x] Code version controlled (Git)
- [x] Results timestamped
- [x] Configuration files (YAML)
- [x] Detailed documentation
- [x] Raw data preserved

---

## Timeline

**Phase 1: Infrastructure Setup** (Complete)
- Ollama integration
- Metric collection framework
- Data generation pipeline

**Phase 2: Experiment Execution** (Complete)
- Experiment 1: ~15 minutes
- Experiment 2: ~25 minutes
- Experiment 3: ~35 minutes
- Experiment 4: ~30 minutes
- **Total runtime**: ~2 hours

**Phase 3: Analysis** (Complete)
- Statistical tests
- Visualization generation
- Results interpretation

**Phase 4: Documentation** (In Progress)
- Research methodology
- Statistical analysis
- Architecture documentation
- User guide

---

## References

**Academic Literature:**

1. Liu, N. F., et al. (2023). "Lost in the Middle: How Language Models Use Long Contexts." *arXiv preprint arXiv:2307.03172*.

2. Vaswani, A., et al. (2017). "Attention is All You Need." *NeurIPS*.

3. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS*.

**Technical Documentation:**

- Ollama Documentation: https://ollama.ai/docs
- Llama 2 Model Card: https://huggingface.co/meta-llama/Llama-2-7b
- Sentence Transformers: https://www.sbert.net/

**Statistical Methods:**

- Field, A. (2013). *Discovering Statistics Using IBM SPSS Statistics*. 4th ed. Sage.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*. 2nd ed. Lawrence Erlbaum.

---

## Appendices

### Appendix A: Sample Queries

**Experiment 1 Example:**
```
Context: [Document 1, Document 2, ..., Document 5]
Query: "What was the temperature on July 15th?"
Ground Truth: "28°C"
```

**Experiment 2 Example:**
```
Context: [2 documents] vs [50 documents]
Query: "Who won the 2023 championship?"
Ground Truth: "Team Alpha"
```

### Appendix B: Statistical Power

**Target Effect Sizes:**
- Experiment 1: d = 0.5 (medium effect)
- Experiment 2: r = 0.3 (medium correlation)
- Experiment 3: η² = 0.06 (medium ANOVA effect)
- Experiment 4: d = 0.5 (medium between-strategies)

**Sample Size Justification:**
- 30 trials per condition → power > 0.80 for d = 0.5
- 3 replications → robust against outliers
- Total queries: 380+ → comprehensive coverage

### Appendix C: Quality Assurance

**Pre-registration**: Methodology documented before analysis

**Blinding**: Not applicable (automated evaluation)

**Peer Review**: Code review by team members

**Validation**: Results cross-checked across runs

---

**Document Version**: 1.0  
**Last Updated**: December 1, 2025  
**Author**: Research Team  
**Status**: Complete

*End of Research Methodology Documentation*
