# Product Requirements Document (PRD)
## Context Windows in Practice: Graduate-Level Research Framework

### Document Information
- **Version**: 1.0
- **Date**: November 2025
- **Project**: LLMs and Multi-Agent Orchestration - Assignment 5
- **Academic Level**: Master's (M.Sc.) - Graduate Research
- **Status**: Research Proposal
- **Compliance**: Level 4 (MIT Academic Standard) + ISO/IEC 25010

---

## 📋 Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Research Goals & Success Metrics](#2-research-goals--success-metrics)
3. [Experiment 1: Lost in the Middle](#3-experiment-1-lost-in-the-middle)
4. [Experiment 2: Context Size Impact](#4-experiment-2-context-size-impact)
5. [Experiment 3: RAG Impact](#5-experiment-3-rag-impact)
6. [Experiment 4: Context Engineering Strategies](#6-experiment-4-context-engineering-strategies)
7. [Non-Functional Requirements](#7-non-functional-requirements)
8. [Technical Architecture](#8-technical-architecture)
9. [Project Structure & Code Quality](#9-project-structure--code-quality)
10. [Testing & Quality Assurance](#10-testing--quality-assurance)
11. [Documentation Requirements](#11-documentation-requirements)
12. [Timeline & Milestones](#12-timeline--milestones)
13. [Deliverables & Evaluation](#13-deliverables--evaluation)

---

## 1. Executive Summary

### 1.1 Graduate-Level Research Vision

This project is a **comprehensive research investigation** at Master's level, exploring fundamental questions about Large Language Model context window behavior. This is **not a coding exercise**—it is an opportunity to:

- **Define original research questions** beyond the provided framework
- **Design rigorous experiments** with statistical validation
- **Discover unexpected patterns** and explain their mechanisms
- **Create publication-quality visualizations** that tell compelling stories
- **Provide actionable insights** for practitioners building production systems

### 1.2 Research Philosophy

**You are the researcher.** The assignment provides a scaffold, not a prescription. Your mission:

1. **Transform basic prompts into sophisticated hypotheses**
2. **Design experiments that test specific, falsifiable claims**
3. **Analyze data with statistical rigor** (hypothesis tests, effect sizes, confidence intervals)
4. **Present findings visually** with charts that communicate insights immediately
5. **Connect results to theory** (attention mechanisms, information theory, cognitive science)
6. **Recommend practices** that improve real-world LLM applications

### 1.3 Problem Statement

As LLMs scale to support massive context windows (128K+ tokens), critical questions remain:

- **Lost in the Middle**: Why do models lose track of information in the middle of long contexts?
- **Scalability**: Does accuracy degrade even within theoretical context limits?
- **RAG Effectiveness**: When does retrieval help vs. when is full context better?
- **Engineering Strategies**: How should production systems manage growing context in multi-turn conversations?

These questions reveal **fundamental properties** of transformer architectures and guide the **future design of AI systems**.

### 1.4 Key Performance Indicators (KPIs)

**Academic Excellence KPIs:**
- Research Originality: ≥2 novel research questions beyond template
- Statistical Rigor: 100% experiments with hypothesis testing + effect sizes + CIs
- Visualization Quality: 100% figures publication-ready (≥300 DPI)
- Documentation Completeness: All 8+ required documents
- Code Quality: ≥85% test coverage
- Reproducibility: 100% with documented seeds

**Technical Performance KPIs:**
- Experiment Execution: Full suite <2 hours
- Data Quality: 100% valid measurements
- Statistical Power: ≥0.80 for primary hypotheses
- Effect Detection: Minimum d=0.5

**Deliverable Quality KPIs:**
- Documentation: ≥10,000 lines total
- Figures: ≥15 publication-quality
- Statistical Tests: ≥10 hypothesis tests
- References: ≥5 academic citations

---

## 2. Research Goals & Success Metrics

### 2.1 Primary Research Goals

**G1: Lost in the Middle Phenomenon**
- Empirically demonstrate position effects on accuracy
- Quantify magnitude of middle penalty
- Test mitigation strategies (prompt engineering, chunking)
- **Success**: Significant effect (p<0.05, d≥0.5) with clear visualization

**G2: Context Size Scaling Laws**
- Measure accuracy degradation as context grows
- Identify optimal context size ranges
- Model functional form (linear, logarithmic, exponential)
- **Success**: Validated scaling model (R²>0.80) with actionable recommendations

**G3: RAG vs. Full Context Comparison**
- Quantify RAG accuracy, latency, and cost benefits
- Identify conditions where RAG excels vs. fails
- Analyze retrieval quality's impact on final accuracy
- **Success**: Clear decision framework for when to use RAG

**G4: Context Engineering Strategy Evaluation**
- Benchmark SELECT, COMPRESS, WRITE, HYBRID strategies
- Measure performance across 10-step agent workflows
- Identify optimal strategy per use case
- **Success**: Evidence-based strategy selection guide

### 2.2 Research Quality Standards

**Minimum Acceptable (70-79%):**
- All experiments completed
- Basic statistics (means, std)
- Simple visualizations
- Functional code

**Good (80-89%):**
- Additional variations
- Hypothesis testing
- Multi-panel plots
- Original insights

**Excellent (90-94%):**
- Novel research questions
- Advanced statistics
- Publication-quality figures
- Theory connections
- Practical recommendations

**Outstanding (95-100%):**
- Paradigm-shifting insights
- Mathematical formulations
- Interactive visualizations
- Community-reusable framework
- External validation

---

## 3. Experiment 1: Lost in the Middle

### 3.1 Research Questions

**Primary (Required):**
- RQ1.1: Is there statistically significant accuracy degradation for middle-positioned facts?

**Advanced (Choose ≥1):**
- RQ1.2: Does document length modulate the middle penalty?
- RQ1.3: Can prompt engineering mitigate the effect?
- RQ1.4: Does fact salience interact with position?
- RQ1.5: Is the effect symmetric (early vs. late middle)?

### 3.2 Mathematical Framework

**Accuracy Model:**
$$
\text{Accuracy}(p) = \beta_0 + \beta_1 \cdot \mathbb{1}_{\text{middle}}(p) + \beta_2 \cdot \mathbb{1}_{\text{end}}(p) + \varepsilon
$$

Where:
- $p$ = fact position (start=baseline, middle, end)
- $\mathbb{1}$ = indicator function
- $\varepsilon \sim \mathcal{N}(0, \sigma^2)$ = error

**Hypothesis:**
$$
H_0: \beta_1 = 0 \quad \text{(no middle penalty)}
$$
$$
H_1: \beta_1 < 0 \quad \text{(middle accuracy lower)}
$$

**Sample Size:**
$$
n = \frac{2(z_{\alpha/2} + z_{\beta})^2 \sigma^2}{\delta^2}
$$

For α=0.05, power=0.80, d=0.5: **n ≥ 64 per condition**

### 3.3 Experimental Design

**FR-01: Document Generation**
- **Count**: 5 documents per trial
- **Length**: 200 ± 20 words (configurable)
- **Fact embedding**: One critical fact per document
- **Positions**: Start (0-20%), Middle (40-60%), End (80-100%)
- **Filler text**: Coherent but non-informative

**FR-02: Experimental Variations (Optional Enhancements)**

*Option A: Document Length*
- Test: 100, 200, 500, 1000 words
- Hypothesis: Longer documents → stronger middle penalty

*Option B: Fact Salience*
- High: "Q3 revenue: $50M"
- Low: "The office has blue walls"
- Hypothesis: Salient facts partially resistant

*Option C: Prompt Engineering*
- Baseline: "What is the CEO's name?"
- Enhanced: "Carefully review all documents. What is the CEO's name?"
- Hypothesis: ≥30% reduction in middle penalty

### 3.4 Data Collection

**Metrics:**
- Accuracy: Binary correct/incorrect + semantic similarity (0-1)
- Latency: Response time per query
- Confidence: Model confidence score (if available)
- Response length: Token count
- Metadata: Document length, fact complexity

**Trials**: ≥3 runs per condition (≥9 total per position)

### 3.5 Statistical Analysis

**Required Tests:**

1. **One-Way ANOVA**
   - DV: Accuracy, IV: Position (3 levels)
   - Report: F-statistic, p-value, effect size (η²)

2. **Post-hoc Pairwise**
   - Tukey HSD or Bonferroni
   - Compare: Start-Middle, Start-End, Middle-End

3. **Effect Size**
   - Cohen's d for pairwise differences
   - Interpretation: small (0.2), medium (0.5), large (0.8)

4. **Confidence Intervals**
   - 95% CI for each position mean
   - Visualize with error bars

**Advanced (Recommended):**
- Non-parametric: Kruskal-Wallis if normality violated
- Power analysis: Actual vs. target power
- Sensitivity analysis: Vary parameters ±20%

### 3.6 Visualization Requirements

**Mandatory:**
- Bar chart: Mean accuracy by position with 95% CI
- Statistical annotations: p-values or significance stars
- Sample sizes noted

**Enhanced (Choose ≥1):**
- Box plots: Show distributions, not just means
- Heatmap: Position × Document Length
- Attention visualization: Model attention by position
- Before-after: Baseline vs. mitigation strategy

### 3.7 Deliverables

1. **Quantitative Summary**: "Middle accuracy X% lower (p<0.05, d=Y, 95% CI: [a,b])"
2. **Mechanism Hypothesis**: Why does this occur? (attention decay, positional encoding, etc.)
3. **Practical Recommendation**: Where to place critical information
4. **Limitations**: Generalizability boundaries
5. **Theory Connection**: Reference to literature (Liu et al., 2023)

---

## 4. Experiment 2: Context Size Impact

### 4.1 Research Questions

**Primary:**
- RQ2.1: What is the functional form of accuracy degradation? (linear, log, exponential)
- RQ2.2: Is there a "cliff" where performance drops sharply?

**Advanced:**
- RQ2.3: How does latency scale with context size?
- RQ2.4: What is optimal size for 90% accuracy + minimum latency?
- RQ2.5: Does task complexity modulate scaling?

### 4.2 Mathematical Models

**Accuracy Degradation:**

*Model 1 (Logarithmic):*
$$
\text{Accuracy}(n) = \alpha - \beta \log(n) + \varepsilon
$$

*Model 2 (Logistic):*
$$
\text{Accuracy}(n) = \frac{\alpha}{1 + e^{\beta(n - n_0)}}
$$

*Model 3 (Polynomial):*
$$
\text{Accuracy}(n) = \beta_0 + \beta_1 n + \beta_2 n^2 + \varepsilon
$$

**Latency Scaling:**
$$
\text{Latency}(n) = \gamma_0 + \gamma_1 n + \gamma_2 n^2
$$

Test: Is $\gamma_2 > 0$ (super-linear)?

### 4.3 Experimental Design

**FR-03: Context Size Progression**
- **Mandatory**: 2, 5, 10, 20, 50 documents
- **Recommended**: Add 3, 7, 15, 30, 40 for smoothness
- **Advanced**: Test 75, 100, 150 to find limits

**FR-04: Task Complexity Variation**

*Simple Task:*
- "What is the main topic of document 5?"
- Direct retrieval, single document

*Complex Task:*
- "Compare arguments in docs 3, 7, 12. Which is most compelling?"
- Multi-document reasoning

**Hypothesis**: Complex tasks degrade faster

### 4.4 Metrics

**Primary:**
- Accuracy: Exact match, F1, semantic similarity
- Latency: Time to first token (TTFT), total time, tokens/sec
- Resources: Token count, RAM usage, GPU util

**Secondary:**
- Answer quality: Length, coherence
- Cost: Token count × price
- Throughput: Queries/second

### 4.5 Statistical Analysis

**Required:**

1. **Correlation Analysis**
   - Pearson r: Accuracy ~ Size
   - Report r, p-value, 95% CI

2. **Regression Modeling**
   - Fit all 3 models (log, logistic, polynomial)
   - Compare: AIC, BIC, R²
   - Select best model
   - Report: Coefficients, p-values, prediction intervals

3. **Breakpoint Analysis**
   - Piecewise regression
   - Identify inflection point $n_0$
   - Test: Is slope change significant?

**Advanced:**
- Model diagnostics: Residual plots, Q-Q plots
- Cross-validation: Train/test split
- Extrapolation: Predict performance at 100K tokens

### 4.6 Optimization: Pareto Frontier

**Multi-Objective:**
$$
\max_n \{\text{Accuracy}(n), -\text{Latency}(n)\}
$$

**Utility Function:**
$$
U(n) = w_1 \cdot \text{Acc}(n) - w_2 \cdot \text{Lat}(n)
$$

**Weight Profiles:**
- High accuracy: $w_1=0.8, w_2=0.2$
- Balanced: $w_1=0.5, w_2=0.5$
- Low latency: $w_1=0.2, w_2=0.8$

Find $n^*$ that maximizes $U(n)$ for each profile

### 4.7 Visualization Requirements

**Mandatory:**
- **Panel A**: Accuracy vs. size (scatter + fitted curve with equation)
- **Panel B**: Latency vs. size (with confidence bands)
- **Panel C**: Pareto frontier (Accuracy vs. Latency)
- All panels: Consistent styling, statistical annotations

**Enhanced:**
- 3D surface: Accuracy ~ Size × Complexity
- Animation: Show degradation progression
- Residual plots: Validate assumptions
- Comparison: Overlay theoretical limits

### 4.8 Deliverables

1. **Scaling Law**: "Accuracy drops X% per doubling (95% CI: [a,b])"
2. **Optimal Range**: "For Llama2-13B: 10-20 docs for >85% accuracy, <5s latency"
3. **Extrapolation**: "Predicted 100K token performance: Y% ± Z%"
4. **Model Comparison**: Compare with OpenAI technical reports
5. **Production Guidance**: "Limit to ≤15 docs for >90% accuracy"

---

## 5. Experiment 3: RAG Impact

### 5.1 Research Questions

**Primary:**
- RQ3.1: Under what conditions does RAG outperform full context?
- RQ3.2: How does retrieval quality affect final accuracy?

**Advanced:**
- RQ3.3: Context size threshold where RAG becomes necessary?
- RQ3.4: Query type impact (factual vs. analytical)?
- RQ3.5: Accuracy-latency-cost tradeoff optimization?

### 5.2 Mathematical Framework

**Retrieval Metrics:**

*Precision@k:*
$$
P@k = \frac{|\text{Retrieved}_k \cap \text{Relevant}|}{k}
$$

*Recall@k:*
$$
R@k = \frac{|\text{Retrieved}_k \cap \text{Relevant}|}{|\text{Relevant}|}
$$

*MRR (Mean Reciprocal Rank):*
$$
\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}
$$

*nDCG@k:*
$$
\text{nDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}, \quad \text{DCG@k} = \sum_{i=1}^{k} \frac{\text{rel}_i}{\log_2(i+1)}
$$

**Hypothesis:**
$$
\text{Accuracy}_{\text{final}} = \beta_0 + \beta_1 \cdot P@k + \varepsilon
$$

Expected: $\beta_1 > 0$, $p < 0.05$

### 5.3 Experimental Design

**FR-05: Document Corpus**
- **Minimum**: 20 documents (Hebrew or English)
- **Recommended**: 50+ documents
- **Topics**: Technology, law, medicine (diverse)
- **Statistics**: Report avg length, vocabulary size, topical diversity

**FR-06: Dual-Mode Comparison**

*Mode A: Full Context*
```python
context = concatenate_all(documents)  # All N docs
response = llm.query(context + query)
metrics = measure(accuracy, latency, tokens, cost)
```

*Mode B: RAG*
```python
relevant_docs = vector_store.similarity_search(query, k=3)
context = concatenate(relevant_docs)  # Only k docs
response = llm.query(context + query)
metrics = measure(accuracy, latency, tokens, cost, precision@k)
```

**FR-07: Parameter Sensitivity**

*Top-k variation:* Test k ∈ {1, 2, 3, 5, 7, 10}
- Hypothesis: Optimal k ∈ [3, 5]
- Too small: Low recall
- Too large: Noise

*Chunk size:* Test {250, 500, 750, 1000} tokens
- Hypothesis: Optimal depends on document structure

*Embedding models:* Compare 2+
- MiniLM-L6 (384 dim, fast)
- Nomic-embed (768 dim, quality)
- Ada-002 (1536 dim, OpenAI)

### 5.4 Implementation Specifications

**Chunking:**
```python
chunks = split_documents(
    documents, 
    chunk_size=500,
    chunk_overlap=50,
    separator="\n\n"
)
```

**Embedding & Storage:**
```python
embeddings = embedding_model.embed(chunks)
vector_store = ChromaDB()
vector_store.add(chunks, embeddings, metadata)
```

**Retrieval:**
```python
results = vector_store.similarity_search(
    query=query,
    k=3,
    filter={"topic": "relevant"}
)
```

### 5.5 Failure Mode Analysis

**When RAG Fails:**

*Type 1: Information Diffusion*
- Query needs synthesis across many docs
- Example: "Overall sentiment toward X?"
- Mitigation: Increase k or use full context

*Type 2: Poor Retrieval*
- Semantic gap between query and documents
- Example: Synonym mismatch
- Mitigation: Query expansion, hybrid search

*Type 3: Context Fragments*
- Answer spans multiple chunks
- Example: Split tables
- Mitigation: Larger chunks, overlap

**When RAG Excels:**

*Type A: Needle in Haystack*
- Specific fact, large corpus (N>20)
- Example: "Q3 2023 revenue?"

*Type B: Latency-Constrained*
- Real-time apps (<1s)
- Example: Chatbot

*Type C: Cost-Sensitive*
- High query volume
- Example: 1M+ queries/day API

### 5.6 Cost-Benefit Analysis

**Economic Model:**
$$
C_{\text{full}} = N \cdot L \cdot p_{\text{token}}
$$
$$
C_{\text{RAG}} = k \cdot L \cdot p_{\text{token}} + C_{\text{embed}} + C_{\text{storage}}
$$
$$
\text{Savings} = C_{\text{full}} - C_{\text{RAG}}
$$

**Break-even:**
$$
N > k + \frac{C_{\text{overhead}}}{L \cdot p_{\text{token}}}
$$

**Latency Model:**
$$
\text{Speedup} = \frac{T_{\text{full}}}{T_{\text{RAG}}} = \frac{\alpha N L + \beta}{T_{\text{retr}} + \alpha k L + \beta}
$$

### 5.7 Statistical Analysis

**Required:**

1. **Paired t-test**: Accuracy_RAG vs. Accuracy_Full
2. **Effect size**: Cohen's d
3. **Correlation**: P@k vs. Final Accuracy
4. **Regression**: Accuracy = f(P@k, k, corpus_size)

**Report:**
- t-statistic, p-value, 95% CI of difference
- "RAG improves accuracy by X% (p<0.05, d=Y)"

### 5.8 Visualization Requirements

**Mandatory:**
- Comparison table: RAG vs. Full (all metrics)
- Side-by-side bar chart: Performance by query
- Scatter: P@k vs. Final Accuracy (with regression line)

**Enhanced:**
- Confusion matrix: RAG correct/incorrect × Full correct/incorrect
- Query complexity analysis: Simple vs. complex
- Case studies: 3 examples with visual explanations
- Cost-performance frontier: Cost vs. Accuracy

### 5.9 Decision Framework Deliverable

Create flowchart or decision tree:
```
if corpus_size ≤ 10:
    return "Full Context"
elif latency_req < 2.0:
    return "RAG"
elif query_type == "factual" and accuracy_req < 0.95:
    return "RAG"
elif query_type == "synthesis":
    return "Full Context"
else:
    return "RAG (default)"
```

Document with justification for each branch.

---

## 6. Experiment 4: Context Engineering Strategies

### 6.1 Research Questions

**Primary:**
- RQ4.1: Which strategy maintains accuracy over 10 steps?
- RQ4.2: Latency and memory efficiency comparison?

**Advanced:**
- RQ4.3: Do optimal strategies vary by action type?
- RQ4.4: Can hybrid strategies outperform single strategies?
- RQ4.5: At what growth rate does each strategy fail?

### 6.2 Mathematical Framework

**Context Growth (Unmanaged):**
$$
|C_t| = |C_0| + \sum_{i=1}^{t} |O_i|
$$

**Strategy-Specific Models:**

*SELECT:*
$$
|C_t| = k \cdot L_{\text{avg}} + |O_t|
$$
Space: $O(k)$ constant

*COMPRESS:*
$$
|C_t| = \begin{cases}
|C_{t-1}| + |O_t| & \text{if } |C_{t-1}| < M \\
r \cdot |C_{t-1}| + |O_t| & \text{if } |C_{t-1}| \geq M
\end{cases}
$$
Space: $O(M)$ bounded

*WRITE:*
$$
|C_t| = |S_t| + |O_t|
$$
Space: $O(F)$ where F = extracted facts

### 6.3 Strategy Implementations

**Strategy 1: SELECT (RAG-Based)**
```python
def select_strategy(history, query, top_k=5):
    """Retrieve most relevant past context"""
    embeddings = embed_text(history)
    relevant = similarity_search(query, embeddings, k=top_k)
    response = llm.query(relevant + query)
    return response, {
        'context_size': count_tokens(relevant),
        'retrieval_time': measure_latency()
    }
```

**Strategy 2: COMPRESS (Summarization)**
```python
def compress_strategy(history, query, max_tokens=2000):
    """Summarize when context exceeds limit"""
    if len(history) > max_tokens:
        history = llm.summarize(history, target=max_tokens//2)
    response = llm.query(history + query)
    return response, {
        'context_size': count_tokens(history),
        'was_compressed': True/False
    }
```

**Strategy 3: WRITE (External Memory)**
```python
def write_strategy(history, query, scratchpad):
    """Extract and store key facts"""
    key_facts = extract_key_info(history)
    scratchpad.store(key_facts)
    relevant_facts = scratchpad.retrieve(query, k=10)
    response = llm.query(relevant_facts + query)
    return response, {
        'context_size': count_tokens(relevant_facts),
        'scratchpad_size': len(scratchpad)
    }
```

**Strategy 4: HYBRID (Your Design)**
```python
def hybrid_strategy(history, query):
    """Combine SELECT + COMPRESS"""
    # Phase 1: SELECT relevant
    relevant = select_top_k(history, query, k=5)
    # Phase 2: COMPRESS if needed
    if len(relevant) > max_tokens:
        relevant = compress(relevant)
    response = llm.query(relevant + query)
    return response, metrics
```

### 6.4 Scenario Design

**Scenario A: Sequential Data Collection**
```
Weather monitoring: 10 days of reports
Query: "Temperature trend over past 3 days?"
Context growth: ~100 tokens/step
Challenge: Recent vs. historical data access
```

**Scenario B: Multi-Step Reasoning**
```
Trip planning with evolving constraints
10 steps: destination → hotels → budget → restaurants → ...
Context growth: ~150 tokens/step
Challenge: Consistency across constraints
```

**Scenario C: Adversarial**
```
Fact-checking with contradictions
Steps 1-4: Consistent info
Step 5: Contradiction
Steps 6-10: Mixed evidence
Challenge: Detect contradictions across distance
```

**Action Types:**
1. Retrieval: "What was mentioned in step 3?"
2. Synthesis: "Summarize decisions so far"
3. Reasoning: "Given constraints, what next?"
4. Comparison: "How does step 7 differ from step 2?"

### 6.5 Performance Metrics

**Primary (At Each Step 1-10):**

*Accuracy:*
$$
A_t = \frac{\text{correct answers at step } t}{\text{total queries at step } t}
$$

*Context Size:*
$$
|C_t| = \text{tokens in context at step } t
$$

*Latency:*
$$
L_t = \text{response time at step } t
$$

*Memory:*
$$
M_t = \text{RAM usage at step } t
$$

**Secondary:**

*Information Retention:*
$$
R_t(d) = \text{accuracy on queries about step } (t-d)
$$

*Consistency:*
$$
C = \frac{\text{consistent answer pairs}}{\text{testable pairs}}
$$

### 6.6 Statistical Analysis

**Required:**

1. **Repeated Measures ANOVA**
   - Accuracy ~ Strategy × Step
   - Report: F-stats, p-values, η² for main effects and interaction

2. **Post-hoc Pairwise**
   - Bonferroni correction
   - "SELECT > COMPRESS (p<0.001, d=0.85)"

3. **Trend Analysis**
   - Accuracy_t = β₀ + β₁·t + ε
   - Test if β₁ < 0 (degradation)

**Advanced:**

4. **AUC (Area Under Curve)**
   - Overall performance across 10 steps
   - Rank strategies by AUC

5. **Time-to-Failure**
   - Define: Accuracy < 0.70
   - Kaplan-Meier survival curves
   - Log-rank test

6. **Efficiency Metrics**
   - Accuracy/Token
   - Accuracy/Second

### 6.7 Decision Matrix

| Criterion | Weight | SELECT | COMPRESS | WRITE | HYBRID |
|-----------|--------|--------|----------|-------|--------|
| Accuracy | 0.30 | ? | ? | ? | ? |
| Latency | 0.20 | ? | ? | ? | ? |
| Memory | 0.15 | ? | ? | ? | ? |
| Scalability | 0.15 | ? | ? | ? | ? |
| Simplicity | 0.10 | ? | ? | ? | ? |
| Robustness | 0.10 | ? | ? | ? | ? |
| **Total** | 1.00 | ? | ? | ? | ? |

**Weighted Score:**
$$
S = \sum_{i} w_i \cdot m_i
$$

Fill based on experimental results, then recommend by use case.

### 6.8 Visualization Requirements

**Mandatory:**
- Line plot: Accuracy over 10 steps (each strategy)
- Box plots: Latency distribution per strategy
- Stacked area: Context size growth

**Enhanced:**
- Radar chart: Multi-dimensional comparison
- Heatmap: Accuracy × Strategy × Step
- Sankey: Information flow (WRITE strategy)
- Animation: Context evolution over time
- Pareto frontier: Accuracy vs. Latency

### 6.9 Deliverables

1. **Strategy Failure Modes**: When/why each fails (with examples)
2. **Crossover Points**: At what step does A outperform B?
3. **Robustness Analysis**: Parameter sensitivity (±20%)
4. **Recommendation Table**: Use case → Strategy mapping

**Example:**
```
Customer service (5-10 turns) → WRITE (fact recall)
Code review (3-5 files) → SELECT (relevant sections)
Document QA (50+ docs) → HYBRID (efficiency)
Long conversations (50+ turns) → COMPRESS (bounded growth)
```

---

## 7. Non-Functional Requirements (ISO/IEC 25010)

### 7.1 Performance Efficiency

**NFR-PE-01: Time Behavior**
- Individual experiment: ≤30 minutes
- Full suite: ≤2 hours
- Progress updates: Every 10 seconds

**NFR-PE-02: Resource Utilization**
- Memory: ≤8GB RAM (excluding model weights)
- Storage: ≤1GB for results
- CPU: Efficient parallelization where applicable

**NFR-PE-03: Capacity**
- Support context sizes: 2-150 documents
- Handle corpus: up to 100 documents
- Scalable to multiple LLM providers

### 7.2 Reliability

**NFR-RE-01: Maturity**
- Zero critical bugs in core logic
- Graceful degradation on errors
- Validated statistical formulas

**NFR-RE-02: Availability**
- Automatic retry on LLM API failures (3 attempts, exponential backoff)
- Checkpoint progress after each experiment
- Resume capability after interruption

**NFR-RE-03: Fault Tolerance**
- Handle network timeouts
- Validate API responses
- Catch and log exceptions

**NFR-RE-04: Recoverability**
- Save intermediate results
- Resume from last checkpoint
- Automatic backup of critical data

### 7.3 Usability

**NFR-US-01: Appropriateness Recognizability**
- Clear README with quick start
- Example configurations provided
- Visual workflow diagram

**NFR-US-02: Learnability**
- Step-by-step installation guide
- Commented example runs
- Troubleshooting section

**NFR-US-03: Operability**
- Single command execution: `python run_experiments.py`
- CLI arguments for quick overrides
- Configuration via YAML files

**NFR-US-04: User Error Protection**
- Validate config before execution
- Clear error messages
- Confirm before long operations

### 7.4 Maintainability

**NFR-MA-01: Modularity**
- Single Responsibility: Each module one purpose
- Low coupling: Minimal dependencies between modules
- High cohesion: Related functions grouped

**NFR-MA-02: Reusability**
- Generic utilities: document loading, metric calculation
- Pluggable LLM interfaces
- Extensible experiment framework

**NFR-MA-03: Analyzability**
- Clear naming conventions
- Comprehensive logging
- Code complexity: <10 cyclomatic complexity per function

**NFR-MA-04: Modifiability**
- New experiments: Add without changing existing
- New metrics: Plug-in architecture
- New LLMs: Interface-based design

**NFR-MA-05: Testability**
- Unit tests: ≥85% coverage
- Integration tests: End-to-end flows
- Mocking: LLM API calls for testing

### 7.5 Security

**NFR-SE-01: Confidentiality**
- API keys: Environment variables only
- No secrets in source code
- .gitignore: Excludes .env, credentials

**NFR-SE-02: Integrity**
- Input validation: Check file formats, parameter ranges
- Data checksums: Verify corpus integrity
- Immutable raw data: Separate from processed results

**NFR-SE-03: Non-repudiation**
- Audit log: All API calls with timestamps
- Result provenance: Track parameters used
- Version control: Git commits for changes

---

## 8. Technical Architecture

### 8.1 System Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Experiment Orchestrator                │
│         (run_experiments.py - Main Entry Point)         │
└────────────┬──────────────┬──────────────┬──────────────┘
             │              │              │
    ┌────────▼─────┐ ┌─────▼──────┐ ┌─────▼──────┐
    │ Experiment 1 │ │Experiment 2│ │Experiment 3│ ...
    └────────┬─────┘ └─────┬──────┘ └─────┬──────┘
             │              │              │
    ┌────────▼──────────────▼──────────────▼──────────────┐
    │             Shared Components Layer                  │
    ├──────────────────────────────────────────────────────┤
    │ • LLM Interface  • Vector Store  • Metrics           │
    │ • Data Loader    • Visualizer    • Statistics        │
    └──────────────────────────────────────────────────────┘
             │              │              │
    ┌────────▼─────┐ ┌─────▼──────┐ ┌─────▼──────┐
    │    Ollama    │ │  ChromaDB  │ │File System │
    └──────────────┘ └────────────┘ └────────────┘
```

### 8.2 Directory Structure

```
LLMs-and-Multi-Agent-Orchestration---Assignment5/
├── README.md                    # Quick start guide
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
├── .env.example                 # Example configuration
├── .gitignore                   # Exclude secrets, data
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── config.py                # Configuration loader
│   ├── llm_interface.py         # LLM abstraction
│   ├── data_generation.py       # Synthetic data
│   ├── metrics.py               # Evaluation metrics
│   ├── statistics.py            # Statistical tests
│   ├── visualization.py         # Plotting functions
│   ├── rag_pipeline.py          # RAG implementation
│   └── experiments/
│       ├── __init__.py
│       ├── experiment_1.py      # Lost in the Middle
│       ├── experiment_2.py      # Context Size
│       ├── experiment_3.py      # RAG Impact
│       └── experiment_4.py      # Strategies
│
├── data/                        # Data files
│   ├── corpora/
│   │   ├── hebrew_corpus/       # Experiment 3 docs
│   │   └── synthetic/           # Generated docs
│   └── ground_truth/
│       ├── queries.json         # Test queries
│       └── answers.json         # Reference answers
│
├── results/                     # Experimental outputs
│   ├── raw/                     # Raw measurements
│   │   ├── exp1_run1.json
│   │   └── ...
│   ├── processed/               # Aggregated stats
│   │   ├── exp1_summary.csv
│   │   └── ...
│   ├── figures/                 # Visualizations
│   │   ├── exp1_accuracy_by_position.png
│   │   └── ...
│   └── reports/                 # Markdown summaries
│       └── experiment_1_report.md
│
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── test_llm_interface.py
│   ├── test_metrics.py
│   ├── test_statistics.py
│   └── test_experiments.py
│
├── config/                      # Configuration files
│   ├── experiments.yaml         # Experiment parameters
│   ├── models.yaml              # LLM configurations
│   └── paths.yaml               # File paths
│
├── scripts/                     # Utility scripts
│   ├── run_experiments.py       # Main orchestrator
│   ├── generate_corpus.py       # Create test data
│   ├── validate_results.py      # Check outputs
│   └── create_report.py         # Generate final report
│
├── docs/                        # Documentation
│   ├── PRD.md                   # This file
│   ├── ARCHITECTURE.md          # System design
│   ├── API.md                   # API documentation
│   ├── RESEARCH_METHODOLOGY.md  # Experimental design
│   ├── STATISTICAL_ANALYSIS.md  # Stats procedures
│   └── USER_GUIDE.md            # Usage instructions
│
└── notebooks/                   # Analysis notebooks
    ├── exploratory_analysis.ipynb
    ├── experiment_1_analysis.ipynb
    └── final_report.ipynb
```

### 8.3 Technology Stack

**Core Languages:**
- Python 3.9+ (primary)

**LLM Integration:**
- `ollama-python` or `langchain` for Ollama
- `openai` (optional, for comparison)

**Vector Database:**
- `chromadb` ≥0.4.0

**Data Science:**
- `numpy` ≥1.24.0
- `pandas` ≥2.0.0
- `scipy` ≥1.10.0 (statistics)
- `scikit-learn` ≥1.3.0 (ML utilities)

**Visualization:**
- `matplotlib` ≥3.7.0
- `seaborn` ≥0.12.0
- `plotly` ≥5.0.0 (interactive)

**Utilities:**
- `pyyaml` ≥6.0 (config)
- `tqdm` ≥4.60.0 (progress bars)
- `loguru` ≥0.7.0 (logging)
- `python-dotenv` ≥1.0.0 (env vars)

**Testing:**
- `pytest` ≥7.0.0
- `pytest-cov` ≥4.0.0 (coverage)
- `pytest-mock` ≥3.10.0 (mocking)

**Code Quality:**
- `black` (formatter)
- `flake8` (linter)
- `mypy` (type checker)
- `isort` (import sorter)

### 8.4 Key Design Patterns

**1. Strategy Pattern (Experiment 4)**
```python
class ContextStrategy(ABC):
    @abstractmethod
    def manage_context(self, history, query): pass

class SelectStrategy(ContextStrategy): ...
class CompressStrategy(ContextStrategy): ...
class WriteStrategy(ContextStrategy): ...
```

**2. Factory Pattern (LLM Interface)**
```python
class LLMFactory:
    @staticmethod
    def create_llm(provider: str):
        if provider == "ollama":
            return OllamaInterface()
        elif provider == "openai":
            return OpenAIInterface()
```

**3. Observer Pattern (Progress Tracking)**
```python
class ExperimentObserver:
    def on_experiment_start(self, name): ...
    def on_step_complete(self, step, metrics): ...
    def on_experiment_complete(self, results): ...
```

**4. Repository Pattern (Data Access)**
```python
class DataRepository:
    def load_corpus(self, name): ...
    def save_results(self, experiment, data): ...
    def load_ground_truth(self, experiment): ...
```

### 8.5 Configuration Management

**Example: config/experiments.yaml**
```yaml
general:
  random_seed: 42
  num_runs: 3
  ollama_model: "llama2:13b"
  output_dir: "results/"

experiment_1:
  num_documents: 5
  words_per_document: 200
  fact_positions: ["start", "middle", "end"]
  query_template: "What is the CEO's name?"

experiment_2:
  context_sizes: [2, 5, 10, 20, 50]
  document_length: 200
  task_types: ["simple", "complex"]

experiment_3:
  corpus_path: "data/corpora/hebrew_corpus/"
  chunk_size: 500
  top_k_values: [1, 2, 3, 5, 7, 10]
  embedding_model: "nomic-embed-text"

experiment_4:
  num_steps: 10
  strategies: ["select", "compress", "write", "hybrid"]
  max_tokens: 2000
```

### 8.6 API Interfaces

**LLM Interface:**
```python
class LLMInterface(ABC):
    @abstractmethod
    def query(self, context: str, query: str) -> Response:
        """Query LLM with context and question"""
        
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding vector"""
        
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
```

**Metrics Interface:**
```python
class MetricsCalculator:
    def accuracy(self, predicted, ground_truth) -> float: ...
    def semantic_similarity(self, text1, text2) -> float: ...
    def precision_at_k(self, retrieved, relevant, k) -> float: ...
```

---

## 9. Project Structure & Code Quality

### 9.1 Package Organization

**setup.py:**
```python
from setuptools import setup, find_packages

setup(
    name="context-windows-research",
    version="1.0.0",
    description="Graduate research on LLM context window behavior",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "ollama-python>=0.1.0",
        "chromadb>=0.4.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "tqdm>=4.60.0",
        "loguru>=0.7.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0", "black", "flake8", "mypy"],
    },
)
```

**src/__init__.py:**
```python
"""
Context Windows Research Framework

A comprehensive toolkit for analyzing LLM context window behavior.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Export main interfaces
from .llm_interface import LLMInterface, OllamaInterface
from .metrics import MetricsCalculator
from .visualization import Visualizer

__all__ = [
    "LLMInterface",
    "OllamaInterface", 
    "MetricsCalculator",
    "Visualizer",
]
```

### 9.2 Code Quality Standards

**PEP 8 Compliance:**
- Line length: ≤88 characters (Black default)
- 4-space indentation
- 2 blank lines between functions
- Snake_case for functions/variables
- PascalCase for classes

**Type Hints:**
```python
def calculate_accuracy(
    predictions: List[str], 
    ground_truth: List[str]
) -> float:
    """
    Calculate accuracy between predictions and ground truth.
    
    Args:
        predictions: List of predicted answers
        ground_truth: List of correct answers
        
    Returns:
        Accuracy score between 0 and 1
        
    Raises:
        ValueError: If lists have different lengths
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Lists must have same length")
    
    correct = sum(p == g for p, g in zip(predictions, ground_truth))
    return correct / len(predictions)
```

**Docstring Standards:**
- Google style docstrings
- Every public function/class/module
- Include: Description, Args, Returns, Raises, Examples

**Code Organization:**
- Files: ≤150 lines
- Functions: ≤50 lines, single responsibility
- Complexity: <10 cyclomatic complexity
- DRY: No code duplication

### 9.3 Naming Conventions

**Variables:**
```python
# Good
num_documents = 10
accuracy_scores = [0.85, 0.90, 0.88]
llm_interface = OllamaInterface()

# Bad
n = 10  # Too short
accuracyScores = [...]  # CamelCase
llmInt = OllamaInterface()  # Unclear abbreviation
```

**Functions:**
```python
# Good
def calculate_mean_accuracy(scores: List[float]) -> float: ...
def load_corpus_from_directory(path: Path) -> List[Document]: ...

# Bad
def calc(x): ...  # Unclear
def loadCorpus(path): ...  # CamelCase
```

**Constants:**
```python
# Good
MAX_CONTEXT_SIZE = 8192
DEFAULT_CHUNK_SIZE = 500
OLLAMA_BASE_URL = "http://localhost:11434"

# Bad
maxContextSize = 8192  # CamelCase
max_context = 8192  # Not uppercase
```

### 9.4 Error Handling

**Custom Exceptions:**
```python
class ContextWindowError(Exception):
    """Base exception for context window operations"""

class ExperimentConfigError(ContextWindowError):
    """Invalid experiment configuration"""

class LLMAPIError(ContextWindowError):
    """LLM API call failed"""
```

**Graceful Degradation:**
```python
def query_llm_with_retry(
    llm: LLMInterface, 
    context: str, 
    query: str,
    max_retries: int = 3
) -> Response:
    """Query LLM with automatic retry on failure"""
    for attempt in range(max_retries):
        try:
            return llm.query(context, query)
        except LLMAPIError as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Attempt {attempt+1} failed: {e}. Retrying...")
            time.sleep(2 ** attempt)  # Exponential backoff
```

### 9.5 Logging

**Configuration:**
```python
from loguru import logger

logger.remove()  # Remove default handler
logger.add(
    "logs/experiment.log",
    rotation="100 MB",
    retention="10 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)
```

**Usage:**
```python
logger.info(f"Starting Experiment 1 with {num_docs} documents")
logger.debug(f"Query: {query}, Context size: {len(context)} tokens")
logger.error(f"LLM API error: {e}")
logger.success(f"Experiment 1 completed in {duration:.2f}s")
```

---

## 10. Testing & Quality Assurance

### 10.1 Test Coverage Requirements

**Target: ≥85% code coverage**

**Coverage by Module:**
- Core utilities: ≥90% (metrics, statistics, data loading)
- Experiments: ≥80% (main experimental logic)
- Visualizations: ≥70% (plotting functions)
- Integration: ≥75% (end-to-end flows)

### 10.2 Unit Tests

**Example: test_metrics.py**
```python
import pytest
from src.metrics import MetricsCalculator

class TestMetricsCalculator:
    @pytest.fixture
    def calculator(self):
        return MetricsCalculator()
    
    def test_accuracy_perfect_match(self, calculator):
        """Test accuracy with perfect predictions"""
        predictions = ["Paris", "London", "Berlin"]
        ground_truth = ["Paris", "London", "Berlin"]
        assert calculator.accuracy(predictions, ground_truth) == 1.0
    
    def test_accuracy_no_match(self, calculator):
        """Test accuracy with no correct predictions"""
        predictions = ["Paris", "Rome", "Madrid"]
        ground_truth = ["London", "Berlin", "Vienna"]
        assert calculator.accuracy(predictions, ground_truth) == 0.0
    
    def test_accuracy_partial_match(self, calculator):
        """Test accuracy with partial match"""
        predictions = ["Paris", "Rome", "Berlin"]
        ground_truth = ["Paris", "London", "Berlin"]
        assert calculator.accuracy(predictions, ground_truth) == pytest.approx(0.667, rel=0.01)
    
    def test_accuracy_different_lengths_raises(self, calculator):
        """Test that different length lists raise ValueError"""
        with pytest.raises(ValueError):
            calculator.accuracy(["Paris"], ["London", "Berlin"])
    
    def test_precision_at_k(self, calculator):
        """Test precision@k calculation"""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1", "doc3", "doc5"]
        assert calculator.precision_at_k(retrieved, relevant, k=3) == pytest.approx(0.667, rel=0.01)
```

### 10.3 Integration Tests

**Example: test_experiment_1_integration.py**
```python
@pytest.mark.integration
def test_experiment_1_end_to_end(tmp_path):
    """Test complete Experiment 1 workflow"""
    # Setup
    config = {
        "num_documents": 3,
        "num_runs": 2,
        "output_dir": str(tmp_path)
    }
    
    # Execute
    experiment = Experiment1(config)
    results = experiment.run()
    
    # Validate
    assert "accuracy_by_position" in results
    assert len(results["accuracy_by_position"]) == 3  # start, middle, end
    assert all(0 <= acc <= 1 for acc in results["accuracy_by_position"].values())
    
    # Check outputs created
    assert (tmp_path / "exp1_summary.csv").exists()
    assert (tmp_path / "figures" / "accuracy_by_position.png").exists()
```

### 10.4 Mocking LLM Calls

**Example: Mock Ollama responses**
```python
@pytest.fixture
def mock_ollama(mocker):
    """Mock Ollama API calls"""
    mock = mocker.patch("src.llm_interface.OllamaInterface.query")
    mock.return_value = Response(
        text="The CEO is David Cohen",
        latency=0.5,
        tokens=10
    )
    return mock

def test_experiment_with_mock_llm(mock_ollama):
    """Test experiment logic without real LLM calls"""
    experiment = Experiment1(config)
    results = experiment.run()
    
    # Verify LLM was called correctly
    assert mock_ollama.call_count == 15  # 5 docs × 3 positions
    mock_ollama.assert_called_with(
        context=pytest.any(str),
        query="What is the CEO's name?"
    )
```

### 10.5 Test Execution

**Run all tests:**
```bash
pytest tests/ -v --cov=src --cov-report=html --cov-report=term
```

**Run specific test category:**
```bash
pytest tests/ -m unit          # Unit tests only
pytest tests/ -m integration   # Integration tests
pytest tests/ -k "experiment_1"  # Tests matching pattern
```

**Coverage report:**
```bash
pytest --cov=src --cov-report=html
# Open htmlcov/index.html in browser
```

### 10.6 Edge Cases & Error Scenarios

**Document Edge Cases:**
- Empty documents
- Documents with only special characters
- Extremely long documents (>10K words)
- Documents in unexpected encoding

**Query Edge Cases:**
- Empty query
- Query in different language than documents
- Ambiguous queries with multiple valid answers
- Queries with no relevant documents

**System Edge Cases:**
- LLM API unavailable
- Insufficient memory for large context
- Disk full during result saving
- Network interruption during long experiment

**Testing:**
```python
def test_empty_document_handling():
    """Test system handles empty documents gracefully"""
    documents = ["", "Valid content", ""]
    result = process_documents(documents)
    assert result is not None
    
def test_llm_api_failure_retry():
    """Test retry logic on LLM failure"""
    with pytest.raises(LLMAPIError):
        query_llm_with_retry(llm, context, query, max_retries=3)
```

---

## 11. Documentation Requirements

### 11.1 README.md Structure

```markdown
# Context Windows in Practice: Research Framework

## Overview
Brief description of project goals and scope.

## Quick Start
```bash
# Clone repository
git clone https://github.com/user/context-windows-research.git
cd context-windows-research

# Install dependencies
pip install -e .

# Run experiments
python scripts/run_experiments.py
```

## Installation
Step-by-step installation instructions.

## Configuration
How to configure experiments via YAML files.

## Usage
Examples of running each experiment.

## Results
How to interpret outputs and visualizations.

## Troubleshooting
Common issues and solutions.

## Contributing
Guidelines for contributions.

## License
MIT License
```

### 11.2 Architecture Documentation

**ARCHITECTURE.md must include:**
- System overview diagram
- Component interaction flows
- Design patterns used
- Technology stack rationale
- Scalability considerations
- Future extensibility points

### 11.3 API Documentation

**API.md must document:**
- All public interfaces
- Function signatures with type hints
- Parameters and return values
- Usage examples
- Error conditions

**Example:**
```markdown
### `LLMInterface.query()`

**Signature:**
```python
def query(self, context: str, query: str, **kwargs) -> Response:
```

**Parameters:**
- `context` (str): The contextual information to provide to the LLM
- `query` (str): The question or prompt
- `**kwargs`: Provider-specific options

**Returns:**
- `Response`: Object containing `text`, `latency`, `tokens`

**Raises:**
- `LLMAPIError`: If API call fails after retries

**Example:**
```python
llm = OllamaInterface(model="llama2")
response = llm.query(
    context="Document content...",
    query="What is the main topic?"
)
print(response.text)
```
```

### 11.4 Research Methodology Documentation

**RESEARCH_METHODOLOGY.md must include:**
- Research questions and hypotheses
- Experimental design decisions
- Control variables
- Randomization procedures
- Blinding (if applicable)
- Sample size justification
- Statistical methods chosen and why
- Threats to validity
- Ethical considerations

### 11.5 Statistical Analysis Documentation

**STATISTICAL_ANALYSIS.md must include:**
- All statistical tests used
- Assumptions checked
- Formulas with LaTeX
- Interpretation guidelines
- Multiple testing corrections
- Effect size calculations
- Power analysis results

### 11.6 Code Documentation (Docstrings)

**Module-level:**
```python
"""
Experiment 1: Lost in the Middle

This module implements the Lost in the Middle experiment, testing
whether LLMs exhibit position bias when retrieving information from
long contexts.

Key Components:
    - DocumentGenerator: Creates synthetic documents with embedded facts
    - Experiment1: Main experimental logic
    - Analyzer: Statistical analysis of results

Example:
    >>> from experiments import Experiment1
    >>> exp = Experiment1(config)
    >>> results = exp.run()
"""
```

**Class-level:**
```python
class Experiment1:
    """
    Lost in the Middle experiment implementation.
    
    Tests whether fact position (start, middle, end) affects retrieval
    accuracy in LLM responses.
    
    Attributes:
        config (dict): Experiment configuration
        num_documents (int): Documents per trial
        num_runs (int): Repetitions per condition
        
    Methods:
        run(): Execute full experiment
        analyze(): Perform statistical analysis
        visualize(): Create result plots
    """
```

**Function-level:**
```python
def embed_fact_at_position(
    document: str, 
    fact: str, 
    position: str
) -> str:
    """
    Embed a fact at specified position in document.
    
    Args:
        document: The base document text
        fact: The fact to embed
        position: One of 'start', 'middle', 'end'
        
    Returns:
        Document with embedded fact
        
    Raises:
        ValueError: If position not in ['start', 'middle', 'end']
        
    Example:
        >>> doc = "This is filler text. More filler."
        >>> fact = "The CEO is Alice."
        >>> result = embed_fact_at_position(doc, fact, 'middle')
        >>> assert "Alice" in result
    """
```

---

## 12. Timeline & Milestones

### 12.1 Four-Week Development Plan

**Week 1: Design & Infrastructure (20% effort)**

*Days 1-2: Project Setup*
- Create repository structure
- Configure development environment
- Install dependencies
- Validate Ollama connection
- Write configuration files

*Days 3-4: Core Infrastructure*
- Implement LLM interface
- Build data loading utilities
- Create metrics calculator
- Setup logging system
- **Milestone**: Infrastructure tests passing

*Days 5-7: Experiments 1-2 Implementation*
- Implement document generation
- Code Experiment 1 logic
- Code Experiment 2 logic
- Write unit tests
- **Milestone**: Experiments 1-2 executable

**Week 2: Execution & Advanced Experiments (30% effort)**

*Days 8-9: Data Collection for Exp 1-2*
- Run Experiment 1 (multiple conditions)
- Run Experiment 2 (all context sizes)
- Monitor and log progress
- **Milestone**: Exp 1-2 data collected

*Days 10-11: Experiments 3-4 Implementation*
- Setup ChromaDB
- Implement RAG pipeline
- Code context strategies
- Write integration tests
- **Milestone**: All experiments implemented

*Days 12-14: Data Collection for Exp 3-4*
- Prepare document corpus
- Run Experiment 3 (RAG analysis)
- Run Experiment 4 (strategies)
- **Milestone**: All data collected

**Week 3: Analysis & Visualization (30% effort)**

*Days 15-16: Statistical Analysis*
- Perform ANOVA, regression, correlation
- Calculate effect sizes and CIs
- Check assumptions
- Document findings
- **Milestone**: Statistical analyses complete

*Days 17-18: Visualization*
- Create all mandatory figures
- Design enhanced visualizations
- Ensure publication quality (300 DPI)
- Add statistical annotations
- **Milestone**: All figures complete

*Days 19-21: Cross-Experiment Synthesis*
- Identify patterns across experiments
- Develop unified theory
- Create decision frameworks
- Write practical recommendations
- **Milestone**: Insights synthesized

**Week 4: Documentation & Polish (20% effort)**

*Days 22-23: Core Documentation*
- Write ARCHITECTURE.md
- Complete API.md
- Finish RESEARCH_METHODOLOGY.md
- Write STATISTICAL_ANALYSIS.md
- **Milestone**: Technical docs complete

*Days 24-25: User-Facing Documentation*
- Polish README.md
- Write USER_GUIDE.md
- Create troubleshooting guide
- Add usage examples
- **Milestone**: User docs complete

*Days 26-27: Quality Assurance*
- Run full test suite
- Check code coverage (≥85%)
- Lint code (Black, flake8)
- Type check (mypy)
- **Milestone**: QA checks passing

*Day 28: Final Review & Submission*
- Review all deliverables
- Generate final report
- Create submission package
- Submit assignment
- **Milestone**: SUBMISSION COMPLETE ✓

### 12.2 Gantt Chart

```
Week 1: Infrastructure & Exp 1-2 Implementation
┌────┬────┬────┬────┬────┬────┬────┐
│ 1  │ 2  │ 3  │ 4  │ 5  │ 6  │ 7  │
└────┴────┴────┴────┴────┴────┴────┘
├─Setup─┤
      ├──Infra──┤
              ├────Exp 1-2────┤

Week 2: Execution & Exp 3-4 Implementation
┌────┬────┬────┬────┬────┬────┬────┐
│ 8  │ 9  │ 10 │ 11 │ 12 │ 13 │ 14 │
└────┴────┴────┴────┴────┴────┴────┘
├Run 1-2┤
      ├Imp 3-4┤
              ├─Run 3-4──┤

Week 3: Analysis & Visualization
┌────┬────┬────┬────┬────┬────┬────┐
│ 15 │ 16 │ 17 │ 18 │ 19 │ 20 │ 21 │
└────┴────┴────┴────┴────┴────┴────┘
├─Stats─┤
      ├─Viz──┤
            ├──Synthesis──┤

Week 4: Documentation & Submission
┌────┬────┬────┬────┬────┬────┬────┐
│ 22 │ 23 │ 24 │ 25 │ 26 │ 27 │ 28 │
└────┴────┴────┴────┴────┴────┴────┘
├Tech Docs┤
        ├User Docs┤
                ├─QA─┤
                    ├Final┤
```

### 12.3 Risk Management

**Risk Matrix:**

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Ollama API instability | Medium | High | Implement retry logic, checkpointing |
| Insufficient statistical power | Medium | High | Run power analysis early, increase n |
| Time overrun | High | Medium | Define MVP, defer enhancements |
| Corpus quality issues | Low | Medium | Validate early, have fallback sources |
| Unexpected findings | Low | Low | Document as discovery, adjust interpretation |

**Contingency Plans:**

*If Ollama unavailable:*
- Use OpenAI API as fallback
- Use cached responses for testing
- Focus on analysis of existing data

*If experiments take too long:*
- Reduce number of runs per condition
- Parallelize where possible
- Focus on mandatory variations only

*If statistical assumptions violated:*
- Use non-parametric alternatives
- Document violations and implications
- Consult statistical references

---

## 13. Deliverables & Evaluation

### 13.1 Complete Deliverables Checklist

**Code Deliverables (40 points):**

- [ ] Source code (`src/`) - modular, documented, tested (15 pts)
- [ ] Test suite (`tests/`) with ≥85% coverage (10 pts)
- [ ] Configuration files (`.yaml`, `.env.example`) (5 pts)
- [ ] Setup & installation (`setup.py`, `requirements.txt`) (5 pts)
- [ ] Utility scripts (`scripts/`) (5 pts)

**Data Deliverables (10 points):**

- [ ] Document corpora (`data/corpora/`) (3 pts)
- [ ] Ground truth (`data/ground_truth/`) (2 pts)
- [ ] Raw results (`results/raw/`) (3 pts)
- [ ] Processed results (`results/processed/`) (2 pts)

**Documentation Deliverables (20 points):**

- [ ] This PRD (`docs/PRD.md`) (3 pts)
- [ ] README.md with quick start (3 pts)
- [ ] ARCHITECTURE.md with diagrams (3 pts)
- [ ] API.md with all interfaces (2 pts)
- [ ] RESEARCH_METHODOLOGY.md (3 pts)
- [ ] STATISTICAL_ANALYSIS.md (3 pts)
- [ ] USER_GUIDE.md (2 pts)
- [ ] Inline code documentation (1 pt)

**Research Deliverables (30 points):**

- [ ] All 4 experiments executed (8 pts)
- [ ] Statistical analyses with hypothesis tests (8 pts)
- [ ] Publication-quality figures (≥15) (8 pts)
- [ ] Final research report synthesizing findings (6 pts)

**Bonus Deliverables (10 points):**

- [ ] Interactive visualizations/dashboard (3 pts)
- [ ] Novel research questions with original findings (3 pts)
- [ ] Comparison with published literature (2 pts)
- [ ] Reusable framework for community (2 pts)

**Total: 100 points (+ 10 bonus)**

### 13.2 Evaluation Rubric (Self-Assessment)

**Project Documentation (20%):**

| Criterion | Points | Self-Score |
|-----------|--------|------------|
| Clear problem definition and goals | 5 | ___ |
| KPIs defined and measurable | 3 | ___ |
| Detailed functional requirements | 5 | ___ |
| Architecture documentation (C4, UML) | 4 | ___ |
| ADRs (Architectural Decision Records) | 3 | ___ |

**README & Code Documentation (15%):**

| Criterion | Points | Self-Score |
|-----------|--------|------------|
| Step-by-step installation | 3 | ___ |
| Detailed usage instructions | 3 | ___ |
| Screenshots/demos | 2 | ___ |
| Configuration guide | 2 | ___ |
| Troubleshooting | 2 | ___ |
| Docstrings (every function/class) | 3 | ___ |

**Project Structure & Code Quality (15%):**

| Criterion | Points | Self-Score |
|-----------|--------|------------|
| Clear modular folder structure | 4 | ___ |
| Separation of code, data, results | 3 | ___ |
| Files ≤150 lines | 2 | ___ |
| Consistent naming conventions | 2 | ___ |
| Single Responsibility Principle | 2 | ___ |
| DRY (no code duplication) | 2 | ___ |

**Configuration & Security (10%):**

| Criterion | Points | Self-Score |
|-----------|--------|------------|
| Separate config files (.yaml, .env) | 3 | ___ |
| No hardcoded values | 2 | ___ |
| Example config (.env.example) | 2 | ___ |
| No API keys in source | 2 | ___ |
| Updated .gitignore | 1 | ___ |

**Testing & QA (15%):**

| Criterion | Points | Self-Score |
|-----------|--------|------------|
| Unit tests with ≥85% coverage | 5 | ___ |
| Edge case testing | 3 | ___ |
| Coverage reports | 2 | ___ |
| Comprehensive error handling | 3 | ___ |
| Debugging logs | 2 | ___ |

**Research & Analysis (15%):**

| Criterion | Points | Self-Score |
|-----------|--------|------------|
| Systematic experiments | 3 | ___ |
| Sensitivity analysis | 2 | ___ |
| Statistical hypothesis testing | 3 | ___ |
| Mathematical formulations (LaTeX) | 2 | ___ |
| Academic references | 2 | ___ |
| Publication-quality visualizations | 3 | ___ |

**UI/UX & Extensibility (10%):**

| Criterion | Points | Self-Score |
|-----------|--------|------------|
| Intuitive interface | 3 | ___ |
| Workflow documentation | 2 | ___ |
| Extension points/hooks | 3 | ___ |
| Plugin architecture | 2 | ___ |

**Total: 100 points** | **Your Score: ___**

### 13.3 Submission Package Structure

```
context-windows-research-submission/
├── README.md                          # Overview and quick start
├── requirements.txt                   # Dependencies
├── setup.py                           # Package installation
├── .env.example                       # Configuration template
├── src/                               # Source code
├── data/                              # Data files
├── results/                           # Experimental outputs
├── tests/                             # Test suite
├── config/                            # Configuration files
├── scripts/                           # Execution scripts
├── docs/                              # Documentation
│   ├── PRD.md                         # This document
│   ├── ARCHITECTURE.md
│   ├── API.md
│   ├── RESEARCH_METHODOLOGY.md
│   ├── STATISTICAL_ANALYSIS.md
│   ├── USER_GUIDE.md
│   └── FINAL_REPORT.md               # Comprehensive findings
└── notebooks/                         # Analysis notebooks
```

### 13.4 Final Report Structure

**FINAL_REPORT.md should include:**

1. **Executive Summary** (1 page)
   - Key findings
   - Main contributions
   - Practical recommendations

2. **Introduction** (2 pages)
   - Motivation
   - Research questions
   - Significance

3. **Methodology** (3 pages)
   - Experimental design
   - Data collection procedures
   - Statistical methods

4. **Experiment 1 Results** (4 pages)
   - Findings with visualizations
   - Statistical analysis
   - Interpretation

5. **Experiment 2 Results** (4 pages)

6. **Experiment 3 Results** (4 pages)

7. **Experiment 4 Results** (4 pages)

8. **Discussion** (3 pages)
   - Cross-experiment patterns
   - Theory connections
   - Limitations

9. **Conclusions** (2 pages)
   - Summary of findings
   - Practical implications
   - Future work

10. **References**
    - Academic citations
    - Technical documentation

**Total: 25-30 pages**

---

## 14. Appendix: Self-Assessment Guide Integration

This PRD fully addresses all categories from the self-assessment guide:

### ✅ Covered Categories

**3.2.1 Project Documentation (20%)**
- ✅ PRD with clear problem, goals, KPIs (§1-2)
- ✅ Detailed functional/non-functional requirements (§3-7)
- ✅ Architecture documentation (§8)
- ✅ ADRs implicitly in design choices

**3.2.2 README & Code Documentation (15%)**
- ✅ README structure defined (§11.1)
- ✅ Docstring standards (§11.6)
- ✅ API documentation (§11.3)

**3.2.3 Project Structure & Code Quality (15%)**
- ✅ Directory structure (§8.2)
- ✅ Code quality standards (§9.2)
- ✅ Naming conventions (§9.3)

**3.2.4 Configuration & Security (10%)**
- ✅ Configuration management (§8.5)
- ✅ Security requirements (§7.5)
- ✅ .env example provided

**3.2.5 Testing & QA (15%)**
- ✅ Test coverage requirements (§10.1)
- ✅ Unit and integration tests (§10.2-10.3)
- ✅ Edge case handling (§10.6)

**3.2.6 Research & Analysis (15%)**
- ✅ Systematic experiments (§3-6)
- ✅ Statistical methods (detailed in each experiment)
- ✅ Mathematical formulations with LaTeX
- ✅ Visualization requirements
- ✅ Academic references expected

**3.2.7 UI/UX & Extensibility (10%)**
- ✅ Intuitive interface design (§7.3)
- ✅ Extension points (§8.4, §9.2)
- ✅ Plugin architecture (Strategy pattern)

**Advanced Topics:**
- ✅ Package organization (§9.1)
- ✅ setup.py, __init__.py
- ✅ Proper imports and relative paths

---

## 15. Quick Reference: Key Formulas

### Experiment 1: Lost in the Middle

**Accuracy Model:**
$$\text{Accuracy}(p) = \beta_0 + \beta_1 \cdot \mathbb{1}_{\text{middle}}(p) + \beta_2 \cdot \mathbb{1}_{\text{end}}(p) + \varepsilon$$

**Sample Size:**
$$n = \frac{2(z_{\alpha/2} + z_{\beta})^2 \sigma^2}{\delta^2}$$

**Cohen's d:**
$$d = \frac{\bar{x}_1 - \bar{x}_2}{s_{\text{pooled}}}$$

### Experiment 2: Context Size Impact

**Logarithmic Model:**
$$\text{Accuracy}(n) = \alpha - \beta \log(n) + \varepsilon$$

**Latency Scaling:**
$$\text{Latency}(n) = \gamma_0 + \gamma_1 n + \gamma_2 n^2$$

**AIC/BIC:**
$$\text{AIC} = 2k - 2\ln(\hat{L}), \quad \text{BIC} = k\ln(n) - 2\ln(\hat{L})$$

### Experiment 3: RAG Impact

**Precision@k:**
$$P@k = \frac{|\text{Retrieved}_k \cap \text{Relevant}|}{k}$$

**nDCG@k:**
$$\text{nDCG@k} = \frac{\sum_{i=1}^{k} \frac{\text{rel}_i}{\log_2(i+1)}}{\text{IDCG@k}}$$

**Cost Savings:**
$$S = (N - k) \cdot L \cdot p_{\text{token}} - C_{\text{overhead}}$$

### Experiment 4: Context Strategies

**Context Growth:**
$$|C_t| = |C_0| + \sum_{i=1}^{t} |O_i|$$

**Weighted Score:**
$$S_{\text{strategy}} = \sum_{i} w_i \cdot m_{i,\text{strategy}}$$

**AUC:**
$$\text{AUC} = \sum_{t=1}^{T-1} \frac{A_t + A_{t+1}}{2} \cdot \Delta t$$

---

## Document Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | Nov 2025 | Initial comprehensive PRD | Research Team |

---

**End of Product Requirements Document**

This PRD provides a complete blueprint for executing the Context Windows research project at Master's level academic standards. Use it as your guide throughout the project lifecycle.

**Next Steps:**
1. Review and internalize this PRD
2. Set up development environment
3. Begin Week 1 tasks
4. Refer back to this document regularly
5. Update as needed based on discoveries

**Good luck with your research! 🚀**
