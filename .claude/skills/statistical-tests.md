---
name: statistical-tests
description: Performs statistical hypothesis tests including ANOVA, t-tests, regression, effect sizes, and assumption checks.
allowed-tools: Bash, Read, Write
---

# Statistical Tests Skill

Provides comprehensive statistical analysis functions using scipy and numpy.

## Available Tests

### Hypothesis Tests
- One-way ANOVA
- Two-way ANOVA
- Independent t-test
- Paired t-test
- Linear regression
- Polynomial regression
- Post-hoc tests (Tukey HSD)

### Effect Sizes
- Cohen's d
- Eta-squared (η²)
- R-squared (R²)

### Assumption Checks
- Shapiro-Wilk (normality)
- Levene's test (homoscedasticity)
- Q-Q plots
- Residual plots

## Usage

```python
from src.analysis.statistics import (
    run_anova,
    calculate_cohens_d,
    check_normality
)

# Run ANOVA
f_stat, p_value = run_anova(groups)

# Calculate effect size
d = calculate_cohens_d(group1, group2)

# Check assumptions
is_normal = check_normality(data)
```

## When to Use

Use this skill for:
- Analyzing experiment results
- Comparing experimental conditions
- Validating statistical assumptions
- Calculating effect sizes
- Fitting predictive models
