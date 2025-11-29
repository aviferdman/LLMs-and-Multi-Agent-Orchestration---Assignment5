---
name: run-all-tests
description: Execute the complete test suite and generate coverage report
---

# Run All Tests Command

Executes the full test suite with coverage reporting.

## What It Does

1. Runs pytest on all test files
2. Generates coverage report
3. Validates ≥85% coverage threshold
4. Saves HTML coverage report
5. Logs results to `agents_log.txt`

## Usage

Say: **"Run all tests"** or **"Execute test suite"**

## Prerequisites

- All experiments implemented
- Test files created by test-engineer

## Expected Outcome

- All tests pass
- Coverage ≥85%
- HTML report in htmlcov/
- Results logged
