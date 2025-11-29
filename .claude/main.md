# Context Windows Research Project - Main Orchestrator

## 🚨 MULTI-AGENT COORDINATION PROTOCOL 🚨

**THIS PROJECT SUPPORTS PARALLEL AGENT EXECUTION**

All agents communicate via a shared log file: `agents_log.txt` in the project root.

## 📋 Shared Log Protocol

**Every agent MUST:**
1. **Read** `agents_log.txt` before starting work to see what others are doing
2. **Write** status updates when starting/completing tasks
3. **Lock** the file during writes (use atomic append operations)
4. **Format** entries as: `[TIMESTAMP] [AGENT_NAME] [STATUS] Message`

**Status Types:**
- `[STARTED]` - Agent began work on a task
- `[PROGRESS]` - Intermediate update
- `[COMPLETED]` - Task finished successfully
- `[ERROR]` - Problem encountered
- `[WAITING]` - Waiting for dependency

**Example Log Entry:**
```
[2025-11-29 14:30:22] [infrastructure-builder] [STARTED] Creating src/ directory structure
[2025-11-29 14:31:05] [infrastructure-builder] [COMPLETED] Directory structure created
[2025-11-29 14:31:10] [experiment-1-dev] [STARTED] Implementing Lost in the Middle experiment
```

## 🎯 Project Overview

**Assignment:** Context Windows in Practice - Graduate-Level Research Framework

**Four Experiments:**
1. **Lost in the Middle** - Information retrieval at different context positions
2. **Context Size Impact** - Accuracy vs context length scaling
3. **RAG Impact** - Full context vs retrieval-augmented generation
4. **Context Engineering** - Strategies for managing growing context

## 🏗️ Architecture

```
LLMs-and-Multi-Agent-Orchestration---Assignment5/
├── src/                    # Source code
│   ├── experiments/        # Experiment implementations
│   ├── llm_interface/      # Ollama integration
│   ├── rag/               # RAG components
│   ├── analysis/          # Statistical analysis
│   └── visualization/     # Plotting utilities
├── data/                   # Data files
├── results/               # Experimental outputs
├── tests/                 # Test suite
├── config/                # Configuration
├── docs/                  # Documentation
├── .claude/               # Agent definitions
└── agents_log.txt         # **Shared coordination log**
```

## 👥 Available Agents (Flat Structure)

All agent files are in `.claude/agents/` (no subfolders):

### Infrastructure & Development
- `infrastructure-builder.md` - Creates project structure and basic code
- `experiment-1-developer.md` - Implements Experiment 1 (Lost in the Middle)
- `experiment-2-developer.md` - Implements Experiment 2 (Context Size Impact)
- `experiment-3-developer.md` - Implements Experiment 3 (RAG Impact)
- `experiment-4-developer.md` - Implements Experiment 4 (Context Engineering)
- `test-engineer.md` - Writes comprehensive test suite

### Analysis & Reporting
- `statistical-analyst.md` - Performs statistical analysis
- `visualization-specialist.md` - Creates publication-quality figures
- `documentation-writer.md` - Writes all documentation

## 📁 Commands & Skills (Also Flat)

**Commands** (in `.claude/commands/`):
- `setup-project.md` - Initialize infrastructure
- `run-experiment-1.md` - Execute Experiment 1
- `run-all-tests.md` - Run test suite
- `generate-final-report.md` - Create final report

**Skills** (in `.claude/skills/`):
- `ollama-interface.md` - LLM interface utilities
- `statistical-tests.md` - Statistical analysis functions
- `plot-generator.md` - Visualization utilities
- `rag-pipeline.md` - RAG implementation utilities

## 🚀 Quick Start Commands

### For Parallel Development (Multiple Terminal Windows)

**Terminal 1 - Infrastructure:**
```
"Set up the project infrastructure: create directories, setup.py, config files, and LLM interface"
```

**Terminal 2 - Experiment 1:**
```
"Implement Experiment 1: Lost in the Middle"
```

**Terminal 3 - Experiment 2:**
```
"Implement Experiment 2: Context Size Impact"
```

**Terminal 4 - Tests:**
```
"Write comprehensive tests for all experiments"
```

Each terminal will:
1. Check `agents_log.txt` for current activity
2. Log their own progress
3. Coordinate with other agents automatically

## 📚 Documentation Reference

- **PRD:** `docs/PRD.md` - Complete product requirements
- **Architecture:** To be created in `docs/ARCHITECTURE.md`
- **API:** To be created in `docs/API.md`

## 🔧 Technical Stack

- **Python:** 3.9+
- **LLM:** Ollama with llama2:13b
- **Vector DB:** ChromaDB
- **Stats:** scipy, numpy, pandas
- **Viz:** matplotlib, seaborn, plotly
- **Testing:** pytest, pytest-cov

## ⚡ Agent Coordination Rules

1. **No Blocking:** Agents work independently unless explicit dependencies exist
2. **Log Everything:** All major actions logged to `agents_log.txt`
3. **Check First:** Always read log before starting work
4. **Announce Intent:** Log `[STARTED]` before beginning tasks
5. **Report Completion:** Log `[COMPLETED]` when done
6. **Handle Conflicts:** If two agents start same task, first logger wins
7. **Dependencies:** Check for `[COMPLETED]` status of prerequisites

## 🎓 Academic Standards

- **Level:** MIT Academic Standard (Level 4)
- **Compliance:** ISO/IEC 25010
- **Coverage:** ≥85% test coverage
- **Documentation:** Comprehensive docstrings, README, architecture docs
- **Research:** Hypothesis testing, statistical analysis, visualizations

## 📞 Getting Help

Refer to:
- `docs/PRD.md` for detailed requirements
- `.claude/FOLDER_STRUCTURE.md` for agent organization
- `.claude/commands/` for common tasks
- `.claude/skills/` for reusable utilities

---

**Ready to coordinate! Launch agents in parallel and watch them collaborate via agents_log.txt**
