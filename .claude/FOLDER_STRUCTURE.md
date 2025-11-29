````markdown
# Claude Folder Structure Guide - Assignment 5

This document explains the organization of the `.claude` folder for the Context Windows Research Project.

## Overview

The `.claude` folder contains all multi-agent orchestration configuration for parallel development of the context windows research project.

```
.claude/
├── main.md                              # Main orchestrator (entry point)
├── FOLDER_STRUCTURE.md                  # This file
├── settings.local.json                  # Local configuration (optional)
│
├── agents/                              # All agent definitions (FLAT - no subfolders)
│   ├── infrastructure-builder.md        # Sets up project structure
│   ├── experiment-1-developer.md        # Implements Lost in the Middle
│   ├── experiment-2-developer.md        # Implements Context Size Impact
│   ├── experiment-3-developer.md        # Implements RAG Impact
│   ├── experiment-4-developer.md        # Implements Context Engineering
│   ├── test-engineer.md                 # Writes test suite
│   ├── statistical-analyst.md           # Performs statistical analysis
│   ├── visualization-specialist.md      # Creates plots and figures
│   └── documentation-writer.md          # Writes documentation
│
├── commands/                            # Runnable commands (FLAT - no subfolders)
│   ├── setup-project.md                 # Initialize infrastructure
│   ├── run-experiment-1.md              # Execute Experiment 1
│   ├── run-all-tests.md                 # Run test suite
│   └── generate-final-report.md         # Create final report
│
└── skills/                              # Reusable skills (FLAT - no subfolders)
    ├── ollama-interface.md              # LLM interface utilities
    ├── statistical-tests.md             # Statistical analysis functions
    ├── plot-generator.md                # Visualization utilities
    └── rag-pipeline.md                  # RAG implementation utilities
```

## Key Principles

### 1. **Flat Structure - No Subfolders**
- All agent files go directly in `agents/` folder
- All command files go directly in `commands/` folder
- All skill files go directly in `skills/` folder
- No subdirectories or categorization folders
- All files are `.md` format

### 2. **Parallel Agent Coordination**
- All agents communicate via `agents_log.txt` in project root
- No central coordinator blocking workflow
- Agents check log before starting work
- Agents announce their activities in real-time

### 3. **Agent Categories (Logical, Not Physical)**

While files are flat, agents serve different roles:

**Infrastructure & Development:**
- infrastructure-builder.md
- experiment-1-developer.md
- experiment-2-developer.md
- experiment-3-developer.md
- experiment-4-developer.md
- test-engineer.md

**Analysis & Reporting:**
- statistical-analyst.md
- visualization-specialist.md
- documentation-writer.md

### 4. **Shared Log Protocol**

Every agent MUST follow this protocol:

```python
# Read log before starting
with open('agents_log.txt', 'r') as f:
    log = f.read()
    # Check if someone else is working on this

# Write status update
import datetime
timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
with open('agents_log.txt', 'a') as f:
    f.write(f"[{timestamp}] [agent-name] [STARTED] Task description\n")
```

**Log Entry Format:**
```
[TIMESTAMP] [AGENT_NAME] [STATUS] Message
```

**Status Types:**
- `[STARTED]` - Beginning work
- `[PROGRESS]` - Intermediate update
- `[COMPLETED]` - Task finished
- `[ERROR]` - Problem encountered
- `[WAITING]` - Waiting for dependency
- `[BLOCKED]` - Cannot proceed

### 5. **File Organization**

**All files are `.md` format in flat folders:**
- Agent definitions in `agents/`: Just the filename, e.g., `infrastructure-builder.md`
- Commands in `commands/`: Just the filename, e.g., `setup-project.md`
- Skills in `skills/`: Just the filename, e.g., `ollama-interface.md`

**Naming Conventions:**
- Agents: `{role}-{specialization}.md` or `{role}.md`
- Commands: `{verb}-{object}.md`
- Skills: `{name}.md` (descriptive names)

### 6. **Agent Definition Structure**

```markdown
---
name: agent-name
description: What this agent does
tools: Bash, Read, Write, Task
model: sonnet
---

# Agent Title

Role and responsibilities description.

## Coordination Protocol

1. Read agents_log.txt
2. Check for conflicts
3. Log [STARTED]
4. Do work
5. Log [COMPLETED]

## Responsibilities

- Task 1
- Task 2

## Dependencies

- Requires X to be completed first
- Waits for Y before proceeding
```

## Usage Patterns

### Single Agent Development

If working alone:
```
User: "Set up the infrastructure"
→ infrastructure-builder agent executes
```

### Parallel Multi-Agent Development

If working with multiple terminals:

**Terminal 1:**
```
User: "Implement Experiment 1"
→ experiment-1-developer starts, logs to agents_log.txt
```

**Terminal 2:**
```
User: "Implement Experiment 2"
→ experiment-2-developer starts, logs to agents_log.txt
→ Reads log, sees Exp 1 in progress, proceeds with Exp 2
```

**Terminal 3:**
```
User: "Write tests for experiments"
→ test-engineer starts, logs to agents_log.txt
→ Reads log, waits for Exp 1/2 to complete
→ Begins writing tests when experiments ready
```

### Checking Progress

Any agent can read `agents_log.txt` to see:
- What's currently being worked on
- What's completed
- What's blocked
- Who's working on what

## Commands vs Agents

**Commands:** User-facing executable workflows
- Located in `commands/`
- Can be run directly
- May invoke multiple agents

**Agents:** Specialized workers
- Located in `agents/` subdirectories
- Invoked by commands or other agents
- Focus on specific tasks

## Skills vs Agents

**Skills:** Reusable utility functions
- Located in `skills/` subdirectories
- No state, pure functionality
- Can be used by any agent

**Agents:** Stateful workers with goals
- Have specific roles and responsibilities
- Coordinate with other agents
- Make decisions based on context

## Best Practices

1. **Always Check the Log First**
   - Read `agents_log.txt` before starting
   - Verify no conflicts
   - Check dependencies

2. **Log Frequently**
   - Log when starting
   - Log progress milestones
   - Log completion
   - Log errors

3. **Handle Dependencies**
   - Check for `[COMPLETED]` status
   - Use `[WAITING]` if blocked
   - Retry after dependencies satisfied

4. **Avoid Duplication**
   - If another agent started same task, coordinate
   - First logger gets priority
   - Second agent can assist or move to different task

5. **Clear Communication**
   - Use descriptive log messages
   - Include file paths when relevant
   - Specify what was done, not just "completed"

## Example Multi-Agent Workflow

**Goal:** Implement and test Experiment 1

**agents_log.txt progression:**

```
[2025-11-29 14:00:00] [infrastructure-builder] [STARTED] Creating src/ directory structure
[2025-11-29 14:01:30] [infrastructure-builder] [COMPLETED] Created src/experiments, src/llm_interface, src/analysis, src/visualization
[2025-11-29 14:02:00] [experiment-1-developer] [STARTED] Implementing Lost in the Middle experiment in src/experiments/experiment1.py
[2025-11-29 14:05:00] [test-engineer] [WAITING] Waiting for experiment-1-developer to complete before writing tests
[2025-11-29 14:15:00] [experiment-1-developer] [PROGRESS] Implemented document generation and position variation logic
[2025-11-29 14:25:00] [experiment-1-developer] [COMPLETED] Experiment 1 implementation done in src/experiments/experiment1.py
[2025-11-29 14:26:00] [test-engineer] [STARTED] Writing tests for Experiment 1 in tests/test_experiment1.py
[2025-11-29 14:35:00] [test-engineer] [COMPLETED] Tests written with 90% coverage
```

## Troubleshooting

**Problem:** Two agents started same task
**Solution:** First logger wins, second agent checks log and moves to different task

**Problem:** Agent waiting for dependency that never completes
**Solution:** Check for `[ERROR]` or `[BLOCKED]` status, intervene manually

**Problem:** Log file becomes very large
**Solution:** Archive old logs, start fresh log for new development session

**Problem:** Can't tell what's happening
**Solution:** Read `agents_log.txt` - it's the single source of truth

## Extending the System

### Adding New Agents

1. Create agent file in appropriate category folder
2. Define name, description, tools, model in frontmatter
3. Include coordination protocol section
4. Document dependencies

### Adding New Commands

1. Create command file in `commands/`
2. Define workflow steps
3. Specify which agents to invoke
4. Include expected outcomes

### Adding New Skills

1. Create folder in `skills/`
2. Add `SKILL.md` with implementation
3. Document inputs, outputs, usage
4. Make it reusable and stateless

## Summary

The `.claude` folder enables:
- ✅ Parallel multi-agent development
- ✅ Coordinated execution via shared log
- ✅ Clear separation of concerns
- ✅ Reusable components
- ✅ Scalable architecture
- ✅ Real-time progress tracking

**Key File:** `agents_log.txt` - The coordination hub for all agent activities!
````
