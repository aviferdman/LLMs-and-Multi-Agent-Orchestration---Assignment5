---
name: setup-project
description: Complete project setup - creates infrastructure, installs dependencies, configures environment
---

# Setup Project Command

This command sets up the entire project from scratch.

## What It Does

1. Invokes **infrastructure-builder** agent to create directory structure and files
2. Installs Python dependencies from requirements.txt
3. Configures environment variables
4. Validates Ollama is running
5. Initializes ChromaDB

## Usage

Simply say: **"Set up the project"** or **"Initialize the infrastructure"**

## Expected Outcome

- Complete directory structure created
- All configuration files in place
- Dependencies installed
- Environment ready for development
- Entry logged in `agents_log.txt`
