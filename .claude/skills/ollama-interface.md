---
name: ollama-interface
description: Interface with Ollama LLM for text generation. Handles retries, errors, and logging.
allowed-tools: Bash, Read, Write
---

# Ollama Interface Skill

Provides reliable interface to Ollama LLM with error handling and retry logic.

## Usage

```python
from src.llm_interface.ollama_client import OllamaClient

client = OllamaClient(
    base_url="http://localhost:11434",
    model="llama2:13b",
    temperature=0.0
)

response = client.generate(
    prompt="Your prompt here",
    max_tokens=2048
)
```

## Features

- Automatic retry with exponential backoff
- Connection error handling
- Timeout management
- Response validation
- Logging of all requests

## Error Handling

- Network errors: Retry up to 3 times
- Timeout errors: Increase timeout and retry
- Model not found: Log error and fail
- Invalid response: Log and raise exception

## When to Use

Use this skill whenever any agent needs to:
- Generate text with LLM
- Test prompts
- Run experiments
- Validate LLM responses
