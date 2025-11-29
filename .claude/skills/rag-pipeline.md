---
name: rag-pipeline
description: Retrieval-Augmented Generation pipeline with ChromaDB. Handles document indexing, retrieval, and generation.
allowed-tools: Bash, Read, Write
---

# RAG Pipeline Skill

Complete RAG pipeline using ChromaDB for document storage and retrieval.

## Components

1. **Document Loader** - Load and chunk documents
2. **Embedder** - Generate embeddings (sentence-transformers)
3. **Indexer** - Store in ChromaDB
4. **Retriever** - Semantic search with filters
5. **Generator** - LLM generation with retrieved context

## Usage

```python
from src.rag.chromadb_client import ChromaDBClient
from src.rag.retriever import Retriever

# Initialize
db = ChromaDBClient(persist_directory="./data/chromadb")
retriever = Retriever(db_client=db, top_k=5)

# Index documents
db.index_documents(documents, collection_name="docs")

# Retrieve
results = retriever.retrieve(query="Your query", top_k=5)

# Generate with context
response = llm.generate(
    prompt=build_rag_prompt(query, results)
)
```

## Features

- Chunking strategies (fixed size, semantic)
- Multiple embedding models
- Metadata filtering
- Similarity thresholds
- Re-ranking
- Metrics (Precision@k, nDCG@k)

## When to Use

Use this skill for:
- Experiment 3 (RAG Impact)
- Document retrieval tasks
- Context-aware generation
- Comparing RAG vs full context
