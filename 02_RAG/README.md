# Lab 02: Retrieval-Augmented Generation (RAG)

Build a RAG system that answers questions from your own documents - completely locally and free.

## Prerequisites

```bash
pip install chromadb ollama
ollama pull llama3.2
ollama pull nomic-embed-text
```

## What You'll Learn

- Creating embeddings with local models
- Vector storage with ChromaDB
- Semantic similarity search
- Building a complete RAG pipeline
- Document chunking strategies
- Metadata filtering

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_rag_with_chromadb.ipynb` | Complete RAG implementation |

## Architecture

```
Question → Embed → Search ChromaDB → Retrieve Docs → Build Prompt → LLM → Answer
```

## Key Concepts

- **Embeddings**: Convert text to vectors that capture meaning
- **Vector DB**: Store and efficiently search embeddings
- **Retrieval**: Find relevant documents for a query
- **Augmentation**: Add retrieved context to the prompt
- **Generation**: LLM generates answer from context

## Time to Complete

Approximately 35 minutes
