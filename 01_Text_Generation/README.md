# Lab 01: Text Generation with Local LLMs

This lab introduces text generation using **Ollama**, which lets you run large language models completely locally and free.

## Prerequisites

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.2

# Install Python package
pip install ollama
```

## What You'll Learn

- Basic text generation and completion
- Text summarization
- Question answering
- Code generation and explanation
- Streaming responses
- System prompts and personas
- Multi-turn conversations
- Model parameters (temperature, etc.)
- JSON output formatting

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_text_generation_with_ollama.ipynb` | Complete guide to text generation |

## Recommended Models

| Model | Command | Use Case |
|-------|---------|----------|
| Llama 3.2 3B | `ollama pull llama3.2` | Fast, general purpose |
| Llama 3.1 8B | `ollama pull llama3.1` | Better quality |
| Mistral 7B | `ollama pull mistral` | Good balance |
| Code Llama | `ollama pull codellama` | Code generation |
| Phi-3 | `ollama pull phi3` | Fast, lightweight |

## Time to Complete

Approximately 25 minutes
