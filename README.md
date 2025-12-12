# Open Source AI Workshop

A **zero-cost, fully open-source** AI workshop covering text generation, RAG, fine-tuning, multimodal applications, and AI agents - all running locally on your machine.

This workshop is inspired by the [Amazon Bedrock Workshop](https://github.com/aws-samples/amazon-bedrock-workshop) but requires **no cloud services, no API keys, and no budget**.

## Prerequisites

- **Python 3.10+**
- **8GB+ RAM** (16GB recommended for larger models)
- **GPU recommended** but not required (CPU inference works, just slower)

### Required Software

1. **Ollama** - Local LLM runtime
   ```bash
   # macOS/Linux
   curl -fsSL https://ollama.com/install.sh | sh

   # Then pull a model
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

2. **Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Workshop Labs

| Lab | Description | Time |
|-----|-------------|------|
| **01 - Text Generation** | Generate text and code with local LLMs (Ollama) | 25 min |
| **02 - RAG** | Build a retrieval-augmented generation system with ChromaDB | 35 min |
| **03 - Model Customization** | Fine-tune models with QLoRA and Hugging Face | 45 min |
| **04 - Image & Multimodal** | Image generation and vision models | 30 min |
| **05 - Agents** | Build AI agents with tool use and function calling | 30 min |

## Tech Stack

| Component | Open Source Tool | Purpose |
|-----------|-----------------|---------|
| LLM Runtime | Ollama | Run LLMs locally |
| Models | Llama 3, Mistral, Phi | Text generation |
| Embeddings | nomic-embed-text, all-MiniLM | Vector embeddings |
| Vector DB | ChromaDB | Store and search embeddings |
| Fine-tuning | Hugging Face + PEFT | Model customization |
| Image Gen | Stable Diffusion | Image generation |
| Vision | LLaVA | Multimodal understanding |
| Agents | LangChain | Agent orchestration |

## Getting Started

```bash
# Clone this repository
git clone https://github.com/timjeromeadams1109/open-source-ai-workshop.git
cd open-source-ai-workshop

# Install dependencies
pip install -r requirements.txt

# Start with Lab 01
jupyter notebook 01_Text_Generation/
```

## Why Open Source?

- **Zero Cost** - No API fees, no cloud bills
- **Privacy** - All data stays on your machine
- **Learning** - Understand how AI works under the hood
- **Customization** - Full control over models and parameters
- **Offline** - Works without internet after initial setup

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
