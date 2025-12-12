# Lab 03: Model Customization

Fine-tune open-source models using **QLoRA** - making customization possible on consumer hardware.

## Prerequisites

```bash
pip install transformers peft datasets accelerate bitsandbytes
```

**Hardware:**
- GPU with 8GB+ VRAM recommended
- CPU works for small models (slower)

## What You'll Learn

- Loading quantized models (4-bit)
- Configuring LoRA adapters
- Preparing training data
- Fine-tuning with Hugging Face
- Saving and loading adapters
- Merging adapters with base models

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_finetuning_with_qlora.ipynb` | Complete QLoRA fine-tuning guide |

## Why QLoRA?

| Aspect | Full Fine-tuning | QLoRA |
|--------|-----------------|-------|
| Memory | 100+ GB | 8-16 GB |
| Trainable Params | 100% | ~0.1% |
| Training Time | Days | Hours |
| Adapter Size | Huge | Few MB |

## Recommended Models for Learning

| Model | Size | Memory Needed |
|-------|------|---------------|
| microsoft/phi-2 | 2.7B | ~4 GB |
| TinyLlama/TinyLlama-1.1B | 1.1B | ~2 GB |
| mistralai/Mistral-7B | 7B | ~6 GB (4-bit) |

## Time to Complete

Approximately 45 minutes (including training time)
