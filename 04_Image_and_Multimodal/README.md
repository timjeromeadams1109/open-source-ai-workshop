# Lab 04: Image Generation and Multimodal AI

Generate images with **Stable Diffusion** and analyze them with **vision models** - all locally.

## Prerequisites

```bash
pip install diffusers transformers accelerate torch Pillow
ollama pull llava  # For vision
```

**Hardware:**
- GPU with 8GB+ VRAM strongly recommended
- CPU works but is very slow (10+ min/image)
- Apple Silicon (MPS) supported

## What You'll Learn

- Text-to-image generation
- Prompt engineering for better results
- Style transfer and modifiers
- Image analysis with vision models
- Building captioning pipelines

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_image_generation.ipynb` | Complete image generation guide |

## Models Used

| Model | Purpose | VRAM |
|-------|---------|------|
| SDXL Turbo | Fast image generation | ~8 GB |
| SD 1.5 | Compatible alternative | ~4 GB |
| LLaVA | Vision understanding | ~8 GB |

## Prompt Tips

```
Subject + Style + Quality + Details

Example:
"A majestic mountain landscape, oil painting style,
highly detailed, dramatic lighting, 8k resolution"
```

## Time to Complete

Approximately 30 minutes
