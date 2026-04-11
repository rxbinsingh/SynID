# SynID: Zero-Shot Identity-Consistent Image Generation

> Generate multiple images of the same character from a text prompt alone — no real reference images, no dataset, no pretraining.

[![arXiv](https://img.shields.io/badge/arXiv-coming_soon-red)](.)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](.)

---

## What is SynID?

SynID is a closed-loop self-distillation pipeline for identity-consistent image generation. Given only a text description, it:

1. Generates synthetic anchor images from the model's own prior
2. Trains a multi-token identity projector on CLIP embeddings
3. Refines the identity via bootstrap candidates and drift correction
4. Trains a lightweight UNet cross-attention adapter on-the-fly (~100 steps, ~5 min)
5. Injects identity at two levels: text embeddings (coarse) + UNet cross-attention (fine)

**No real images. No external dataset. No pretraining.**

The reference implementation is packaged in an app-first way:

- `synid_ui.py` launches the interface immediately
- heavy models are loaded only when the user clicks `Load Pipelines`
- the UI can expose a mobile QR via the Gradio share URL
- research/evaluation code remains available through the backend files

---

## Results

| Character | FW-CLIP | Pairwise Consistency |
|-----------|---------|---------------------|
| Woman (brunette) | 0.9515 | 0.9435 ± 0.022 |
| Elderly man | 0.9508 | 0.9430 ± 0.031 |
| Anime girl | 0.9655 | 0.9541 ± 0.013 |
| Young man | 0.9407 | 0.9528 ± 0.019 |
| Woman (redhead) | 0.9625 | 0.9411 ± 0.013 |
| **Mean** | **0.9542** | **0.9469 ± 0.020** |

**Full system ArcFace: 0.791** (comparable to Arc2Face 0.79, zero real images)

### Ablation

| Config | CLIP | ArcFace | Time |
|--------|------|---------|------|
| Baseline (single anchor) | 0.9603 | 0.7562 | 55s |
| + Multi-anchor ensemble | 0.9676 | 0.7581 | 78s |
| + Bootstrap + Drift correction | 0.9665 | 0.7560 | 154s |
| **Full system (+ UNet adapter)** | **0.9690** | **0.7912** | 158s |

---

## Quick Start

### App UI

**Google Colab (T4 GPU recommended)**

Upload these files into the same Colab working directory:

- `synid_ui.py`
- `identity_projection_complete.py`
- `evaluation_harness.py`

Then run:

```python
!pip install -q gradio diffusers transformers accelerate controlnet_aux safetensors huggingface_hub insightface onnxruntime-gpu
exec(open("synid_ui.py").read())
```

Usage flow:

1. Open the Gradio app
2. Click `Load Pipelines`
3. Create a character
4. Generate, evaluate, or use pose-free mode
5. Scan the QR block if you want to open the app on mobile

**Local (GPU required)**

```bash
pip install -r requirements.txt
python synid_ui.py
```

### Research Backend / Scripting

If you want the backend without the UI:

```python
from identity_projection_complete import (
    init_synid_backend,
    create_character,
    attach_identity_adapters,
    register_adapter_hooks,
    generate_with_adapter,
    save_checkpoint,
    load_checkpoint,
    pipe,
)

init_synid_backend()

profile = create_character(
    identity_prompt="young woman, brown eyes, dark hair, photorealistic",
    anchor_seed=1234,
    num_identity_tokens=4,
    train_steps=250,
)

adapters = attach_identity_adapters(pipe.unet, identity_dim=768, scale=0.5)
hooks = register_adapter_hooks(pipe.unet)

image = generate_with_adapter(
    profile.identity_tokens,
    profile.character_core_prompt + ", bright smile, studio portrait",
    profile.pose_image,
    pipe.unet,
    pipe,
    seed=5555,
)
```

### Full Benchmark

The five-character benchmark is now opt-in rather than auto-running on import.

```python
from identity_projection_complete import init_synid_backend, run_full_benchmark

init_synid_backend()
run_full_benchmark()
```

---

## Files

| File | Description |
|------|-------------|
| `synid_ui.py` | App-first Gradio interface with staged backend loading and mobile QR |
| `identity_projection_complete.py` | Backend pipeline: initialization, identity learning, adapter training, checkpointing |
| `evaluation_harness.py` | Evaluation suites + ablation study |
| `synid_paper.md` | Paper (Markdown) |
| `synid_paper.tex` | Paper (LaTeX, arXiv-ready) |
| `requirements.txt` | Dependencies |

---

## App Flow

The repository now follows an app-style runtime flow:

1. Launch `synid_ui.py`
2. Load core pipelines explicitly with `Load Pipelines`
3. Create a character profile from text
4. Reuse the same profile for generation, evaluation, saving, and mobile access

This keeps startup responsive and avoids notebook-style auto-execution of the full benchmark.

---

## How It Works

```
Text prompt
    ↓
Multi-anchor ensemble (4 anchors, softmax-weighted by CLIP similarity)
    ↓
Multi-token projector (512 → 1024 → 4×768, diversity + ArcFace loss)
    ↓
Bootstrap refinement (8 expression candidates, ArcFace-scored selection)
    ↓
Drift correction (2 rounds, self-correcting feedback loop)
    ↓
UNet adapter training (200 steps MSE + CLIP + ArcFace, on 8 synthetic images)
    ↓
Generation (text injection + UNet cross-attention, adaptive identity scale)
```

---

## Key Differences from IP-Adapter

| | IP-Adapter FaceID | SynID |
|--|-------------------|-------|
| Real reference image | Required | Not needed |
| Training data | ~1M image pairs | 4 synthetic images |
| Pretraining time | Days | None |
| Per-character time | Inference only | ~5 min |
| CLIP identity | 0.854 | **0.969** |
| ArcFace | 0.132* | **0.791** |

*IP-Adapter run without pose conditioning

---

## Citation

```bibtex
@article{synid2025,
  title={SynID: Zero-Shot Identity-Consistent Image Generation via Synthetic Bootstrapping and On-the-Fly UNet Adaptation},
  author={Singh, Robin},
  journal={arXiv preprint},
  year={2025}
}
```

---

## Status

🚧 **Work in progress** — actively improving. Current focus:
- [ ] Remove ControlNet pose dependence entirely
- [ ] Stronger face-specific loss (full ArcFace integration)
- [ ] Multi-character consistency (same identity, different prompts)
- [ ] SDXL support
