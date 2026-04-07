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

## Quick Start (Google Colab, T4 GPU)

```python
# Cell 1 — Install
!pip install -q diffusers transformers accelerate controlnet_aux safetensors huggingface_hub insightface onnxruntime-gpu

# Cell 2 — Run full pipeline
exec(open("identity_projection_complete.py").read())

# Cell 3 — Run evaluation + ablation
exec(open("evaluation_harness.py").read())
```

Upload both `.py` files to Colab before running. Runtime: ~70 min for 5 characters on T4.

---

## Files

| File | Description |
|------|-------------|
| `identity_projection_complete.py` | Full pipeline — single file, everything included |
| `evaluation_harness.py` | Evaluation suites + ablation study |
| `synid_paper.md` | Paper (Markdown) |
| `synid_paper.tex` | Paper (LaTeX, arXiv-ready) |
| `requirements.txt` | Dependencies |

---

## Customizing Your Character

Edit the `TEST_CHARACTERS` list in `identity_projection_complete.py`:

```python
TEST_CHARACTERS = [
    {
        "name": "my_character",
        "seed": 1234,
        "prompt": "young woman, blue eyes, blonde hair, freckles, upper body portrait, photorealistic"
    },
]
```

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
  author={Anonymous},
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
