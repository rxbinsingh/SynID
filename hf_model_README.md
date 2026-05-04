---
language:
  - en
license: mit
tags:
  - text-to-image
  - stable-diffusion
  - controlnet
  - identity-consistent
  - diffusion
  - gradio
  - pytorch
library_name: diffusers
pipeline_tag: text-to-image
---

# SynID

**Zero-shot identity-consistent image generation from text alone.**

[![Paper](https://img.shields.io/badge/paper-ResearchGate-00CCBB?logo=researchgate&logoColor=white)](https://doi.org/10.13140/RG.2.2.30671.85925)
[![GitHub](https://img.shields.io/badge/github-SynID-181717?logo=github)](https://github.com/rxbinsingh/SynID)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rxbinsingh/SynID/blob/main/LICENSE)

---

SynID generates multiple consistent images of the same character from a text description.  
No real reference photos. No dataset. No pretraining. ~5 minutes per character on a T4 GPU.

## How it works

```
Text prompt
  → Multi-anchor ensemble       (4 synthetic anchors, CLIP-weighted)
  → Multi-token projector       (CLIP → 4×768 identity tokens, ArcFace loss)
  → Bootstrap refinement        (20 candidates, diversity-enforced selection)
  → Drift correction            (2 rounds, self-correcting feedback loop)
  → UNet adapter training       (~260 steps on 8 synthetic images)
  → Generation                  (dual-level injection: text + UNet cross-attention)
```

The pipeline is a closed-loop self-distillation system — the diffusion model generates its own training data, refines its own identity embedding, and trains its own adapter entirely from text.

---

## Quick start

**Google Colab (T4 GPU recommended)**

```python
!pip install -q gradio diffusers transformers accelerate controlnet_aux \
    safetensors huggingface_hub insightface onnxruntime torchvision

!git clone https://github.com/rxbinsingh/SynID
%cd SynID

exec(open("synid_ui.py").read())
```

Then in the UI:
1. Click **Load Pipelines**
2. Enter a character description → **Create Character**
3. Generate variations, run evaluation, or use pose-free mode

**Local (GPU required)**

```bash
git clone https://github.com/rxbinsingh/SynID
cd SynID
pip install -r requirements.txt
python synid_ui.py
```

**Scripting**

```python
from identity_projection_complete import (
    init_synid_backend, create_character,
    attach_identity_adapters, register_adapter_hooks,
    generate_with_adapter, pipe,
)

init_synid_backend()

profile = create_character(
    identity_prompt="young woman, brown eyes, dark hair, photorealistic",
    anchor_seed=1234,
    num_identity_tokens=4,
    train_steps=250,
)

adapters = attach_identity_adapters(pipe.unet, identity_dim=768, scale=0.5)
hooks    = register_adapter_hooks(pipe.unet)

image = generate_with_adapter(
    profile.identity_tokens,
    profile.character_core_prompt + ", bright smile, studio portrait",
    profile.pose_image,
    pipe.unet, pipe,
    seed=5555,
)
image.save("output.png")
```

---

## Results

### Multi-character benchmark

| Character | CLIP Identity | Pairwise Consistency |
|---|---|---|
| Woman (brunette) | 0.9515 | 0.9435 ± 0.022 |
| Elderly man | 0.9508 | 0.9430 ± 0.031 |
| Anime girl | 0.9655 | 0.9541 ± 0.013 |
| Young man | 0.9407 | 0.9528 ± 0.019 |
| Woman (redhead) | 0.9625 | 0.9411 ± 0.013 |
| **Mean** | **0.9542** | **0.9469 ± 0.020** |

**ArcFace (full system): 0.791** — comparable to Arc2Face (~0.79) trained on 21M real faces.

### Ablation

| Configuration | CLIP | ArcFace | Time |
|---|---|---|---|
| Baseline — single anchor, no bootstrap, no adapter | 0.9603 | 0.7562 | 55 s |
| + Multi-anchor ensemble | 0.9676 | 0.7581 | 78 s |
| + Bootstrap + drift correction | 0.9665 | 0.7560 | 154 s |
| **Full system — + UNet adapter** | **0.9690** | **0.7912** | 158 s |

### Comparison with prior methods

| Method | CLIP | ArcFace | Real image required | Training data |
|---|---|---|---|---|
| IP-Adapter FaceID | 0.854 | 0.132 | Yes | ~1M pairs |
| Arc2Face | — | ~0.79 | Yes | ~21M faces |
| PhotoMaker | — | ~0.618 | Yes (multiple) | Real |
| **SynID (ours)** | **0.969** | **0.791** | **No** | **8 synthetic** |

---

## Repository structure

| File | Description |
|---|---|
| `synid_ui.py` | Gradio UI — staged pipeline loading, character creation, generation, evaluation |
| `identity_projection_complete.py` | Full backend — initialization, identity learning, adapter training, checkpointing |
| `evaluation_harness.py` | Evaluation suites (quick / full), ablation study, standardized benchmark |
| `app.py` | HuggingFace Space entry point |
| `requirements.txt` | Python dependencies |

---

## Base models used

This pipeline downloads and uses the following models at runtime:

- [Lykon/DreamShaper](https://huggingface.co/Lykon/DreamShaper) — base diffusion model (SD 1.5)
- [lllyasviel/control_v11p_sd15_openpose](https://huggingface.co/lllyasviel/control_v11p_sd15_openpose) — ControlNet pose conditioning
- [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) — CLIP image/text encoder
- [lllyasviel/ControlNet](https://huggingface.co/lllyasviel/ControlNet) — OpenPose detector

No weights are stored in this repository. All models are downloaded automatically on first run.

---

## Hardware requirements

- CUDA GPU required (T4 or better, 8 GB+ VRAM)
- ~5 minutes per character on T4
- CPU-only is not supported

---

## Paper

> **SynID: Zero-Shot Identity-Consistent Image Generation via Synthetic Bootstrapping and On-the-Fly UNet Adaptation**  
> Robin Singh, 2025  
> [https://doi.org/10.13140/RG.2.2.30671.85925](https://doi.org/10.13140/RG.2.2.30671.85925)

```bibtex
@article{singh2025synid,
  title   = {SynID: Zero-Shot Identity-Consistent Image Generation via
             Synthetic Bootstrapping and On-the-Fly UNet Adaptation},
  author  = {Singh, Robin},
  year    = {2025},
  doi     = {10.13140/RG.2.2.30671.85925},
  url     = {https://doi.org/10.13140/RG.2.2.30671.85925}
}
```

---

## License

[MIT](https://github.com/rxbinsingh/SynID/blob/main/LICENSE) © 2025 Robin Singh
