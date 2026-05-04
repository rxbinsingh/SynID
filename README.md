# SynID

**Zero-shot identity-consistent image generation from text alone.**

[![Paper](https://img.shields.io/badge/paper-ResearchGate-00CCBB?logo=researchgate&logoColor=white)](https://doi.org/10.13140/RG.2.2.30671.85925)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-SynID-FFD21E)](https://huggingface.co/rxbinsingh/SynID)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)

---

SynID generates multiple consistent images of the same character from a text description — no real reference photos, no dataset, no pretraining. It runs in about five minutes per character on a single T4 GPU.

The pipeline is a closed-loop self-distillation system: the diffusion model generates its own training data, refines its own identity embedding, and trains its own adapter — entirely from text.

---

## How it works

```
Text prompt
    │
    ▼
Multi-anchor ensemble
  4 synthetic anchors, softmax-weighted by CLIP similarity
    │
    ▼
Multi-token identity projector
  CLIP embedding → 4 × 768 identity tokens
  Trained with text alignment + diversity + ArcFace losses
    │
    ▼
Bootstrap refinement
  20 expression-diverse candidates generated and scored
  Top-K selected with diversity enforcement
  Projector retrained on refined embedding
    │
    ▼
Drift correction
  Probe image generated, CLIP drift measured
  Projector fine-tuned to close the gap (2 rounds)
    │
    ▼
UNet adapter training
  Lightweight cross-attention adapters on all transformer blocks
  Trained on 8 synthetic images: MSE + CLIP + ArcFace losses (~260 steps)
    │
    ▼
Generation
  Dual-level identity injection:
    · Text embedding (coarse, adaptive scale)
    · UNet cross-attention (fine-grained, every denoising step)
  Identity-aware negative conditioning
```

---

## Results

### Multi-character benchmark (5 characters)

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

## Quick start

### Colab (recommended — T4 GPU)

Upload `synid_ui.py`, `identity_projection_complete.py`, and `evaluation_harness.py` to the same Colab working directory, then run:

```python
!pip install -q gradio diffusers transformers accelerate controlnet_aux \
    safetensors huggingface_hub insightface onnxruntime torchvision

exec(open("synid_ui.py").read())
```

Then in the UI:
1. Click **Load Pipelines** — loads DreamShaper, ControlNet, CLIP, OpenPose
2. Enter a character description and click **Create Character**
3. Use the **Generate**, **Pose-Free**, or **Evaluate** tabs
4. Scan the QR code to open the app on mobile

### Local (GPU required)

```bash
git clone https://github.com/rxbinsingh/SynID
cd SynID
pip install -r requirements.txt
python synid_ui.py
```

> GPU users: replace `onnxruntime` with `onnxruntime-gpu` in `requirements.txt` for faster face detection.

### Scripting / research backend

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
hooks    = register_adapter_hooks(pipe.unet)

image = generate_with_adapter(
    profile.identity_tokens,
    profile.character_core_prompt + ", bright smile, studio portrait",
    profile.pose_image,
    pipe.unet,
    pipe,
    seed=5555,
)
image.save("output.png")
```

### Full benchmark (5 characters)

```python
from identity_projection_complete import init_synid_backend, run_full_benchmark

init_synid_backend()
run_full_benchmark()
```

---

## Files

| File | Description |
|---|---|
| `synid_ui.py` | Gradio UI — staged pipeline loading, character creation, generation, evaluation, mobile QR |
| `identity_projection_complete.py` | Full backend — initialization, identity learning, adapter training, checkpointing |
| `evaluation_harness.py` | Evaluation suites (quick / full), ablation study, standardized benchmark |
| `requirements.txt` | Python dependencies |

---

## Checkpointing

Save and reload a trained character profile:

```python
from identity_projection_complete import save_checkpoint, load_checkpoint, attach_identity_adapters

# save
save_checkpoint(profile, adapters, "/path/to/checkpoints/my_character")

# load
adapters = attach_identity_adapters(pipe.unet, identity_dim=768, scale=0.5)
profile  = load_checkpoint(adapters, "/path/to/checkpoints/my_character")
```

Export as a portable `.character` archive:

```python
from identity_projection_complete import export_character
export_character("my_character", checkpoint_dir="/path/to/checkpoints")
```

---

## Requirements

- Python 3.9+
- CUDA GPU (T4 or better recommended; 8 GB+ VRAM)
- See `requirements.txt` for full dependency list

Key dependencies: `torch`, `diffusers`, `transformers`, `controlnet_aux`, `insightface`, `gradio`

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
  year    = {2026},
  doi     = {10.13140/RG.2.2.30671.85925},
  url     = {https://doi.org/10.13140/RG.2.2.30671.85925}
}
```

---

## License

[MIT](LICENSE) © 2025 Robin Singh
