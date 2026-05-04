---
title: SynID
emoji: 🪪
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
short_description: Zero-shot identity-consistent image generation from text alone
tags:
  - text-to-image
  - identity
  - diffusion
  - stable-diffusion
  - controlnet
  - gradio
---

# SynID

**Zero-shot identity-consistent image generation from text alone.**

[![Paper](https://img.shields.io/badge/paper-ResearchGate-00CCBB?logo=researchgate&logoColor=white)](https://doi.org/10.13140/RG.2.2.30671.85925)
[![GitHub](https://img.shields.io/badge/github-SynID-181717?logo=github)](https://github.com/rxbinsingh/SynID)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rxbinsingh/SynID/blob/main/LICENSE)

---

SynID generates multiple consistent images of the same character from a text description.  
No real reference photos. No dataset. No pretraining. ~5 minutes per character on a T4 GPU.

## How to use

1. Click **Load Pipelines** — loads DreamShaper, ControlNet, CLIP, and OpenPose (~2 min on first run)
2. Enter a character description and click **Create Character**
3. Use the **Expressions & Scenes**, **Pose-Free**, or **Evaluate** tabs to generate images
4. Scan the QR code to open the app on your phone

## How it works

The pipeline is a closed-loop self-distillation system:

```
Text prompt
  → Multi-anchor ensemble (4 synthetic anchors, CLIP-weighted)
  → Multi-token identity projector (CLIP → 4×768 tokens, ArcFace loss)
  → Bootstrap refinement (20 candidates, diversity-enforced selection)
  → Drift correction (2 rounds, self-correcting feedback)
  → UNet adapter training (~260 steps on 8 synthetic images)
  → Generation (dual-level injection: text + UNet cross-attention)
```

## Results

| Character | CLIP Identity | Pairwise Consistency |
|---|---|---|
| Woman (brunette) | 0.9515 | 0.9435 ± 0.022 |
| Elderly man | 0.9508 | 0.9430 ± 0.031 |
| Anime girl | 0.9655 | 0.9541 ± 0.013 |
| Young man | 0.9407 | 0.9528 ± 0.019 |
| Woman (redhead) | 0.9625 | 0.9411 ± 0.013 |
| **Mean** | **0.9542** | **0.9469 ± 0.020** |

**ArcFace: 0.791** — comparable to Arc2Face (~0.79) trained on 21M real faces.

## Paper

> Robin Singh (2025). *SynID: Zero-Shot Identity-Consistent Image Generation via Synthetic Bootstrapping and On-the-Fly UNet Adaptation.*  
> [https://doi.org/10.13140/RG.2.2.30671.85925](https://doi.org/10.13140/RG.2.2.30671.85925)

## Note on cold starts

This Space runs on a GPU that sleeps after inactivity. The first load after sleep takes ~30 seconds — this is normal. Pipeline model downloads happen once and are cached.
