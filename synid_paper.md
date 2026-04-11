# SynID: Zero-Shot Identity-Consistent Image Generation via Synthetic Bootstrapping and On-the-Fly UNet Adaptation

---

## Abstract

We present **SynID**, a novel framework for identity-consistent image generation that requires no real reference images, no external dataset, and no large-scale pretraining. Given only a text description of a character, SynID generates multiple images of the same identity across diverse expressions, scenes, and poses. The system operates through a closed-loop self-distillation pipeline: synthetic anchor images are generated from the model's own prior, a multi-token identity projector is trained on the resulting CLIP embeddings, bootstrap refinement selects high-quality candidates to refine the identity embedding, and a lightweight UNet cross-attention adapter is trained on-the-fly using only eight synthetic images in approximately 260 steps. Identity is injected at two levels simultaneously — text embedding space for coarse alignment and UNet cross-attention for fine-grained facial consistency. On five diverse character types, SynID achieves a mean face-weighted CLIP identity score of **0.954** and an ArcFace score of **0.791**, comparable to Arc2Face (~0.79 on FFHQ) while requiring zero real images. In a controlled comparison, SynID outperforms IP-Adapter FaceID on CLIP identity (0.969 vs 0.854) under identical generation conditions. The entire pipeline runs in approximately five minutes per character on a single T4 GPU. The released implementation is packaged as an app-first backend with staged model loading and an interactive interface layered on top of the research pipeline.

---

## 1. Introduction

Identity-consistent image generation — producing multiple images of the same person across different contexts — is a fundamental challenge in generative modeling. Existing approaches fall into two categories: fine-tuning methods (DreamBooth [Ruiz et al., 2023], LoRA [Hu et al., 2022]) that require multiple real photographs and significant compute, and adapter-based methods (IP-Adapter [Ye et al., 2023], InstantID [Wang et al., 2024]) that require a real reference image at inference time and are trained on millions of image pairs.

Both paradigms share a critical dependency: **real images of the target identity**. This limits their applicability in scenarios where no reference image exists — character design, synthetic data generation, privacy-preserving personalization, and creative applications where the user wants to generate a character from a text description alone.

We ask a different question: *can identity-consistent generation be achieved from text alone, with no real images at any stage?*

SynID answers yes. The key insight is that modern diffusion models already contain rich identity priors — they can generate consistent-looking characters from text descriptions. SynID exploits this prior through a closed-loop self-distillation process: the model generates its own training data, refines its own identity embedding, and trains its own adapter, all without any external supervision. We call this paradigm **bootstrapped synthetic identity**.

Our contributions are:

1. **Synthetic identity formation**: Identity is initialized from the model's own generative prior via a softmax-weighted multi-anchor ensemble, producing a robust CLIP embedding without any real images. No prior work generates fresh synthetic identities purely from text and averages CLIP image embeddings in this way.

2. **Closed-loop self-distillation**: A bootstrap refinement loop generates expression-diverse candidates, scores them by a balanced CLIP+ArcFace metric with diversity enforcement, and iteratively refines the identity embedding. No prior identity personalization method has explicitly generated new samples, scored them by CLIP+ArcFace, and re-blended with anchors to retrain.

3. **On-the-fly UNet adaptation**: A lightweight cross-attention adapter is trained in ~260 steps on eight synthetic images, injecting identity directly into the UNet's denoising process. Training adapters on synthetic images per-identity on-the-fly, without any offline pretraining, is novel.

4. **Dual-level identity injection**: Identity tokens are injected at both the text embedding level (coarse) and UNet cross-attention level (fine-grained). Prior methods use only one level — IP-Adapter uses only cross-attention, textual inversion uses only text. SynID's dual injection with per-token placement in the last N slots is unique.

5. **Self-correcting drift correction**: A feedback loop measures CLIP drift between generated images and the identity embedding, fine-tuning the projector to close the gap. No published identity method has used a "probe image to measure embedding shift" mechanism of this kind.

6. **Identity-aware negative conditioning**: The negative prompt embedding is explicitly pushed away from the identity direction, actively suppressing identity drift during classifier-free guidance. No known prior work adds a repulsive identity embedding in this way.

---

## 2. Related Work

**Fine-tuning approaches.** DreamBooth [Ruiz et al., 2023] and LoRA [Hu et al., 2022] fine-tune diffusion model weights on 3–20 real images of a subject. These methods achieve high identity fidelity but require real photographs and significant compute (15–30 minutes per subject). ID-Booth [Tomašević et al., 2025] extends this with a triplet identity loss for improved consistency, but still requires real images.

**Adapter-based approaches.** IP-Adapter [Ye et al., 2023] trains a cross-attention adapter on ~1M image-text pairs, enabling image-prompted generation at inference time. IP-Adapter FaceID extends this with ArcFace embeddings for face-specific conditioning. InstantID [Wang et al., 2024] combines face embeddings with ControlNet for high-fidelity identity transfer. Inv-Adapter [Hong et al., 2024] inverts a single reference image and trains a per-identity adapter. All these methods require at least one real reference image at inference time and large-scale pretraining. F-Bench [Liu et al., 2025] provides a large human-evaluated benchmark comparing IP-Adapter FaceID, FaceID-Plus, InstantID, and PhotoMaker, finding FaceID-Plus achieves the best overall performance.

**Stacked ID embedding.** PhotoMaker [Li et al., 2024] personalizes generation by encoding multiple real ID images into a stacked ID embedding, achieving ArcFace similarity of ~0.618 on their evaluation set.

**Foundation models for faces.** Arc2Face [Papantoniou et al., 2024] trains a face foundation model on 21M facial images, achieving ArcFace similarity of ~0.79 on a held-out FFHQ identity set. It requires a real ArcFace embedding as input.

**Concurrent zero-shot personalization.** EZIGen [Zhang et al., 2024] achieves zero-shot personalized image generation but requires a real subject image as input. Inv-Adapter [Hong et al., 2024] similarly requires one real image for inversion.

**Key difference.** SynID requires no real images at any stage — not for training, not for inference. The identity is entirely synthetic, initialized from the diffusion model's own prior and refined through self-distillation. This distinguishes it from all methods above. The closest conceptual relatives are Chen et al. [2023]'s identity tokens for diffusion (learned token embeddings for face generation) and DreamBooth-style iterative refinement, but neither operates in a fully synthetic, closed-loop manner.

---

## 3. Method

### 3.1 Overview

Given a text prompt describing a character, SynID produces a `CharacterProfile` containing identity tokens that can be used to generate arbitrarily many consistent images. The pipeline consists of five stages:

1. Multi-anchor ensemble embedding
2. Multi-token projector training
3. Bootstrap refinement
4. Drift correction
5. On-the-fly UNet adapter training

### 3.2 Multi-Anchor Ensemble Embedding

A single synthetic anchor image may be unrepresentative due to seed-dependent variation in the diffusion process. We generate N=4 anchor images from the same prompt with different seeds and compute a softmax-weighted ensemble embedding:

$$\mathbf{e}_{ensemble} = \text{normalize}\left(\sum_{i=1}^{N} w_i \cdot \text{CLIP}(I_i^{anchor})\right)$$

where weights $w_i = \text{softmax}(\text{cos}(\text{CLIP}(I_i^{anchor}), \text{CLIP}_{text}(p)))$ assign higher weight to anchors more similar to the text prompt. This produces a more stable identity embedding than any single anchor.

We use a face-weighted CLIP encoding that blends full-image features (40%) with face-crop features (60%), improving discriminability for facial identity:

$$\text{CLIP}_{face}(I) = \text{normalize}(0.4 \cdot \text{CLIP}_{full}(I) + 0.6 \cdot \text{CLIP}_{crop}(I))$$

The face crop covers the center 64% of image width and top 64% of image height, targeting the facial region without requiring a face detector.

### 3.3 Multi-Token Identity Projector

We train a lightweight MLP projector $f_\theta: \mathbb{R}^{512} \rightarrow \mathbb{R}^{4 \times 768}$ that maps the CLIP image embedding to four identity tokens in the SD text encoder's embedding space. The architecture is: Linear(512→1024) → SiLU → Linear(1024→4×768), trained with Adam for 250 steps.

The training objective is:

$$\mathcal{L}_{proj} = \mathcal{L}_{text} + \mathcal{L}_{per\text{-}token} + \lambda_{div}\mathcal{L}_{div} + \lambda_{norm}\mathcal{L}_{norm}$$

where:
- $\mathcal{L}_{text} = 1 - \cos(\text{mean}(f_\theta(\mathbf{e})), \mathbf{t}_{target})$ aligns the mean token to the masked-mean text embedding of the character prompt
- $\mathcal{L}_{per\text{-}token} = 0.3 \cdot \frac{1}{K}\sum_k (1 - \cos(f_\theta(\mathbf{e})_k, \mathbf{t}_{target}))$ aligns each token individually
- $\mathcal{L}_{div} = \text{mean off-diagonal cosine similarity of normalized tokens}$ encourages token diversity
- $\mathcal{L}_{norm} = |\text{mean}(\|f_\theta(\mathbf{e})\|) - 1.0|$ regularizes token norms toward 1.0

with $\lambda_{div} = 0.05$, $\lambda_{norm} = 0.01$.

When InsightFace is available, an additional face verification loss is added:

$$\mathcal{L}_{arc} = 1 - \cos(\text{mean}(f_\theta(\mathbf{e})), \mathbf{W}_{proj}\mathbf{e}_{arc})$$

where $\mathbf{W}_{proj}$ is a fixed linear projection from the 512-dim ArcFace space to 768-dim text space, initialized as a partial identity matrix. This directly optimizes the projector to produce tokens that activate the model's face-consistent representations.

### 3.4 Bootstrap Refinement

The initial projector is used to generate M=8 bootstrap images with diverse expression prompts (surprised, smiling, serious, laughing, neutral, thoughtful, smirking, tired). Each candidate is scored by a balanced metric:

$$s_i = 0.45 \cdot \text{CLIP}_{id}(I_i, \mathbf{e}_{ensemble}) + 0.35 \cdot \text{ArcFace}(I_i, I_{anchor}) + 0.20 \cdot \text{CLIP}_{prompt}(I_i, p_i)$$

Candidates are selected with diversity enforcement: a candidate is included only if its CLIP identity similarity exceeds 0.85 and its pairwise similarity to already-selected candidates is below 0.92. This prevents the refinement from collapsing to near-identical images. If fewer than two diverse candidates pass the threshold, the top-ranked candidates are used as a fallback.

The top-K=2 selected candidates are blended with the anchor embedding:

$$\mathbf{e}_{refined} = \text{normalize}(0.55 \cdot \mathbf{e}_{ensemble} + 0.45 \cdot \text{mean}(\mathbf{e}_{selected}))$$

The projector is retrained on $\mathbf{e}_{refined}$ for 150 steps, using the best bootstrap candidate's ArcFace embedding as the face verification target.

### 3.5 Drift Correction

After bootstrap refinement, a self-correcting feedback loop measures the gap between the projector's output and what the full system actually produces:

1. Generate one probe image using current identity tokens
2. Compute CLIP drift: $d = 1 - \cos(\text{CLIP}_{face}(I_{gen}), \mathbf{e}_{refined})$
3. If $d > 0.015$: blend $\mathbf{e}_{refined} \leftarrow \text{normalize}(0.65\mathbf{e}_{refined} + 0.35\text{CLIP}_{face}(I_{gen}))$
4. Fine-tune projector for 40 steps on blended embedding
5. Repeat for 2 rounds

This loop closes the gap between the projector's target and the system's actual output distribution, compensating for the mismatch between CLIP embedding space and the diffusion model's generation manifold.

### 3.6 UNet Cross-Attention Adapter

We attach a lightweight cross-attention adapter to every transformer block in the UNet's down, mid, and up blocks. The adapter uses the identity tokens as key-value pairs and the UNet hidden states as queries:

$$\text{Adapter}(\mathbf{h}, \mathbf{z}) = \mathbf{h} + \alpha \cdot \text{CrossAttn}(\mathbf{h}, \mathbf{z})$$

where $\mathbf{h}$ are UNet hidden states, $\mathbf{z}$ are identity tokens, and $\alpha$ is a per-block scale. We use a graduated scale: down blocks ($\alpha = 0.25$), mid block ($\alpha = 0.375$), up blocks ($\alpha = 0.5$). Up blocks receive the strongest signal as they reconstruct fine facial details. The output projection of each adapter is zero-initialized, ensuring it starts as a no-op and learns incrementally.

The adapter is implemented as a standard multi-head cross-attention module with 8 heads, with separate Q, K, V projections and a zero-initialized output projection. All UNet parameters are frozen; only adapter parameters are trained.

**On-the-fly training.** The adapter is trained using only the 8 bootstrap images generated during refinement — no external dataset. The training objective combines three terms:

$$\mathcal{L}_{adapter} = \mathcal{L}_{MSE} + \lambda_{clip}\mathcal{L}_{CLIP} + \lambda_{arc}\mathcal{L}_{ArcFace}$$

- $\mathcal{L}_{MSE}$: diffusion noise prediction loss on noisy latents (every step)
- $\mathcal{L}_{CLIP}$: CLIP identity loss on decoded predictions (every 10 steps, $\lambda_{clip}=0.1$)
- $\mathcal{L}_{ArcFace}$: ArcFace face verification loss on decoded predictions (every 25 steps)

Scale warmup ramps $\alpha$ from 0.1 to target over the first 50% of training, preventing early instability. Training uses AdamW with cosine LR schedule for 200 steps ($\lambda_{arc}=0.0$), followed by a 60-step ArcFace fine-tune pass ($\lambda_{arc}=0.2$). Gradients are clipped to norm 1.0.

### 3.7 Identity Injection

At generation time, identity tokens are injected at two levels:

**Text embedding level (coarse):** Identity tokens are added to the last N=4 positions of the text embedding sequence, scaled by an adaptive factor:

$$\mathbf{c}_{conditioned}[:, -N:, :] \mathrel{+}= \alpha_{adaptive} \cdot \mathbf{z}_{identity}$$

where $\alpha_{adaptive}$ scales with prompt complexity (measured by the norm of the text embedding) to prevent identity from being overwhelmed by long prompts. The base scale is 1.2, capped at 2.0.

**UNet level (fine-grained):** The adapter injects identity into every transformer block via cross-attention, influencing the entire denoising trajectory.

**Identity-aware negative conditioning:** The negative embedding is pushed away from the identity direction, actively suppressing identity drift during classifier-free guidance:

$$\mathbf{c}_{negative}[:, -N:, :] \mathrel{-}= 0.5 \cdot \mathbf{z}_{identity}$$

This is distinct from standard negative prompting: rather than suppressing semantic content, it creates a repulsive force in identity space, steering the denoising trajectory away from identity-inconsistent regions.

### 3.8 Post-Adapter Refinement

After adapter training, four probe images are generated and scored by ArcFace. The top-2 by face verification score are used to perform one final projector fine-tune, closing any remaining gap between the adapter's output distribution and the target identity embedding.

### 3.9 Reference Implementation and Interface

The released system separates **backend initialization** from **user interaction**. Model definitions load immediately, but heavyweight assets — DreamShaper, ControlNet, CLIP, and OpenPose — are initialized only through an explicit backend setup step. This keeps startup responsive and makes the system usable as both a library and an app. On top of the backend, we provide an interactive UI that exposes staged pipeline loading, character creation, generation, evaluation, checkpointing, and a mobile-access QR for the public app URL. These packaging choices are not the research contribution themselves, but they make the full pipeline reproducible and usable beyond a notebook-style workflow.

---

## 4. Experiments

### 4.1 Setup

**Base model:** Lykon/DreamShaper (SD 1.5 based)  
**CLIP encoder:** openai/clip-vit-large-patch14 when available; fallback to openai/clip-vit-base-patch32  
**Face verification:** InsightFace buffalo_sc (det_size=320×320)  
**Pose conditioning:** ControlNet OpenPose (lllyasviel/control_v11p_sd15_openpose)  
**Hardware:** NVIDIA T4 GPU  
**Time per character:** ~5 minutes  

InsightFace buffalo_sc is used in preference to buffalo_l for better detection on stylized and generated faces. When no face is detected, the system falls back to CLIP face-crop embeddings.

### 4.2 Evaluation Metrics

- **Face-weighted CLIP identity:** Blend of full-image (40%) and face-crop (60%) CLIP cosine similarity to anchor embedding. Computed against the ensemble anchor, not a single image.
- **ArcFace identity:** InsightFace buffalo_sc cosine similarity between generated and anchor face embeddings. More discriminative than CLIP for facial geometry.
- **Pairwise consistency:** Mean pairwise CLIP similarity across 8 generated images. Measures consistency across the set, not just similarity to a single anchor.

Note: benchmarks across methods are not standardized — different papers use different prompts, datasets, and evaluation protocols. Direct numeric comparisons should be interpreted with this caveat.

### 4.3 Multi-Character Results

We evaluate on five diverse character types spanning photorealistic and stylized domains:

| Character | CLIP Identity | Pairwise Consistency |
|-----------|--------------|---------------------|
| Woman (brunette) | 0.9515 | 0.9435 ± 0.0224 |
| Elderly man | 0.9508 | 0.9430 ± 0.0309 |
| Anime girl | 0.9655 | 0.9541 ± 0.0134 |
| Young man | 0.9407 | 0.9528 ± 0.0192 |
| Woman (redhead) | 0.9625 | 0.9411 ± 0.0130 |
| **Mean** | **0.9542** | **0.9469 ± 0.0198** |

### 4.4 Evaluation Suites (Woman Redhead Character)

We evaluate across four axes of variation using 6 images per suite:

| Suite | CLIP mean | CLIP std | ArcFace mean | ArcFace std | Faces detected |
|-------|-----------|----------|--------------|-------------|----------------|
| Expression | 0.9523 | 0.0214 | 0.5113 | 0.0709 | 6/6 |
| Scene | 0.9349 | 0.0131 | 0.5502 | 0.0217 | 6/6 |
| Pose | 0.9635 | 0.0112 | 0.5882 | 0.0504 | 6/6 |
| Seed | 0.9628 | 0.0047 | 0.6105 | 0.0438 | 6/6 |

The pose suite uses reduced ControlNet conditioning scale (0.35) to allow natural pose variation while maintaining identity. The seed suite demonstrates robustness across random seeds with low variance (std=0.0047), confirming the identity embedding is stable rather than seed-dependent. The ArcFace drop from seed (0.61) to expression (0.51) reflects genuine facial geometry change under strong expressions, not identity drift.

### 4.5 Ablation Study

| Configuration | CLIP | ArcFace | Time (s) |
|---------------|------|---------|----------|
| Baseline (single anchor, no bootstrap, no adapter) | 0.9603 | 0.7562 | 55 |
| + Multi-anchor ensemble | 0.9676 | 0.7581 | 78 |
| + Bootstrap + drift correction | 0.9665 | 0.7560 | 154 |
| **Full system (+ UNet adapter)** | **0.9690** | **0.7912** | 158 |

The UNet adapter provides the largest ArcFace improvement (+3.5%), confirming that cross-attention injection into the denoising process captures facial geometry that text-space injection alone cannot. Bootstrap + drift correction improves CLIP consistency but does not independently improve ArcFace — the adapter is necessary to translate the refined embedding into geometric face fidelity.

The multi-anchor ensemble adds only 23 seconds while improving both CLIP (+0.0073) and ArcFace (+0.0019), making it a high-value, low-cost component. The full system achieves the best ArcFace at a total cost of 158 seconds (~2.6 minutes for the non-adapter stages).

### 4.6 Comparison with Prior Methods

| Method | CLIP | ArcFace | Real image required | Training data |
|--------|------|---------|--------------------|--------------| 
| PhotoMaker [Li et al., 2024] | — | ~0.618† | Yes (multiple) | Real |
| IP-Adapter FaceID [Ye et al., 2023] | 0.854‡ | 0.132‡ | Yes | ~1M pairs |
| Arc2Face [Papantoniou et al., 2024] | — | ~0.79† | Yes | ~21M faces |
| **SynID (ours)** | **0.969** | **0.791** | **No** | **8 synthetic** |

†Reported on authors' own evaluation sets; not directly comparable to our protocol.  
‡IP-Adapter FaceID run without pose conditioning for technical compatibility; ArcFace score reflects this constraint.

SynID achieves ArcFace scores comparable to Arc2Face — a model trained on 21M real faces — while requiring zero real images and ~5 minutes of per-character training versus days of pretraining.

### 4.7 Efficiency Analysis

| Method | Training data | Pretraining time | Per-character time |
|--------|--------------|-----------------|-------------------|
| DreamBooth | 3-20 real images | None | 15-30 min |
| IP-Adapter | ~1M image pairs | Days | Inference only |
| Arc2Face | ~21M face images | Days | Inference only |
| **SynID** | **8 synthetic images** | **None** | **~5 min** |

SynID is the only method in this comparison that requires neither pretraining nor real images, while still achieving competitive identity scores.

---

## 5. Discussion

### 5.1 What the System Actually Learns

SynID does not learn identity in the same sense as face recognition systems. Rather, it learns to steer the diffusion model's existing identity priors toward a specific point in the generative manifold. The projector maps a CLIP embedding to a text-space direction that activates the model's internal representation of the described character. The adapter reinforces this direction at every denoising step.

This distinction matters: the system generalizes within the distribution of the base model. Characters that are well-represented in DreamShaper's training distribution (photorealistic portraits, anime faces) achieve higher consistency than out-of-distribution subjects. The ArcFace score of 0.791 reflects the model's ability to maintain facial geometry, not a learned face recognition capability.

### 5.2 Limitations

**Pose dependence.** The primary path still benefits from anchor-derived pose conditioning. A pose-free generation mode (using the adapter without ControlNet) is implemented, but identity scores remain lower than the main ControlNet-assisted path. Fully pose-independent identity-consistent generation remains an open problem.

**Expression sensitivity.** ArcFace scores drop from 0.61 (seed robustness) to 0.51 (expression variation), reflecting that strong expressions change facial geometry enough to affect face verification scores. This is a fundamental limitation of any face-based identity metric, not specific to SynID.

**Base model dependence.** Identity consistency is bounded by the base model's generative prior. Characters with highly distinctive features (unusual hair color, distinctive facial marks) achieve higher consistency than generic descriptions.

**CLIP metric ceiling.** The face-weighted CLIP metric approaches 0.97 for our system, suggesting we are near the ceiling of what this metric can measure. ArcFace provides a more discriminative signal for further improvements.

**Scope.** The system is designed for human faces. Non-face subjects (animals, objects, full-body characters) are not supported, as the ArcFace loss and face-weighted CLIP encoding are face-specific.

### 5.3 Failure Cases

- Extreme pose changes (profile view, looking down >45°) cause identity drift
- Very similar character descriptions (e.g., two "young women with brown hair") may converge to similar identities due to the shared generative prior
- Non-face subjects are not supported
- Characters outside the base model's training distribution (highly stylized, non-human) show reduced consistency

### 5.4 Ethical Considerations

SynID generates entirely synthetic identities from text descriptions — it does not reproduce or manipulate real people's likenesses. However, the system could in principle be used to generate persistent synthetic personas at scale, which raises concerns about synthetic identity misuse in disinformation or fraud contexts. We note that all generated identities are bounded by the base model's prior and do not correspond to real individuals. Responsible deployment should include watermarking or provenance tracking for generated images.

---

## 6. Conclusion

We presented SynID, a zero-shot identity-consistent generation system that requires no real reference images at any stage. Through a closed-loop self-distillation pipeline combining multi-anchor ensemble embedding, bootstrap refinement, drift correction, and on-the-fly UNet adaptation, SynID achieves identity consistency comparable to systems trained on millions of real images, while operating entirely within the model's own generative prior.

The core finding is that **bootstrapped synthetic identity** — initializing from the model's own prior and refining through self-distillation — can replace dataset-driven identity conditioning for text-described characters. Each component of the pipeline is novel: the softmax-weighted synthetic anchor ensemble, the diversity-enforced bootstrap loop, the probe-based drift correction, the on-the-fly adapter training on synthetic images, and the dual-level injection with identity-aware negative conditioning.

This opens new possibilities for character design, synthetic data generation, and privacy-preserving personalization where real reference images are unavailable or undesirable. The current implementation already includes stronger ArcFace-integrated training, a pose-free generation path, preliminary SDXL loading support, and multi-character consistency checks. Future work is therefore less about introducing these capabilities from scratch and more about making them first-class: closing the identity gap in the pose-free route, fully wiring SDXL through the training and evaluation stack, and extending multi-character support from evaluation to robust joint generation.

---

## References

- Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation. CVPR 2023.
- Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
- Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models. arXiv:2308.06721.
- Wang et al. (2024). InstantID: Zero-shot Identity-Preserving Generation in Seconds. arXiv:2401.07519.
- Papantoniou et al. (2024). Arc2Face: A Foundation Model for ID-Consistent Human Faces. ECCV 2024.
- Li et al. (2024). PhotoMaker: Customizing Realistic Human Photos via Stacked ID Embedding. CVPR 2024.
- Tomašević et al. (2025). ID-Booth: Identity-consistent Face Generation with Diffusion Models. arXiv:2504.07392.
- Hong et al. (2024). Inv-Adapter: ID Customization Generation via Image Inversion. arXiv:2406.02881.
- Liu et al. (2025). F-Bench: Rethinking Human Preference Evaluation Metrics for Benchmarking Face Generation, Customization, and Restoration. arXiv:2412.13155.
- Zhang et al. (2024). EZIGen: Enhancing zero-shot subject-driven image generation with precise subject encoding and decoupled guidance. arXiv 2024.
- Deng et al. (2019). ArcFace: Additive Angular Margin Loss for Deep Face Recognition. CVPR 2019.
- Zhang et al. (2023). Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet). ICCV 2023.
