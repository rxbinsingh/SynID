# ============================================================
# Evaluation Harness — Publication-Ready
# Run AFTER identity_projection_complete.py
# Produces: ablation table, pose robustness, efficiency stats
# ============================================================

import time
import torch
import torch.nn.functional as F
import numpy as np
from IPython.display import display
from PIL import Image
from typing import List, Dict

# ── Reuse ArcFace already loaded by identity_projection_complete.py ──
def arcface_score(image: Image.Image, anchor_image: Image.Image) -> float:
    """
    ArcFace cosine similarity using the already-loaded _arcface_app.
    Falls back to encode_arcface (CLIP face crop) if InsightFace unavailable.
    Returns None if no face detected.
    """
    if _arcface_mode == "insightface" and _arcface_app is not None:
        import numpy as _np
        def _get(img):
            arr = _np.array(img.convert("RGB"))
            faces = _arcface_app.get(arr)
            if not faces: return None
            return torch.tensor(faces[0].normed_embedding).float().unsqueeze(0)
        e1 = _get(image); e2 = _get(anchor_image)
        if e1 is None or e2 is None: return None
        return float(F.cosine_similarity(e1, e2).mean())
    else:
        e1 = encode_arcface(image); e2 = encode_arcface(anchor_image)
        if e1 is None or e2 is None: return None
        return float(F.cosine_similarity(e1, e2).mean())


# ── Evaluation suite ──────────────────────────────────────────
def run_evaluation_suite(profile, adapters_active=True):
    results = {}

    expression_prompts = [
        profile.character_core_prompt + ", bright smile, cheerful",
        profile.character_core_prompt + ", surprised, wide eyes, open mouth",
        profile.character_core_prompt + ", serious, focused gaze",
        profile.character_core_prompt + ", sad, downcast eyes",
        profile.character_core_prompt + ", laughing, happy",
        profile.character_core_prompt + ", angry, furrowed brows",
    ]
    scene_prompts = [
        profile.character_core_prompt + ", coffee shop, warm bokeh lighting",
        profile.character_core_prompt + ", outdoors, golden hour sunlight",
        profile.character_core_prompt + ", rainy window, soft diffused light",
        profile.character_core_prompt + ", studio headshot, clean white background",
        profile.character_core_prompt + ", neon city night, colorful reflections",
        profile.character_core_prompt + ", forest, dappled natural light",
    ]
    pose_prompts = [
        profile.character_core_prompt + ", looking slightly left, 3/4 view",
        profile.character_core_prompt + ", looking slightly right, 3/4 view",
        profile.character_core_prompt + ", looking up, chin raised",
        profile.character_core_prompt + ", looking down, contemplative",
        profile.character_core_prompt + ", side profile view",
        profile.character_core_prompt + ", over shoulder glance",
    ]
    seed_prompt = profile.character_core_prompt + ", neutral expression"
    seed_list   = [1001, 2002, 3003, 4004, 5005, 6006]

    def eval_set(prompts, seeds, suite_name, controlnet_scale=0.55):
        clip_scores, arc_scores, images = [], [], []
        for i, prompt in enumerate(prompts):
            seed = seeds[i % len(seeds)]
            iscale = adaptive_identity_scale(prompt)
            if adapters_active:
                img = generate_with_adapter(
                    profile.identity_tokens, prompt, profile.pose_image,
                    pipe.unet, pipe, seed=seed, identity_scale=iscale,
                    generation_steps=30, guidance_scale=7.5,
                    controlnet_scale=controlnet_scale)
            else:
                img = generate_with_tokens(
                    profile.identity_tokens, prompt, profile.pose_image,
                    seed=seed, identity_scale=iscale, generation_steps=30,
                    guidance_scale=7.5, controlnet_scale=controlnet_scale)

            clip_id, _, _ = candidate_score(img, profile.base_identity_embedding, prompt)
            arc_id = arcface_score(img, profile.anchor_image)
            clip_scores.append(clip_id)
            if arc_id is not None: arc_scores.append(arc_id)
            images.append(img)
            arc_str = f"{arc_id:.4f}" if arc_id is not None else "no face"
            print(f"  [{suite_name}] {i+1}: CLIP={clip_id:.4f} | ArcFace={arc_str}")

        return {
            "images": images,
            "clip_mean": float(np.mean(clip_scores)),
            "clip_std":  float(np.std(clip_scores)),
            "arc_mean":  float(np.mean(arc_scores)) if arc_scores else None,
            "arc_std":   float(np.std(arc_scores))  if arc_scores else None,
            "n_face_detected": len(arc_scores),
        }

    print("\n── Expression suite ──")
    results["expression"] = eval_set(expression_prompts, [5001,5002,5003,5004,5005,5006], "expression")
    display(show_grid(results["expression"]["images"], cols=3))

    print("\n── Scene suite ──")
    results["scene"] = eval_set(scene_prompts, [6001,6002,6003,6004,6005,6006], "scene", controlnet_scale=0.45)
    display(show_grid(results["scene"]["images"], cols=3))

    print("\n── Pose suite (controlnet_scale=0.35 — more pose freedom) ──")
    results["pose"] = eval_set(pose_prompts, [7001,7002,7003,7004,7005,7006], "pose", controlnet_scale=0.35)
    display(show_grid(results["pose"]["images"], cols=3))

    print("\n── Seed robustness ──")
    results["seed"] = eval_set([seed_prompt]*6, seed_list, "seed")
    display(show_grid(results["seed"]["images"], cols=3))

    return results


# ── Ablation study ────────────────────────────────────────────
def run_ablation(char_prompt, anchor_seed):
    print("\n" + "="*60 + "\nABLATION STUDY\n" + "="*60)

    ablation_results = {}
    eval_prompts = [
        char_prompt + ", bright smile",
        char_prompt + ", surprised expression",
        char_prompt + ", serious expression",
        char_prompt + ", calm neutral",
    ]
    eval_seeds = [5001, 5002, 5003, 5004]

    def _eval_config(tokens, pose_img, anchor_img, anchor_emb, use_adapter=False, adp=None):
        clips, arcs = [], []
        for prompt, seed in zip(eval_prompts, eval_seeds):
            iscale = adaptive_identity_scale(prompt)
            if use_adapter and adp is not None:
                img = generate_with_adapter(tokens, prompt, pose_img,
                                            pipe.unet, pipe, seed=seed,
                                            identity_scale=iscale, generation_steps=30)
            else:
                img = generate_with_tokens(tokens, prompt, pose_img,
                                           seed=seed, identity_scale=iscale, generation_steps=30)
            c, _, _ = candidate_score(img, anchor_emb, prompt)
            a = arcface_score(img, anchor_img)
            clips.append(c)
            if a is not None: arcs.append(a)
        return float(np.mean(clips)), (float(np.mean(arcs)) if arcs else None)

    # A: baseline — single anchor, no bootstrap, no adapter
    print("\n[A] Baseline: single anchor, no bootstrap, no adapter")
    t0 = time.time()
    emb_a, imgs_a = build_ensemble_embedding(char_prompt + ", calm expression", [anchor_seed])
    pose_a = pose_detector(imgs_a[0])
    proj_a, _ = train_projector(emb_a, char_prompt, num_tokens=4, train_steps=250)
    with torch.inference_mode():
        tok_a = proj_a(emb_a.float()).to(dtype)
    clip_a, arc_a = _eval_config(tok_a, pose_a, imgs_a[0], emb_a)
    t_a = time.time() - t0
    ablation_results["A"] = {"clip": clip_a, "arc": arc_a, "time_s": t_a,
                              "label": "Baseline (single anchor, no bootstrap, no adapter)"}
    arc_str = f"{arc_a:.4f}" if arc_a else "N/A"
    print(f"  CLIP={clip_a:.4f} | ArcFace={arc_str} | time={t_a:.0f}s")

    # B: + multi-anchor ensemble
    print("\n[B] + Multi-anchor ensemble")
    t0 = time.time()
    emb_b, imgs_b = build_ensemble_embedding(
        char_prompt + ", calm expression",
        [anchor_seed, anchor_seed+111, anchor_seed+222, anchor_seed+333])
    pose_b = pose_detector(imgs_b[0])
    proj_b, _ = train_projector(emb_b, char_prompt, num_tokens=4, train_steps=250)
    with torch.inference_mode():
        tok_b = proj_b(emb_b.float()).to(dtype)
    clip_b, arc_b = _eval_config(tok_b, pose_b, imgs_b[0], emb_b)
    t_b = time.time() - t0
    ablation_results["B"] = {"clip": clip_b, "arc": arc_b, "time_s": t_b,
                              "label": "+ Multi-anchor ensemble"}
    arc_str = f"{arc_b:.4f}" if arc_b else "N/A"
    print(f"  CLIP={clip_b:.4f} | ArcFace={arc_str} | time={t_b:.0f}s")

    # C: + bootstrap + drift correction (no adapter)
    print("\n[C] + Bootstrap + drift correction (no adapter)")
    t0 = time.time()
    prof_c = create_character(identity_prompt=char_prompt, anchor_seed=anchor_seed,
                              num_identity_tokens=4, train_steps=250,
                              bootstrap_top_k=2, correction_rounds=2)
    clip_c, arc_c = _eval_config(prof_c.identity_tokens, prof_c.pose_image,
                                  prof_c.anchor_image, prof_c.base_identity_embedding)
    t_c = time.time() - t0
    ablation_results["C"] = {"clip": clip_c, "arc": arc_c, "time_s": t_c,
                              "label": "+ Bootstrap + drift correction"}
    arc_str = f"{arc_c:.4f}" if arc_c else "N/A"
    print(f"  CLIP={clip_c:.4f} | ArcFace={arc_str} | time={t_c:.0f}s")

    # D: full system (+ UNet adapter)
    print("\n[D] Full system (+ UNet adapter + ArcFace fine-tune)")
    t0 = time.time()
    adp = attach_identity_adapters(pipe.unet, identity_dim=768, scale=0.5)
    hks = register_adapter_hooks(pipe.unet)
    train_adapter_on_bootstrap(
        pipe.unet, pipe.vae, pipe.text_encoder, pipe.tokenizer, adp,
        prof_c.identity_tokens,
        [c.image for c in prof_c.bootstrap_candidates],
        [c.prompt for c in prof_c.bootstrap_candidates],
        prof_c.base_identity_embedding,
        train_steps=200, lr=1e-5, clip_loss_weight=0.1, arcface_loss_weight=0.0)
    train_adapter_on_bootstrap(
        pipe.unet, pipe.vae, pipe.text_encoder, pipe.tokenizer, adp,
        prof_c.identity_tokens,
        [c.image for c in prof_c.bootstrap_candidates],
        [c.prompt for c in prof_c.bootstrap_candidates],
        prof_c.base_identity_embedding,
        train_steps=60, lr=1e-5, clip_loss_weight=0.1, arcface_loss_weight=0.2)
    clip_d, arc_d = _eval_config(prof_c.identity_tokens, prof_c.pose_image,
                                  prof_c.anchor_image, prof_c.base_identity_embedding,
                                  use_adapter=True, adp=adp)
    t_d = time.time() - t0
    ablation_results["D"] = {"clip": clip_d, "arc": arc_d, "time_s": t_d,
                              "label": "Full system (+ UNet adapter)"}
    arc_str = f"{arc_d:.4f}" if arc_d else "N/A"
    print(f"  CLIP={clip_d:.4f} | ArcFace={arc_str} | time={t_d:.0f}s")
    for h in hks: h.remove()

    return ablation_results, prof_c


# ── Print publication table ───────────────────────────────────
def print_results_table(eval_results, ablation_results, char_name):
    print("\n" + "="*70)
    print(f"RESULTS TABLE — {char_name}")
    print("="*70)

    print(f"\n{'Suite':<20} {'CLIP mean':>10} {'CLIP std':>10} {'ArcFace mean':>13} {'ArcFace std':>12} {'Faces':>6}")
    print("-"*70)
    for suite, res in eval_results.items():
        arc_m = f"{res['arc_mean']:.4f}" if res['arc_mean'] is not None else "N/A"
        arc_s = f"{res['arc_std']:.4f}"  if res['arc_std']  is not None else "N/A"
        print(f"{suite:<20} {res['clip_mean']:>10.4f} {res['clip_std']:>10.4f} {arc_m:>13} {arc_s:>12} {res['n_face_detected']:>6}")

    print(f"\n{'Config':<45} {'CLIP':>8} {'ArcFace':>10} {'Time(s)':>9}")
    print("-"*70)
    for key, res in ablation_results.items():
        arc = f"{res['arc']:.4f}" if res['arc'] is not None else "N/A"
        print(f"{res['label']:<45} {res['clip']:>8.4f} {arc:>10} {res['time_s']:>9.0f}")

    print(f"\n{'Method':<35} {'Data':>18} {'Time':>10} {'CLIP':>8} {'ArcFace':>10}")
    print("-"*70)
    print(f"{'IP-Adapter (ViT-H)':<35} {'~1M image pairs':>18} {'~days':>10} {'~0.85':>8} {'~0.65':>10}")
    d = ablation_results['D']
    arc_d = f"{d['arc']:.4f}" if d['arc'] is not None else "N/A"
    print(f"{'Ours (SynID)':<35} {'4 synthetic images':>18} {'~5 min':>10} {d['clip']:>8.4f} {arc_d:>10}")
    print("="*70)
    print(f"\nArcFace mode used: {_arcface_mode}")


# ── RUN ───────────────────────────────────────────────────────
print(f"ArcFace mode: {_arcface_mode}")
print(f"\nRunning full evaluation on: {profile.character_core_prompt[:60]}...")

eval_results = run_evaluation_suite(profile, adapters_active=True)

ablation_char = "anime girl, long silver hair, violet eyes, soft bangs, detailed face, upper body portrait, white background, sharp lineart"
ablation_results, ablation_profile = run_ablation(ablation_char, anchor_seed=5678)

print_results_table(eval_results, ablation_results, profile.character_core_prompt[:40])
