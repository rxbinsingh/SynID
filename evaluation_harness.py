# ============================================================
# SynID Evaluation Harness — Publication-Ready + UI-callable
# Run AFTER identity_projection_complete.py:
#   exec(open("evaluation_harness.py").read())
# Or call individual functions from the UI:
#   results = eval_quick(profile)
#   results = eval_full(profile)
#   run_ablation(char_prompt, anchor_seed)
# ============================================================

import os, time
import torch
import torch.nn.functional as F
import numpy as np
from IPython.display import display
from PIL import Image
from typing import List, Dict, Optional

# ── ArcFace score ─────────────────────────────────────────────
def arcface_score(image: Image.Image, anchor_image: Image.Image) -> Optional[float]:
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


# ── FID (CLIP-proxy) ──────────────────────────────────────────
def compute_fid_approx(real_images: List[Image.Image], gen_images: List[Image.Image]) -> float:
    def _feats(imgs):
        embs = []
        for img in imgs:
            inp = clip_processor(images=img, return_tensors="pt")
            with torch.no_grad():
                out = clip_model.vision_model(pixel_values=inp["pixel_values"].to(device))
                embs.append(clip_model.visual_projection(out.pooler_output).float().cpu())
        return torch.cat(embs, dim=0)
    real_f = _feats(real_images); gen_f = _feats(gen_images)
    mu_r, mu_g = real_f.mean(0), gen_f.mean(0)
    diff_sq = (mu_r - mu_g).pow(2).sum()
    if real_f.shape[0] < 2 or gen_f.shape[0] < 2: return float(diff_sq)
    sig_r = torch.cov(real_f.T); sig_g = torch.cov(gen_f.T)
    try:
        eigvals = torch.linalg.eigvalsh(sig_r @ sig_g).clamp(min=0)
        fid = float(diff_sq + sig_r.trace() + sig_g.trace() - 2*eigvals.sqrt().sum())
    except Exception: fid = float(diff_sq)
    return max(fid, 0.0)


# ── Save helper ───────────────────────────────────────────────
def save_images(images: List[Image.Image], folder: str = "/content/synid_output",
                prefix: str = "synid", return_paths: bool = False):
    """Save a list of PIL images to folder. Returns paths if requested."""
    os.makedirs(folder, exist_ok=True)
    paths = []
    for i, img in enumerate(images):
        path = f"{folder}/{prefix}_{i+1:03d}.png"
        img.save(path)
        paths.append(path)
    print(f"  Saved {len(images)} images → {folder}/")
    return paths if return_paths else None


# ── Quick eval (4 expressions, fast) ─────────────────────────
def eval_quick(profile, adapters_active=True, save_dir=None) -> Dict:
    """
    Fast evaluation: 4 expressions only. ~2 min.
    Returns dict with clip_mean, arc_mean, images.
    Callable from UI.
    """
    prompts = [
        profile.character_core_prompt + ", bright smile, cheerful",
        profile.character_core_prompt + ", surprised, wide eyes",
        profile.character_core_prompt + ", serious, focused gaze",
        profile.character_core_prompt + ", neutral expression",
    ]
    clips, arcs, images = [], [], []
    for i, prompt in enumerate(prompts):
        iscale = adaptive_identity_scale(prompt)
        if adapters_active:
            img = generate_with_adapter(
                profile.identity_tokens, prompt, profile.pose_image,
                pipe.unet, pipe, seed=5001+i, identity_scale=iscale,
                generation_steps=30, guidance_scale=7.5)
        else:
            img = generate_with_tokens(
                profile.identity_tokens, prompt, profile.pose_image,
                seed=5001+i, identity_scale=iscale, generation_steps=30)
        c, _, _ = candidate_score(img, profile.base_identity_embedding, prompt)
        a = arcface_score(img, profile.anchor_image)
        clips.append(c); images.append(img)
        if a is not None: arcs.append(a)
        arc_str = f"{a:.4f}" if a is not None else "no face"
        print(f"  [{i+1}/4] CLIP={c:.4f} | ArcFace={arc_str}")
    result = {
        "images": images,
        "clip_mean": float(np.mean(clips)), "clip_std": float(np.std(clips)),
        "arc_mean": float(np.mean(arcs)) if arcs else None,
        "arc_std": float(np.std(arcs)) if arcs else None,
    }
    arc_mean_str = f"{result['arc_mean']:.4f}" if result["arc_mean"] is not None else "N/A"
    print(f"  Quick eval → CLIP={result['clip_mean']:.4f} | ArcFace={arc_mean_str}")
    if save_dir: save_images(images, save_dir, "quick_eval")
    return result


# ── Full eval suite ───────────────────────────────────────────
def eval_full(profile, adapters_active=True, save_dir=None) -> Dict:
    """
    Full evaluation: expression + scene + pose + seed suites (24 images).
    Returns dict of suite results. Callable from UI.
    """
    results = {}
    suites = {
        "expression": {
            "prompts": [
                profile.character_core_prompt + ", bright smile, cheerful",
                profile.character_core_prompt + ", surprised, wide eyes, open mouth",
                profile.character_core_prompt + ", serious, focused gaze",
                profile.character_core_prompt + ", sad, downcast eyes",
                profile.character_core_prompt + ", laughing, happy",
                profile.character_core_prompt + ", angry, furrowed brows",
            ],
            "seeds": [5001,5002,5003,5004,5005,5006],
            "controlnet_scale": 0.55,
        },
        "scene": {
            "prompts": [
                profile.character_core_prompt + ", coffee shop, warm bokeh lighting",
                profile.character_core_prompt + ", outdoors, golden hour sunlight",
                profile.character_core_prompt + ", rainy window, soft diffused light",
                profile.character_core_prompt + ", studio headshot, clean white background",
                profile.character_core_prompt + ", neon city night, colorful reflections",
                profile.character_core_prompt + ", forest, dappled natural light",
            ],
            "seeds": [6001,6002,6003,6004,6005,6006],
            "controlnet_scale": 0.45,
        },
        "pose": {
            "prompts": [
                profile.character_core_prompt + ", looking slightly left, 3/4 view",
                profile.character_core_prompt + ", looking slightly right, 3/4 view",
                profile.character_core_prompt + ", looking up, chin raised",
                profile.character_core_prompt + ", looking down, contemplative",
                profile.character_core_prompt + ", side profile view",
                profile.character_core_prompt + ", over shoulder glance",
            ],
            "seeds": [7001,7002,7003,7004,7005,7006],
            "controlnet_scale": 0.35,
        },
        "seed": {
            "prompts": [profile.character_core_prompt + ", neutral expression"] * 6,
            "seeds": [1001,2002,3003,4004,5005,6006],
            "controlnet_scale": 0.55,
        },
    }

    for suite_name, cfg in suites.items():
        print(f"\n── {suite_name} suite ──")
        clips, arcs, images = [], [], []
        for i, (prompt, seed) in enumerate(zip(cfg["prompts"], cfg["seeds"])):
            iscale = adaptive_identity_scale(prompt)
            if adapters_active:
                img = generate_with_adapter(
                    profile.identity_tokens, prompt, profile.pose_image,
                    pipe.unet, pipe, seed=seed, identity_scale=iscale,
                    generation_steps=30, guidance_scale=7.5,
                    controlnet_scale=cfg["controlnet_scale"])
            else:
                img = generate_with_tokens(
                    profile.identity_tokens, prompt, profile.pose_image,
                    seed=seed, identity_scale=iscale, generation_steps=30,
                    controlnet_scale=cfg["controlnet_scale"])
            c, _, _ = candidate_score(img, profile.base_identity_embedding, prompt)
            a = arcface_score(img, profile.anchor_image)
            clips.append(c); images.append(img)
            if a is not None: arcs.append(a)
            arc_str = f"{a:.4f}" if a is not None else "no face"
            print(f"  [{suite_name}] {i+1}: CLIP={c:.4f} | ArcFace={arc_str}")
        display(show_grid(images, cols=3))
        if save_dir: save_images(images, save_dir, suite_name)
        results[suite_name] = {
            "images": images,
            "clip_mean": float(np.mean(clips)), "clip_std": float(np.std(clips)),
            "arc_mean": float(np.mean(arcs)) if arcs else None,
            "arc_std": float(np.std(arcs)) if arcs else None,
            "n_face_detected": len(arcs),
        }
    return results


# ── Ablation ──────────────────────────────────────────────────
def run_ablation(char_prompt, anchor_seed) -> Dict:
    print("\n" + "="*60 + "\nABLATION STUDY\n" + "="*60)
    ablation_results = {}
    eval_prompts = [char_prompt+s for s in [", bright smile", ", surprised expression",
                                             ", serious expression", ", calm neutral"]]
    eval_seeds = [5001, 5002, 5003, 5004]

    def _eval(tokens, pose_img, anchor_img, anchor_emb, use_adapter=False, adp=None):
        clips, arcs = [], []
        for prompt, seed in zip(eval_prompts, eval_seeds):
            iscale = adaptive_identity_scale(prompt)
            img = (generate_with_adapter(tokens, prompt, pose_img, pipe.unet, pipe,
                                         seed=seed, identity_scale=iscale, generation_steps=30)
                   if use_adapter and adp else
                   generate_with_tokens(tokens, prompt, pose_img,
                                        seed=seed, identity_scale=iscale, generation_steps=30))
            c, _, _ = candidate_score(img, anchor_emb, prompt)
            a = arcface_score(img, anchor_img)
            clips.append(c)
            if a is not None: arcs.append(a)
        return float(np.mean(clips)), (float(np.mean(arcs)) if arcs else None)

    # A: baseline
    print("\n[A] Baseline")
    t0 = time.time()
    emb_a, imgs_a = build_ensemble_embedding(char_prompt + ", calm expression", [anchor_seed])
    pose_a = pose_detector(imgs_a[0])
    proj_a, _ = train_projector(emb_a, char_prompt, num_tokens=4, train_steps=250)
    with torch.inference_mode(): tok_a = proj_a(emb_a.float()).to(dtype)
    clip_a, arc_a = _eval(tok_a, pose_a, imgs_a[0], emb_a)
    ablation_results["A"] = {"clip": clip_a, "arc": arc_a, "time_s": time.time()-t0,
                              "label": "Baseline (single anchor, no bootstrap, no adapter)"}
    arc_a_str = f"{arc_a:.4f}" if arc_a is not None else "N/A"
    print(f"  CLIP={clip_a:.4f} | ArcFace={arc_a_str} | {ablation_results['A']['time_s']:.0f}s")

    # B: + ensemble
    print("\n[B] + Multi-anchor ensemble")
    t0 = time.time()
    emb_b, imgs_b = build_ensemble_embedding(char_prompt + ", calm expression",
        [anchor_seed, anchor_seed+111, anchor_seed+222, anchor_seed+333])
    pose_b = pose_detector(imgs_b[0])
    proj_b, _ = train_projector(emb_b, char_prompt, num_tokens=4, train_steps=250)
    with torch.inference_mode(): tok_b = proj_b(emb_b.float()).to(dtype)
    clip_b, arc_b = _eval(tok_b, pose_b, imgs_b[0], emb_b)
    ablation_results["B"] = {"clip": clip_b, "arc": arc_b, "time_s": time.time()-t0,
                              "label": "+ Multi-anchor ensemble"}
    arc_b_str = f"{arc_b:.4f}" if arc_b is not None else "N/A"
    print(f"  CLIP={clip_b:.4f} | ArcFace={arc_b_str} | {ablation_results['B']['time_s']:.0f}s")

    # C: + bootstrap + drift
    print("\n[C] + Bootstrap + drift correction")
    t0 = time.time()
    prof_c = create_character(identity_prompt=char_prompt, anchor_seed=anchor_seed,
                              num_identity_tokens=4, train_steps=250,
                              bootstrap_top_k=2, correction_rounds=2)
    clip_c, arc_c = _eval(prof_c.identity_tokens, prof_c.pose_image,
                           prof_c.anchor_image, prof_c.base_identity_embedding)
    ablation_results["C"] = {"clip": clip_c, "arc": arc_c, "time_s": time.time()-t0,
                              "label": "+ Bootstrap + drift correction"}
    arc_c_str = f"{arc_c:.4f}" if arc_c is not None else "N/A"
    print(f"  CLIP={clip_c:.4f} | ArcFace={arc_c_str} | {ablation_results['C']['time_s']:.0f}s")

    # D: full system
    print("\n[D] Full system (+ UNet adapter)")
    t0 = time.time()
    adp = attach_identity_adapters(pipe.unet, identity_dim=768, scale=0.5)
    hks = register_adapter_hooks(pipe.unet)
    train_adapter_on_bootstrap(
        pipe.unet, pipe.vae, pipe.text_encoder, pipe.tokenizer, adp,
        prof_c.identity_tokens,
        [c.image for c in prof_c.bootstrap_candidates],
        [c.prompt for c in prof_c.bootstrap_candidates],
        prof_c.base_identity_embedding,
        train_steps=200, lr=1e-5, clip_loss_weight=0.1, arcface_loss_weight=0.1)
    train_adapter_on_bootstrap(
        pipe.unet, pipe.vae, pipe.text_encoder, pipe.tokenizer, adp,
        prof_c.identity_tokens,
        [c.image for c in prof_c.bootstrap_candidates],
        [c.prompt for c in prof_c.bootstrap_candidates],
        prof_c.base_identity_embedding,
        train_steps=60, lr=2e-6, clip_loss_weight=0.1, arcface_loss_weight=0.3)
    clip_d, arc_d = _eval(prof_c.identity_tokens, prof_c.pose_image,
                           prof_c.anchor_image, prof_c.base_identity_embedding,
                           use_adapter=True, adp=adp)
    ablation_results["D"] = {"clip": clip_d, "arc": arc_d, "time_s": time.time()-t0,
                              "label": "Full system (+ UNet adapter)"}
    arc_d_str = f"{arc_d:.4f}" if arc_d is not None else "N/A"
    print(f"  CLIP={clip_d:.4f} | ArcFace={arc_d_str} | {ablation_results['D']['time_s']:.0f}s")
    for h in hks: h.remove()

    return ablation_results, prof_c


# ── Print table ───────────────────────────────────────────────
def print_results_table(eval_results, ablation_results, char_name, ablation_profile=None):
    print("\n" + "="*70)
    print(f"RESULTS TABLE — {char_name}")
    print("="*70)
    print(f"\n{'Suite':<20} {'CLIP':>8} {'±':>6} {'ArcFace':>9} {'±':>6} {'Faces':>6}")
    print("-"*60)
    for suite, res in eval_results.items():
        arc_m = f"{res['arc_mean']:.4f}" if res['arc_mean'] is not None else "N/A"
        arc_s = f"{res['arc_std']:.4f}"  if res['arc_std']  is not None else "N/A"
        print(f"{suite:<20} {res['clip_mean']:>8.4f} {res['clip_std']:>6.4f} {arc_m:>9} {arc_s:>6} {res.get('n_face_detected',''):>6}")
    print(f"\n{'Config':<50} {'CLIP':>8} {'ArcFace':>10} {'Time':>8}")
    print("-"*80)
    for res in ablation_results.values():
        arc  = f"{res['arc']:.4f}"  if res['arc']  else "N/A"
        clip = f"{res['clip']:.4f}" if res['clip'] else "N/A"
        print(f"{res['label']:<50} {clip:>8} {arc:>10} {res['time_s']:>7.0f}s")
    print(f"\n{'Method':<35} {'Data':>18} {'Time':>10} {'CLIP':>8} {'ArcFace':>10}")
    print("-"*75)
    print(f"{'IP-Adapter FaceID':<35} {'~1M image pairs':>18} {'~days':>10} {'0.854':>8} {'0.132':>10}")
    print(f"{'Arc2Face (reported)':<35} {'~21M faces':>18} {'~days':>10} {'---':>8} {'~0.79':>10}")
    d = ablation_results.get('D', {})
    arc_d  = f"{d['arc']:.4f}"  if d.get('arc')  else "N/A"
    clip_d = f"{d['clip']:.4f}" if d.get('clip') else "N/A"
    n = len(ablation_profile.bootstrap_candidates) if ablation_profile else "?"
    print(f"{'Ours (SynID)':<35} {f'{n} synthetic images':>18} {'~5 min':>10} {clip_d:>8} {arc_d:>10}")
    print("="*70)
    print(f"\nArcFace mode: {_arcface_mode}")


# ── Standardized benchmark ────────────────────────────────────
STANDARD_BENCHMARK_CHARACTERS = [
    {"name": "std_woman_1",   "seed": 1234,
     "prompt": "young woman, brown eyes, dark brown hair, natural makeup, upper body portrait, neutral background, photorealistic"},
    {"name": "std_man_1",     "seed": 4321,
     "prompt": "middle-aged man, short dark hair, brown eyes, clean shaven, upper body portrait, neutral background, photorealistic"},
    {"name": "std_elderly_1", "seed": 5678,
     "prompt": "elderly woman, white hair, blue eyes, kind expression, upper body portrait, neutral background, photorealistic"},
    {"name": "std_young_1",   "seed": 2468,
     "prompt": "young man, curly black hair, dark eyes, light stubble, upper body portrait, neutral background, photorealistic"},
    {"name": "std_woman_2",   "seed": 9876,
     "prompt": "woman, blonde hair, green eyes, freckles, upper body portrait, neutral background, photorealistic"},
    {"name": "std_man_2",     "seed": 3579,
     "prompt": "man, bald, grey beard, strong jawline, upper body portrait, neutral background, photorealistic"},
    {"name": "std_asian_1",   "seed": 7531,
     "prompt": "young asian woman, straight black hair, dark eyes, upper body portrait, neutral background, photorealistic"},
    {"name": "std_anime_1",   "seed": 8642,
     "prompt": "anime character, blue hair, violet eyes, detailed face, upper body portrait, white background, sharp lineart"},
]

def run_standardized_benchmark(num_chars=8, gen_per_char=6, save_dir=None):
    print("\n" + "="*70 + "\nSTANDARDIZED BENCHMARK\n" + "="*70)
    results = []
    for char in STANDARD_BENCHMARK_CHARACTERS[:num_chars]:
        print(f"\n── {char['name']} ──")
        prof = create_character(identity_prompt=char["prompt"], anchor_seed=char["seed"],
                                num_identity_tokens=4, train_steps=250,
                                bootstrap_top_k=2, correction_rounds=2)
        adp = attach_identity_adapters(pipe.unet, identity_dim=768, scale=0.5)
        hks = register_adapter_hooks(pipe.unet)
        train_adapter_on_bootstrap(
            pipe.unet, pipe.vae, pipe.text_encoder, pipe.tokenizer, adp,
            prof.identity_tokens,
            [c.image for c in prof.bootstrap_candidates],
            [c.prompt for c in prof.bootstrap_candidates],
            prof.base_identity_embedding,
            train_steps=200, lr=1e-5, clip_loss_weight=0.1, arcface_loss_weight=0.1)
        train_adapter_on_bootstrap(
            pipe.unet, pipe.vae, pipe.text_encoder, pipe.tokenizer, adp,
            prof.identity_tokens,
            [c.image for c in prof.bootstrap_candidates],
            [c.prompt for c in prof.bootstrap_candidates],
            prof.base_identity_embedding,
            train_steps=60, lr=2e-6, clip_loss_weight=0.1, arcface_loss_weight=0.3)
        eval_prompts = [char["prompt"]+s for s in
                        [", bright smile", ", surprised", ", serious", ", neutral", ", laughing", ", thoughtful"]]
        clips, arcs, gen_imgs = [], [], []
        for i, prompt in enumerate(eval_prompts[:gen_per_char]):
            img = generate_with_adapter(prof.identity_tokens, prompt, prof.pose_image,
                                        pipe.unet, pipe, seed=5001+i,
                                        identity_scale=adaptive_identity_scale(prompt),
                                        generation_steps=30)
            c, _, _ = candidate_score(img, prof.base_identity_embedding, prompt)
            a = arcface_score(img, prof.anchor_image)
            clips.append(c); gen_imgs.append(img)
            if a is not None: arcs.append(a)
            arc_str = f"{a:.4f}" if a is not None else "N/A"
            print(f"  {i+1}: CLIP={c:.4f} | ArcFace={arc_str}")
        fid = compute_fid_approx(prof.anchor_images, gen_imgs)
        clip_mean = float(np.mean(clips))
        arc_mean  = float(np.mean(arcs)) if arcs else None
        results.append({"name": char["name"], "clip": clip_mean, "arc": arc_mean, "fid": fid})
        arc_mean_str = f"{arc_mean:.4f}" if arc_mean is not None else "N/A"
        print(f"  → CLIP={clip_mean:.4f} | ArcFace={arc_mean_str} | FID≈{fid:.1f}")
        if save_dir: save_images(gen_imgs, f"{save_dir}/{char['name']}", "eval")
        for h in hks: h.remove()
    print(f"\n{'Character':<20} {'CLIP':>8} {'ArcFace':>10} {'FID':>8}")
    print("-"*50)
    for r in results:
        arc = f"{r['arc']:.4f}" if r['arc'] is not None else "N/A"
        print(f"{r['name']:<20} {r['clip']:>8.4f} {arc:>10} {r['fid']:>8.1f}")
    mean_clip = np.mean([r['clip'] for r in results])
    mean_arc  = np.mean([r['arc'] for r in results if r['arc']])
    mean_fid  = np.mean([r['fid'] for r in results])
    print(f"{'MEAN':<20} {mean_clip:>8.4f} {mean_arc:>10.4f} {mean_fid:>8.1f}")
    return results


# ── RUN (when exec'd directly) ────────────────────────────────
if 'profile' in dir():
    print(f"ArcFace mode: {_arcface_mode}")
    print(f"Running full evaluation: {profile.character_core_prompt[:60]}...")

    eval_results = eval_full(profile, adapters_active=True)

    ablation_char = "anime girl, long silver hair, violet eyes, soft bangs, detailed face, upper body portrait, white background, sharp lineart"
    ablation_results, ablation_profile = run_ablation(ablation_char, anchor_seed=5678)

    print_results_table(eval_results, ablation_results,
                        profile.character_core_prompt[:40], ablation_profile)

    print("\n── FID score (CLIP-proxy) ──")
    fid_imgs = eval_results["seed"]["images"] + eval_results["expression"]["images"]
    fid_score = compute_fid_approx(profile.anchor_images, fid_imgs)
    print(f"  FID≈{fid_score:.2f} (lower = better)")

    RUN_BENCHMARK = False
    if RUN_BENCHMARK:
        run_standardized_benchmark(num_chars=8, gen_per_char=6)
    else:
        print("\n── Standardized benchmark skipped (set RUN_BENCHMARK=True) ──")
else:
    print("Evaluation harness loaded. Call eval_quick(profile), eval_full(profile), or run_ablation(prompt, seed).")
