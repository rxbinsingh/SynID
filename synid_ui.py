# ============================================================
# SynID — Clean Studio UI
# Run in Colab:
#   !pip install -q gradio
#   exec(open("synid_ui.py").read())
# ============================================================

import os
from urllib.parse import quote
import gradio as gr
import torch
import numpy as np
from PIL import Image


def _exec_local_file(filename):
    base_dir = os.getcwd()
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        exec(f.read(), globals())

CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

* { font-family: 'IBM Plex Sans', sans-serif !important; box-sizing: border-box; }

:root {
    --bg: #f4f6fb;
    --surface: #ffffff;
    --surface-2: #f8fafc;
    --border: #d9e2ec;
    --accent: #0f4c81;
    --accent-2: #2563eb;
    --text: #142132;
    --muted: #5b6b7f;
    --soft: #edf2f7;
    --shadow: 0 10px 35px rgba(15, 23, 42, 0.06);
}

body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    min-height: 100vh;
}

.gradio-container::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(circle at top left, rgba(37, 99, 235, 0.08), transparent 32%),
        radial-gradient(circle at top right, rgba(15, 76, 129, 0.07), transparent 28%),
        linear-gradient(180deg, rgba(255,255,255,0.75), rgba(244,246,251,0.96));
    pointer-events: none;
    z-index: 0;
}

.gradio-container { max-width: 1220px !important; margin: 0 auto !important; padding: 0 1rem 2rem !important; }
.gradio-container > * { position: relative; z-index: 1; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 2.6rem 0 1.8rem;
}
.hero-title {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: clamp(2.2rem, 5vw, 3.6rem);
    font-weight: 700;
    color: var(--text);
    letter-spacing: -1.6px;
    line-height: 1.05;
    margin-bottom: 0.35rem;
}
.hero-tag {
    font-size: 0.72rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.65rem;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: 5px 12px;
    font-size: 0.7rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--accent);
    margin: 0 4px;
    box-shadow: 0 4px 10px rgba(15, 23, 42, 0.03);
}
.hero-copy {
    max-width: 720px;
    margin: 0.9rem auto 0;
    color: var(--muted);
    font-size: 0.98rem;
    line-height: 1.6;
}

.setup-grid {
    display: grid;
    grid-template-columns: 1.15fr 0.85fr;
    gap: 1rem;
}

.mobile-card {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 162px;
    border: 1px dashed var(--border);
    border-radius: 16px;
    background: linear-gradient(180deg, #ffffff, #f8fbff);
    padding: 1rem;
    text-align: center;
}

.mobile-wrap {
    width: 100%;
}

.mobile-title {
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent);
    font-weight: 600;
    margin-bottom: 0.55rem;
}

.mobile-copy {
    font-size: 0.9rem;
    line-height: 1.55;
    color: var(--muted);
    margin-bottom: 0.75rem;
}

.qr-shell {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: white;
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 0.75rem;
    box-shadow: var(--shadow);
}

.qr-shell img {
    width: 176px;
    height: 176px;
    border-radius: 10px;
}

.mobile-note {
    margin-top: 0.7rem;
    font-size: 0.78rem;
    color: var(--muted);
}

/* ── Cards ── */
.card {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    padding: 1.5rem !important;
    box-shadow: var(--shadow) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
.card:hover { border-color: #c6d4e1 !important; box-shadow: 0 14px 38px rgba(15, 23, 42, 0.08) !important; }

.section-label {
    font-size: 0.68rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent);
    font-weight: 600;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Inputs ── */
textarea, input[type="text"], input[type="number"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s !important;
}
textarea:focus, input:focus {
    border-color: var(--accent-2) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.10) !important;
}
label span { color: var(--muted) !important; font-size: 0.75rem !important; letter-spacing: 1px !important; text-transform: uppercase !important; }

/* ── Buttons ── */
button.primary, .gr-button-primary {
    background: var(--text) !important;
    border: 1px solid var(--text) !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.6px !important;
    text-transform: uppercase !important;
    padding: 0.7rem 1.5rem !important;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.10) !important;
    transition: all 0.2s !important;
    cursor: pointer !important;
}
button.primary:hover, .gr-button-primary:hover {
    transform: translateY(-1px) !important;
    background: #223248 !important;
    border-color: #223248 !important;
}
button.secondary {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--muted) !important;
    font-size: 0.8rem !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.15s !important;
}
button.secondary:hover { border-color: var(--accent) !important; color: var(--accent) !important; }

/* ── Sliders ── */
input[type="range"] { accent-color: var(--accent-2) !important; }

/* ── Gallery ── */
.gallery-item img { border-radius: 12px !important; }
.gallery-item { border-radius: 12px !important; overflow: hidden !important; border: 1px solid var(--border) !important; background: var(--surface-2) !important; }

/* ── Status ── */
.status-text textarea {
    background: transparent !important;
    border: none !important;
    color: var(--muted) !important;
    font-size: 0.8rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Tabs ── */
.tab-nav button {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: var(--muted) !important;
    font-size: 0.8rem !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    padding: 0.5rem 1rem !important;
    border-radius: 0 !important;
}
.tab-nav button.selected {
    color: var(--text) !important;
    border-bottom-color: var(--accent-2) !important;
}

/* ── Presets ── */
.preset-btn {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 100px !important;
    color: var(--accent) !important;
    font-size: 0.75rem !important;
    padding: 0.4rem 1rem !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
}
.preset-btn:hover { background: #eef4fb !important; border-color: #bfd1e2 !important; }

/* ── Metrics ── */
.metric-row {
    display: flex;
    gap: 8px;
    margin-top: 8px;
}
.metric {
    flex: 1;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 8px 12px;
    text-align: center;
}
.metric-val { font-family: 'IBM Plex Mono', monospace; font-size: 1.1rem; color: var(--accent); font-weight: 600; }
.metric-lbl { font-size: 0.6rem; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-top: 2px; }
"""

# ── State ─────────────────────────────────────────────────────
_profiles = {}   # name → CharacterProfile
_adapters_map = {}  # name → adapters
_hooks_map = {}     # name → hooks
_current_name = [None]
_last_images = []   # last generated images for save feature
_backend_state = {
    "ready": False,
    "summary": "Pipelines not loaded",
}
_launch_state = {
    "local_url": "",
    "share_url": "",
}

def _clear_hooks(name):
    for h in _hooks_map.get(name, []): h.remove()
    _hooks_map[name] = []

def _clear_all_hooks():
    for name in list(_hooks_map.keys()):
        _clear_hooks(name)

def _is_backend_ready():
    return bool(_backend_state.get("ready")) or bool(globals().get("BACKEND_READY", False))

def _load_backend(progress=gr.Progress()):
    try:
        progress(0.05, desc="loading backend definitions")
        if "init_synid_backend" not in globals():
            _exec_local_file("identity_projection_complete.py")
        progress(0.18, desc="loading core pipelines")
        backend_info = init_synid_backend()
        progress(0.88, desc="loading evaluation tools")
        if "eval_quick" not in globals() or "eval_full" not in globals():
            _exec_local_file("evaluation_harness.py")
        _backend_state["ready"] = True
        _backend_state["summary"] = (
            f"Ready · device={backend_info['device']} · clip={backend_info['clip_dim']} "
            f"· posefree={'on' if backend_info['posefree'] else 'off'}"
        )
        progress(1.0, desc="ready")
        enabled = gr.update(interactive=True)
        return (
            _backend_state["summary"],
            enabled, enabled, enabled, enabled, enabled, enabled,
        )
    except Exception as e:
        _backend_state["ready"] = False
        _backend_state["summary"] = f"error: {e}"
        disabled = gr.update(interactive=False)
        return (_backend_state["summary"], disabled, disabled, disabled, disabled, disabled, disabled)

def _mobile_access_markup():
    active_url = _launch_state.get("share_url") or _launch_state.get("local_url") or ""
    if not active_url:
        return (
            "Starting…",
            """
            <div class="mobile-card">
              <div class="mobile-wrap">
                <div class="mobile-title">Mobile Access</div>
                <div class="mobile-copy">The public app link will appear here after launch.</div>
              </div>
            </div>
            """,
        )
    qr_url = f"https://api.qrserver.com/v1/create-qr-code/?size=220x220&data={quote(active_url, safe='')}"
    markup = f"""
    <div class="mobile-card">
      <div class="mobile-wrap">
        <div class="mobile-title">Mobile Access</div>
        <div class="mobile-copy">Scan this QR on your phone to open the live SynID app.</div>
        <div class="qr-shell">
          <img src="{qr_url}" alt="SynID mobile QR" />
        </div>
        <div class="mobile-note">Use the share URL for mobile. Localhost links work only on the current machine.</div>
      </div>
    </div>
    """
    return active_url, markup

PRESETS = {
    "Anime Girl":  "anime girl, long silver hair, violet eyes, soft bangs, small mole under left eye, detailed face, upper body portrait, white background, sharp lineart",
    "Redhead":     "young woman, curly red hair, green eyes, freckles, upper body portrait, neutral background, photorealistic, high quality",
    "Elder":       "elderly man, grey beard, deep wrinkles, kind blue eyes, weathered face, upper body portrait, photorealistic",
    "Young Man":   "young man, short black hair, sharp jawline, brown eyes, light stubble, upper body portrait, photorealistic",
    "Brunette":    "young woman, brown eyes, dark brown hair, natural makeup, soft facial features, upper body portrait, photorealistic",
}

def create_char(prompt, seed, train_steps, progress=gr.Progress()):
    if not _is_backend_ready():
        return None, "load the pipelines first"
    if not prompt.strip():
        return None, "enter a description first"
    name = prompt[:30].strip().replace(" ", "_")
    _clear_all_hooks()
    progress(0.05, desc="generating anchors...")
    try:
        profile = create_character(
            identity_prompt=prompt, anchor_seed=int(seed),
            anchor_steps=30, anchor_guidance_scale=8.0,
            num_identity_tokens=4, train_steps=int(train_steps),
            bootstrap_top_k=2, bootstrap_generation_steps=20,
            refine_anchor_weight=0.55, correction_rounds=2,
            progress_callback=progress)
        progress(0.66, desc="attaching adapters")
        adp = attach_identity_adapters(pipe.unet, identity_dim=768, scale=0.5)
        hks = register_adapter_hooks(pipe.unet)
        train_adapter_on_bootstrap(
            pipe.unet, pipe.vae, pipe.text_encoder, pipe.tokenizer,
            adp, profile.identity_tokens,
            [c.image for c in profile.bootstrap_candidates],
            [c.prompt for c in profile.bootstrap_candidates],
            profile.base_identity_embedding,
            train_steps=200, lr=1e-5, clip_loss_weight=0.1, arcface_loss_weight=0.1,
            progress_callback=progress, progress_start=0.68, progress_end=0.90)
        train_adapter_on_bootstrap(
            pipe.unet, pipe.vae, pipe.text_encoder, pipe.tokenizer,
            adp, profile.identity_tokens,
            [c.image for c in profile.bootstrap_candidates],
            [c.prompt for c in profile.bootstrap_candidates],
            profile.base_identity_embedding,
            train_steps=60, lr=2e-6, clip_loss_weight=0.1, arcface_loss_weight=0.3,
            progress_callback=progress, progress_start=0.90, progress_end=0.98)
        _profiles[name] = profile
        _adapters_map[name] = adp
        _hooks_map[name] = hks
        _current_name[0] = name
        progress(1.0, desc="character ready")
        return profile.anchor_image, f"✓ ready — {name}"
    except Exception as e:
        return None, f"error: {e}"

def gen_variations(expression, n_imgs, id_scale, progress=gr.Progress()):
    global _last_images
    if not _is_backend_ready():
        return [], "load the pipelines first"
    name = _current_name[0]
    if not name or name not in _profiles:
        return [], "create a character first"
    profile = _profiles[name]
    prompt = profile.character_core_prompt
    if expression.strip(): prompt += ", " + expression.strip()
    imgs = []; scores = []
    for i in range(int(n_imgs)):
        progress(i/int(n_imgs))
        img = generate_with_adapter(
            profile.identity_tokens, prompt, profile.pose_image,
            pipe.unet, pipe, seed=5001+i*137,
            identity_scale=float(id_scale), generation_steps=30)
        c, _, _ = candidate_score(img, profile.base_identity_embedding, prompt)
        imgs.append(img); scores.append(c)
    _last_images = imgs
    avg = sum(scores)/len(scores)
    return imgs, f"avg identity {avg:.4f} · {len(imgs)} images"

def gen_posefree(n_imgs, progress=gr.Progress()):
    global _last_images
    if not _is_backend_ready():
        return [], "load the pipelines first"
    name = _current_name[0]
    if not name or name not in _profiles:
        return [], "create a character first"
    profile = _profiles[name]
    if 'posefree_pipe' not in dir() or posefree_pipe is None:
        globals()['posefree_pipe'] = load_posefree_pipe()
    imgs = []; scores = []
    for i in range(int(n_imgs)):
        progress(i/int(n_imgs))
        img, s = generate_posefree(
            profile.identity_tokens, profile.character_core_prompt,
            pipe.unet, posefree_pipe, profile.base_identity_embedding,
            seed=7001+i*137, identity_scale=1.2, generation_steps=30)
        imgs.append(img); scores.append(s)
    _last_images = imgs
    avg = sum(scores)/len(scores)
    return imgs, f"pose-free avg identity {avg:.4f}"

def run_eval_ui(mode, do_save, progress=gr.Progress()):
    if not _is_backend_ready():
        return [], "load the pipelines first"
    name = _current_name[0]
    if not name or name not in _profiles:
        return [], "create a character first"
    profile = _profiles[name]
    save_dir = f"/content/synid_output/{name}" if do_save else None
    progress(0.1, desc="evaluating...")
    if "quick" in mode:
        res = eval_quick(profile, adapters_active=True, save_dir=save_dir)
        imgs = res["images"]
        arc_str = f"{res['arc_mean']:.4f}" if res["arc_mean"] is not None else "N/A"
        status = f"quick eval · CLIP={res['clip_mean']:.4f} · ArcFace={arc_str}"
    else:
        res = eval_full(profile, adapters_active=True, save_dir=save_dir)
        imgs = []
        for suite_res in res.values(): imgs.extend(suite_res["images"])
        clip_avg = float(np.mean([r["clip_mean"] for r in res.values()]))
        status = f"full eval · avg CLIP={clip_avg:.4f}"
    progress(1.0)
    return imgs, status

def save_last_images(folder, prefix):
    if not _last_images:
        return "no images yet — generate something first"
    import os
    os.makedirs(folder, exist_ok=True)
    for i, img in enumerate(_last_images):
        img.save(f"{folder}/{prefix}_{i+1:03d}.png")
    return f"✓ saved {len(_last_images)} images → {folder}/"

def multi_char_consistency(progress=gr.Progress()):
    if not _is_backend_ready():
        return [], "load the pipelines first"
    if len(_profiles) < 2:
        return [], "need at least 2 characters"
    shared = ", neutral expression, studio lighting, upper body portrait"
    all_imgs = []
    for i, (name, profile) in enumerate(_profiles.items()):
        progress(i/len(_profiles), desc=name)
        adp = _adapters_map.get(name)
        if adp is None: continue
        hks = register_adapter_hooks(pipe.unet)
        for seed in [1001, 2002, 3003]:
            img = generate_with_adapter(
                profile.identity_tokens, profile.character_core_prompt + shared,
                profile.pose_image, pipe.unet, pipe,
                seed=seed, identity_scale=adaptive_identity_scale(profile.character_core_prompt),
                generation_steps=25)
            all_imgs.append(img)
        for h in hks: h.remove()
    return all_imgs, f"consistency check · {len(_profiles)} characters · 3 seeds each"

def use_preset(preset_name):
    return PRESETS.get(preset_name, "")

# ── Build UI ──────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="SynID", theme=gr.themes.Base()) as demo:

    gr.HTML("""
    <div class="hero">
        <div class="hero-tag">Synthetic Identity Diffusion</div>
        <div class="hero-title">SynID</div>
        <div style="margin-top:0.75rem">
            <span class="hero-badge">synthetic anchors</span>
            <span class="hero-badge">adapter training</span>
            <span class="hero-badge">dual conditioning</span>
        </div>
        <div class="hero-copy">
            Build a reusable character identity from text, refine it with synthetic examples,
            and generate consistent variations from a single workspace.
        </div>
    </div>
    """)

    with gr.Row():
        with gr.Column():
            gr.HTML('<div class="setup-grid">')
            with gr.Column(elem_classes="card"):
                gr.HTML('<div class="section-label">Setup</div>')
                gr.Markdown(
                    "Load the core diffusion and evaluation pipelines first. "
                    "This initializes DreamShaper, ControlNet, CLIP, OpenPose, and evaluation helpers."
                )
                with gr.Row():
                    load_btn = gr.Button("Load Pipelines", variant="primary")
                    load_status = gr.Textbox(
                        value="Pipelines not loaded",
                        show_label=False,
                        interactive=False,
                        elem_classes="status-text",
                    )
            with gr.Column():
                launch_url = gr.Textbox(
                    value="Starting…",
                    label="App URL",
                    interactive=False,
                )
                mobile_qr = gr.HTML("""
                <div class="mobile-card">
                  <div class="mobile-wrap">
                    <div class="mobile-title">Mobile Access</div>
                    <div class="mobile-copy">The public app link will appear here after launch.</div>
                  </div>
                </div>
                """)
            gr.HTML('</div>')

    with gr.Row(equal_height=False):

        # ── Left: Create ──────────────────────────────────────
        with gr.Column(scale=1, elem_classes="card"):
            gr.HTML('<div class="section-label">① describe your character</div>')

            prompt_box = gr.Textbox(
                placeholder="young woman, curly red hair, green eyes, freckles...",
                lines=3, show_label=False)

            gr.HTML('<div style="margin:0.5rem 0 0.25rem;font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;color:#475569">quick presets</div>')
            with gr.Row():
                for p in list(PRESETS.keys())[:3]:
                    gr.Button(p, elem_classes="preset-btn", size="sm").click(
                        lambda x=p: PRESETS[x], outputs=prompt_box)
            with gr.Row():
                for p in list(PRESETS.keys())[3:]:
                    gr.Button(p, elem_classes="preset-btn", size="sm").click(
                        lambda x=p: PRESETS[x], outputs=prompt_box)

            with gr.Row():
                seed_in   = gr.Number(value=42, label="Seed", precision=0)
                steps_in  = gr.Slider(100, 400, value=250, step=50, label="Train Steps")

            create_btn = gr.Button("Create Character", variant="primary", size="lg", interactive=False)
            anchor_out = gr.Image(show_label=False, height=260)
            status_out = gr.Textbox(show_label=False, interactive=False, elem_classes="status-text")

        # ── Right: Generate ───────────────────────────────────
        with gr.Column(scale=1, elem_classes="card"):
            gr.HTML('<div class="section-label">② generate</div>')

            with gr.Tabs():
                with gr.Tab("expressions & scenes"):
                    expr_box = gr.Textbox(
                        placeholder="bright smile · coffee shop · golden hour · surprised...",
                        lines=1, show_label=False)
                    with gr.Row():
                        n_imgs   = gr.Slider(1, 8, value=4, step=1, label="Images")
                        id_scale = gr.Slider(0.8, 2.5, value=1.4, step=0.1, label="Identity Scale")
                    gen_btn    = gr.Button("Generate", variant="primary", interactive=False)
                    gen_gallery = gr.Gallery(show_label=False, columns=2, height=340)
                    gen_status  = gr.Textbox(show_label=False, interactive=False, elem_classes="status-text")

                with gr.Tab("pose-free"):
                    gr.HTML('<p style="color:#475569;font-size:0.8rem;margin-bottom:1rem">no ControlNet — identity adapter only · more pose diversity</p>')
                    pf_n   = gr.Slider(1, 4, value=4, step=1, label="Images")
                    pf_btn = gr.Button("Generate Pose-Free", variant="primary", interactive=False)
                    pf_gallery = gr.Gallery(show_label=False, columns=2, height=340)
                    pf_status  = gr.Textbox(show_label=False, interactive=False, elem_classes="status-text")

                with gr.Tab("multi-character"):
                    gr.HTML('<p style="color:#475569;font-size:0.8rem;margin-bottom:1rem">generate all created characters with the same prompt — verifies identity is distinct and consistent</p>')
                    mc_btn     = gr.Button("Run Consistency Check", variant="primary", interactive=False)
                    mc_gallery = gr.Gallery(show_label=False, columns=3, height=340)
                    mc_status  = gr.Textbox(show_label=False, interactive=False, elem_classes="status-text")

                with gr.Tab("evaluate"):
                    gr.HTML('<p style="color:#475569;font-size:0.8rem;margin-bottom:1rem">run evaluation suite on current character — expression · scene · pose · seed</p>')
                    with gr.Row():
                        eval_mode = gr.Radio(["quick (4 imgs)", "full (24 imgs)"], value="quick (4 imgs)", label="Mode", show_label=False)
                        eval_save = gr.Checkbox(label="Save images", value=False)
                    eval_btn     = gr.Button("Run Evaluation", variant="primary", interactive=False)
                    eval_gallery = gr.Gallery(show_label=False, columns=3, height=340)
                    eval_status  = gr.Textbox(show_label=False, interactive=False, elem_classes="status-text")

    # ── Save panel ────────────────────────────────────────────
    with gr.Row(elem_classes="card"):
        gr.HTML('<div class="section-label">③ save</div>')
        with gr.Column(scale=2):
            save_folder = gr.Textbox(value="/content/synid_output", label="Save Folder", show_label=True)
            save_prefix = gr.Textbox(value="synid", label="Filename Prefix", show_label=True)
        with gr.Column(scale=1):
            save_btn    = gr.Button("Save Last Generated", variant="primary", interactive=False)
            save_status = gr.Textbox(show_label=False, interactive=False, elem_classes="status-text")

    # ── Wire ──────────────────────────────────────────────────
    load_btn.click(
        _load_backend,
        inputs=[],
        outputs=[load_status, create_btn, gen_btn, pf_btn, mc_btn, eval_btn, save_btn],
    )

    demo.load(
        _mobile_access_markup,
        inputs=[],
        outputs=[launch_url, mobile_qr],
    )

    create_btn.click(create_char,
        inputs=[prompt_box, seed_in, steps_in],
        outputs=[anchor_out, status_out])

    gen_btn.click(gen_variations,
        inputs=[expr_box, n_imgs, id_scale],
        outputs=[gen_gallery, gen_status])

    pf_btn.click(gen_posefree,
        inputs=[pf_n],
        outputs=[pf_gallery, pf_status])

    mc_btn.click(multi_char_consistency,
        inputs=[],
        outputs=[mc_gallery, mc_status])

    eval_btn.click(run_eval_ui,
        inputs=[eval_mode, eval_save],
        outputs=[eval_gallery, eval_status])

    save_btn.click(save_last_images,
        inputs=[save_folder, save_prefix],
        outputs=[save_status])

print("Launching SynID UI...")
launch_result = demo.launch(share=True, debug=False, quiet=True, prevent_thread_lock=True)
if isinstance(launch_result, tuple):
    if len(launch_result) >= 3:
        _launch_state["local_url"] = launch_result[1] or ""
        _launch_state["share_url"] = launch_result[2] or ""
    elif len(launch_result) >= 2:
        _launch_state["local_url"] = launch_result[1] or ""

if _launch_state["share_url"]:
    print(f"Mobile/share URL: {_launch_state['share_url']}")
elif _launch_state["local_url"]:
    print(f"Local URL: {_launch_state['local_url']}")
