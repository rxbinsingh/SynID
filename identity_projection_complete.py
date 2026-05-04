# ============================================================
# Identity Projection — Complete Pipeline
# App-first backend: definitions load immediately, heavy models load only
# when init_synid_backend() is called.
# ============================================================
# Run in Colab:
#   !pip install -q diffusers transformers accelerate controlnet_aux safetensors huggingface_hub
#   exec(open("identity_projection_complete.py").read())
#   init_synid_backend()
# ============================================================

# ── Imports ──────────────────────────────────────────────────
import os, json, zipfile
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from controlnet_aux import OpenposeDetector
from diffusers import (
    ControlNetModel, DDPMScheduler,
    StableDiffusionControlNetPipeline, StableDiffusionPipeline,
)
from IPython.display import display
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32
print("device:", device, "| dtype:", dtype)

# ── Backend state ────────────────────────────────────────────
# USE_POSEFREE=True → skip ControlNet entirely, use base_pipe for everything
# USE_SDXL=True     → use SDXL instead of SD1.5 (needs A100/V100, ~14GB VRAM)
USE_POSEFREE = False
USE_SDXL     = False
BACKEND_READY = False

controlnet = None
pipe = None
base_pipe = None
posefree_pipe = None
sdxl_pipe = None
clip_model = None
clip_processor = None
pose_detector = None
CLIP_DIM = 768

def init_synid_backend(use_posefree=None, use_sdxl=None, reload=False):
    global USE_POSEFREE, USE_SDXL, BACKEND_READY
    global controlnet, pipe, base_pipe, posefree_pipe, sdxl_pipe
    global clip_model, clip_processor, pose_detector, CLIP_DIM

    if use_posefree is not None:
        USE_POSEFREE = bool(use_posefree)
    if use_sdxl is not None:
        USE_SDXL = bool(use_sdxl)

    if BACKEND_READY and not reload:
        print("SynID backend already loaded.")
        return {
            "device": device,
            "dtype": str(dtype),
            "clip_dim": CLIP_DIM,
            "posefree": USE_POSEFREE,
            "sdxl": USE_SDXL,
        }

    print("Loading SynID core pipelines...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_openpose", torch_dtype=dtype).to(device)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "Lykon/DreamShaper", controlnet=controlnet, torch_dtype=dtype).to(device)
    base_pipe = StableDiffusionPipeline.from_pretrained(
        "Lykon/DreamShaper", torch_dtype=dtype).to(device)
    pipe.enable_attention_slicing()
    base_pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    pipe.safety_checker = None
    base_pipe.safety_checker = None
    posefree_pipe = base_pipe

    sdxl_pipe = None
    if USE_SDXL:
        try:
            from diffusers import StableDiffusionXLPipeline
            sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16, variant="fp16").to(device)
            sdxl_pipe.enable_attention_slicing()
            sdxl_pipe.safety_checker = None
            print("SDXL loaded. Use attach_identity_adapters(sdxl_pipe.unet, identity_dim=2048)")
        except Exception as _e:
            print(f"SDXL load failed: {_e}")

    try:
        clip_model     = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        CLIP_DIM = 768
        print("CLIP ViT-L/14 loaded.")
    except Exception as _e:
        clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        CLIP_DIM = 512
        print(f"CLIP ViT-B/32 loaded (ViT-L unavailable: {_e})")
    clip_model.eval()

    pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    BACKEND_READY = True
    print("SynID backend ready.")
    return {
        "device": device,
        "dtype": str(dtype),
        "clip_dim": CLIP_DIM,
        "posefree": USE_POSEFREE,
        "sdxl": USE_SDXL,
    }

# ── Data classes ─────────────────────────────────────────────
@dataclass
class CandidateResult:
    prompt: str; seed: int
    identity_similarity: float; prompt_similarity: float
    combined_score: float; image: Image.Image

@dataclass
class CharacterProfile:
    character_core_prompt: str; anchor_prompt: str; anchor_seed: int
    anchor_image: Image.Image; anchor_images: List[Image.Image]
    pose_image: Image.Image
    base_identity_embedding: torch.Tensor
    refined_identity_embedding: torch.Tensor
    identity_tokens: torch.Tensor; projector_loss: float
    bootstrap_candidates: List[CandidateResult] = field(default_factory=list)
    drift_history: List[float] = field(default_factory=list)

# ── Multi-token projector — dynamic CLIP dim, 8 tokens ───────
class MultiTokenIdentityProjector(nn.Module):
    def __init__(self, in_dim=None, hidden_dim=1024, num_tokens=8, out_dim=768):
        super().__init__()
        if in_dim is None: in_dim = CLIP_DIM
        self.num_tokens = num_tokens; self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),  # extra layer for capacity
            nn.Linear(hidden_dim, num_tokens * out_dim))
    def forward(self, x):
        return self.net(x.float()).view(x.shape[0], self.num_tokens, self.out_dim)

# ── Dual-token projector (improvement 2) ─────────────────────
class DualTokenProjector(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=1024, num_tokens=4, out_dim=768):
        super().__init__()
        self.num_tokens = num_tokens; self.out_dim = out_dim
        self.encoder = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU())
        self.face_head  = nn.Linear(hidden_dim, num_tokens * out_dim)
        self.style_head = nn.Linear(hidden_dim, num_tokens * out_dim)
    def forward(self, x):
        h = self.encoder(x.float())
        face  = self.face_head(h).view(x.shape[0], self.num_tokens, self.out_dim)
        style = self.style_head(h).view(x.shape[0], self.num_tokens, self.out_dim)
        return face, style

# ── UNet adapter ──────────────────────────────────────────────
class IdentityCrossAttention(nn.Module):
    def __init__(self, hidden_dim, identity_dim=768, num_heads=8):
        super().__init__()
        self.num_heads = num_heads; self.head_dim = hidden_dim // num_heads
        self.to_q   = nn.Linear(hidden_dim,    hidden_dim, bias=False)
        self.to_k   = nn.Linear(identity_dim,  hidden_dim, bias=False)
        self.to_v   = nn.Linear(identity_dim,  hidden_dim, bias=False)
        self.to_out = nn.Linear(hidden_dim,    hidden_dim, bias=False)
        nn.init.zeros_(self.to_out.weight)
    def forward(self, hs, id_tokens):
        hs = hs.float(); id_tokens = id_tokens.float()
        B, S, C = hs.shape; H = self.num_heads; D = self.head_dim
        q = self.to_q(hs).view(B,S,H,D).transpose(1,2)
        k = self.to_k(id_tokens).view(B,-1,H,D).transpose(1,2)
        v = self.to_v(id_tokens).view(B,-1,H,D).transpose(1,2)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2,-1)) * (D**-0.5), dim=-1)
        out  = torch.matmul(attn, v).transpose(1,2).contiguous().view(B,S,C)
        return self.to_out(out)

class IdentityAdapter(nn.Module):
    def __init__(self, hidden_dim, identity_dim=768, scale=1.0):
        super().__init__()
        self.cross_attn = IdentityCrossAttention(hidden_dim, identity_dim)
        # deeper adapter: MLP after cross-attn for more expressive identity transform
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.scale = scale
    def forward(self, hs, id_tokens):
        delta = self.cross_attn(hs, id_tokens)
        delta = delta + self.mlp(delta)  # residual MLP refinement
        return hs + self.scale * delta

# ── Utilities (exact from novel_pipeline_colab.py) ────────────
def build_generator(seed):
    return torch.Generator(device="cuda").manual_seed(seed) if device=="cuda" else torch.Generator().manual_seed(seed)

def normalize(x):
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-6)

def token_diversity_loss(tokens):
    if tokens.shape[1] <= 1:
        return torch.zeros(1, device=tokens.device, dtype=tokens.dtype).mean()
    n = normalize(tokens)
    sim = torch.matmul(n, n.transpose(1,2))
    eye = torch.eye(sim.shape[-1], device=sim.device, dtype=torch.bool).unsqueeze(0)
    return sim.masked_select(~eye).mean()

def encode_sd_text(prompt, return_mask=False):
    enc = pipe.tokenizer(prompt, padding="max_length",
        max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        emb = pipe.text_encoder(enc.input_ids.to(device))[0]
    return (emb, enc.attention_mask.to(device)) if return_mask else emb

def masked_mean_sd(prompt):
    emb, mask = encode_sd_text(prompt, return_mask=True)
    m = mask.unsqueeze(-1).float()
    return normalize((emb.float()*m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0))

def encode_clip_image(image):
    """
    Face-weighted CLIP encoding.
    Blends full-image embedding (0.4) with face-crop embedding (0.6).
    More face-specific signal — improves identity discriminability
    especially for male faces with less distinctive global features.
    """
    # full image
    inputs_full = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        out_full  = clip_model.vision_model(pixel_values=inputs_full["pixel_values"].to(device))
        feat_full = clip_model.visual_projection(out_full.pooler_output)

    # face crop (center 60% width, top 65% height)
    w, h = image.size
    face_crop = image.crop((int(w*.18), int(h*.03), int(w*.82), int(h*.67))).resize((224, 224))
    inputs_face = clip_processor(images=face_crop, return_tensors="pt")
    with torch.no_grad():
        out_face  = clip_model.vision_model(pixel_values=inputs_face["pixel_values"].to(device))
        feat_face = clip_model.visual_projection(out_face.pooler_output)

    # weighted blend: face gets 60% weight
    blended = 0.4 * feat_full.float() + 0.6 * feat_face.float()
    return normalize(blended)

def encode_clip_text(text):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        out  = clip_model.text_model(input_ids=inputs["input_ids"].to(device),
                                     attention_mask=inputs["attention_mask"].to(device))
        feat = clip_model.text_projection(out.pooler_output)
    return normalize(feat.float())

def candidate_score(image, anchor_identity, target_prompt, id_w=0.65, prompt_w=0.35):
    img_emb    = encode_clip_image(image)
    prompt_emb = encode_clip_text(target_prompt)
    id_sim     = float(F.cosine_similarity(img_emb, anchor_identity).mean())
    prompt_sim = float(F.cosine_similarity(img_emb, prompt_emb).mean())
    return id_sim, prompt_sim, id_w*id_sim + prompt_w*prompt_sim

def adaptive_identity_scale(prompt, base_scale=1.4):
    """
    Scale up identity injection when prompt is complex.
    IMPROVED: base_scale raised 1.2→1.4, and expression keywords
    (smile, laugh, surprised etc.) get an extra boost since expressions
    compete strongly with identity in the embedding space.
    """
    with torch.no_grad():
        text_emb = encode_sd_text(prompt)
        prompt_strength = float(text_emb.norm(dim=-1).mean().item())
    ratio = min(max(prompt_strength / 12.0, 0.0), 1.0)
    scale = base_scale * (1.0 + 0.25 * ratio)
    # expression boost: smiling/laughing/surprised change facial geometry most
    expression_keywords = ["smile", "smiling", "laugh", "laughing", "surprised",
                           "happy", "cheerful", "excited", "angry", "sad", "cry"]
    if any(kw in prompt.lower() for kw in expression_keywords):
        scale *= 1.2  # 20% extra identity pressure on expressive prompts
    return min(scale, 2.5)

def inject_identity_tokens(text_emb, identity_tokens, scale=1.0):
    c = text_emb.clone(); n = identity_tokens.shape[1]
    c[:,-n:,:] = c[:,-n:,:] + scale * identity_tokens.to(text_emb.dtype)
    return c

def clear_negative_slots(neg_emb, num_tokens):
    c = neg_emb.clone(); c[:,-num_tokens:,:] = 0; return c

def inject_dual_tokens(text_emb, face_tokens, style_tokens, face_scale=1.0, style_scale=1.0):
    c = text_emb.clone(); n = face_tokens.shape[1]
    c[:,-n:,:]    = c[:,-n:,:]    + face_scale  * face_tokens.to(c.dtype)
    c[:,-2*n:-n,:] = c[:,-2*n:-n,:] + style_scale * style_tokens.to(c.dtype)
    return c

def show_grid(images, cols=2):
    w, h = images[0].size
    rows = (len(images)+cols-1)//cols
    grid = Image.new("RGB", (cols*w, rows*h), (255,255,255))
    for i, img in enumerate(images):
        grid.paste(img, ((i%cols)*w, (i//cols)*h))
    return grid

# ── ArcFace: InsightFace with CLIP face crop fallback ─────────
import subprocess as _sp
try:
    _sp.run(["pip", "install", "-q", "insightface", "onnxruntime"], check=False,
            stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
except Exception:
    pass  # pip not available; InsightFace may already be installed

_arcface_app  = None
_arcface_mode = "clip_crop"

_cosface_app = None  # second face encoder for ensemble loss

def _init_arcface():
    global _arcface_app, _arcface_mode, _cosface_app
    try:
        from insightface.app import FaceAnalysis
        # buffalo_sc is less strict than buffalo_l — better for stylized/generated faces
        app = FaceAnalysis(name="buffalo_sc",
                           providers=["CUDAExecutionProvider","CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(320, 320))
        _arcface_app  = app
        _arcface_mode = "insightface"
        print("ArcFace (InsightFace buffalo_sc) loaded.")
        # CosFace: use buffalo_l which uses a CosFace-trained backbone
        try:
            app2 = FaceAnalysis(name="buffalo_l",
                                providers=["CUDAExecutionProvider","CPUExecutionProvider"])
            app2.prepare(ctx_id=0, det_size=(320, 320))
            _cosface_app = app2
            print("CosFace (InsightFace buffalo_l) loaded for ensemble loss.")
        except Exception as e2:
            print(f"CosFace (buffalo_l) unavailable ({e2}), using ArcFace only.")
    except Exception as e:
        print(f"InsightFace unavailable ({e}), using CLIP face crop.")
        _arcface_app  = None
        _arcface_mode = "clip_crop"

def encode_cosface(image: Image.Image) -> Optional[torch.Tensor]:
    """CosFace embedding via buffalo_l. Returns None if unavailable or no face."""
    if _cosface_app is None: return None
    img_np = np.array(image.convert("RGB"))
    faces  = _cosface_app.get(img_np)
    if not faces:
        img_np_large = np.array(image.convert("RGB").resize((640, 640)))
        faces = _cosface_app.get(img_np_large)
    if faces:
        emb = torch.tensor(faces[0].normed_embedding).float().unsqueeze(0)
        return F.normalize(emb, dim=-1).to(device)
    return None

def encode_arcface(image: Image.Image, face_detector=None, face_encoder=None) -> Optional[torch.Tensor]:
    """
    Real ArcFace via InsightFace if available.
    IMPORTANT: when InsightFace is active, return either a real ArcFace embedding
    or None. Do not fall back to CLIP crop in that branch, otherwise we mix
    512-d ArcFace vectors with 768-d CLIP vectors and break cosine losses.
    Only use the CLIP crop fallback when InsightFace itself is unavailable.
    """
    if _arcface_mode == "insightface" and _arcface_app is not None:
        img_np = np.array(image.convert("RGB"))
        faces  = _arcface_app.get(img_np)
        if not faces:
            # retry on resized image — helps with stylized/small faces
            img_np_large = np.array(image.convert("RGB").resize((640, 640)))
            faces = _arcface_app.get(img_np_large)
        if faces:
            emb = torch.tensor(faces[0].normed_embedding).float().unsqueeze(0)
            return F.normalize(emb, dim=-1).to(device)
        # no face detected; keep representation type consistent
        return None
    # fallback: CLIP face crop
    enc = face_encoder or clip_model
    w, h = image.size
    crop = image.crop((int(w*.18), int(h*.03), int(w*.82), int(h*.67))).resize((224,224))
    inputs = clip_processor(images=crop, return_tensors="pt")
    with torch.no_grad():
        out  = enc.vision_model(pixel_values=inputs["pixel_values"].to(device))
        feat = enc.visual_projection(out.pooler_output)
    return F.normalize(feat.float(), dim=-1)

def load_arcface():
    _init_arcface()
    return _arcface_mode, clip_model

def arcface_identity_loss(gen_image, anchor_face_emb, mtcnn=None, arcface=None):
    gen_face = encode_arcface(gen_image)
    if gen_face is None or anchor_face_emb is None: return None
    if gen_face.shape[-1] != anchor_face_emb.shape[-1]:
        return None
    return float(F.cosine_similarity(gen_face, anchor_face_emb).mean())

# initialise immediately
_init_arcface()

# ── Projector training (exact from novel_pipeline_colab.py) ──
def train_projector(identity_embedding, character_core_prompt,
                    num_tokens=4, hidden_dim=1024, train_steps=300,
                    lr=1e-4, diversity_weight=0.05, norm_weight=0.01,
                    anchor_face_emb=None, arcface_weight=0.5):
    """
    Multi-token projector with full ArcFace integration.
    arcface_weight=0.5: strong face geometry signal on both mean token and per-token.
    Uses orthogonal projection for face→text space mapping.
    """
    projector = MultiTokenIdentityProjector(in_dim=CLIP_DIM, num_tokens=num_tokens, hidden_dim=hidden_dim).to(device)
    projector.train()
    optimizer = torch.optim.Adam(projector.parameters(), lr=lr)
    target = masked_mean_sd(character_core_prompt).detach()
    face_target = None
    if anchor_face_emb is not None and _arcface_mode == "insightface":
        face_proj = nn.Linear(anchor_face_emb.shape[-1], 768, bias=False).to(device)
        nn.init.orthogonal_(face_proj.weight)
        with torch.no_grad():
            face_target = normalize(face_proj(anchor_face_emb.float())).detach()
    emb = identity_embedding.float(); last_loss = 0.0
    for step in range(train_steps):
        optimizer.zero_grad(set_to_none=True)
        tokens  = projector(emb)
        summary = normalize(tokens.mean(dim=1))
        text_loss    = (1 - F.cosine_similarity(summary, target)).mean()
        per_tok      = normalize(tokens.view(-1, tokens.shape[-1]))
        per_tok_loss = (1 - F.cosine_similarity(per_tok, target.expand(per_tok.shape[0], -1))).mean() * 0.3
        div_loss     = token_diversity_loss(tokens)
        norm_loss    = (tokens.norm(dim=-1).mean() - 1.0).abs()
        loss = text_loss + per_tok_loss + diversity_weight * div_loss + norm_weight * norm_loss
        if face_target is not None:
            # mean token → face direction
            face_loss = (1 - F.cosine_similarity(summary, face_target)).mean()
            # per-token → face direction (full ArcFace integration)
            per_tok_face = (1 - F.cosine_similarity(
                per_tok, face_target.expand(per_tok.shape[0], -1))).mean() * 0.3
            loss = loss + arcface_weight * face_loss + arcface_weight * per_tok_face
        loss.backward(); optimizer.step()
        last_loss = float(loss.item())
        if step==0 or (step+1)%50==0 or step==train_steps-1:
            print(f"  step {step+1} | loss {last_loss:.6f} | text {float(text_loss):.6f} | div {float(div_loss):.6f}")
    projector.eval(); return projector, last_loss

# ── Dual projector training ───────────────────────────────────
def train_dual_projector(identity_embedding, character_core_prompt,
                         style_prompt=None, num_tokens=4, train_steps=300, lr=1e-4):
    style_prompt = style_prompt or character_core_prompt
    projector = DualTokenProjector(num_tokens=num_tokens).to(device)
    projector.train()
    optimizer = torch.optim.Adam(projector.parameters(), lr=lr)
    face_target  = masked_mean_sd(character_core_prompt).detach()
    style_target = masked_mean_sd(style_prompt).detach()
    emb = identity_embedding.float(); last_loss = 0.0
    def _n(x): return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    for step in range(train_steps):
        optimizer.zero_grad(set_to_none=True)
        face_tokens, style_tokens = projector(emb)
        face_loss  = (1 - F.cosine_similarity(_n(face_tokens.mean(dim=1)),  face_target)).mean()
        style_loss = (1 - F.cosine_similarity(_n(style_tokens.mean(dim=1)), style_target)).mean()
        ortho_loss = F.cosine_similarity(
            _n(face_tokens.view(-1, face_tokens.shape[-1])),
            _n(style_tokens.view(-1, style_tokens.shape[-1]))).mean().abs()
        loss = face_loss + style_loss + 0.1 * ortho_loss
        loss.backward(); optimizer.step()
        last_loss = float(loss.item())
        if step==0 or (step+1)%50==0 or step==train_steps-1:
            print(f"  step {step+1} | face {float(face_loss):.4f} | style {float(style_loss):.4f} | ortho {float(ortho_loss):.4f}")
    projector.eval(); return projector, last_loss

# ── Generation helpers ────────────────────────────────────────
def generate_with_tokens(identity_tokens, prompt, pose_image,
                         seed=1234, identity_scale=1.0, generation_steps=30,
                         guidance_scale=7.5, negative_prompt="", controlnet_scale=0.55):
    text_emb = encode_sd_text(prompt); neg_emb = encode_sd_text(negative_prompt or "")
    conditioned     = inject_identity_tokens(text_emb, identity_tokens, identity_scale)
    neg_conditioned = clear_negative_slots(neg_emb, identity_tokens.shape[1])
    expression_keywords = ["smile","smiling","laugh","laughing","surprised",
                           "happy","cheerful","excited","angry","sad","cry"]
    if any(kw in prompt.lower() for kw in expression_keywords):
        controlnet_scale = min(controlnet_scale, 0.35)
    # pose-free mode: use base_pipe (no ControlNet)
    if USE_POSEFREE or pose_image is None:
        return base_pipe(prompt_embeds=conditioned, negative_prompt_embeds=neg_conditioned,
                         num_inference_steps=generation_steps, guidance_scale=guidance_scale,
                         generator=build_generator(seed)).images[0]
    return pipe(prompt_embeds=conditioned, negative_prompt_embeds=neg_conditioned,
                image=pose_image, controlnet_conditioning_scale=controlnet_scale,
                num_inference_steps=generation_steps, guidance_scale=guidance_scale,
                generator=build_generator(seed)).images[0]

def generate_batch_with_tokens(identity_tokens, prompts, pose_image,
                                seeds, identity_scale=1.0, generation_steps=20,
                                guidance_scale=7.5, controlnet_scale=0.55, batch_size=4):
    """
    Batched bootstrap generation — generates multiple images in grouped forward passes.
    ~3-4x faster than sequential on T4 for 20 prompts.
    Each image gets its own seed via per-image generator list.
    """
    all_images = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_seeds   = seeds[i:i+batch_size]
        cond_list, neg_list = [], []
        for prompt in batch_prompts:
            text_emb = encode_sd_text(prompt)
            neg_emb  = encode_sd_text("")
            cond_list.append(inject_identity_tokens(text_emb, identity_tokens, identity_scale))
            neg_list.append(clear_negative_slots(neg_emb, identity_tokens.shape[1]))
        cond_batch = torch.cat(cond_list, dim=0)
        neg_batch  = torch.cat(neg_list,  dim=0)
        # per-image generators for seed diversity within the batch
        generators = [build_generator(s) for s in batch_seeds]
        print(f"  Batch {i//batch_size+1}: generating {len(batch_prompts)} images...")
        if USE_POSEFREE or pose_image is None:
            imgs = base_pipe(
                prompt_embeds=cond_batch,
                negative_prompt_embeds=neg_batch,
                num_inference_steps=generation_steps,
                guidance_scale=guidance_scale,
                generator=generators,
            ).images
        else:
            pose_batch = [pose_image] * len(batch_prompts)
            imgs = pipe(
                prompt_embeds=cond_batch,
                negative_prompt_embeds=neg_batch,
                image=pose_batch,
                controlnet_conditioning_scale=controlnet_scale,
                num_inference_steps=generation_steps,
                guidance_scale=guidance_scale,
                generator=generators,
            ).images
        all_images.extend(imgs)
    return all_images

# ── Multi-anchor ensemble (exact from novel_pipeline_colab.py) 
def build_ensemble_embedding(identity_prompt, anchor_seeds,
                              anchor_steps=30, anchor_guidance_scale=8.0,
                              anchor_negative_prompt=""):
    text_target = encode_clip_text(identity_prompt)
    anchor_images, embeddings = [], []
    # batch all anchors in one call — each gets a different latent via different seeds
    print(f"Generating {len(anchor_seeds)} anchors (batched)...")
    # build per-image generators for diversity
    generators = [build_generator(s) for s in anchor_seeds]
    imgs = base_pipe(
        prompt=[identity_prompt] * len(anchor_seeds),
        negative_prompt=[anchor_negative_prompt or ""] * len(anchor_seeds),
        num_inference_steps=anchor_steps,
        guidance_scale=anchor_guidance_scale,
        generator=generators,
    ).images
    for img in imgs:
        anchor_images.append(img); embeddings.append(encode_clip_image(img))
    stacked = torch.cat(embeddings, dim=0)
    sims    = F.cosine_similarity(stacked, text_target.expand_as(stacked), dim=-1)
    weights = torch.softmax(sims, dim=0).unsqueeze(-1)
    ensemble = normalize((stacked*weights).sum(dim=0, keepdim=True))
    w_list = weights.squeeze(-1).tolist()
    if isinstance(w_list, float): w_list = [w_list]
    print(f"Anchor weights: {[round(w,3) for w in w_list]}")
    return ensemble, anchor_images

# ── Bootstrap refinement (exact from novel_pipeline_colab.py) 
def bootstrap_refine(base_identity, projector, character_core_prompt,
                     bootstrap_prompts, bootstrap_seeds, pose_image,
                     bootstrap_top_k=2, bootstrap_identity_scale=1.0,
                     bootstrap_generation_steps=20, bootstrap_guidance_scale=7.5,
                     bootstrap_controlnet_scale=0.55, refine_anchor_weight=0.55,
                     num_tokens=4, train_steps=150, projector_lr=1e-4,
                     anchor_face_emb=None, anchor_image=None):
    with torch.no_grad():
        bootstrap_tokens = projector(base_identity).to(dtype)
    candidates, candidate_embeddings = [], []

    # BATCHED generation — ~3-4x faster than sequential on T4
    seeds_list = [int(bootstrap_seeds[i % len(bootstrap_seeds)]) for i in range(len(bootstrap_prompts))]
    print(f"  Batched bootstrap: {len(bootstrap_prompts)} images in groups of 4...")
    all_images = generate_batch_with_tokens(
        bootstrap_tokens, bootstrap_prompts, pose_image, seeds_list,
        identity_scale=bootstrap_identity_scale,
        generation_steps=bootstrap_generation_steps,
        guidance_scale=bootstrap_guidance_scale,
        controlnet_scale=bootstrap_controlnet_scale,
        batch_size=4)

    for i, (prompt, image) in enumerate(zip(bootstrap_prompts, all_images)):
        seed = seeds_list[i]
        img_emb = encode_clip_image(image)
        id_sim, prompt_sim, score = candidate_score(image, base_identity, prompt)
        if anchor_face_emb is not None:
            arc_emb = encode_arcface(image)
            if arc_emb is not None and arc_emb.shape[-1] == anchor_face_emb.shape[-1]:
                arc_sim = float(F.cosine_similarity(arc_emb, anchor_face_emb).mean())
                score   = 0.45 * id_sim + 0.35 * arc_sim + 0.20 * prompt_sim
                print(f"  Bootstrap {i+1}: CLIP={id_sim:.4f} | ArcFace={arc_sim:.4f} | prompt={prompt_sim:.4f} | score={score:.4f}")
            else:
                print(f"  Bootstrap {i+1}: identity={id_sim:.4f} | prompt={prompt_sim:.4f} | score={score:.4f}")
        else:
            print(f"  Bootstrap {i+1}: identity={id_sim:.4f} | prompt={prompt_sim:.4f} | score={score:.4f}")
        candidates.append(CandidateResult(prompt=prompt, seed=seed,
            identity_similarity=id_sim, prompt_similarity=prompt_sim,
            combined_score=score, image=image))
        candidate_embeddings.append(img_emb)
    ranked   = sorted(range(len(candidates)), key=lambda i: candidates[i].combined_score, reverse=True)
    # diversity-enforced selection: avoid blending near-identical candidates
    selected = []
    selected_embs = []
    for idx in ranked:
        if candidates[idx].identity_similarity < 0.85:
            continue  # skip low-identity candidates
        if selected_embs:
            # check diversity against already selected
            new_emb = candidate_embeddings[idx]
            too_similar = any(
                float(F.cosine_similarity(new_emb, se).mean()) > 0.92
                for se in selected_embs
            )
            if too_similar:
                continue
        selected.append(idx)
        selected_embs.append(candidate_embeddings[idx])
        if len(selected) >= bootstrap_top_k:
            break
    # fallback if not enough diverse candidates
    if len(selected) < 2:
        selected = ranked[:min(bootstrap_top_k, len(ranked))]
    sel_emb  = torch.stack([candidate_embeddings[i] for i in selected], dim=0)
    refined_identity = normalize(
        refine_anchor_weight * base_identity + (1.0-refine_anchor_weight) * normalize(sel_emb.mean(dim=0)))
    print("Selected scores:", [round(candidates[i].combined_score,4) for i in selected])
    print("Retraining projector on refined identity...")
    # use best bootstrap candidate's ArcFace embedding for face-aware retraining
    best_face_emb = encode_arcface(candidates[selected[0]].image)
    final_projector, final_loss = train_projector(refined_identity, character_core_prompt,
                                                   num_tokens=num_tokens, train_steps=train_steps,
                                                   lr=projector_lr,
                                                   anchor_face_emb=best_face_emb)
    return final_projector, final_loss, refined_identity, candidates

# ── Drift correction (exact from novel_pipeline_colab.py) ─────
def drift_correction(projector, refined_identity, character_core_prompt, pose_image,
                     correction_rounds=2, correction_steps=40, correction_lr=5e-5,
                     generation_seed=9999, generation_steps=20,
                     identity_scale=1.0, controlnet_scale=0.55):
    drift_history = []; embedding = refined_identity.float()
    for r in range(correction_rounds):
        print(f"Drift correction round {r+1}/{correction_rounds}")
        with torch.inference_mode():
            tokens = projector(embedding).to(dtype)
        gen_image = generate_with_tokens(tokens, character_core_prompt, pose_image,
                                         seed=generation_seed+r, identity_scale=identity_scale,
                                         generation_steps=generation_steps, controlnet_scale=controlnet_scale)
        gen_emb = encode_clip_image(gen_image).float()
        drift   = float(1 - F.cosine_similarity(gen_emb, embedding).mean())
        drift_history.append(drift); print(f"  drift={drift:.4f}")
        if drift < 0.015: print("  drift low, stopping"); break
        blended = normalize(0.65*embedding + 0.35*gen_emb)
        projector.train()
        opt = torch.optim.Adam(projector.parameters(), lr=correction_lr)
        target = masked_mean_sd(character_core_prompt).detach()
        for _ in range(correction_steps):
            opt.zero_grad(set_to_none=True)
            t = projector(blended); s = normalize(t.mean(dim=1))
            loss = (1-F.cosine_similarity(s,target)).mean() + 0.05*token_diversity_loss(t)
            loss.backward(); opt.step()
        projector.eval(); embedding = blended
    return projector, embedding, drift_history

# ── UNet adapter attachment (exact from unet_adapter_colab.py) 
def attach_identity_adapters(unet, identity_dim=768, scale=0.5):
    for p in unet.parameters(): p.requires_grad_(False)
    adapters = nn.ModuleList(); dev = next(unet.parameters()).device
    for block in unet.down_blocks:
        if hasattr(block,"attentions"):
            for ab in block.attentions:
                for tb in ab.transformer_blocks:
                    hd = tb.attn2.to_out[0].out_features
                    a = IdentityAdapter(hd, identity_dim, scale=scale*0.5).to(dev).float()
                    tb._identity_adapter = a; adapters.append(a)
    if hasattr(unet.mid_block,"attentions"):
        for ab in unet.mid_block.attentions:
            for tb in ab.transformer_blocks:
                hd = tb.attn2.to_out[0].out_features
                a = IdentityAdapter(hd, identity_dim, scale=scale*0.75).to(dev).float()
                tb._identity_adapter = a; adapters.append(a)
    for block in unet.up_blocks:
        if hasattr(block,"attentions"):
            for ab in block.attentions:
                for tb in ab.transformer_blocks:
                    hd = tb.attn2.to_out[0].out_features
                    a = IdentityAdapter(hd, identity_dim, scale=scale).to(dev).float()
                    tb._identity_adapter = a; adapters.append(a)
    n_down = sum(1 for b in unet.down_blocks if hasattr(b,"attentions") for a in b.attentions for _ in a.transformer_blocks)
    n_mid  = sum(1 for a in unet.mid_block.attentions for _ in a.transformer_blocks) if hasattr(unet.mid_block,"attentions") else 0
    n_up   = len(adapters) - n_down - n_mid
    print(f"Attached {len(adapters)} adapters: {n_down} down | {n_mid} mid | {n_up} up [float32]")
    return adapters

def register_adapter_hooks(unet):
    hooks = []
    def make_hook(tb):
        def hook(module, input, output):
            tokens = getattr(unet, "_current_identity_tokens", None)
            if tokens is None: return output
            adapter = getattr(tb, "_identity_adapter", None)
            if adapter is None: return output
            if isinstance(output, tuple):
                hs = output[0]; dt = hs.dtype
                return (adapter(hs, tokens).to(dt),) + output[1:]
            return adapter(output, tokens).to(output.dtype)
        return hook
    for block in unet.down_blocks:
        if hasattr(block,"attentions"):
            for ab in block.attentions:
                for tb in ab.transformer_blocks:
                    if hasattr(tb,"_identity_adapter"):
                        hooks.append(tb.register_forward_hook(make_hook(tb)))
    if hasattr(unet.mid_block,"attentions"):
        for ab in unet.mid_block.attentions:
            for tb in ab.transformer_blocks:
                if hasattr(tb,"_identity_adapter"):
                    hooks.append(tb.register_forward_hook(make_hook(tb)))
    for block in unet.up_blocks:
        if hasattr(block,"attentions"):
            for ab in block.attentions:
                for tb in ab.transformer_blocks:
                    if hasattr(tb,"_identity_adapter"):
                        hooks.append(tb.register_forward_hook(make_hook(tb)))
    print(f"Registered {len(hooks)} hooks."); return hooks

def set_identity_tokens(unet, tokens):
    unet._current_identity_tokens = tokens

# ── Adapter training (exact from unet_adapter_colab.py) ──────
# ── Adapter training — proven git1 approach + ArcFace every 10 steps ──
def train_adapter_on_bootstrap(
    unet, vae, text_encoder, tokenizer, adapters,
    identity_tokens, bootstrap_images, bootstrap_prompts,
    base_identity_embedding, train_steps=200, lr=1e-5,
    clip_loss_weight=0.1, arcface_loss_weight=0.1, noise_scheduler=None,
    progress_callback=None, progress_start=0.0, progress_end=1.0):
    """
    Proven git1 approach: MSE + CLIP (every 10 steps) + ArcFace (every 10 steps)
    using noisy reconstruction decode. ArcFace fires 2.5x more than original (10 vs 25).
    """
    if noise_scheduler is None:
        noise_scheduler = pipe.scheduler  # reuse already-loaded scheduler — no extra download
    optimizer  = torch.optim.AdamW(adapters.parameters(), lr=lr, weight_decay=1e-4)
    lr_sched   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_steps)
    dev        = next(unet.parameters()).device
    vae_scale  = vae.config.scaling_factor

    latents_list, text_embs_list = [], []
    for img in bootstrap_images:
        arr = np.array(img.resize((512,512))).astype("float32")/127.5 - 1.0
        img_t = torch.tensor(arr).permute(2,0,1).unsqueeze(0).to(dev).half()
        with torch.no_grad():
            latents_list.append(vae.encode(img_t).latent_dist.sample() * vae_scale)
    for prompt in bootstrap_prompts:
        ids = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad(): text_embs_list.append(text_encoder(ids)[0])

    current_clip_anchor = encode_clip_image(bootstrap_images[0]).float().to(dev)
    anchor_emb = base_identity_embedding.float().to(dev)
    if anchor_emb.shape[-1] != current_clip_anchor.shape[-1]:
        print(
            f"  base_identity_embedding dim {anchor_emb.shape[-1]} "
            f"!= current CLIP dim {current_clip_anchor.shape[-1]} | "
            "recomputing anchor embedding from current CLIP model"
        )
        anchor_emb = current_clip_anchor
    _af             = encode_arcface(bootstrap_images[0])
    anchor_face_emb = _af.float().to(dev) if _af is not None else None
    if anchor_face_emb is None:
        print("  No face in anchor, skipping ArcFace loss")
    id_tokens   = identity_tokens.float().clone().detach().to(dev)
    orig_scales = [a.scale for a in adapters]

    print(f"Training adapter ({train_steps} steps, clip_w={clip_loss_weight}, arc_w={arcface_loss_weight})...")
    last_loss = 0.0; last_clip = 0.0; last_face = 0.0
    for step in range(train_steps):
        warmup = min(1.0, step/(train_steps*0.5))
        for a, s in zip(adapters, orig_scales): a.scale = 0.1 + warmup*(s-0.1)
        set_identity_tokens(unet, id_tokens)
        idx   = step % len(latents_list)
        noise = torch.randn_like(latents_list[idx])
        t     = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=dev).long()
        noisy = noise_scheduler.add_noise(latents_list[idx], noise, t)
        optimizer.zero_grad(set_to_none=True)
        noise_pred = unet(noisy, t, encoder_hidden_states=text_embs_list[idx]).sample
        mse_loss  = F.mse_loss(noise_pred.float(), noise.float())
        clip_loss = torch.tensor(0.0, device=dev)
        face_loss = torch.tensor(0.0, device=dev)
        run_clip    = (step % 10 == 0)
        run_arcface = (step % 10 == 0) and anchor_face_emb is not None  # every 10 (was 25)
        if run_clip or run_arcface:
            with torch.no_grad():
                pred_lat = ((noisy.float()-noise_pred.float())/vae_scale).half()
                decoded  = vae.decode(pred_lat).sample
                import torchvision.transforms.functional as _tvf
                decoded_pil = _tvf.to_pil_image(
                    ((decoded.float().clamp(-1,1)+1)/2)[0].cpu())
            if run_clip:
                gen_emb   = encode_clip_image(decoded_pil).float().to(dev)
                if gen_emb.shape[-1] == anchor_emb.shape[-1]:
                    clip_loss = (1 - F.cosine_similarity(gen_emb, anchor_emb)).mean()
                else:
                    if step == 0:
                        print(
                            f"  Skipping CLIP loss: generated dim {gen_emb.shape[-1]} "
                            f"!= anchor dim {anchor_emb.shape[-1]}"
                        )
            if run_arcface:
                gen_face = encode_arcface(decoded_pil)
                if gen_face is not None and gen_face.shape[-1] == anchor_face_emb.shape[-1]:
                    face_loss = (1 - F.cosine_similarity(gen_face.float().to(dev), anchor_face_emb)).mean()
        loss = mse_loss + clip_loss_weight*clip_loss + arcface_loss_weight*face_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapters.parameters(), max_norm=1.0)
        optimizer.step(); lr_sched.step()
        last_loss = float(mse_loss.item())
        if float(clip_loss) > 0: last_clip = float(clip_loss)
        if float(face_loss) > 0: last_face = float(face_loss)
        if progress_callback is not None and (
            step == 0 or (step + 1) % 10 == 0 or step == train_steps - 1
        ):
            frac = (step + 1) / max(train_steps, 1)
            progress_value = progress_start + (progress_end - progress_start) * frac
            progress_callback(progress_value, desc=f"training adapter {step+1}/{train_steps}")
        if step==0 or (step+1)%25==0 or step==train_steps-1:
            print(f"  step {step+1}/{train_steps} | mse {last_loss:.6f} | clip {last_clip:.4f} | face {last_face:.4f} | scale {adapters[0].scale:.2f}")
    set_identity_tokens(unet, None)
    print(f"Adapter done. Final MSE: {last_loss:.6f}"); return last_loss

# ── Adapter generation ────────────────────────────────────────
@torch.inference_mode()
def generate_with_adapter(identity_tokens, prompt, pose_image, unet, pipe,
                          seed=1234, identity_scale=1.0, generation_steps=30,
                          guidance_scale=7.5, negative_prompt="", controlnet_scale=0.55):
    set_identity_tokens(unet, identity_tokens.to(next(unet.parameters()).device))
    text_emb = encode_sd_text(prompt); neg_emb = encode_sd_text(negative_prompt or "")
    conditioned = inject_identity_tokens(text_emb, identity_tokens, identity_scale)
    n = identity_tokens.shape[1]; neg_mod = neg_emb.clone()
    neg_mod[:,-n:,:] = neg_mod[:,-n:,:] - 0.5 * identity_tokens.to(neg_emb.dtype)
    expression_keywords = ["smile","smiling","laugh","laughing","surprised",
                           "happy","cheerful","excited","angry","sad","cry"]
    if any(kw in prompt.lower() for kw in expression_keywords):
        controlnet_scale = min(controlnet_scale, 0.35)
    # pose-free mode: use posefree_pipe (no ControlNet)
    if USE_POSEFREE or pose_image is None:
        image = posefree_pipe(
            prompt_embeds=conditioned, negative_prompt_embeds=neg_mod,
            num_inference_steps=generation_steps, guidance_scale=guidance_scale,
            generator=build_generator(seed)).images[0]
    else:
        image = pipe(prompt_embeds=conditioned, negative_prompt_embeds=neg_mod,
                     image=pose_image, controlnet_conditioning_scale=controlnet_scale,
                     num_inference_steps=generation_steps, guidance_scale=guidance_scale,
                     generator=build_generator(seed)).images[0]
    set_identity_tokens(unet, None); return image

# ── Consistency auto-tune (exact from unet_adapter_colab.py) ─
def consistency_auto_tune(adapters, unet, vae, text_encoder, tokenizer,
                          profile, pipe, consistency_threshold=0.93,
                          max_extra_rounds=2, tune_steps=50, tune_lr=5e-6, eval_seeds=None):
    if eval_seeds is None: eval_seeds = [5555,6666,7777,8888]
    avg_consistency = 0.0
    for round_idx in range(max_extra_rounds):
        print(f"Consistency check round {round_idx+1}/{max_extra_rounds}")
        eval_images = [generate_with_adapter(
            profile.identity_tokens, profile.character_core_prompt,
            profile.pose_image, unet, pipe, seed=s,
            identity_scale=1.0, generation_steps=20, controlnet_scale=0.55)
            for s in eval_seeds]
        embs = torch.cat([encode_clip_image(img) for img in eval_images], dim=0)
        n = embs.shape[0]
        sims = [float(F.cosine_similarity(embs[i:i+1], embs[j:j+1]).mean())
                for i in range(n) for j in range(i+1,n)]
        avg_consistency = sum(sims)/len(sims)
        print(f"  avg pairwise consistency: {avg_consistency:.4f} (threshold: {consistency_threshold})")
        if avg_consistency >= consistency_threshold: print("  OK"); break
        print(f"  below threshold, fine-tuning {tune_steps} steps...")
        train_adapter_on_bootstrap(
            unet, vae, text_encoder, tokenizer, adapters,
            profile.identity_tokens,
            [c.image for c in profile.bootstrap_candidates],
            [c.prompt for c in profile.bootstrap_candidates],
            profile.base_identity_embedding,
            train_steps=tune_steps, lr=tune_lr)
    return avg_consistency

# ── Seed quality predictor — fast CLIP-based scoring ─────────
@torch.inference_mode()
def select_best_seeds(identity_tokens, prompt, unet, pipe, anchor_embedding,
                      candidate_seeds, pose_image=None, num_select=4,
                      quality_threshold=0.88, identity_scale=1.0):
    """Instant seed selection — just return first N seeds. No scoring needed."""
    selected = list(candidate_seeds[:num_select])
    print(f"  Seeds selected: {selected}")
    return selected

# ── Improvement 3: Pose-free generation ──────────────────────
def load_posefree_pipe():
    """Pose-free pipe reuses base_pipe — no extra download or VRAM."""
    base_pipe.unet = pipe.unet  # share UNet so adapters work
    print("Pose-free pipeline ready (reusing base_pipe).")
    return base_pipe

@torch.inference_mode()
def generate_posefree(identity_tokens, prompt, unet, posefree_pipe, anchor_embedding,
                      seed=1234, identity_scale=1.0, generation_steps=30,
                      guidance_scale=7.5, drift_threshold=0.10, max_retries=2):
    best_image, best_score, current_scale = None, 0.0, identity_scale
    for attempt in range(max_retries+1):
        set_identity_tokens(unet, identity_tokens.to(next(unet.parameters()).device))
        text_emb = encode_sd_text(prompt); neg_emb = encode_sd_text("")
        conditioned = inject_identity_tokens(text_emb, identity_tokens, current_scale)
        n = identity_tokens.shape[1]; neg_mod = neg_emb.clone()
        neg_mod[:,-n:,:] = neg_mod[:,-n:,:] - 0.8 * identity_tokens.to(neg_emb.dtype)
        image = posefree_pipe(prompt_embeds=conditioned, negative_prompt_embeds=neg_mod,
                              num_inference_steps=generation_steps, guidance_scale=guidance_scale,
                              generator=build_generator(seed+attempt*100)).images[0]
        set_identity_tokens(unet, None)
        id_sim = float(F.cosine_similarity(encode_clip_image(image).float(),
                                           anchor_embedding.float()).mean())
        if id_sim > best_score: best_score, best_image = id_sim, image
        if id_sim >= (1-drift_threshold): break
        current_scale = min(current_scale*1.4, 3.0)
    return best_image, best_score

# ── Improvement 5: Meta-learned adapter init ─────────────────
def save_adapter_to_library(adapters, character_name, library_dir="/content/adapter_library"):
    os.makedirs(library_dir, exist_ok=True)
    path = f"{library_dir}/{character_name}_adapter.pt"
    torch.save(adapters.state_dict(), path)
    print(f"Saved adapter to library: {path}")

def build_meta_init(adapters, library_dir="/content/adapter_library"):
    files = [f for f in os.listdir(library_dir) if f.endswith("_adapter.pt")]
    if not files: print("No adapters in library yet."); return False
    print(f"Building meta-init from {len(files)} saved adapters...")
    state_dicts = [torch.load(f"{library_dir}/{f}", map_location=device) for f in files]
    avg_state = {k: torch.stack([sd[k].float() for sd in state_dicts]).mean(dim=0)
                 for k in state_dicts[0].keys()}
    adapters.load_state_dict({k: v.to(next(adapters.parameters()).device) for k,v in avg_state.items()})
    print(f"Meta-init loaded from {len(files)} characters."); return True

# ── Checkpoint save/load ──────────────────────────────────────
def save_checkpoint(profile, adapters, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(profile.identity_tokens,            f"{save_dir}/identity_tokens.pt")
    torch.save(profile.refined_identity_embedding, f"{save_dir}/refined_embedding.pt")
    torch.save(profile.base_identity_embedding,    f"{save_dir}/base_embedding.pt")
    torch.save(adapters.state_dict(),              f"{save_dir}/adapter_weights.pt")
    if profile.pose_image is not None:
        profile.pose_image.save(f"{save_dir}/pose_image.png")
    profile.anchor_image.save(f"{save_dir}/anchor_image.png")
    with open(f"{save_dir}/metadata.json","w") as f:
        json.dump({"character_core_prompt": profile.character_core_prompt,
                   "anchor_prompt": profile.anchor_prompt,
                   "anchor_seed": profile.anchor_seed,
                   "projector_loss": profile.projector_loss,
                   "drift_history": profile.drift_history}, f, indent=2)
    print(f"Checkpoint saved: {save_dir}")

def export_character(name, checkpoint_dir="/content/checkpoints"):
    src = f"{checkpoint_dir}/{name}"; out = f"{checkpoint_dir}/{name}.character"
    with zipfile.ZipFile(out,"w",zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(src): zf.write(f"{src}/{fname}", arcname=fname)
    print(f"Exported: {out} ({os.path.getsize(out)/1e6:.1f} MB)"); return out

# ── create_character (exact proven pipeline) ──────────────────
def create_character(
    identity_prompt, character_core_prompt=None,
    anchor_seeds=None, anchor_seed=1234,
    anchor_steps=30, anchor_guidance_scale=8.0, anchor_negative_prompt="",
    num_identity_tokens=8, train_steps=250, projector_lr=1e-4,
    bootstrap_prompts=None, bootstrap_seeds=None,
    bootstrap_top_k=4, bootstrap_identity_scale=1.0,
    bootstrap_generation_steps=20, bootstrap_guidance_scale=7.5,
    bootstrap_controlnet_scale=0.55, refine_anchor_weight=0.55,
    correction_rounds=2, progress_callback=None,
):
    core_prompt   = character_core_prompt or identity_prompt
    anchor_prompt = identity_prompt + ", calm expression"
    if anchor_seeds is None:
        anchor_seeds = [anchor_seed, anchor_seed+111, anchor_seed+222, anchor_seed+333]
    if bootstrap_prompts is None:
        bootstrap_prompts = [
            core_prompt + ", surprised expression, raised eyebrows, open mouth",
            core_prompt + ", bright smile, cheerful expression",
            core_prompt + ", serious expression, looking slightly left",
            core_prompt + ", laughing expression, happy eyes",
            core_prompt + ", calm neutral expression, soft lighting",
            core_prompt + ", looking up slightly, thoughtful expression",
            core_prompt + ", slight smirk, confident expression",
            core_prompt + ", tired expression, soft eyes",
            core_prompt + ", gentle smile, warm expression",
            core_prompt + ", focused expression, looking directly at camera",
            core_prompt + ", pensive expression, looking slightly right",
            core_prompt + ", relaxed expression, slight head tilt",
            core_prompt + ", excited expression, wide smile",
            core_prompt + ", shy expression, soft downward gaze",
            core_prompt + ", determined expression, strong eye contact",
            core_prompt + ", curious expression, slight head tilt left",
            core_prompt + ", content expression, peaceful look",
            core_prompt + ", playful expression, slight wink",
            core_prompt + ", stoic expression, neutral face",
            core_prompt + ", warm expression, soft eyes, slight smile",
        ]
    if bootstrap_seeds is None:
        bootstrap_seeds = [1111,2222,3333,4444,5555,6666,7777,8888,
                           9111,9222,9333,9444,9555,9666,9777,9888,
                           7111,7222,7333,7444]

    if progress_callback is not None:
        progress_callback(0.08, desc="generating anchors")
    print("="*50+"\nSTEP 1: Multi-anchor ensemble\n"+"="*50)
    ensemble_embedding, anchor_images = build_ensemble_embedding(
        anchor_prompt, anchor_seeds, anchor_steps, anchor_guidance_scale, anchor_negative_prompt)
    try:
        pose_image = pose_detector(anchor_images[0])
    except Exception as e:
        pose_image = None
        print(f"  Pose extraction failed ({e}); falling back to pose-free generation.")
    if pose_image is None:
        print("  Pose detector returned no pose image; falling back to pose-free generation.")

    if progress_callback is not None:
        progress_callback(0.20, desc="extracting pose and identity")
    print("="*50+"\nSTEP 2: Initial projector training\n"+"="*50)
    # compute ArcFace embedding of primary anchor for face-aware projector training
    anchor_face_emb = encode_arcface(anchor_images[0])
    if anchor_face_emb is not None:
        print(f"  ArcFace anchor embedding ready (mode: {_arcface_mode})")
    else:
        print("  No face detected in anchor, training without ArcFace loss")
    projector, _ = train_projector(ensemble_embedding, core_prompt,
                                   num_tokens=num_identity_tokens, train_steps=train_steps,
                                   lr=projector_lr, anchor_face_emb=anchor_face_emb)

    if progress_callback is not None:
        progress_callback(0.32, desc="training identity projector")
    print("="*50+"\nSTEP 3: Bootstrap refinement\n"+"="*50)
    projector, projector_loss, refined_identity, candidates = bootstrap_refine(
        base_identity=ensemble_embedding, projector=projector,
        character_core_prompt=core_prompt,
        bootstrap_prompts=bootstrap_prompts, bootstrap_seeds=bootstrap_seeds,
        pose_image=pose_image, bootstrap_top_k=bootstrap_top_k,
        bootstrap_identity_scale=bootstrap_identity_scale,
        bootstrap_generation_steps=bootstrap_generation_steps,
        bootstrap_guidance_scale=bootstrap_guidance_scale,
        bootstrap_controlnet_scale=bootstrap_controlnet_scale,
        refine_anchor_weight=refine_anchor_weight,
        num_tokens=num_identity_tokens, train_steps=train_steps,
        projector_lr=projector_lr,
        anchor_face_emb=anchor_face_emb, anchor_image=anchor_images[0])

    if progress_callback is not None:
        progress_callback(0.52, desc="bootstrap refinement")
    if correction_rounds > 0:
        print("="*50+"\nSTEP 4: Drift correction\n"+"="*50)
        projector, refined_identity, drift_history = drift_correction(
            projector, refined_identity, core_prompt, pose_image,
            correction_rounds=correction_rounds)
    else:
        drift_history = []

    if progress_callback is not None:
        progress_callback(0.60, desc="finalizing identity tokens")

    with torch.inference_mode():
        identity_tokens = projector(refined_identity.float()).to(dtype)

    print(f"\nCharacter ready. Loss: {projector_loss:.6f} | Drift: {[round(d,4) for d in drift_history]}")
    return CharacterProfile(
        character_core_prompt=core_prompt, anchor_prompt=anchor_prompt,
        anchor_seed=int(anchor_seeds[0]), anchor_image=anchor_images[0],
        anchor_images=anchor_images, pose_image=pose_image,
        base_identity_embedding=ensemble_embedding,
        refined_identity_embedding=refined_identity,
        identity_tokens=identity_tokens, projector_loss=projector_loss,
        bootstrap_candidates=candidates, drift_history=drift_history)

# ── load_checkpoint ──────────────────────────────────────────
def load_checkpoint(adapters, save_dir):
    adapters.load_state_dict(torch.load(f"{save_dir}/adapter_weights.pt", map_location=device))
    tokens   = torch.load(f"{save_dir}/identity_tokens.pt",    map_location=device)
    refined  = torch.load(f"{save_dir}/refined_embedding.pt",  map_location=device)
    base_emb = torch.load(f"{save_dir}/base_embedding.pt",     map_location=device)
    pose_path = f"{save_dir}/pose_image.png"
    pose_img  = Image.open(pose_path) if os.path.exists(pose_path) else None
    anchor_img= Image.open(f"{save_dir}/anchor_image.png")
    with open(f"{save_dir}/metadata.json") as f: meta = json.load(f)
    profile = CharacterProfile(
        character_core_prompt=meta["character_core_prompt"],
        anchor_prompt=meta["anchor_prompt"], anchor_seed=meta["anchor_seed"],
        anchor_image=anchor_img, anchor_images=[anchor_img], pose_image=pose_img,
        base_identity_embedding=base_emb, refined_identity_embedding=refined,
        identity_tokens=tokens, projector_loss=meta["projector_loss"],
        drift_history=meta.get("drift_history", []))
    print(f"Loaded: {meta['character_core_prompt'][:60]}")
    return profile

# ── RUN — 5 diverse characters (opt-in) ──────────────────────
TEST_CHARACTERS = [
    {"name": "woman_brunette", "seed": 1234,
     "prompt": "young woman, brown eyes, dark brown hair, natural makeup, soft facial features, upper body portrait, neutral background, photorealistic, high quality, sharp focus"},
    {"name": "elderly_man",    "seed": 4321,
     "prompt": "elderly man, grey beard, deep wrinkles, kind blue eyes, weathered face, upper body portrait, neutral background, photorealistic, high quality"},
    {"name": "anime_girl",     "seed": 5678,
     "prompt": "anime girl, long silver hair, violet eyes, soft bangs, small mole under left eye, detailed face, upper body portrait, white background, high quality, sharp lineart"},
    {"name": "young_man",      "seed": 2468,
     "prompt": "young man, short black hair, sharp jawline, brown eyes, light stubble, upper body portrait, neutral background, photorealistic, high quality"},
    {"name": "woman_redhead",  "seed": 9876,
     "prompt": "young woman, curly red hair, green eyes, freckles, warm smile, upper body portrait, neutral background, photorealistic, high quality"},
]

EVAL_EXPRESSIONS = [
    ", bright smile, cheerful expression",
    ", surprised expression, wide eyes, open mouth",
    ", serious expression, looking slightly left",
    ", calm neutral expression",
]

# ── Speed config — set FAST_MODE=True to cut runtime ~50% ────
# FAST_MODE skips cross-validation, pose-free, and reduces eval images
FAST_MODE = True
AUTO_RUN_FULL_BENCHMARK = False

def run_full_benchmark():
    all_scores = {}

    for char in TEST_CHARACTERS:
        print(f"\n{'='*60}\nCharacter: {char['name']}\n{'='*60}")

        if 'hooks' in dir() and hooks:
            for h in hooks: h.remove()

        profile = create_character(
            identity_prompt=char["prompt"],
            anchor_seed=char["seed"],
            anchor_steps=30, anchor_guidance_scale=8.0,
            num_identity_tokens=4, train_steps=250,
            bootstrap_top_k=2, bootstrap_identity_scale=1.0,
            bootstrap_generation_steps=20,
            refine_anchor_weight=0.55,
            correction_rounds=2,
        )
        display(profile.anchor_image)

        adapters = attach_identity_adapters(pipe.unet, identity_dim=768, scale=0.5)
        hooks    = register_adapter_hooks(pipe.unet)
        train_adapter_on_bootstrap(
            unet=pipe.unet, vae=pipe.vae,
            text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer,
            adapters=adapters, identity_tokens=profile.identity_tokens,
            bootstrap_images=[c.image for c in profile.bootstrap_candidates],
            bootstrap_prompts=[c.prompt for c in profile.bootstrap_candidates],
            base_identity_embedding=profile.base_identity_embedding,
            train_steps=200, lr=1e-5, clip_loss_weight=0.1, arcface_loss_weight=0.1,
        )

        print("  ArcFace fine-tune (60 steps)...")
        train_adapter_on_bootstrap(
            unet=pipe.unet, vae=pipe.vae,
            text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer,
            adapters=adapters, identity_tokens=profile.identity_tokens,
            bootstrap_images=[c.image for c in profile.bootstrap_candidates],
            bootstrap_prompts=[c.prompt for c in profile.bootstrap_candidates],
            base_identity_embedding=profile.base_identity_embedding,
            train_steps=60, lr=2e-6, clip_loss_weight=0.1, arcface_loss_weight=0.3,
        )

        print("  Post-adapter identity refinement...")
        _anchor_face_emb = encode_arcface(profile.anchor_image)
        probe_seeds = [5555, 6666, 7777, 8888]
        probe_images, probe_scores = [], []
        for ps in probe_seeds:
            pimg = generate_with_adapter(
                profile.identity_tokens, char["prompt"], profile.pose_image,
                pipe.unet, pipe, seed=ps,
                identity_scale=adaptive_identity_scale(char["prompt"]),
                generation_steps=30, guidance_scale=7.5, controlnet_scale=0.55)
            pid, _, _ = candidate_score(pimg, profile.base_identity_embedding, char["prompt"])
            probe_images.append(pimg); probe_scores.append(pid)

        if _anchor_face_emb is not None:
            arc_probe_scores = []
            for idx, pimg in enumerate(probe_images):
                arc_emb = encode_arcface(pimg)
                arc_probe_scores.append(
                    float(F.cosine_similarity(arc_emb, _anchor_face_emb).mean())
                    if arc_emb is not None and arc_emb.shape[-1] == _anchor_face_emb.shape[-1]
                    else probe_scores[idx])
            top2 = sorted(range(len(probe_images)), key=lambda i: arc_probe_scores[i], reverse=True)[:2]
            print(f"  Post-refinement top ArcFace scores: {[round(arc_probe_scores[i],4) for i in top2]}")
        else:
            top2 = sorted(range(len(probe_images)), key=lambda i: probe_scores[i], reverse=True)[:2]

        top_embs = torch.stack([encode_clip_image(probe_images[i]) for i in top2], dim=0)
        refined_emb = normalize(
            0.6 * profile.refined_identity_embedding.float() +
            0.4 * normalize(top_embs.mean(dim=0)))

        proj_refine = MultiTokenIdentityProjector(in_dim=CLIP_DIM, num_tokens=8, hidden_dim=1024).to(device)
        proj_refine.train()
        opt_r = torch.optim.Adam(proj_refine.parameters(), lr=5e-5)
        target_r = masked_mean_sd(char["prompt"]).detach()
        for _ in range(50):
            opt_r.zero_grad(set_to_none=True)
            t = proj_refine(refined_emb.float())
            s = normalize(t.mean(dim=1))
            loss_r = (1 - F.cosine_similarity(s, target_r)).mean()
            loss_r += 0.05 * token_diversity_loss(t)
            loss_r.backward(); opt_r.step()
        proj_refine.eval()
        with torch.inference_mode():
            profile.identity_tokens = proj_refine(refined_emb.float()).to(dtype)
            profile.refined_identity_embedding = refined_emb

        print(f"\nGenerating variations for {char['name']}...")
        candidate_pool = [5555,6666,7777,8888,9999,1010,2020,3030]
        best_seeds = select_best_seeds(
            profile.identity_tokens, char["prompt"] + ", surprised expression",
            pipe.unet, pipe, profile.base_identity_embedding,
            candidate_pool, pose_image=profile.pose_image, num_select=4)

        gen_images = []
        for seed in best_seeds:
            img = generate_with_adapter(
                profile.identity_tokens, char["prompt"] + ", surprised expression",
                profile.pose_image, pipe.unet, pipe,
                seed=seed, identity_scale=adaptive_identity_scale(char["prompt"] + ", surprised expression"),
                generation_steps=30, guidance_scale=7.5, controlnet_scale=0.55)
            id_sim, _, score = candidate_score(img, profile.base_identity_embedding, char["prompt"])
            print(f"  seed {seed}: identity={id_sim:.4f} | score={score:.4f}")
            gen_images.append(img)
        display(show_grid(gen_images, cols=2))

        eval_prompts = [char["prompt"] + e for e in EVAL_EXPRESSIONS]
        scores = []
        eval_images = []
        for i, prompt in enumerate(eval_prompts):
            seed = 5001 + i
            img = generate_with_adapter(
                profile.identity_tokens, prompt, profile.pose_image,
                pipe.unet, pipe, seed=seed,
                identity_scale=adaptive_identity_scale(prompt),
                generation_steps=30, guidance_scale=7.5, controlnet_scale=0.55)
            id_sim, prompt_sim, score = candidate_score(img, profile.base_identity_embedding, prompt)
            print(f"  expr {i+1}: identity={id_sim:.4f} | prompt={prompt_sim:.4f} | score={score:.4f}")
            scores.append(id_sim)
            eval_images.append(img)
        display(show_grid(eval_images, cols=2))

        avg_id = sum(scores) / len(scores)
        all_scores[char["name"]] = avg_id
        print(f"  avg identity: {avg_id:.4f}")

        cv_seeds = [5001,5002,5003,5004,5005,5006,5007,5008]
        print(f"  Pairwise consistency ({len(cv_seeds)} seeds)...")
        cv_embs = []
        for cvs in cv_seeds:
            cvi = generate_with_adapter(
                profile.identity_tokens, char["prompt"], profile.pose_image,
                pipe.unet, pipe, seed=cvs,
                identity_scale=adaptive_identity_scale(char["prompt"]),
                generation_steps=20, guidance_scale=7.5, controlnet_scale=0.55)
            cv_embs.append(encode_clip_image(cvi))
        pw_sims = [float(F.cosine_similarity(cv_embs[i], cv_embs[j]).mean())
                   for i in range(len(cv_embs)) for j in range(i+1, len(cv_embs))]
        pw_mean = sum(pw_sims)/len(pw_sims)
        pw_std  = float(torch.tensor(pw_sims).std())
        print(f"  Pairwise: mean={pw_mean:.4f} std={pw_std:.4f}")
        all_scores[char["name"] + "_pw_mean"] = pw_mean
        all_scores[char["name"] + "_pw_std"]  = pw_std

        print("  Pose-free generation...")
        if 'posefree_pipe' not in dir():
            posefree_pipe = load_posefree_pipe()
        pf_seeds = best_seeds[:4]
        posefree_images = []
        pf_id_scores = []
        for seed in pf_seeds:
            img, id_sim = generate_posefree(
                profile.identity_tokens, char["prompt"],
                pipe.unet, posefree_pipe, profile.base_identity_embedding,
                seed=seed, identity_scale=1.0, generation_steps=30)
            print(f"  posefree seed {seed}: identity={id_sim:.4f}")
            posefree_images.append(img)
            pf_id_scores.append(id_sim)
        pf_avg = sum(pf_id_scores) / len(pf_id_scores) if pf_id_scores else 0.0
        all_scores[char["name"] + "_posefree"] = pf_avg
        print(f"  Pose-free avg identity: {pf_avg:.4f}")
        display(show_grid(posefree_images, cols=2))

        save_checkpoint(profile, adapters, f"/content/checkpoints/{char['name']}")
        save_adapter_to_library(adapters, char["name"])
        export_character(char["name"], "/content/checkpoints")

    print(f"\n{'='*60}\nFINAL RESULTS\n{'='*60}")
    char_names = [c["name"] for c in TEST_CHARACTERS]
    identity_scores = [all_scores[n] for n in char_names]
    print(f"\n{'Character':<20} {'AvgID':>8} {'Pairwise':>12} {'PoseFree':>10}")
    print("-"*55)
    for name in char_names:
        avg_id = all_scores.get(name, 0)
        pw_m   = all_scores.get(name+"_pw_mean", 0)
        pw_s   = all_scores.get(name+"_pw_std",  0)
        pf     = all_scores.get(name+"_posefree", 0)
        print(f"{name:<20} {avg_id:>8.4f} {pw_m:>6.4f}±{pw_s:<5.4f} {pf:>10.4f}")
    overall    = sum(identity_scores)/len(identity_scores)
    overall_pf = sum(all_scores.get(n+"_posefree",0) for n in char_names)/len(char_names)
    print(f"\n{'MEAN':<20} {overall:>8.4f} {'':>12} {overall_pf:>10.4f}")
    print(f"\nOverall avg identity (with ControlNet): {overall:.4f}")
    print(f"Overall avg identity (pose-free):       {overall_pf:.4f}")

    print(f"\n{'='*60}\nMULTI-CHARACTER CONSISTENCY\n{'='*60}")
    shared_prompt_suffix = ", neutral expression, studio lighting, upper body portrait"
    mc_results = {}
    for char in TEST_CHARACTERS:
        try:
            adp_mc = attach_identity_adapters(pipe.unet, identity_dim=768, scale=0.5)
            hks_mc = register_adapter_hooks(pipe.unet)
            prof_mc = load_checkpoint(adp_mc, f"/content/checkpoints/{char['name']}")
            imgs_mc = []
            for seed in [1001, 2002, 3003]:
                img = generate_with_adapter(
                    prof_mc.identity_tokens,
                    char["prompt"] + shared_prompt_suffix,
                    prof_mc.pose_image, pipe.unet, pipe,
                    seed=seed, identity_scale=adaptive_identity_scale(char["prompt"]),
                    generation_steps=25, guidance_scale=7.5)
                imgs_mc.append(img)
            embs_mc = [encode_clip_image(i) for i in imgs_mc]
            pw = [float(F.cosine_similarity(embs_mc[i], embs_mc[j]).mean())
                  for i in range(len(embs_mc)) for j in range(i+1, len(embs_mc))]
            mc_results[char["name"]] = sum(pw)/len(pw)
            print(f"  {char['name']:20s}  consistency={mc_results[char['name']]:.4f}")
            display(show_grid(imgs_mc, cols=3))
            for h in hks_mc: h.remove()
        except Exception as e:
            print(f"  {char['name']}: skipped ({e})")

    build_meta_init(adapters, "/content/adapter_library")
    print("Meta-init ready. Future characters will warm-start from this prior.")

    return all_scores

if AUTO_RUN_FULL_BENCHMARK:
    run_full_benchmark()
else:
    print("SynID definitions loaded. Call init_synid_backend() to load models, or set AUTO_RUN_FULL_BENCHMARK=True to run the full benchmark.")
