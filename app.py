# app.py
# BirdLookalike — Mobile-friendly (Camera + Upload) + 1-to-1 best bird match + confidence + image
# - Uses Streamlit st.camera_input for mobile camera capture
# - Uses MediaPipe Tasks FaceDetector (no mp.solutions dependency)
# - Robust CLIP embedding extraction (avoids BaseModelOutputWithPooling .norm issues)
#
# Required repo files for Streamlit Cloud if you want to DISPLAY bird images:
#   bird_embeddings.npy
#   bird_metadata.json
#   birds/...(images)   OR  birds_preview/...(previews) + modify display code accordingly
#
# pip deps (requirements.txt):
#   streamlit numpy pillow torch transformers mediapipe

import os
import json
import urllib.request
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPModel

# -------------------------
# Repo assets
# -------------------------
EMB_FILE = "bird_embeddings.npy"
META_FILE = "bird_metadata.json"

# -------------------------
# MediaPipe Tasks Face Detector model (auto download)
# -------------------------
MODEL_DIR = "models"
FACE_MODEL_PATH = os.path.join(MODEL_DIR, "blaze_face_short_range.tflite")
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)

st.set_page_config(page_title="BirdLookalike (Mobile Camera)", layout="wide")


# -------------------------
# Path resolver (works on Streamlit Cloud + local)
# -------------------------
def find_repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for p in [here] + list(here.parents):
        if (p / META_FILE).exists() or (p / EMB_FILE).exists():
            return p
    return here

REPO_ROOT = find_repo_root()

def resolve_asset(path_str: str) -> Path:
    """
    Resolve:
    - relative paths like 'birds/...'
    - accidental absolute local paths like '/Users/.../birds/...'
    - paths relative to repo root
    """
    s = str(path_str).replace("\\", "/")

    # If absolute path contains /birds/, keep only the tail from birds/
    if s.startswith("/") and "/birds/" in s:
        s = "birds/" + s.split("/birds/", 1)[1]

    # First try relative to repo root
    cand = REPO_ROOT / s
    if cand.exists():
        return cand

    # Fix duplicated prefix like birds/birds/...
    if s.startswith("birds/"):
        cand2 = REPO_ROOT / s[len("birds/"):]
        if cand2.exists():
            return cand2

    # As a last resort, try relative to this script directory
    cand3 = Path(__file__).resolve().parent / s
    return cand3


# -------------------------
# Device helper (CPU / MPS / CUDA)
# -------------------------
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# -------------------------
# Load CLIP (cached)
# -------------------------
@st.cache_resource
def load_clip():
    device = get_device()
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor, device


# -------------------------
# Load bird index (cached)
# -------------------------
@st.cache_data
def load_index():
    emb_path = resolve_asset(EMB_FILE)
    meta_path = resolve_asset(META_FILE)

    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {EMB_FILE} or {META_FILE} in repo. "
            f"Found: {emb_path.exists()} / {meta_path.exists()}"
        )

    embs = np.load(str(emb_path)).astype(np.float32)
    with open(str(meta_path), "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Normalize once for cosine similarity
    embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)

    return embs, meta


# -------------------------
# MediaPipe face model download
# -------------------------
def ensure_face_model() -> bool:
    os.makedirs(str(REPO_ROOT / MODEL_DIR), exist_ok=True)
    face_model = REPO_ROOT / FACE_MODEL_PATH

    if face_model.exists():
        return True
    try:
        urllib.request.urlretrieve(FACE_MODEL_URL, str(face_model))
        return True
    except Exception:
        return False


# -------------------------
# Face crop using MediaPipe Tasks (no mp.solutions)
# -------------------------
def face_crop_pil_tasks(img: Image.Image, expand=1.35) -> Image.Image:
    """
    Returns cropped face image if detection works.
    Otherwise returns original img.
    """
    try:
        import mediapipe as mp

        if not ensure_face_model():
            return img

        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(REPO_ROOT / FACE_MODEL_PATH)),
            running_mode=RunningMode.IMAGE,
            min_detection_confidence=0.5,
        )

        np_img = np.array(img.convert("RGB"))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_img)

        with FaceDetector.create_from_options(options) as detector:
            result = detector.detect(mp_image)

        if not result.detections:
            return img

        def score(det):
            return det.categories[0].score if det.categories else 0.0

        det = max(result.detections, key=score)
        bbox = det.bounding_box  # origin_x, origin_y, width, height (pixels)

        x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
        H, W = np_img.shape[:2]

        cx, cy = x + w / 2.0, y + h / 2.0
        size = max(w, h) * expand

        x1 = int(max(0, cx - size / 2.0))
        y1 = int(max(0, cy - size / 2.0))
        x2 = int(min(W, cx + size / 2.0))
        y2 = int(min(H, cy + size / 2.0))

        if x2 <= x1 or y2 <= y1:
            return img

        crop = np_img[y1:y2, x1:x2]
        return Image.fromarray(crop)

    except Exception:
        return img


# -------------------------
# Robust CLIP embedding extraction -> ALWAYS returns (B, 512) Tensor
# -------------------------
@torch.no_grad()
def get_clip_image_embeds(model: CLIPModel, pixel_values: torch.Tensor) -> torch.Tensor:
    # Preferred path
    if hasattr(model, "get_image_features"):
        out = model.get_image_features(pixel_values=pixel_values)
        if torch.is_tensor(out):
            return out

    # Robust path: vision_model -> visual_projection
    if hasattr(model, "vision_model") and hasattr(model, "visual_projection"):
        vision_out = model.vision_model(pixel_values=pixel_values)
        pooled = getattr(vision_out, "pooler_output", None)
        if pooled is None:
            pooled = vision_out[1]
        embeds = model.visual_projection(pooled)
        return embeds

    # Last resort: forward + common fields
    out = model(pixel_values=pixel_values)
    for name in ["image_embeds", "pooler_output", "last_hidden_state"]:
        if hasattr(out, name):
            t = getattr(out, name)
            if torch.is_tensor(t):
                if name == "last_hidden_state":
                    t = t[:, 0, :]
                return t

    raise TypeError(f"Could not extract tensor embeddings from model output: {type(out)}")


@torch.no_grad()
def embed_one(img: Image.Image, model, processor, device) -> np.ndarray:
    inputs = processor(images=[img.convert("RGB")], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    feat = get_clip_image_embeds(model, pixel_values)
    feat = feat / feat.norm(dim=-1, keepdim=True)  # normalize
    return feat.squeeze(0).detach().cpu().numpy().astype(np.float32)


# -------------------------
# Confidence scoring (softmax over Top-N similarities)
# -------------------------
def softmax(x):
    x = np.array(x, dtype=np.float32)
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


def best_match_with_confidence(embs, q, topN=50, temperature=20.0):
    sims = embs @ q
    idx_sorted = np.argsort(-sims)

    best_i = int(idx_sorted[0])
    best_sim = float(sims[best_i])

    topN = min(int(topN), len(sims))
    top_sims = sims[idx_sorted[:topN]]
    probs = softmax(top_sims * float(temperature))
    best_conf = float(probs[0])

    return best_i, best_sim, best_conf, idx_sorted, sims


# -------------------------
# UI
# -------------------------
st.title("🪶 BirdLookalike — Mobile Camera + Best Bird + Confidence")

model, processor, device = load_clip()
embs, meta = load_index()

col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    source = st.radio("Input source", ["📷 Camera", "🖼️ Upload"], horizontal=True)
    use_crop = st.checkbox("Auto-crop face (recommended)", value=True)

    st.markdown("### Confidence settings")
    topN = st.slider("Top-N competitors (for confidence)", 10, 200, min(50, len(meta)))
    temperature = st.slider("Temperature", 5.0, 40.0, 20.0, 1.0)

    if source == "📷 Camera":
        up = st.camera_input("Take a photo")
    else:
        up = st.file_uploader("Upload a face photo", type=["jpg", "jpeg", "png", "webp"])

    st.caption(f"Device: {device} | Indexed bird images: {len(meta)}")

if not up:
    st.info("Capture or upload a photo to continue.")
    st.stop()

# Works for both camera_input and file_uploader
img = Image.open(BytesIO(up.getvalue())).convert("RGB")
img_in = face_crop_pil_tasks(img) if use_crop else img

q = embed_one(img_in, model, processor, device)

best_i, best_sim, best_conf, idx_sorted, sims = best_match_with_confidence(
    embs, q, topN=topN, temperature=temperature
)

best = meta[best_i]

with col1:
    st.subheader("Input (face)")
    st.image(img_in, use_container_width=True)

with col2:
    st.subheader("✅ Best match (1-to-1)")
    st.markdown(f"## **{best['species']}**")
    st.metric("Cosine similarity", f"{best_sim:.3f}")
    st.metric("Confidence (softmax over Top-N)", f"{best_conf * 100:.1f}%")

    # Display bird image if present in repo
    bird_path = resolve_asset(best["path"])
    if bird_path.exists():
        bird_img = Image.open(str(bird_path)).convert("RGB")
        st.image(bird_img, caption=f"{best['species']} | {bird_path.name}", use_container_width=True)
    else:
        st.warning(
            "Bird image file not found in this deployment. "
            "To show images on Streamlit Cloud, include the birds/ folder in the GitHub repo "
            "or switch to a birds_preview/ approach."
        )

    st.markdown("---")
    st.subheader("Runner-ups (context)")
    runner_k = min(5, len(meta) - 1)
    for j in idx_sorted[1:1 + runner_k]:
        m = meta[int(j)]
        st.write(f"- {m['species']}  (sim={float(sims[int(j)]):.3f})")
