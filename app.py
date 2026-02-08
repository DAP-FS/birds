# app.py
# BirdLookalike — 1-to-1 best bird match + confidence + image
# Works on Mac even if mediapipe.solutions is missing (uses MediaPipe Tasks API).

import os
import json
import urllib.request
import numpy as np
import streamlit as st
from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPModel

# -------------------------
# Files created by build_index.py
# -------------------------
EMB_PATH = "bird_embeddings.npy"
META_PATH = "bird_metadata.json"

# -------------------------
# MediaPipe Tasks Face Detector model (auto download)
# -------------------------
MODEL_DIR = "models"
FACE_MODEL_PATH = os.path.join(MODEL_DIR, "blaze_face_short_range.tflite")
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)

st.set_page_config(page_title="BirdLookalike (1-to-1)", layout="wide")


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
    if not (os.path.exists(EMB_PATH) and os.path.exists(META_PATH)):
        raise FileNotFoundError(
            f"Missing {EMB_PATH} or {META_PATH}. Run: python3 build_index.py"
        )

    embs = np.load(EMB_PATH).astype(np.float32)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Normalize once (cosine similarity)
    embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    return embs, meta


# -------------------------
# MediaPipe model download
# -------------------------
def ensure_face_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(FACE_MODEL_PATH):
        return True
    try:
        urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)
        return True
    except Exception:
        return False


# -------------------------
# Face crop using MediaPipe Tasks (no mp.solutions)
# -------------------------
def face_crop_pil_tasks(img: Image.Image, expand=1.35):
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
            base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            min_detection_confidence=0.5,
        )

        np_img = np.array(img.convert("RGB"))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_img)

        with FaceDetector.create_from_options(options) as detector:
            result = detector.detect(mp_image)

        if not result.detections:
            return img

        # choose best detection by score
        def score(det):
            return det.categories[0].score if det.categories else 0.0

        det = max(result.detections, key=score)
        bbox = det.bounding_box  # pixel coords: origin_x, origin_y, width, height

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
# Robust CLIP image embedding extractor -> ALWAYS returns (B, 512)
# -------------------------
@torch.no_grad()
def get_clip_image_embeds(model: CLIPModel, pixel_values: torch.Tensor):
    """
    Returns projected CLIP image embeddings (B, 512).
    Works even if get_image_features returns a non-tensor on some installs.
    """
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
            pooled = vision_out[1]  # fallback
        embeds = model.visual_projection(pooled)
        return embeds

    # Last resort: forward + check common fields
    out = model(pixel_values=pixel_values)
    for name in ["image_embeds", "pooler_output", "last_hidden_state"]:
        if hasattr(out, name):
            t = getattr(out, name)
            if torch.is_tensor(t):
                if name == "last_hidden_state":
                    t = t[:, 0, :]  # CLS token
                return t

    raise TypeError(f"Could not extract tensor embeddings from model output: {type(out)}")


@torch.no_grad()
def embed_one(img: Image.Image, model, processor, device):
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


def best_match_with_confidence(embs, meta, q, topN=50, temperature=20.0):
    """
    1-to-1 best bird image match.
    Returns: best_index, best_similarity, best_confidence (0..1), top_sorted_indices, sims
    """
    sims = embs @ q  # cosine similarity
    idx_sorted = np.argsort(-sims)

    best_i = int(idx_sorted[0])
    best_sim = float(sims[best_i])

    topN = min(topN, len(sims))
    top_idx = idx_sorted[:topN]
    top_sims = sims[top_idx]

    probs = softmax(top_sims * temperature)
    best_conf = float(probs[0])  # first is best

    return best_i, best_sim, best_conf, idx_sorted, sims


# -------------------------
# UI
# -------------------------
st.title("🪶 BirdLookalike — One best bird match + confidence")

model, processor, device = load_clip()
embs, meta = load_index()

col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    up = st.file_uploader("Upload a face photo", type=["jpg", "jpeg", "png", "webp"])
    use_crop = st.checkbox("Auto-crop face (recommended)", value=True)

    st.markdown("### Confidence settings")
    topN = st.slider("Top-N competitors (for confidence)", 10, 200, min(50, len(meta)))
    temperature = st.slider("Temperature", 5.0, 40.0, 20.0, 1.0)

    st.caption(f"Device: {device} | Indexed bird images: {len(meta)}")
    st.caption(f"Model: {model.__class__.__name__}")

if not up:
    st.info("Upload a face photo to see the closest bird.")
    st.stop()

img = Image.open(up).convert("RGB")
img_in = face_crop_pil_tasks(img) if use_crop else img

q = embed_one(img_in, model, processor, device)

best_i, best_sim, best_conf, idx_sorted, sims = best_match_with_confidence(
    embs, meta, q, topN=topN, temperature=temperature
)

best = meta[best_i]
bird_img = Image.open(best["path"]).convert("RGB")

with col1:
    st.subheader("Input (face)")
    st.image(img_in, use_container_width=True)

with col2:
    st.subheader("✅ Best match (1-to-1)")
    st.markdown(f"## **{best['species']}**")
    st.metric("Cosine similarity", f"{best_sim:.3f}")
    st.metric("Confidence (softmax over Top-N)", f"{best_conf * 100:.1f}%")
    st.image(bird_img, caption=f"{best['species']} | {os.path.basename(best['path'])}", use_container_width=True)

    st.markdown("---")
    st.subheader("Runner-ups (for context)")
    runner_k = min(5, len(meta) - 1)
    for j in idx_sorted[1:1 + runner_k]:
        m = meta[int(j)]
        st.write(f"- {m['species']}  (sim={float(sims[int(j)]):.3f})")
