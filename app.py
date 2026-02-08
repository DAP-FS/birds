# app.py
# BirdLookalike — Mobile-friendly (Camera + Upload) + robust face crop (FaceLandmarker mesh) + DEBUG
# + 1-to-1 best bird match + confidence + (optional) bird image if present in repo.
#
# Mobile UI improvements:
# - layout="centered"
# - downsized display images (keeps screen compact)
# - bird image shown inside expander
#
# IMPORTANT (Streamlit Cloud):
# - MediaPipe often fails on Python 3.13. Prefer Python 3.12 in Streamlit Cloud settings.
#
# Repo must contain (for matching):
#   bird_embeddings.npy
#   bird_metadata.json
# Optional (for displaying bird images):
#   birds/... images referenced by bird_metadata.json (or switch to birds_preview approach)

import os
import json
import urllib.request
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

import torch
from transformers import CLIPProcessor, CLIPModel

# -------------------------
# Repo assets
# -------------------------
EMB_FILE = "bird_embeddings.npy"
META_FILE = "bird_metadata.json"

# -------------------------
# MediaPipe FaceLandmarker model (download to cache, not repo)
# -------------------------
CACHE_DIR = Path.home() / ".cache" / "birdlookalike"
MP_MODELS = CACHE_DIR / "models"
MP_MODELS.mkdir(parents=True, exist_ok=True)

FACE_LANDMARKER_PATH = MP_MODELS / "face_landmarker.task"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

# -------------------------
# Mobile-friendly display settings
# -------------------------
DISPLAY_MAX_W = 360  # good on most phones (adjust if you like)


def resize_for_display(img: Image.Image, max_w: int = DISPLAY_MAX_W) -> Image.Image:
    w, h = img.size
    if w <= max_w:
        return img
    new_h = int(h * (max_w / w))
    return img.resize((max_w, new_h))


st.set_page_config(page_title="BirdLookalike (Mobile)", layout="centered")


# -------------------------
# Path helpers (Cloud-safe)
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
    - absolute local paths like '/Users/.../birds/...'
    """
    s = str(path_str).replace("\\", "/")

    # Convert absolute local path to repo-relative birds/...
    if s.startswith("/") and "/birds/" in s:
        s = "birds/" + s.split("/birds/", 1)[1]

    cand = REPO_ROOT / s
    if cand.exists():
        return cand

    # fix accidental duplication: birds/birds/...
    if s.startswith("birds/"):
        cand2 = REPO_ROOT / s[len("birds/") :]
        if cand2.exists():
            return cand2

    return (Path(__file__).resolve().parent / s)


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

    embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    return embs, meta


# -------------------------
# MediaPipe FaceLandmarker: ensure model + create landmarker
# -------------------------
def ensure_face_landmarker_model() -> bool:
    if FACE_LANDMARKER_PATH.exists():
        return True
    try:
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, str(FACE_LANDMARKER_PATH))
        return True
    except Exception:
        return False


def create_face_landmarker():
    """
    Creates a FaceLandmarker instance.
    Returns None if mediapipe or model is unavailable.
    """
    try:
        import mediapipe as mp
    except Exception:
        return None

    if not ensure_face_landmarker_model():
        return None

    try:
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(FACE_LANDMARKER_PATH)),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return FaceLandmarker.create_from_options(options)
    except Exception:
        return None


def face_crop_pil_exact(img: Image.Image, margin: float = 0.05, debug: bool = False):
    """
    Face crop using FaceLandmarker mesh landmarks.
    Returns: (cropped_img, info_dict)

    If anything fails, returns original image with info['ok']=False and reason.
    """
    info = {"ok": False, "reason": "", "faces": 0, "model_path": str(FACE_LANDMARKER_PATH)}

    # Fix iPhone/Android EXIF rotation
    img = ImageOps.exif_transpose(img).convert("RGB")

    landmarker = create_face_landmarker()
    if landmarker is None:
        info["reason"] = "FaceLandmarker unavailable (mediapipe install / model download / runtime issue)"
        return img, info

    try:
        import mediapipe as mp

        # detect on downscaled image for speed, crop on original using normalized coords
        work = img.copy()
        work.thumbnail((720, 720))
        np_work = np.array(work)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_work)

        result = landmarker.detect(mp_image)

        if not getattr(result, "face_landmarks", None):
            info["reason"] = "No face landmarks detected"
            return img, info

        info["faces"] = len(result.face_landmarks)
        lm = result.face_landmarks[0]

        xs = np.array([p.x for p in lm], dtype=np.float32)
        ys = np.array([p.y for p in lm], dtype=np.float32)

        minx, maxx = float(xs.min()), float(xs.max())
        miny, maxy = float(ys.min()), float(ys.max())

        bw = max(1e-6, maxx - minx)
        bh = max(1e-6, maxy - miny)

        minx -= margin * bw
        maxx += margin * bw
        miny -= margin * bh
        maxy += margin * bh

        # clamp to [0,1]
        minx = max(0.0, min(1.0, minx))
        maxx = max(0.0, min(1.0, maxx))
        miny = max(0.0, min(1.0, miny))
        maxy = max(0.0, min(1.0, maxy))

        # crop on original image using normalized coords
        orig = np.array(img)
        H, W = orig.shape[:2]
        x1, x2 = int(minx * W), int(maxx * W)
        y1, y2 = int(miny * H), int(maxy * H)

        if debug:
            info["norm_box"] = [minx, miny, maxx, maxy]
            info["px_box"] = [x1, y1, x2, y2]
            info["orig_size"] = [W, H]

        if x2 <= x1 or y2 <= y1:
            info["reason"] = "Invalid crop box"
            return img, info

        crop = orig[y1:y2, x1:x2]
        info["ok"] = True
        info["reason"] = "cropped"
        return Image.fromarray(crop), info

    except Exception as e:
        info["reason"] = f"Landmarker runtime error: {e}"
        return img, info
    finally:
        try:
            landmarker.close()
        except Exception:
            pass


# -------------------------
# Robust CLIP embedding extraction -> ALWAYS returns (B, 512)
# -------------------------
@torch.no_grad()
def get_clip_image_embeds(model: CLIPModel, pixel_values: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "get_image_features"):
        out = model.get_image_features(pixel_values=pixel_values)
        if torch.is_tensor(out):
            return out

    if hasattr(model, "vision_model") and hasattr(model, "visual_projection"):
        vision_out = model.vision_model(pixel_values=pixel_values)
        pooled = getattr(vision_out, "pooler_output", None)
        if pooled is None:
            pooled = vision_out[1]
        return model.visual_projection(pooled)

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
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).detach().cpu().numpy().astype(np.float32)


# -------------------------
# Confidence scoring (softmax over Top-N)
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
st.title("🪶 BirdLookalike — Mobile Camera + Compact UI + Face Crop Debug")

model, processor, device = load_clip()
embs, meta = load_index()

# Mobile compact controls
source = st.radio("Input source", ["📷 Camera", "🖼️ Upload"], horizontal=True)

with st.expander("⚙️ Settings", expanded=False):
    use_crop = st.checkbox("Crop face (FaceLandmarker mesh)", value=True)
    crop_margin = st.slider(
        "Crop margin (tight → loose)",
        0.02, 0.15, 0.05, 0.01,
        help="Smaller = tighter. If forehead/hair is cut, increase slightly."
    )
    debug_crop = st.checkbox("Debug face crop", value=False)

    st.markdown("### Confidence")
    topN = st.slider("Top-N competitors", 10, 200, min(50, len(meta)))
    temperature = st.slider("Temperature", 5.0, 40.0, 20.0, 1.0)

st.caption(f"Device: {device} | Indexed bird images: {len(meta)}")

# Input widget
if source == "📷 Camera":
    up = st.camera_input("Take a photo")
else:
    up = st.file_uploader("Upload a face photo", type=["jpg", "jpeg", "png", "webp"])

if not up:
    st.info("Capture or upload a photo to continue.")
    st.stop()

# Works for both camera_input and file_uploader
img = Image.open(BytesIO(up.getvalue())).convert("RGB")

# Face crop
if use_crop:
    img_in, crop_info = face_crop_pil_exact(img, margin=crop_margin, debug=debug_crop)
    if debug_crop:
        st.json(crop_info)
else:
    img_in = ImageOps.exif_transpose(img).convert("RGB")

# Embed + match
q = embed_one(img_in, model, processor, device)
best_i, best_sim, best_conf, idx_sorted, sims = best_match_with_confidence(
    embs, q, topN=topN, temperature=temperature
)
best = meta[best_i]

# Display (compact)
st.subheader("Input (used for embedding)")
st.image(resize_for_display(img_in), width=DISPLAY_MAX_W)

st.subheader("✅ Best match")
st.markdown(f"### **{best['species']}**")
c1, c2 = st.columns(2)
c1.metric("Similarity", f"{best_sim:.3f}")
c2.metric("Confidence", f"{best_conf * 100:.1f}%")

# Bird image (optional) in expander to keep UI clean on mobile
bird_path = resolve_asset(best["path"])
if bird_path.exists():
    with st.expander("🖼️ Show matched bird image", expanded=False):
        bird_img = Image.open(str(bird_path)).convert("RGB")
        st.image(resize_for_display(bird_img), caption=f"{best['species']} | {bird_path.name}", width=DISPLAY_MAX_W)
else:
    st.warning(
        "Bird image file not found in this deployment. "
        "To show bird images on Streamlit Cloud, include the birds/ folder in the GitHub repo "
        "or use a birds_preview/ folder (one image per species)."
    )

with st.expander("Runner-ups", expanded=False):
    runner_k = min(5, len(meta) - 1)
    for j in idx_sorted[1 : 1 + runner_k]:
        m = meta[int(j)]
        st.write(f"- {m['species']}  (sim={float(sims[int(j)]):.3f})")
