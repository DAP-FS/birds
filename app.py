# app.py
# BirdLookalike — Mobile (camera/upload) + compact UI + robust face crop:
#   1) MediaPipe FaceLandmarker mesh (best)
#   2) OpenCV Haar cascade fallback (reliable on cloud)
# + CLIP embedding match + confidence + optional bird image display
#
# Streamlit Cloud note:
# - If MediaPipe fails, use Python 3.12 (recommended) and/or rely on OpenCV fallback.
#
# requirements.txt:
#   streamlit numpy pillow torch transformers mediapipe opencv-python-headless

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
# Cache directory (safe on Streamlit Cloud)
# -------------------------
CACHE_DIR = Path.home() / ".cache" / "birdlookalike"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# MediaPipe task model
MP_MODELS = CACHE_DIR / "models"
MP_MODELS.mkdir(parents=True, exist_ok=True)

FACE_LANDMARKER_PATH = MP_MODELS / "face_landmarker.task"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

# OpenCV Haar cascade (fallback)
OPENCV_MODELS = CACHE_DIR / "opencv"
OPENCV_MODELS.mkdir(parents=True, exist_ok=True)

HAAR_PATH = OPENCV_MODELS / "haarcascade_frontalface_default.xml"
HAAR_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

# Mobile-friendly display
DISPLAY_MAX_W = 360

st.set_page_config(page_title="BirdLookalike (Mobile)", layout="centered")


def resize_for_display(img: Image.Image, max_w: int = DISPLAY_MAX_W) -> Image.Image:
    w, h = img.size
    if w <= max_w:
        return img
    new_h = int(h * (max_w / w))
    return img.resize((max_w, new_h))


# -------------------------
# Path helpers
# -------------------------
def find_repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for p in [here] + list(here.parents):
        if (p / META_FILE).exists() or (p / EMB_FILE).exists():
            return p
    return here


REPO_ROOT = find_repo_root()


def resolve_asset(path_str: str) -> Path:
    s = str(path_str).replace("\\", "/")
    # Convert absolute local path to repo-relative birds/...
    if s.startswith("/") and "/birds/" in s:
        s = "birds/" + s.split("/birds/", 1)[1]
    cand = REPO_ROOT / s
    if cand.exists():
        return cand
    if s.startswith("birds/"):
        cand2 = REPO_ROOT / s[len("birds/") :]
        if cand2.exists():
            return cand2
    return Path(__file__).resolve().parent / s


# -------------------------
# Download helper
# -------------------------
def download_if_missing(url: str, dst: Path) -> bool:
    if dst.exists():
        return True
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(dst))
        return True
    except Exception:
        return False


# -------------------------
# Device helper
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
        raise FileNotFoundError(f"Missing {EMB_FILE} or {META_FILE} in the repo.")
    embs = np.load(str(emb_path)).astype(np.float32)
    with open(str(meta_path), "r", encoding="utf-8") as f:
        meta = json.load(f)
    embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    return embs, meta


# -------------------------
# Face crop backends
# -------------------------
def crop_with_mediapipe_landmarks(img: Image.Image, margin: float, debug: bool):
    """
    Returns (cropped_img, info_dict). If fails, info['ok']=False with details.
    """
    info = {
        "ok": False,
        "backend": "mediapipe",
        "reason": "",
        "faces": 0,
        "model_path": str(FACE_LANDMARKER_PATH),
    }

    # Fix rotations (very important for mobile photos)
    img = ImageOps.exif_transpose(img).convert("RGB")

    # Import mediapipe
    try:
        import mediapipe as mp
    except Exception as e:
        info["reason"] = f"mediapipe import failed: {e}"
        return img, info

    # Download model
    if not download_if_missing(FACE_LANDMARKER_URL, FACE_LANDMARKER_PATH):
        info["reason"] = "failed to download face_landmarker.task"
        return img, info

    # Create landmarker
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
        landmarker = FaceLandmarker.create_from_options(options)
    except Exception as e:
        info["reason"] = f"FaceLandmarker create failed: {e}"
        return img, info

    try:
        # Detect on downscaled copy for speed, but crop on original using normalized coords
        work = img.copy()
        work.thumbnail((720, 720))
        np_work = np.array(work)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_work)

        result = landmarker.detect(mp_image)
        if not getattr(result, "face_landmarks", None):
            info["reason"] = "no face landmarks detected"
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

        orig = np.array(img)
        H, W = orig.shape[:2]
        x1, x2 = int(minx * W), int(maxx * W)
        y1, y2 = int(miny * H), int(maxy * H)

        if debug:
            info["norm_box"] = [minx, miny, maxx, maxy]
            info["px_box"] = [x1, y1, x2, y2]
            info["orig_size"] = [W, H]

        if x2 <= x1 or y2 <= y1:
            info["reason"] = "invalid crop box"
            return img, info

        crop = orig[y1:y2, x1:x2]
        info["ok"] = True
        info["reason"] = "cropped"
        return Image.fromarray(crop), info

    except Exception as e:
        info["reason"] = f"landmarker runtime error: {e}"
        return img, info
    finally:
        try:
            landmarker.close()
        except Exception:
            pass


def crop_with_opencv_haar(img: Image.Image, margin: float, debug: bool):
    """
    OpenCV Haar cascade fallback. Returns (cropped_img, info_dict).
    """
    info = {
        "ok": False,
        "backend": "opencv_haar",
        "reason": "",
        "faces": 0,
        "model_path": str(HAAR_PATH),
    }

    img = ImageOps.exif_transpose(img).convert("RGB")

    try:
        import cv2
    except Exception as e:
        info["reason"] = f"opencv import failed: {e}"
        return img, info

    if not download_if_missing(HAAR_URL, HAAR_PATH):
        info["reason"] = "failed to download haarcascade xml"
        return img, info

    # Downscale for detection speed; map back to original
    work = img.copy()
    work.thumbnail((720, 720))
    np_work = np.array(work)
    gray = cv2.cvtColor(np_work, cv2.COLOR_RGB2GRAY)

    cascade = cv2.CascadeClassifier(str(HAAR_PATH))
    if cascade.empty():
        info["reason"] = "cascade classifier failed to load"
        return img, info

    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if faces is None or len(faces) == 0:
        info["reason"] = "no face detected by haar cascade"
        return img, info

    info["faces"] = int(len(faces))

    # pick the largest face
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])

    # map coords to original
    sx = img.size[0] / work.size[0]
    sy = img.size[1] / work.size[1]
    x, y, w, h = int(x * sx), int(y * sy), int(w * sx), int(h * sy)

    # expand
    cx, cy = x + w / 2, y + h / 2
    size = max(w, h) * (1.0 + margin * 2.0)
    x1 = int(max(0, cx - size / 2))
    y1 = int(max(0, cy - size / 2))
    x2 = int(min(img.size[0], cx + size / 2))
    y2 = int(min(img.size[1], cy + size / 2))

    if debug:
        info["px_box"] = [x1, y1, x2, y2]
        info["orig_size"] = [img.size[0], img.size[1]]

    if x2 <= x1 or y2 <= y1:
        info["reason"] = "invalid crop box"
        return img, info

    crop = np.array(img)[y1:y2, x1:x2]
    info["ok"] = True
    info["reason"] = "cropped"
    return Image.fromarray(crop), info


def face_crop_auto(img: Image.Image, margin: float, debug: bool):
    """
    Try MediaPipe first, then OpenCV Haar fallback.
    """
    cropped, info = crop_with_mediapipe_landmarks(img, margin=margin, debug=debug)
    if info.get("ok"):
        return cropped, info

    cropped2, info2 = crop_with_opencv_haar(img, margin=margin, debug=debug)
    if info2.get("ok"):
        return cropped2, info2

    # both failed -> return original with combined info
    return img, {"ok": False, "backend": "none", "reason": f"mediapipe: {info.get('reason')} | opencv: {info2.get('reason')}"}


# -------------------------
# CLIP embedding (robust)
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
    raise TypeError(f"Could not extract embeddings from model output: {type(out)}")


@torch.no_grad()
def embed_one(img: Image.Image, model, processor, device) -> np.ndarray:
    inputs = processor(images=[img.convert("RGB")], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    feat = get_clip_image_embeds(model, pixel_values)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).detach().cpu().numpy().astype(np.float32)


# -------------------------
# Confidence
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
st.title("🪶 BirdLookalike — Mobile Camera + Robust Face Crop")

model, processor, device = load_clip()
embs, meta = load_index()

source = st.radio("Input source", ["📷 Camera", "🖼️ Upload"], horizontal=True)

# Defaults
use_crop = True
crop_margin = 0.06
debug_crop = False
topN = min(50, len(meta))
temperature = 20.0

with st.expander("⚙️ Settings", expanded=False):
    use_crop = st.checkbox("Crop face (MediaPipe → OpenCV fallback)", value=True)
    crop_margin = st.slider("Crop margin", 0.02, 0.20, float(crop_margin), 0.01)
    debug_crop = st.checkbox("Debug crop", value=False)
    topN = st.slider("Top-N competitors", 10, 200, int(topN))
    temperature = st.slider("Temperature", 5.0, 40.0, float(temperature), 1.0)

st.caption(f"Device: {device} | Birds indexed: {len(meta)}")

if source == "📷 Camera":
    up = st.camera_input("Take a photo")
else:
    up = st.file_uploader("Upload a face photo", type=["jpg", "jpeg", "png", "webp"])

if not up:
    st.info("Capture or upload a photo to continue.")
    st.stop()

img = Image.open(BytesIO(up.getvalue())).convert("RGB")

if use_crop:
    img_in, crop_info = face_crop_auto(img, margin=crop_margin, debug=debug_crop)
    if debug_crop:
        st.json(crop_info)
else:
    img_in = ImageOps.exif_transpose(img).convert("RGB")

q = embed_one(img_in, model, processor, device)
best_i, best_sim, best_conf, idx_sorted, sims = best_match_with_confidence(
    embs, q, topN=topN, temperature=temperature
)
best = meta[best_i]

st.subheader("Input (used for embedding)")
st.image(resize_for_display(img_in), width=DISPLAY_MAX_W)

st.subheader("✅ Best match")
st.markdown(f"### **{best.get('species','Unknown')}**")
c1, c2 = st.columns(2)
c1.metric("Similarity", f"{best_sim:.3f}")
c2.metric("Confidence", f"{best_conf * 100:.1f}%")

bird_path = resolve_asset(best.get("path", ""))
if bird_path.exists():
    with st.expander("🖼️ Show matched bird image", expanded=False):
        bird_img = Image.open(str(bird_path)).convert("RGB")
        st.image(resize_for_display(bird_img), caption=f"{best.get('species','Unknown')} | {bird_path.name}", width=DISPLAY_MAX_W)
else:
    st.warning(
        "Bird image file not found in this deployment. Include birds/ in the GitHub repo to display it."
    )

with st.expander("Runner-ups", expanded=False):
    for j in idx_sorted[1:6]:
        m = meta[int(j)]
        st.write(f"- {m.get('species','Unknown')}  (sim={float(sims[int(j)]):.3f})")
