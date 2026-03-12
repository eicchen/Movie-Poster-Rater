#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations

import math
import pickle
import sys
from copy import deepcopy
from itertools import combinations, product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import easyocr
import numpy as np
from PIL import Image
from pgmpy.inference import VariableElimination
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# Paths to the model pickles (resolved relative to this file)
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
genre_pickle = str(_THIS_DIR / ".." / "data" / "poster_genre_bn_bundle.pkl")
rating_pickle = str(_THIS_DIR / ".." / "data" / "poster_rating_bn_bundle.pkl")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_SIDE = 1024
_DATA_DIR = _THIS_DIR / ".." / "data"
FACE_MODEL_PATH = _DATA_DIR / "face_detector.tflite"
YOLO_WEIGHTS = str(_DATA_DIR / "yolov8n.pt")

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

CLIP_PROMPTS = {
    "photographic": "a photographic movie poster",
    "illustrated": "an illustrated movie poster",
    "dark_moody": "a dark moody movie poster",
    "bright_colorful": "a bright colorful movie poster",
    "minimalist": "a minimalist movie poster",
    "crowded": "a crowded movie poster with many elements",
    "romantic": "a romantic movie poster",
    "horror": "a horror movie poster",
    "comedy": "a comedy movie poster",
    "action": "an action movie poster",
    "retro": "a retro vintage movie poster",
    "family_friendly": "a family-friendly movie poster",
    "serious_drama": "a serious drama movie poster",
    "sci_fi": "a science fiction movie poster",
    "historical": "a historical period movie poster",
    "explosion": "a movie poster with an explosion",
    "weapons": "a movie poster featuring weapons",
    "gun": "a movie poster featuring a gun",
    "vehicle": "a movie poster featuring a vehicle",
    "car_chase": "a movie poster with a car chase",
    "face_profile": "a movie poster with a face in profile",
    "face_obscured": "a movie poster with an obscured face",
    "masked_face": "a movie poster with a masked face",
    "serif_type": "a movie poster with serif typography",
    "sans_type": "a movie poster with sans-serif typography",
    "script_type": "a movie poster with script handwriting typography",
    "condensed_bold_type": "a movie poster with bold condensed typography",
}

# ---------------------------------------------------------------------------
# Lazy-loaded model singletons
# ---------------------------------------------------------------------------
_TORCH = None
_DEVICE = None

_YOLO_MODEL = None
_EASYOCR_READER = None

_MP = None
_MP_PYTHON = None
_MP_VISION = None
_FACE_DETECTOR = None
_FACE_DETECTOR_ERROR = None

_OPEN_CLIP = None
_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_TOKENIZER = None
_CLIP_TEXT_FEATURES = None

_NSFW2 = None
_NSFW2_ERROR = None


def get_torch():
    global _TORCH
    if _TORCH is None:
        import torch
        _TORCH = torch
    return _TORCH


def get_device() -> str:
    global _DEVICE
    if _DEVICE is None:
        torch = get_torch()
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return _DEVICE


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def load_rgb(path: str, max_side: int = MAX_SIDE) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return np.array(img, dtype=np.uint8)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


def luminance(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b != 0 else 0.0


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


# ---------------------------------------------------------------------------
# Heavy model getters
# ---------------------------------------------------------------------------
def get_yolo_model():
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        from ultralytics import YOLO
        _YOLO_MODEL = YOLO(YOLO_WEIGHTS)
    return _YOLO_MODEL


def get_easyocr_reader():
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        import easyocr
        torch = get_torch()
        _EASYOCR_READER = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    return _EASYOCR_READER


def get_mediapipe_face_detector():
    global _MP, _MP_PYTHON, _MP_VISION, _FACE_DETECTOR, _FACE_DETECTOR_ERROR

    if _FACE_DETECTOR is not None:
        return _FACE_DETECTOR, None

    if _FACE_DETECTOR_ERROR is not None:
        return None, _FACE_DETECTOR_ERROR

    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        _MP = mp
        _MP_PYTHON = python
        _MP_VISION = vision
    except Exception as e:
        _FACE_DETECTOR_ERROR = f"mediapipe not available: {e}"
        return None, _FACE_DETECTOR_ERROR

    if not FACE_MODEL_PATH.exists():
        _FACE_DETECTOR_ERROR = f"face detector model not found at {FACE_MODEL_PATH.resolve()}"
        return None, _FACE_DETECTOR_ERROR

    try:
        base_options = _MP_PYTHON.BaseOptions(model_asset_path=str(FACE_MODEL_PATH))
        options = _MP_VISION.FaceDetectorOptions(base_options=base_options)
        _FACE_DETECTOR = _MP_VISION.FaceDetector.create_from_options(options)
        return _FACE_DETECTOR, None
    except Exception as e:
        _FACE_DETECTOR_ERROR = f"mediapipe face model issue: {e}"
        return None, _FACE_DETECTOR_ERROR


def get_clip_components():
    global _OPEN_CLIP, _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_TOKENIZER, _CLIP_TEXT_FEATURES

    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_TOKENIZER, _CLIP_TEXT_FEATURES

    torch = get_torch()
    import open_clip

    device = get_device()

    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME,
        pretrained=CLIP_PRETRAINED,
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)

    prompt_texts = list(CLIP_PROMPTS.values())
    text_tokens = tokenizer(prompt_texts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    _OPEN_CLIP = open_clip
    _CLIP_MODEL = model
    _CLIP_PREPROCESS = preprocess
    _CLIP_TOKENIZER = tokenizer
    _CLIP_TEXT_FEATURES = text_features

    return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_TOKENIZER, _CLIP_TEXT_FEATURES


def get_opennsfw2():
    global _NSFW2, _NSFW2_ERROR
    if _NSFW2 is not None:
        return _NSFW2, None
    if _NSFW2_ERROR is not None:
        return None, _NSFW2_ERROR

    try:
        import opennsfw2
        _NSFW2 = opennsfw2
        return _NSFW2, None
    except Exception as e:
        _NSFW2_ERROR = f"opennsfw2 not available: {e}"
        return None, _NSFW2_ERROR


def initialize_feature_models(verbose: bool = True):
    """Optional one-time warmup for notebook use."""
    status = {}

    try:
        get_torch()
        status["device"] = get_device()
    except Exception as e:
        status["device_error"] = str(e)

    try:
        get_yolo_model()
        status["yolo"] = "ok"
    except Exception as e:
        status["yolo"] = f"error: {e}"

    try:
        get_easyocr_reader()
        status["easyocr"] = "ok"
    except Exception as e:
        status["easyocr"] = f"error: {e}"

    detector, err = get_mediapipe_face_detector()
    status["mediapipe_face"] = "ok" if detector is not None else f"error: {err}"

    try:
        get_clip_components()
        status["clip"] = "ok"
    except Exception as e:
        status["clip"] = f"error: {e}"

    nsfw2, err = get_opennsfw2()
    status["opennsfw2"] = "ok" if nsfw2 is not None else f"error: {err}"

    if verbose:
        for k, v in status.items():
            print(f"{k}: {v}")

    return status


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------
def color_features(
    rgb: np.ndarray,
    palette_k: int = 5,
    sample_pixels: int = 20000,
    quant_bins: int = 32,
) -> Dict[str, Any]:
    hsv = rgb_to_hsv(rgb)
    Y = luminance(rgb)

    brightness = float(np.mean(Y) / 255.0)
    contrast = float(np.std(Y) / 255.0)
    saturation = float(np.mean(hsv[..., 1].astype(np.float32)) / 255.0)

    hue = hsv[..., 0].astype(np.float32)
    warm = np.logical_or(hue <= 20, hue >= 160)
    cool = np.logical_and(hue >= 80, hue <= 140)
    warm_ratio = float(np.mean(warm))
    cool_ratio = float(np.mean(cool))
    warmth_score = float(warm_ratio - cool_ratio)

    q = np.clip((rgb.astype(np.int32) * quant_bins) // 256, 0, quant_bins - 1)
    ids = q[..., 0] * (quant_bins * quant_bins) + q[..., 1] * quant_bins + q[..., 2]
    hist = np.bincount(ids.reshape(-1), minlength=quant_bins**3).astype(np.float32)
    p = hist / (hist.sum() + 1e-9)
    p_nz = p[p > 0]
    color_entropy = float(-np.sum(p_nz * np.log(p_nz + 1e-12)))

    dominant_color_count = int(np.sum(p > (1.0 / (quant_bins**3)) * 50.0))

    neon = np.logical_and(hsv[..., 1] >= 200, hsv[..., 2] >= 200)
    neon_ratio = float(np.mean(neon))

    skin = (
        (np.logical_or(hsv[..., 0] <= 25, hsv[..., 0] >= 160))
        & (hsv[..., 1] >= 40)
        & (hsv[..., 1] <= 200)
        & (hsv[..., 2] >= 60)
    )
    skin_tone_ratio = float(np.mean(skin))

    flat = rgb.reshape(-1, 3)
    n = flat.shape[0]
    if n > sample_pixels:
        idx = np.random.choice(n, size=sample_pixels, replace=False)
        samp = flat[idx]
    else:
        samp = flat

    km = KMeans(n_clusters=palette_k, n_init="auto", random_state=0)
    km.fit(samp.astype(np.float32))
    centers = km.cluster_centers_.astype(np.float32)
    counts = np.bincount(km.labels_, minlength=palette_k).astype(np.float32)
    order = np.argsort(-counts)
    centers = centers[order]
    counts = counts[order]
    weights = (counts / (counts.sum() + 1e-9)).tolist()

    palette = [
        {"rgb": centers[i].round(1).tolist(), "weight": float(weights[i])}
        for i in range(palette_k)
    ]

    return {
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "warm_ratio": warm_ratio,
        "cool_ratio": cool_ratio,
        "warmth_score": warmth_score,
        "color_entropy": color_entropy,
        "dominant_color_count": dominant_color_count,
        "neon_ratio": neon_ratio,
        "skin_tone_ratio": skin_tone_ratio,
        "dominant_palette": palette,
    }


def edge_texture_layout_features(rgb: np.ndarray) -> Dict[str, Any]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    edges = cv2.Canny(gray, threshold1=80, threshold2=160)
    edge_density = float(np.mean(edges > 0))

    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11 + 1), density=True)
    lbp_entropy = float(-np.sum(lbp_hist * np.log(lbp_hist + 1e-9)))

    mid = w // 2
    left = gray[:, :mid].astype(np.float32)
    right = gray[:, w - mid:].astype(np.float32)
    right_m = np.fliplr(right)
    mae = float(np.mean(np.abs(left - right_m)) / 255.0)
    symmetry_score = float(1.0 - mae)

    mag = cv2.Laplacian(gray, cv2.CV_32F)
    wts = np.abs(mag)
    wts_sum = float(wts.sum() + 1e-9)
    ys, xs = np.indices(gray.shape)
    visual_center_y = float((ys * wts).sum() / wts_sum) / float(h)
    visual_center_x = float((xs * wts).sum() / wts_sum) / float(w)

    left_mass = float(wts[:, :mid].sum())
    right_mass = float(wts[:, w - mid:].sum())
    visual_balance_lr = float(abs(left_mass - right_mass) / (left_mass + right_mass + 1e-9))

    midy = h // 2
    top_mass = float(wts[:midy, :].sum())
    bot_mass = float(wts[h - midy:, :].sum())
    top_heavy = float((top_mass - bot_mass) / (top_mass + bot_mass + 1e-9))
    bottom_heavy = float(-top_heavy)

    thirds = [(1/3, 1/3), (2/3, 1/3), (1/3, 2/3), (2/3, 2/3)]
    dists = [math.sqrt((visual_center_x - tx) ** 2 + (visual_center_y - ty) ** 2) for tx, ty in thirds]
    dmin = min(dists)
    rule_of_thirds_score = float(1.0 - clamp01(dmin / math.sqrt(2.0)))

    return {
        "edge_density": edge_density,
        "lbp_entropy": lbp_entropy,
        "symmetry_score": symmetry_score,
        "visual_center_x": visual_center_x,
        "visual_center_y": visual_center_y,
        "visual_balance_lr": visual_balance_lr,
        "top_heavy": top_heavy,
        "bottom_heavy": bottom_heavy,
        "rule_of_thirds_score": rule_of_thirds_score,
    }


def lighting_blur_negative_space_features(rgb: np.ndarray) -> Dict[str, Any]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    Y = gray.astype(np.float32) / 255.0

    shadow_ratio = float(np.mean(Y < 0.18))
    highlight_ratio = float(np.mean(Y > 0.85))

    mu = float(np.mean(Y))
    sigma = float(np.std(Y) + 1e-9)
    luminance_skew = float(np.mean(((Y - mu) / sigma) ** 3))

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    blur_score = float(lap.var())
    blur_score_norm = float(1.0 - math.exp(-blur_score / 200.0))

    g = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    thr = float(np.percentile(grad, 60.0))
    negative_space_ratio = float(np.mean(grad <= thr))

    return {
        "shadow_ratio": shadow_ratio,
        "highlight_ratio": highlight_ratio,
        "luminance_skew": luminance_skew,
        "blur_score": blur_score,
        "blur_score_norm": blur_score_norm,
        "negative_space_ratio": negative_space_ratio,
    }


def face_features_mediapipe(rgb: np.ndarray) -> Dict[str, Any]:
    detector, err = get_mediapipe_face_detector()
    if detector is None:
        return {
            "face_count": None,
            "largest_face_ratio": None,
            "avg_face_ratio": None,
            "face_centered_score": None,
            "face_error": err,
        }

    try:
        mp_image = _MP.Image(image_format=_MP.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
    except Exception as e:
        return {
            "face_count": None,
            "largest_face_ratio": None,
            "avg_face_ratio": None,
            "face_centered_score": None,
            "face_error": f"mediapipe face model issue: {e}",
        }

    h, w, _ = rgb.shape
    boxes: List[Tuple[float, float, float, float]] = []
    for det in result.detections:
        bb = det.bounding_box
        boxes.append((bb.origin_x, bb.origin_y, bb.width, bb.height))

    face_count = int(len(boxes))
    if face_count == 0:
        return {
            "face_count": 0,
            "largest_face_ratio": 0.0,
            "avg_face_ratio": 0.0,
            "face_centered_score": 0.0,
        }

    areas = [max(0.0, bw) * max(0.0, bh) for _, _, bw, bh in boxes]
    largest_face_ratio = float(max(areas) / float(h * w))
    avg_face_ratio = float(np.mean(areas) / float(h * w))

    cx_img, cy_img = w / 2.0, h / 2.0
    dists = []
    for x, y, bw, bh in boxes:
        cx = x + bw / 2.0
        cy = y + bh / 2.0
        d = ((cx - cx_img) ** 2 + (cy - cy_img) ** 2) ** 0.5
        dists.append(d)

    dmin = min(dists)
    dmax = (cx_img**2 + cy_img**2) ** 0.5
    face_centered_score = float(1.0 - max(0.0, min(1.0, dmin / (dmax + 1e-9))))

    return {
        "face_count": face_count,
        "largest_face_ratio": largest_face_ratio,
        "avg_face_ratio": avg_face_ratio,
        "face_centered_score": face_centered_score,
    }


def object_features_yolov8(rgb: np.ndarray) -> Dict[str, Any]:
    try:
        torch = get_torch()
        model = get_yolo_model()
    except Exception as e:
        return {
            "object_count": None,
            "object_topk": None,
            "object_density": None,
            "person_count": None,
            "vehicle_count": None,
            "object_flags": None,
            "object_error": f"ultralytics not available: {e}",
        }

    try:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        device_arg = 0 if torch.cuda.is_available() else "cpu"
        pred = model.predict(source=bgr, verbose=False, conf=0.25, device=device_arg)[0]
    except Exception as e:
        return {
            "object_count": None,
            "object_topk": None,
            "object_density": None,
            "person_count": None,
            "vehicle_count": None,
            "object_flags": None,
            "object_error": f"yolo inference failed: {e}",
        }

    names = pred.names
    if pred.boxes is None or pred.boxes.cls is None:
        det_names: List[str] = []
    else:
        cls = pred.boxes.cls.detach().cpu().numpy().astype(int)
        det_names = [names[i] for i in cls]

    h, w, _ = rgb.shape
    object_count = int(len(det_names))
    object_density = float(object_count / float(h * w) * 1e5)

    person_count = int(sum(n == "person" for n in det_names))
    vehicle_set = {"car", "bus", "truck", "motorcycle", "bicycle"}
    vehicle_count = int(sum(n in vehicle_set for n in det_names))

    flags = {
        "has_person": person_count > 0,
        "has_vehicle": vehicle_count > 0,
        "has_animal": any(n in {"dog", "cat", "horse", "bird", "bear"} for n in det_names),
    }

    return {
        "object_count": object_count,
        "object_topk": det_names[:10],
        "object_density": object_density,
        "person_count": person_count,
        "vehicle_count": vehicle_count,
        "object_flags": flags,
    }


def ocr_features_easyocr(rgb: np.ndarray) -> Dict[str, Any]:
    try:
        reader = get_easyocr_reader()
    except Exception as e:
        return {
            "ocr_text": None,
            "text_area_ratio": None,
            "uppercase_ratio": None,
            "num_text_boxes": None,
            "largest_text_area_ratio": None,
            "title_position_y": None,
            "stroke_width_mean": None,
            "stroke_width_std": None,
            "serifness_proxy": None,
            "ocr_error": f"easyocr not available: {e}",
        }

    try:
        result = reader.readtext(rgb)
    except Exception as e:
        return {
            "ocr_text": None,
            "text_area_ratio": None,
            "uppercase_ratio": None,
            "num_text_boxes": None,
            "largest_text_area_ratio": None,
            "title_position_y": None,
            "stroke_width_mean": None,
            "stroke_width_std": None,
            "serifness_proxy": None,
            "ocr_error": f"easyocr inference failed: {e}",
        }

    h, w, _ = rgb.shape
    total_area = float(h * w)
    text_area = 0.0
    texts: List[str] = []
    areas: List[float] = []
    centers_y: List[float] = []
    boxes_pts: List[np.ndarray] = []

    for bbox, text, conf in result:
        pts = np.array(bbox, dtype=np.float32)
        x_min, y_min = float(pts[:, 0].min()), float(pts[:, 1].min())
        x_max, y_max = float(pts[:, 0].max()), float(pts[:, 1].max())
        area = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
        cy = (y_min + y_max) / 2.0

        text_area += area
        texts.append(str(text))
        areas.append(area)
        centers_y.append(cy)
        boxes_pts.append(pts)

    joined = " ".join(texts).strip()
    alpha = [c for c in joined if c.isalpha()]
    uppercase = [c for c in alpha if c.isupper()]
    uppercase_ratio = safe_div(len(uppercase), len(alpha))

    num_text_boxes = int(len(result))
    text_area_ratio = float(text_area / (total_area + 1e-9))

    if len(areas) == 0:
        largest_text_area_ratio = 0.0
        title_position_y = 0.0
        stroke_width_mean = 0.0
        stroke_width_std = 0.0
        serifness_proxy = 0.0
    else:
        imax = int(np.argmax(areas))
        largest_text_area_ratio = float(areas[imax] / (total_area + 1e-9))
        title_position_y = float(centers_y[imax] / float(h))

        pts = boxes_pts[imax].astype(np.int32)
        x_min = int(np.clip(pts[:, 0].min(), 0, w - 1))
        x_max = int(np.clip(pts[:, 0].max(), 0, w - 1))
        y_min = int(np.clip(pts[:, 1].min(), 0, h - 1))
        y_max = int(np.clip(pts[:, 1].max(), 0, h - 1))

        crop = rgb[y_min:y_max + 1, x_min:x_max + 1]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            stroke_width_mean = 0.0
            stroke_width_std = 0.0
            serifness_proxy = 0.0
        else:
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            bw = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
            )
            dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
            vals = dist[dist > 0]
            if vals.size == 0:
                stroke_width_mean = 0.0
                stroke_width_std = 0.0
                serifness_proxy = 0.0
            else:
                sw = 2.0 * vals
                stroke_width_mean = float(np.mean(sw))
                stroke_width_std = float(np.std(sw))

                neigh = cv2.filter2D((bw > 0).astype(np.uint8), -1, np.ones((3, 3), np.uint8))
                endpoints = np.logical_and(bw > 0, neigh == 2)
                serifness_proxy = float(np.mean(endpoints))

    return {
        "ocr_text": joined[:500],
        "text_area_ratio": text_area_ratio,
        "uppercase_ratio": float(uppercase_ratio),
        "num_text_boxes": num_text_boxes,
        "largest_text_area_ratio": largest_text_area_ratio,
        "title_position_y": title_position_y,
        "stroke_width_mean": float(stroke_width_mean),
        "stroke_width_std": float(stroke_width_std),
        "serifness_proxy": float(serifness_proxy),
    }


def nsfw_features_open_nsfw2(path: str) -> Dict[str, Any]:
    nsfw2, err = get_opennsfw2()
    if nsfw2 is None:
        return {"nsfw_score": None, "nsfw_error": err}

    try:
        score = float(nsfw2.predict_image(path))
        return {"nsfw_score": score}
    except Exception as e:
        return {"nsfw_score": None, "nsfw_error": f"opennsfw2 failed: {e}"}


def clip_features_openclip(rgb: np.ndarray) -> Dict[str, Any]:
    try:
        torch = get_torch()
        device = get_device()
        model, preprocess, tokenizer, text_features = get_clip_components()
    except Exception as e:
        return {
            "clip_embedding": None,
            "clip_prompt_scores": None,
            "clip_model": None,
            "clip_error": f"open_clip/torch not available: {e}",
        }

    try:
        pil = Image.fromarray(rgb)
        image_tensor = preprocess(pil).unsqueeze(0).to(device)

        with torch.no_grad():
            img_feat = model.encode_image(image_tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        sims = (img_feat @ text_features.T).squeeze(0).detach().cpu().numpy()
        clip_prompt_scores = {k: float(v) for k, v in zip(CLIP_PROMPTS.keys(), sims)}

        return {
            "clip_embedding": img_feat.squeeze(0).detach().cpu().numpy().astype(np.float32),
            "clip_prompt_scores": clip_prompt_scores,
            "clip_model": {"model": CLIP_MODEL_NAME, "pretrained": CLIP_PRETRAINED},
        }
    except Exception as e:
        return {
            "clip_embedding": None,
            "clip_prompt_scores": None,
            "clip_model": None,
            "clip_error": f"clip inference failed: {e}",
        }


# ---------------------------------------------------------------------------
# One-call extractor
# ---------------------------------------------------------------------------
def extract_poster_nodes(path: str) -> Dict[str, Any]:
    rgb = load_rgb(path, max_side=MAX_SIDE)

    nodes: Dict[str, Any] = {
        "path": path,
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
    }

    nodes.update(color_features(rgb))
    nodes.update(edge_texture_layout_features(rgb))
    nodes.update(lighting_blur_negative_space_features(rgb))
    nodes.update(face_features_mediapipe(rgb))
    nodes.update(object_features_yolov8(rgb))
    nodes.update(ocr_features_easyocr(rgb))
    nodes.update(clip_features_openclip(rgb))
    nodes.update(nsfw_features_open_nsfw2(path))

    return nodes


def enrich_poster_features(poster_data):
    poster_data = dict(poster_data)

    if "visual_center_x" in poster_data and "visual_center_y" in poster_data:
        poster_data["center_distance"] = (
            ((poster_data["visual_center_x"] - 0.5) ** 2 +
             (poster_data["visual_center_y"] - 0.5) ** 2) ** 0.5
        )

    if "brightness" in poster_data:
        poster_data["darkness"] = 1.0 - poster_data["brightness"]

    if "contrast" in poster_data and "saturation" in poster_data:
        poster_data["color_contrast_combo"] = (
            poster_data["contrast"] * poster_data["saturation"]
        )

    if "warmth_score" in poster_data and "brightness" in poster_data:
        poster_data["warm_bright_combo"] = (
            poster_data["warmth_score"] * poster_data["brightness"]
        )

    if "symmetry_score" in poster_data and "edge_density" in poster_data:
        poster_data["symmetry_edge_combo"] = (
            poster_data["symmetry_score"] * poster_data["edge_density"]
        )

    return poster_data


def acquire_poster_data(path):
    return enrich_poster_features(extract_poster_nodes(path))


# ---------------------------------------------------------------------------
# Bayesian network helpers
# ---------------------------------------------------------------------------
def flatten_poster_dict(d, parent_key="", sep="."):
    flat = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            flat.update(flatten_poster_dict(v, new_key, sep=sep))
        else:
            flat[new_key] = v
    return flat


def bin_value(value, edges, labels):
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None

    for i in range(len(labels)):
        left = edges[i]
        right = edges[i + 1]

        if i < len(labels) - 1:
            if left <= value < right:
                return labels[i]
        else:
            if left <= value <= right:
                return labels[i]

    return None


def poster_dict_to_bn_evidence(poster_data, bundle):
    flat = flatten_poster_dict(poster_data)
    bin_specs = bundle["bin_specs"]

    evidence = {}

    for feature_name, spec in bin_specs.items():
        if feature_name not in flat:
            continue

        binned = bin_value(flat[feature_name], spec["edges"], spec["labels"])
        col_name = f"{feature_name}_bin"

        if binned is not None:
            evidence[col_name] = binned

    bool_map = {
        "object_flags.has_person": ("present", "absent"),
        "object_flags.has_vehicle": ("present", "absent"),
        "object_flags.has_animal": ("present", "absent"),
    }

    for key, (true_val, false_val) in bool_map.items():
        if key in flat:
            if flat[key] is True:
                evidence[key] = true_val
            elif flat[key] is False:
                evidence[key] = false_val

    return evidence


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_genre_from_poster_dict(
    poster_data,
    bundle_path=genre_pickle,
    debug=False,
):
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    raw_evidence = poster_dict_to_bn_evidence(poster_data, bundle)

    model_nodes = set(model.nodes())

    used_evidence = {}
    dropped_evidence = {}

    for k, v in raw_evidence.items():
        if k in model_nodes:
            used_evidence[k] = v
        else:
            dropped_evidence[k] = v

    if debug:
        print("\n===== BN EVIDENCE USED =====")
        for k, v in sorted(used_evidence.items()):
            print(f"{k}: {v}")

        print("\n===== BN EVIDENCE DROPPED =====")
        for k, v in sorted(dropped_evidence.items()):
            print(f"{k}: {v}")

        print("\nTotal used:", len(used_evidence))
        print("Total dropped:", len(dropped_evidence))

    infer = VariableElimination(model)

    q = infer.query(
        variables=["genre"],
        evidence=used_evidence,
        show_progress=False,
    )

    probs = {
        genre: float(prob)
        for genre, prob in zip(q.state_names["genre"], q.values)
    }

    if debug:
        print("\n===== GENRE PROBABILITIES =====")
        for g, p in sorted(probs.items(), key=lambda x: -x[1]):
            print(f"{g}: {p:.4f}")

    return probs


def predict_rating_from_poster_dict(
    poster_data,
    bundle_path=rating_pickle,
    debug=False,
):
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    bin_specs = bundle["bin_specs"]

    poster_data = enrich_poster_features(poster_data)

    raw_evidence = poster_dict_to_bn_evidence(poster_data, bundle)

    model_nodes = set(model.nodes())

    evidence = {k: v for k, v in raw_evidence.items() if k in model_nodes}

    if debug:
        print("\n===== BN EVIDENCE USED =====")
        for k, v in sorted(evidence.items()):
            print(f"{k}: {v}")

    infer = VariableElimination(model)

    q = infer.query(
        variables=["rating_bin"],
        evidence=evidence,
        show_progress=False,
    )

    probs = {
        r: float(p)
        for r, p in zip(q.state_names["rating_bin"], q.values)
    }

    if debug:
        print("\n===== RATING PROBABILITIES =====")
        for r, p in sorted(probs.items(), key=lambda x: -x[1]):
            print(f"{r}: {p:.4f}")

    return probs


# ---------------------------------------------------------------------------
# Counterfactual analysis
# ---------------------------------------------------------------------------
def get_bn_evidence_for_poster(poster_data, bundle_path):
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]

    poster_data = enrich_poster_features(poster_data)
    raw_evidence = poster_dict_to_bn_evidence(poster_data, bundle)

    model_nodes = set(model.nodes())
    evidence = {k: v for k, v in raw_evidence.items() if k in model_nodes}

    return evidence, bundle


def query_target_prob(model, target_node, target_value, evidence):
    infer = VariableElimination(model)
    q = infer.query(
        variables=[target_node],
        evidence=evidence,
        show_progress=False,
    )

    state_names = q.state_names[target_node]
    probs = dict(zip(state_names, q.values))
    return float(probs[target_value]), probs


def get_node_states(model, node):
    cpd = model.get_cpds(node)
    if cpd is None:
        return []
    return list(cpd.state_names[node])


def find_min_changes_for_target(
    model,
    evidence,
    target_node,
    target_value,
    candidate_nodes=None,
    max_changes=3,
    threshold=0.50,
    prefer_higher_prob=True,
):
    """
    Search for the fewest feature-node changes that make target_value likely.

    Returns a dict with:
      - base_prob
      - best_solution
      - all_solutions_found

    best_solution contains:
      - changed_nodes
      - new_evidence
      - target_prob
      - full_probs
    """

    if candidate_nodes is None:
        candidate_nodes = [n for n in evidence.keys() if n != target_node]

    valid_candidates = []
    node_states = {}

    for node in candidate_nodes:
        if node not in model.nodes():
            continue
        states = get_node_states(model, node)
        if len(states) > 1:
            valid_candidates.append(node)
            node_states[node] = states

    base_prob, base_probs = query_target_prob(model, target_node, target_value, evidence)

    if base_prob >= threshold:
        return {
            "base_prob": base_prob,
            "base_probs": base_probs,
            "best_solution": {
                "changed_nodes": {},
                "new_evidence": dict(evidence),
                "target_prob": base_prob,
                "full_probs": base_probs,
            },
            "all_solutions_found": [],
        }

    all_solutions = []
    best_solution = None

    for k in range(1, max_changes + 1):
        found_at_this_k = []

        for nodes_subset in combinations(valid_candidates, k):
            replacement_choices = []

            for node in nodes_subset:
                current_val = evidence[node]
                other_states = [s for s in node_states[node] if s != current_val]
                replacement_choices.append(other_states)

            for new_values in product(*replacement_choices):
                new_evidence = dict(evidence)
                changed_nodes = {}

                for node, val in zip(nodes_subset, new_values):
                    new_evidence[node] = val
                    changed_nodes[node] = {
                        "from": evidence[node],
                        "to": val,
                    }

                target_prob, full_probs = query_target_prob(
                    model, target_node, target_value, new_evidence
                )

                sol = {
                    "num_changes": k,
                    "changed_nodes": changed_nodes,
                    "new_evidence": new_evidence,
                    "target_prob": target_prob,
                    "full_probs": full_probs,
                }

                all_solutions.append(sol)

                if target_prob >= threshold:
                    found_at_this_k.append(sol)

        if found_at_this_k:
            if prefer_higher_prob:
                best_solution = max(found_at_this_k, key=lambda s: s["target_prob"])
            else:
                best_solution = found_at_this_k[0]

            return {
                "base_prob": base_prob,
                "base_probs": base_probs,
                "best_solution": best_solution,
                "all_solutions_found": found_at_this_k,
            }

    if all_solutions:
        best_solution = max(all_solutions, key=lambda s: s["target_prob"])

    return {
        "base_prob": base_prob,
        "base_probs": base_probs,
        "best_solution": best_solution,
        "all_solutions_found": all_solutions,
    }


def ChangeGenre(data, target_genre):
    evidence, bundle = get_bn_evidence_for_poster(data, genre_pickle)

    genre_model = bundle["model"]

    result = find_min_changes_for_target(
        model=genre_model,
        evidence=evidence,
        target_node="genre",
        target_value=target_genre,
        max_changes=4,
        threshold=0.50,
    )

    changed_nodes = result["best_solution"]["changed_nodes"]
    new_distribution = result["best_solution"]["full_probs"]

    return {"changed_nodes": changed_nodes, "new_distribution": new_distribution}


def ChangeRating(data, target_rating):
    evidence_r, bundle_r = get_bn_evidence_for_poster(data, rating_pickle)

    rating_model = bundle_r["model"]

    result = find_min_changes_for_target(
        model=rating_model,
        evidence=evidence_r,
        target_node="rating_bin",
        target_value=target_rating,
        max_changes=4,
        threshold=0.50,
    )

    changed_nodes = result["best_solution"]["changed_nodes"]
    new_distribution = result["best_solution"]["full_probs"]

    return {"changed_nodes": changed_nodes, "new_distribution": new_distribution}


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
# # Optionally, run this on startup. Otherwise, the first time you run this it'll take a bit longer
# initialize_feature_models()
#
# # First, get the features from your poster
# poster_path = "eeaaoposter.jpg"
# poster_data = acquire_poster_data(poster_path)
#
# # Now given those features, we can predict genre as so:
# probs = predict_genre_from_poster_dict(poster_data, genre_pickle)
# for genre, prob in sorted(probs.items(), key=lambda x: -x[1]):
#     print(f"{genre}: {prob:.4f}")
#
# # And likewise with predicting the movie's rating
# rating_probs = predict_rating_from_poster_dict(
#     poster_data,
#     rating_pickle,
# )
# print(rating_probs)
#
# # Getting the minimum nodes needed to change projected genre
# print(ChangeGenre(poster_data, 'Comedy'))
#
# # And minimum nodes to change projected rating
# print(ChangeRating(poster_data, 'high'))
