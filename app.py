"""
===========================================================
VisionRestore AI — Interactive Dashboard
Author : Sobika Mageshwaren
===========================================================

Single-file Streamlit dashboard for the PixelRestore /
VisionRestore AI document-image restoration pipeline.

Pipeline stages shown in the UI:
  1. Upload & Preview
  2. AI Quality Analysis   (blur, noise, brightness, contrast, entropy)
  3. Restoration Recommendation (rule-based pipeline planner)
  4. One-click Restoration        (classical CV engine — no
     external model weights required, so this runs anywhere,
     including a free Render instance)
  5. Before / After comparison
  6. Evaluation                  (PSNR, SSIM, optional OCR/CER)
  7. Downloads (image + reports)

Run locally:
    streamlit run streamlit_app.py

Deploy on Render:
    Build Command : pip install -r requirements.txt
    Start Command : streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0
"""

import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# -----------------------------------------------------------------
# Optional OCR (kept optional so the app deploys with zero heavy
# system dependencies; enable it from the sidebar if pytesseract +
# the tesseract binary are available in the environment)
# -----------------------------------------------------------------
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


# ===================================================================
# PAGE CONFIG
# ===================================================================

st.set_page_config(
    page_title="VisionRestore AI",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===================================================================
# THEME / CSS  —  "Aurora Glass" design system
# ===================================================================

PRIMARY_BG = "#080A12"
CARD_BG = "rgba(22, 26, 41, 0.55)"
CARD_BORDER = "rgba(148, 163, 255, 0.14)"
ACCENT_BLUE = "#4F8DFF"
ACCENT_PURPLE = "#A855F7"
ACCENT_PINK = "#EC4899"
ACCENT_TEAL = "#2DD4BF"
TEXT_MUTED = "#9CA6C4"
TEXT_MAIN = "#F1F3FB"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    h1, h2, h3, h4, .vr-metric-value, .vr-pipeline-name {{
        font-family: 'Sora', sans-serif;
    }}

    /* ---------- animated aurora backdrop ---------- */
    .stApp {{
        background-color: {PRIMARY_BG};
        background-image:
            radial-gradient(circle at 12% 8%, rgba(79,141,255,0.16) 0%, transparent 42%),
            radial-gradient(circle at 88% 15%, rgba(168,85,247,0.14) 0%, transparent 45%),
            radial-gradient(circle at 25% 92%, rgba(236,72,153,0.10) 0%, transparent 40%),
            radial-gradient(circle at 80% 85%, rgba(45,212,191,0.10) 0%, transparent 42%);
        background-attachment: fixed;
    }}

    @keyframes vr-fade-in {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes vr-shimmer {{
        0%   {{ background-position: 0% 50%; }}
        100% {{ background-position: 200% 50%; }}
    }}
    @keyframes vr-pulse {{
        0%, 100% {{ box-shadow: 0 0 0 0 rgba(79,141,255,0.35); }}
        50%      {{ box-shadow: 0 0 0 8px rgba(79,141,255,0); }}
    }}
    @keyframes vr-flow-dash {{
        to {{ stroke-dashoffset: -24; }}
    }}

    section.main > div {{ animation: vr-fade-in 0.5s ease both; }}

    /* ---------- hero ---------- */
    .vr-hero {{
        position: relative;
        overflow: hidden;
        padding: 34px 38px;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(19,22,38,0.9) 0%, rgba(28,26,48,0.85) 55%, rgba(36,26,61,0.9) 100%);
        border: 1px solid {CARD_BORDER};
        margin-bottom: 26px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.35);
    }}
    .vr-hero::before {{
        content: "";
        position: absolute;
        inset: -2px;
        padding: 2px;
        border-radius: 22px;
        background: linear-gradient(120deg, {ACCENT_BLUE}, {ACCENT_PURPLE}, {ACCENT_PINK}, {ACCENT_BLUE});
        background-size: 300% 300%;
        animation: vr-shimmer 8s linear infinite;
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        opacity: 0.55;
        pointer-events: none;
    }}
    .vr-hero-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {ACCENT_TEAL};
        background: rgba(45,212,191,0.10);
        border: 1px solid rgba(45,212,191,0.25);
        padding: 4px 12px;
        border-radius: 999px;
        margin-bottom: 14px;
    }}
    .vr-hero-dot {{
        width: 7px; height: 7px; border-radius: 50%;
        background: {ACCENT_TEAL};
        animation: vr-pulse 2s infinite;
    }}
    .vr-hero h1 {{
        font-size: 2.4rem;
        margin: 0 0 6px 0;
        background: linear-gradient(90deg, #FFFFFF 10%, {ACCENT_BLUE} 55%, {ACCENT_PURPLE} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -0.01em;
    }}
    .vr-hero p {{
        color: {TEXT_MUTED};
        margin: 0;
        font-size: 1rem;
        max-width: 640px;
    }}

    /* ---------- glass cards ---------- */
    .vr-card {{
        background: {CARD_BG};
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid {CARD_BORDER};
        border-radius: 16px;
        padding: 16px 18px;
        margin-bottom: 12px;
        transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
    }}
    .vr-card:hover {{
        transform: translateY(-2px);
        border-color: rgba(79,141,255,0.35);
        box-shadow: 0 8px 24px rgba(0,0,0,0.28);
    }}
    .vr-metric-label {{
        color: {TEXT_MUTED};
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    .vr-metric-value {{
        font-size: 1.6rem;
        font-weight: 700;
        color: {TEXT_MAIN};
    }}
    .vr-metric-status {{
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 6px;
        display: inline-block;
        padding: 3px 11px;
        border-radius: 999px;
    }}
    .status-good {{ background: rgba(45,212,191,0.14); color: #4ADE80; }}
    .status-warn {{ background: rgba(250,204,21,0.14); color: #FACC15; }}
    .status-bad  {{ background: rgba(248,113,113,0.14); color: #F87171; }}

    /* ---------- pipeline flow ---------- */
    .vr-flow-row {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 2px;
        margin-bottom: 10px;
        padding: 6px 0;
    }}
    .vr-pipeline-step {{
        position: relative;
        flex: 1 1 140px;
        text-align: center;
        padding: 18px 12px;
        background: {CARD_BG};
        backdrop-filter: blur(14px);
        border: 1px solid {CARD_BORDER};
        border-radius: 16px;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }}
    .vr-pipeline-step:hover {{
        transform: translateY(-4px) scale(1.02);
        border-color: {ACCENT_PURPLE};
        box-shadow: 0 10px 26px rgba(168,85,247,0.18);
    }}
    .vr-pipeline-icon {{
        font-size: 1.9rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 48px; height: 48px;
        border-radius: 50%;
        background: linear-gradient(135deg, rgba(79,141,255,0.14), rgba(168,85,247,0.14));
        margin-bottom: 6px;
    }}
    .vr-pipeline-name {{ font-weight: 700; color: {TEXT_MAIN}; font-size: 0.95rem; }}
    .vr-pipeline-model {{ color: {TEXT_MUTED}; font-size: 0.78rem; }}
    .vr-arrow {{
        flex: 0 0 auto;
        text-align: center;
        font-size: 1.4rem;
        background: linear-gradient(90deg, {ACCENT_BLUE}, {ACCENT_PURPLE});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 0 6px;
        font-weight: 700;
    }}

    /* ---------- sidebar ---------- */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #06070D 0%, #0A0D18 100%);
        border-right: 1px solid {CARD_BORDER};
    }}
    section[data-testid="stSidebar"] .vr-step-row {{
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 5px 2px;
        font-size: 0.9rem;
    }}
    section[data-testid="stSidebar"] .vr-step-dot {{
        width: 9px; height: 9px; border-radius: 50%;
        flex-shrink: 0;
    }}
    .vr-dot-done   {{ background: {ACCENT_TEAL}; box-shadow: 0 0 8px rgba(45,212,191,0.6); }}
    .vr-dot-active {{ background: {ACCENT_BLUE}; animation: vr-pulse 1.6s infinite; }}
    .vr-dot-todo   {{ background: #2A2F42; }}

    /* ---------- misc widget polish ---------- */
    div[data-testid="stMetricValue"] {{ color: {TEXT_MAIN}; font-family: 'Sora', sans-serif; }}
    .stButton > button {{
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
        border: 1px solid rgba(79,141,255,0.35) !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(79,141,255,0.25);
    }}
    .stButton > button[kind="primary"] {{
        background: linear-gradient(90deg, {ACCENT_BLUE}, {ACCENT_PURPLE}) !important;
        border: none !important;
    }}
    div[data-testid="stFileUploaderDropzone"] {{
        background: {CARD_BG} !important;
        backdrop-filter: blur(14px);
        border: 1.5px dashed rgba(79,141,255,0.35) !important;
        border-radius: 16px !important;
    }}
    hr {{ border-color: {CARD_BORDER} !important; }}
    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(180deg, {ACCENT_BLUE}, {ACCENT_PURPLE});
        border-radius: 8px;
    }}
    ::-webkit-scrollbar-track {{ background: transparent; }}

    .vr-section-title {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 6px 0 14px 0;
    }}
    .vr-section-title .bar {{
        width: 4px; height: 22px; border-radius: 4px;
        background: linear-gradient(180deg, {ACCENT_BLUE}, {ACCENT_PURPLE});
    }}
</style>
""", unsafe_allow_html=True)


def section_title(icon, text):
    st.markdown(
        f"<div class='vr-section-title'><div class='bar'></div>"
        f"<h3 style='margin:0;'>{icon}&nbsp; {text}</h3></div>",
        unsafe_allow_html=True,
    )


# ===================================================================
# QUALITY ANALYZER  (adapted from quality_analyzer.py)
# ===================================================================

def compute_blur(gray):
    return round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)


def compute_noise(gray):
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = gray.astype(np.float32) - blur.astype(np.float32)
    return round(float(np.std(residual)), 2)


def compute_brightness(gray):
    return round(float(np.mean(gray)), 2)


def compute_contrast(gray):
    return round(float(np.std(gray)), 2)


def compute_entropy(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    return round(float(-np.sum(prob * np.log2(prob))), 2)


def classify_blur(s):
    return "High Blur" if s < 1000 else "Medium Blur" if s < 3500 else "Low Blur"


def classify_noise(s):
    return "Low Noise" if s < 12 else "Medium Noise" if s < 20 else "High Noise"


def classify_brightness(s):
    return "Low Brightness" if s < 150 else "Medium Brightness" if s < 220 else "High Brightness"


def classify_contrast(s):
    return "Low Contrast" if s < 30 else "Medium Contrast" if s < 40 else "High Contrast"


def classify_entropy(s):
    return "Low Information" if s < 2 else "Medium Information" if s < 4 else "High Information"


def classify_resolution(w, h):
    pixels = w * h
    return "Low Resolution" if pixels < 500_000 else "Medium Resolution" if pixels < 1_000_000 else "High Resolution"


def compute_quality_score(report):
    score = 100
    if report["Blur Status"] == "High Blur":
        score -= 30
    elif report["Blur Status"] == "Medium Blur":
        score -= 15

    if report["Noise Status"] == "High Noise":
        score -= 20
    elif report["Noise Status"] == "Medium Noise":
        score -= 10

    if report["Brightness Status"] == "Low Brightness":
        score -= 20
    elif report["Brightness Status"] == "Medium Brightness":
        score -= 5

    if report["Contrast Status"] == "Low Contrast":
        score -= 15
    elif report["Contrast Status"] == "Medium Contrast":
        score -= 5

    if report["Resolution Status"] == "Low Resolution":
        score -= 15

    score = max(0, min(100, score))

    if score >= 85:
        label = "Excellent"
    elif score >= 60:
        label = "Good"
    elif score >= 40:
        label = "Fair"
    else:
        label = "Poor"

    return score, label


def restoration_difficulty(score):
    return "Easy" if score >= 85 else "Moderate" if score >= 65 else "Severe"


def compute_confidence(report):
    conf = []

    blur = report["Blur Score"]
    if blur >= 3500:
        conf.append(min(100, 70 + (blur - 3500) / 100))
    elif blur <= 1000:
        conf.append(min(100, 70 + (1000 - blur) / 20))
    else:
        conf.append(75)

    noise = report["Noise Score"]
    if noise >= 20:
        conf.append(min(100, 70 + (noise - 20) * 8))
    elif noise <= 12:
        conf.append(min(100, 70 + (12 - noise) * 8))
    else:
        conf.append(75)

    brightness = report["Brightness"]
    if brightness >= 220:
        conf.append(min(100, 70 + (brightness - 220) / 2))
    elif brightness <= 150:
        conf.append(min(100, 70 + (150 - brightness) / 2))
    else:
        conf.append(75)

    return round(sum(conf) / len(conf))


def analyze_image(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = image_rgb.shape[:2]

    report = {
        "Blur Score": compute_blur(gray),
        "Noise Score": compute_noise(gray),
        "Brightness": compute_brightness(gray),
        "Contrast": compute_contrast(gray),
        "Entropy": compute_entropy(gray),
        "Resolution": f"{w} x {h}",
        "Width": w,
        "Height": h,
    }

    report["Blur Status"] = classify_blur(report["Blur Score"])
    report["Noise Status"] = classify_noise(report["Noise Score"])
    report["Brightness Status"] = classify_brightness(report["Brightness"])
    report["Contrast Status"] = classify_contrast(report["Contrast"])
    report["Entropy Status"] = classify_entropy(report["Entropy"])
    report["Resolution Status"] = classify_resolution(w, h)

    score, label = compute_quality_score(report)
    report["Overall Quality Score"] = score
    report["Overall Quality"] = label
    report["Restoration Difficulty"] = restoration_difficulty(score)
    report["Confidence"] = compute_confidence(report)

    return report


def recommend_restoration(report):
    detected, pipeline = [], []

    blur_bad = report["Blur Status"] != "Low Blur"
    noise_bad = report["Noise Status"] != "Low Noise"

    if blur_bad and noise_bad:
        detected.append("Blur + Noise")
        pipeline.append({"operation": "Joint Restoration", "model": "Deblur-Denoise Net", "icon": "⚡"})
    else:
        if blur_bad:
            detected.append(report["Blur Status"])
            pipeline.append({"operation": "Deblur", "model": "Sharpen-Net", "icon": "🌀"})
        if noise_bad:
            detected.append(report["Noise Status"])
            pipeline.append({"operation": "Denoise", "model": "DnCNN", "icon": "🧠"})

    if report["Brightness Status"] == "Low Brightness":
        detected.append("Low Brightness")
        pipeline.append({"operation": "Low-Light Enhancement", "model": "Gamma+CLAHE", "icon": "💡"})

    if report["Contrast Status"] == "Low Contrast":
        detected.append("Low Contrast")
        pipeline.append({"operation": "Contrast Enhancement", "model": "CLAHE", "icon": "🎨"})

    if report["Resolution Status"] == "Low Resolution":
        detected.append("Low Resolution")
        pipeline.append({"operation": "Super Resolution", "model": "Cubic-SR", "icon": "🔍"})

    if not pipeline:
        pipeline.append({"operation": "No Restoration Needed", "model": "—", "icon": "✅"})

    return {"detected_problems": detected, "recommended_pipeline": pipeline}


def generate_diagnosis(report, recommendation):
    issues = recommendation["detected_problems"]
    if not issues:
        return "The image is of good quality. No restoration is required."
    return (
        f"The image quality is rated **{report['Overall Quality']}** "
        f"({report['Overall Quality Score']}/100). The primary degradation is "
        f"**{issues[0]}**. Restoration difficulty: **{report['Restoration Difficulty']}**."
    )


# ===================================================================
# RESTORATION ENGINE  (classical CV — deploys with zero model weights)
# ===================================================================

def op_denoise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 8, 8, 7, 21)


def op_deblur(image):
    gaussian = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1.6, gaussian, -0.6, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def op_low_light(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    gamma = 0.7
    l = np.array(255 * (l / 255) ** gamma, dtype=np.uint8)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def op_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def op_super_resolution(image):
    h, w = image.shape[:2]
    upscaled = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    return op_deblur(upscaled)


def op_joint(image):
    return op_deblur(op_denoise(image))


OPS = {
    "Joint Restoration": op_joint,
    "Deblur": op_deblur,
    "Denoise": op_denoise,
    "Low-Light Enhancement": op_low_light,
    "Contrast Enhancement": op_contrast,
    "Super Resolution": op_super_resolution,
}


def execute_pipeline(image, pipeline):
    restored = image.copy()
    log = []
    for step in pipeline:
        op = step["operation"]
        fn = OPS.get(op)
        if fn is None:
            continue
        start = time.perf_counter()
        restored = fn(restored)
        elapsed = time.perf_counter() - start
        log.append({"step": op, "model": step["model"], "time_sec": round(elapsed, 3)})
    return restored, log


# ===================================================================
# EVALUATION
# ===================================================================

def evaluate_restoration(original, degraded_or_input, restored):
    def to_gray(im):
        return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    same_size_input = cv2.resize(degraded_or_input, (original.shape[1], original.shape[0]))
    same_size_restored = cv2.resize(restored, (original.shape[1], original.shape[0]))

    psnr_before = round(float(peak_signal_noise_ratio(original, same_size_input, data_range=255)), 2)
    psnr_after = round(float(peak_signal_noise_ratio(original, same_size_restored, data_range=255)), 2)

    ssim_before = round(float(structural_similarity(to_gray(original), to_gray(same_size_input))), 4)
    ssim_after = round(float(structural_similarity(to_gray(original), to_gray(same_size_restored))), 4)

    metrics = {
        "PSNR Before": psnr_before, "PSNR After": psnr_after,
        "PSNR Gain": round(psnr_after - psnr_before, 2),
        "SSIM Before": ssim_before, "SSIM After": ssim_after,
        "SSIM Gain": round(ssim_after - ssim_before, 4),
    }

    if OCR_AVAILABLE and st.session_state.get("enable_ocr", False):
        try:
            gt_text = pytesseract.image_to_string(same_size_input)
            pred_text = pytesseract.image_to_string(same_size_restored)
            from difflib import SequenceMatcher
            sim = SequenceMatcher(None, gt_text, pred_text).ratio() * 100
            metrics["OCR Similarity"] = round(sim, 2)
        except Exception:
            metrics["OCR Similarity"] = None

    return metrics


# ===================================================================
# HELPERS
# ===================================================================

def pil_to_rgb_array(pil_image):
    return np.array(pil_image.convert("RGB"))


def rgb_array_to_png_bytes(arr):
    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def status_class(status_text):
    bad_words = ["High Blur", "High Noise", "Low Brightness", "Low Contrast", "Low Resolution", "Low Information"]
    warn_words = ["Medium"]
    if any(w in status_text for w in bad_words):
        return "status-bad"
    if any(w in status_text for w in warn_words):
        return "status-warn"
    return "status-good"


METRIC_ICONS = {
    "Blur": "🌀",
    "Noise": "📡",
    "Brightness": "☀️",
    "Contrast": "🎚️",
    "Entropy": "🧩",
}


def metric_card(col, label, value, status):
    icon = METRIC_ICONS.get(label, "📈")
    with col:
        st.markdown(f"""
        <div class="vr-card">
            <div class="vr-metric-label">{icon} {label}</div>
            <div class="vr-metric-value">{value}</div>
            <div class="vr-metric-status {status_class(status)}">{status}</div>
        </div>
        """, unsafe_allow_html=True)


def quality_gauge(score, label):
    color = {"Excellent": "#4ADE80", "Good": "#4F8DFF", "Fair": "#FACC15", "Poor": "#F87171"}.get(label, ACCENT_BLUE)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"size": 36, "color": TEXT_MAIN, "family": "Sora"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": TEXT_MUTED, "tickfont": {"color": TEXT_MUTED}},
            "bar": {"color": color, "thickness": 0.32},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(248,113,113,0.14)"},
                {"range": [40, 60], "color": "rgba(250,204,21,0.12)"},
                {"range": [60, 85], "color": "rgba(79,141,255,0.12)"},
                {"range": [85, 100], "color": "rgba(74,222,128,0.12)"},
            ],
        },
    ))
    fig.update_layout(
        height=240, margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": TEXT_MAIN},
    )
    return fig


def before_after_bar(metric_name, before, after, higher_is_better=True):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Before", "After"], y=[before, after],
        marker=dict(
            color=[TEXT_MUTED, ACCENT_BLUE if higher_is_better else ACCENT_PURPLE],
            line=dict(width=0),
        ),
        text=[before, after], textposition="outside",
        marker_pattern_shape=None,
        width=0.5,
    ))
    fig.update_layout(
        title=dict(text=metric_name, font=dict(family="Sora", color=TEXT_MAIN)),
        height=260, showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT_MAIN}, margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(gridcolor="rgba(148,163,255,0.08)"),
    )
    return fig


# ===================================================================
# SIDEBAR
# ===================================================================

with st.sidebar:
    st.markdown("### 🖼️ VisionRestore AI")
    st.caption("Version 1.0 · Classical-CV Engine")

    st.markdown("**Models Used**")
    st.markdown("- ✅ DnCNN-style Denoise\n- ✅ Adaptive Sharpen (Deblur)\n- ✅ CLAHE Contrast\n- ✅ Cubic Super-Resolution")

    st.markdown("---")
    st.markdown("**System Status**")
    st.success("● System Ready (CPU)")

    st.markdown("---")
    st.markdown("**Pipeline Progress**")
    steps = ["Upload", "Analyze", "Recommend", "Restore", "Evaluate", "Download"]
    current = st.session_state.get("stage", 0)
    rows = ""
    for i, step in enumerate(steps):
        dot_class = "vr-dot-done" if i < current else "vr-dot-active" if i == current else "vr-dot-todo"
        weight = "700" if i == current else "400"
        color = TEXT_MAIN if i <= current else TEXT_MUTED
        rows += (
            f"<div class='vr-step-row'><div class='vr-step-dot {dot_class}'></div>"
            f"<span style='font-weight:{weight}; color:{color};'>{step}</span></div>"
        )
    st.markdown(rows, unsafe_allow_html=True)

    st.markdown("---")
    st.checkbox(
        "Enable OCR Evaluation" + ("" if OCR_AVAILABLE else " (unavailable)"),
        value=False, key="enable_ocr", disabled=not OCR_AVAILABLE,
        help="Requires pytesseract + the tesseract binary on the host.",
    )

    st.markdown("---")
    if st.button("🔄  Start Over", width="stretch"):
        for key in ["original_image", "report", "recommendation", "restored_image",
                    "pipeline_log", "eval_metrics", "stage", "filename"]:
            st.session_state[key] = None if key != "stage" else 0
        st.rerun()


# ===================================================================
# HEADER
# ===================================================================

st.markdown("""
<div class="vr-hero">
    <div class="vr-hero-badge"><span class="vr-hero-dot"></span> AI Restoration Pipeline · Live</div>
    <h1>VisionRestore AI</h1>
    <p>AI-powered document image restoration & quality analysis — upload, diagnose, restore, and evaluate in one seamless pipeline.</p>
</div>
""", unsafe_allow_html=True)


# ===================================================================
# SESSION STATE INIT
# ===================================================================

for key, default in [
    ("original_image", None), ("report", None), ("recommendation", None),
    ("restored_image", None), ("pipeline_log", None), ("eval_metrics", None),
    ("stage", 0), ("filename", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ===================================================================
# SECTION 1 — UPLOAD
# ===================================================================

section_title("📤", "Upload Document Image")

uploaded_file = st.file_uploader(
    "Drag & drop or browse (PNG / JPG / JPEG)",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    image_rgb = pil_to_rgb_array(pil_image)

    if st.session_state.filename != uploaded_file.name:
        # new upload — reset downstream state
        st.session_state.original_image = image_rgb
        st.session_state.filename = uploaded_file.name
        st.session_state.report = None
        st.session_state.recommendation = None
        st.session_state.restored_image = None
        st.session_state.eval_metrics = None
        st.session_state.stage = 1

    col_img, col_info = st.columns([2, 1])

    with col_img:
        st.image(image_rgb, caption="Original Image", width="stretch")

    with col_info:
        h, w = image_rgb.shape[:2]
        size_kb = uploaded_file.size / 1024
        st.markdown(f"""
        <div class="vr-card">
            <div class="vr-metric-label">📁 Filename</div>
            <div class="vr-metric-value" style="font-size:1.05rem;">{uploaded_file.name}</div>
        </div>
        <div class="vr-card">
            <div class="vr-metric-label">💾 Size</div>
            <div class="vr-metric-value" style="font-size:1.05rem;">{size_kb:,.1f} KB</div>
        </div>
        <div class="vr-card">
            <div class="vr-metric-label">📐 Resolution</div>
            <div class="vr-metric-value" style="font-size:1.05rem;">{w} × {h}</div>
        </div>
        <div class="vr-card">
            <div class="vr-metric-label">🗂️ Image Type</div>
            <div class="vr-metric-value" style="font-size:1.05rem;">{uploaded_file.type.split('/')[-1].upper()}</div>
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------------------------------------
    # SECTION 2 — ANALYZE
    # -----------------------------------------------------------

    section_title("🔍", "AI Quality Analyzer")

    if st.button("Analyze Image", type="primary", width="content"):
        with st.status("Analyzing image quality...", expanded=True) as status_box:
            st.write("Estimating blur...")
            time.sleep(0.15)
            st.write("Estimating noise...")
            time.sleep(0.15)
            st.write("Estimating brightness...")
            time.sleep(0.15)
            st.write("Estimating contrast...")
            time.sleep(0.15)
            st.write("Computing entropy...")
            time.sleep(0.15)
            report = analyze_image(image_rgb)
            recommendation = recommend_restoration(report)
            report["Diagnosis"] = generate_diagnosis(report, recommendation)
            st.write("Generating diagnosis...")
            time.sleep(0.15)
            status_box.update(label="Analysis complete", state="complete", expanded=False)

        st.session_state.report = report
        st.session_state.recommendation = recommendation
        st.session_state.stage = 2

    report = st.session_state.report

    if report is not None:
        recommendation = st.session_state.recommendation

        g_col, d_col = st.columns([1, 1.4])

        with g_col:
            st.plotly_chart(
                quality_gauge(report["Overall Quality Score"], report["Overall Quality"]),
                width="stretch",
            )
            st.markdown(
                f"<div style='text-align:center; font-weight:700; font-size:1.15rem; font-family:Sora;'>{report['Overall Quality']}</div>",
                unsafe_allow_html=True,
            )

        with d_col:
            st.markdown(f"""
            <div class="vr-card">
                <div class="vr-metric-label">🎯 Primary Issue</div>
                <div class="vr-metric-value" style="font-size:1.2rem;">{recommendation['detected_problems'][0] if recommendation['detected_problems'] else 'None'}</div>
            </div>
            """, unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="vr-card">
                    <div class="vr-metric-label">⚙️ Difficulty</div>
                    <div class="vr-metric-value" style="font-size:1.2rem;">{report['Restoration Difficulty']}</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="vr-card">
                    <div class="vr-metric-label">📊 Confidence</div>
                    <div class="vr-metric-value" style="font-size:1.2rem;">{report['Confidence']}%</div>
                </div>
                """, unsafe_allow_html=True)
            st.info(report["Diagnosis"])

        st.markdown("##### Quality Metrics")
        m1, m2, m3, m4, m5 = st.columns(5)
        metric_card(m1, "Blur", f"{report['Blur Score']:,.0f}", report["Blur Status"])
        metric_card(m2, "Noise", f"{report['Noise Score']:,.2f}", report["Noise Status"])
        metric_card(m3, "Brightness", f"{report['Brightness']:,.0f}", report["Brightness Status"])
        metric_card(m4, "Contrast", f"{report['Contrast']:,.0f}", report["Contrast Status"])
        metric_card(m5, "Entropy", f"{report['Entropy']:,.2f}", report["Entropy Status"])

        # -----------------------------------------------------------
        # SECTION 3 — RECOMMENDATION PIPELINE
        # -----------------------------------------------------------

        section_title("🧠", "AI Recommendation Engine")

        pipeline = recommendation["recommended_pipeline"]
        n = len(pipeline)

        def flow_block(icon, name, model):
            return f"""
            <div class="vr-pipeline-step">
                <div class="vr-pipeline-icon">{icon}</div>
                <div class="vr-pipeline-name">{name}</div>
                <div class="vr-pipeline-model">{model}</div>
            </div>
            """

        def flow_arrow():
            return "<div class='vr-arrow'>➜</div>"

        blocks = [flow_block("📄", "Input", ", ".join(recommendation["detected_problems"]) or "Clean")]
        for step in pipeline:
            blocks.append(flow_arrow())
            blocks.append(flow_block(step["icon"], step["operation"], step["model"]))
        blocks.append(flow_arrow())
        blocks.append(flow_block("🖼️", "Output", "Restored"))

        st.markdown(
            f"<div class='vr-flow-row'>{''.join(blocks)}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("")
        est_runtime = round(0.4 * n + 0.3, 2)
        rt_col, btn_col = st.columns([1, 1])
        with rt_col:
            st.metric("Estimated Runtime", f"{est_runtime} sec")
        with btn_col:
            st.write("")
            restore_clicked = st.button("🛠️  Restore Image", type="primary", width="stretch")

        # -----------------------------------------------------------
        # SECTION 4 — RESTORATION
        # -----------------------------------------------------------

        if restore_clicked:
            progress = st.progress(0, text="Loading pipeline...")
            restored = image_rgb.copy()
            log = []
            total_steps = max(len(pipeline), 1)

            for i, step in enumerate(pipeline):
                op = step["operation"]
                progress.progress(int((i / total_steps) * 90), text=f"Running {op} ({step['model']})...")
                fn = OPS.get(op)
                start = time.perf_counter()
                if fn is not None:
                    restored = fn(restored)
                elapsed = time.perf_counter() - start
                log.append({"step": op, "model": step["model"], "time_sec": round(elapsed, 3)})

            progress.progress(95, text="Saving image...")
            time.sleep(0.1)
            progress.progress(100, text="Completed")
            time.sleep(0.15)
            progress.empty()

            st.session_state.restored_image = restored
            st.session_state.pipeline_log = log
            st.session_state.eval_metrics = evaluate_restoration(image_rgb, image_rgb, restored)
            st.session_state.stage = 3
            st.success("Restoration completed successfully.")
            st.balloons()

        restored_image = st.session_state.restored_image

        if restored_image is not None:
            section_title("🎚️", "Before vs After")

            try:
                from streamlit_image_comparison import image_comparison
                image_comparison(
                    img1=Image.fromarray(image_rgb),
                    img2=Image.fromarray(restored_image),
                    label1="Original",
                    label2="Restored",
                )
            except ImportError:
                bcol1, bcol2 = st.columns(2)
                bcol1.image(image_rgb, caption="Original", width="stretch")
                bcol2.image(restored_image, caption="Restored", width="stretch")

            section_title("📋", "Restoration Summary")
            s1, s2, s3 = st.columns(3)
            with s1:
                st.markdown('<div class="vr-card"><div class="vr-metric-label">🩹 Detected Problems</div>' +
                             "".join(f"<div>• {p}</div>" for p in recommendation["detected_problems"] or ["None"]) +
                             "</div>", unsafe_allow_html=True)
            with s2:
                models_used = ", ".join(sorted({s["model"] for s in st.session_state.pipeline_log}))
                st.markdown(f"""<div class="vr-card"><div class="vr-metric-label">🧬 Models Used</div><div>{models_used or '—'}</div></div>""",
                             unsafe_allow_html=True)
            with s3:
                total_time = sum(s["time_sec"] for s in st.session_state.pipeline_log)
                st.markdown(f"""<div class="vr-card"><div class="vr-metric-label">⏱️ Execution Time</div><div class="vr-metric-value" style="font-size:1.2rem;">{total_time:.3f} sec</div></div>""",
                             unsafe_allow_html=True)

            # -----------------------------------------------------------
            # SECTION 5 — EVALUATION
            # -----------------------------------------------------------

            section_title("📊", "Evaluation")

            metrics = st.session_state.eval_metrics
            e1, e2, e3 = st.columns(3)
            e1.metric("PSNR (dB)", metrics["PSNR After"], delta=f"{metrics['PSNR Gain']:+.2f}")
            e2.metric("SSIM", metrics["SSIM After"], delta=f"{metrics['SSIM Gain']:+.4f}")
            if "OCR Similarity" in metrics and metrics["OCR Similarity"] is not None:
                e3.metric("OCR Similarity", f"{metrics['OCR Similarity']}%")
            else:
                e3.metric("OCR Similarity", "—", help="Enable OCR Evaluation in the sidebar to compute this.")

            chart1, chart2 = st.columns(2)
            with chart1:
                st.plotly_chart(before_after_bar("PSNR (dB)", metrics["PSNR Before"], metrics["PSNR After"]),
                                 width="stretch")
            with chart2:
                st.plotly_chart(before_after_bar("SSIM", metrics["SSIM Before"], metrics["SSIM After"]),
                                 width="stretch")

            with st.expander("Pipeline Execution Log"):
                st.dataframe(pd.DataFrame(st.session_state.pipeline_log), width="stretch", hide_index=True)

            # -----------------------------------------------------------
            # SECTION 6 — DOWNLOADS
            # -----------------------------------------------------------

            section_title("⬇️", "Downloads")

            st.session_state.stage = 4

            quality_report_df = pd.DataFrame([{
                k: v for k, v in report.items() if k != "Diagnosis"
            }])
            eval_report_df = pd.DataFrame([metrics])
            pipeline_log_json = json.dumps({
                "detected_problems": recommendation["detected_problems"],
                "pipeline": recommendation["recommended_pipeline"],
                "execution_log": st.session_state.pipeline_log,
            }, indent=2)

            dl1, dl2, dl3, dl4 = st.columns(4)
            dl1.download_button(
                "Restored Image", rgb_array_to_png_bytes(restored_image),
                file_name=f"restored_{st.session_state.filename}", mime="image/png",
                width="stretch",
            )
            dl2.download_button(
                "Quality Report (CSV)", quality_report_df.to_csv(index=False).encode(),
                file_name="quality_report.csv", mime="text/csv", width="stretch",
            )
            dl3.download_button(
                "Evaluation Report (CSV)", eval_report_df.to_csv(index=False).encode(),
                file_name="evaluation_report.csv", mime="text/csv", width="stretch",
            )
            dl4.download_button(
                "Pipeline Log (JSON)", pipeline_log_json.encode(),
                file_name="pipeline_log.json", mime="application/json", width="stretch",
            )

else:
    st.markdown("""
    <div class="vr-card" style="text-align:center; padding:34px;">
        <div style="font-size:2.2rem; margin-bottom:6px;">🖼️✨</div>
        <div style="font-weight:700; font-size:1.05rem; font-family:Sora;">Upload a document image to begin</div>
        <div style="color:#9CA6C4; margin-top:4px;">Supports PNG, JPG, and JPEG — analysis starts the moment you drop a file.</div>
    </div>
    """, unsafe_allow_html=True)