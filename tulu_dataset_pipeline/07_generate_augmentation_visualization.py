"""
=============================================================================
07_generate_augmentation_visualization.py
Tulu Kalpuga — IEEE Research Pipeline
Task: Generate Augmentation Visualization Figure

⚠️  Run this ONLY AFTER augmentation (06_augmentation.py) has been executed
    with user approval.

Purpose:
    - Show sequential augmentation flow:
      Original → Rotation → Scaling → Gaussian Noise → Gaussian Blur
    - Save publication-quality figures for IEEE paper:
      1. augmentation_pipeline_flow.png (Sequential pipeline flow figure)
      2. augmentation_examples.png      (Grid visualization with combined effects)

=============================================================================
"""

import sys
import random
from pathlib import Path

try:
    import cv2
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    print("[OK] Libraries loaded successfully.")
except ImportError as e:
    print(f"[FATAL] Missing library: {e}")
    sys.exit(1)

SCRIPT_DIR   = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR.parent
PIPELINE_DIR = SCRIPT_DIR
SPLIT_DIR    = DATASET_ROOT / "dataset_split"
TRAIN_ORIG   = SPLIT_DIR / "train_original"

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Augmentation parameters (matching 06_augmentation.py)
ROT_RANGE   = (-15.0, 15.0)
SCALE_RANGE = (0.9, 1.1)
NOISE_SIGMA = 10.0
BLUR_KERNEL = (3, 3)


def apply_rotation(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def apply_scaling(img, scale):
    h, w = img.shape[:2]
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if scale > 1.0:
        y0 = (new_h - h) // 2
        x0 = (new_w - w) // 2
        return scaled[y0:y0 + h, x0:x0 + w]
    else:
        pad_y = (h - new_h) // 2
        pad_x = (w - new_w) // 2
        return cv2.copyMakeBorder(scaled, pad_y, h - new_h - pad_y,
                                  pad_x, w - new_w - pad_x,
                                  cv2.BORDER_REFLECT_101)


def apply_noise(img, sigma, seed=42):
    """
    IEEE float32 implementation:
        n ~ N(0, σ²), σ = 10
        I' = clip(float32(I) + n, 0, 255)
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=img.shape)
    noisy = img.astype(np.float32) + noise.astype(np.float32)
    return np.clip(noisy, 0.0, 255.0).astype(np.uint8)


def apply_blur(img, kernel):
    return cv2.GaussianBlur(img, kernel, 0)


def bgr_to_rgb(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


print("\n" + "=" * 70)
print("  Tulu Kalpuga — Augmentation Visualization Generator")
print("=" * 70)

if not TRAIN_ORIG.exists():
    print("[ERROR] train_original directory not found. Run 04_create_split.py first.")
    sys.exit(1)

classes = sorted([d.name for d in TRAIN_ORIG.iterdir() if d.is_dir()])
sample_cls = "ka" if "ka" in classes else classes[0]
cls_dir = TRAIN_ORIG / sample_cls
sample_files = sorted([f for f in cls_dir.iterdir() if f.suffix.lower() in {".png", ".jpg"}])

if not sample_files:
    print(f"[ERROR] No images found in {cls_dir}")
    sys.exit(1)

sample_path = sample_files[0]
original_bgr = cv2.imread(str(sample_path))

if original_bgr is None:
    print(f"[ERROR] Could not read image: {sample_path}")
    sys.exit(1)

print(f"  Sample character class : '{sample_cls}'")
print(f"  Sample image file      : {sample_path.name}")

# Generate sequential pipeline steps
step0_orig  = original_bgr.copy()
step1_rot   = apply_rotation(step0_orig, 12.0)
step2_scale = apply_scaling(step1_rot, 1.08)
step3_noise = apply_noise(step2_scale, NOISE_SIGMA, seed=42)
step4_blur  = apply_blur(step3_noise, BLUR_KERNEL)

sequential_steps = [
    (bgr_to_rgb(step0_orig),  "1. Original",        "Input image (RGBA/RGB)\nReference character"),
    (bgr_to_rgb(step1_rot),   "2. Rotation",        "θ = +12.0°\nUniform(-15°, +15°)"),
    (bgr_to_rgb(step2_scale), "3. Scaling",         "s = 1.08\nUniform(0.9, 1.1)"),
    (bgr_to_rgb(step3_noise), "4. Gaussian Noise",  "σ = 10.0\nn ~ N(0, σ²) in float32"),
    (bgr_to_rgb(step4_blur),  "5. Gaussian Blur",   "3×3 kernel\nSpatial smoothing"),
]

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: SEQUENTIAL PIPELINE FLOW (Original → Rotation → Scaling → Noise → Blur)
# ─────────────────────────────────────────────────────────────────────────────

fig1 = plt.figure(figsize=(15, 3.8), facecolor="white")
gs1 = gridspec.GridSpec(1, 5, figure=fig1, wspace=0.35)

for idx, (img, title, subtitle) in enumerate(sequential_steps):
    ax = fig1.add_subplot(gs1[0, idx])
    ax.imshow(img)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8, color="#111111")
    ax.axis("off")
    
    # Border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#2b5c8f" if idx == 0 else "#666666")
        spine.set_linewidth(2.0 if idx == 0 else 1.2)

    # Subtitle text below image
    ax.text(0.5, -0.18, subtitle, transform=ax.transAxes,
            ha="center", va="top", fontsize=8.5, color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f7fa", edgecolor="#dcdcdc", lw=0.8))

    # Add Arrow between subplots
    if idx < 4:
        fig1.text(0.185 + idx * 0.191, 0.55, "➔", fontsize=22, fontweight="bold",
                  color="#2b5c8f", ha="center", va="center")

fig1.suptitle(
    f"IEEE Research Pipeline — Controlled Data Augmentation Flow (Class: '{sample_cls}')",
    fontsize=12, fontweight="bold", y=1.03, color="#111111"
)

plt.figtext(
    0.5, -0.08,
    "Augmentation strictly applied ONLY to train_original/. Validation (499) and Test (499) sets remain untouched.",
    ha="center", color="#555555", fontsize=8.5, style="italic"
)

flow_path = PIPELINE_DIR / "augmentation_pipeline_flow.png"
plt.savefig(str(flow_path), dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig1)

print(f"  [CREATED] Saved sequential flow figure: {flow_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: GRID VISUALIZATION FOR IEEE PAPER (Dark theme with individual & combinations)
# ─────────────────────────────────────────────────────────────────────────────

rot_only   = apply_rotation(original_bgr, 12.0)
scale_only = apply_scaling(original_bgr, 1.08)
noise_only = apply_noise(original_bgr, NOISE_SIGMA, seed=42)
blur_only  = apply_blur(original_bgr, BLUR_KERNEL)

comb1 = apply_noise(apply_rotation(original_bgr, -8.0), NOISE_SIGMA, seed=43)
comb2 = apply_blur(apply_scaling(apply_rotation(original_bgr, 10.0), 0.93), BLUR_KERNEL)
comb3 = apply_noise(apply_blur(apply_scaling(original_bgr, 1.05), BLUR_KERNEL), NOISE_SIGMA, seed=44)

grid_images = [
    (bgr_to_rgb(original_bgr), "Original Reference"),
    (bgr_to_rgb(rot_only),     "Rotation (θ = +12°)"),
    (bgr_to_rgb(scale_only),   "Scaling (s = 1.08)"),
    (bgr_to_rgb(noise_only),   f"Gaussian Noise (σ = {NOISE_SIGMA:.0f})"),
    (bgr_to_rgb(blur_only),    "Gaussian Blur (3×3)"),
    (bgr_to_rgb(comb1),        "Rot + Noise"),
    (bgr_to_rgb(comb2),        "Rot + Scale + Blur"),
    (bgr_to_rgb(comb3),        "Scale + Blur + Noise"),
]

fig2 = plt.figure(figsize=(16, 5), facecolor="#1a1a2e")
fig2.patch.set_facecolor("#1a1a2e")

gs2 = gridspec.GridSpec(2, 4, figure=fig2, hspace=0.5, wspace=0.3)

for idx, (img, label) in enumerate(grid_images):
    row = idx // 4
    col = idx % 4
    ax = fig2.add_subplot(gs2[row, col])
    ax.imshow(img)
    ax.set_title(label, color="white", fontsize=9.5, fontweight="bold", pad=6)
    ax.axis("off")

    for spine in ax.spines.values():
        spine.set_edgecolor("#00d4ff" if idx == 0 else "#e94560")
        spine.set_linewidth(2.5 if idx == 0 else 1.5)

fig2.suptitle(
    f"Tulu Lipi Data Augmentation Catalog — Class: '{sample_cls}'\n"
    "Individual Transformations & Multi-Stage Stochastic Combinations",
    color="white", fontsize=11.5, fontweight="bold", y=1.02
)

plt.figtext(
    0.5, -0.04,
    "Augmentation strictly applied ONLY to train_original/. Validation (499) and Test (499) sets remain untouched.",
    ha="center", color="#aaaaaa", fontsize=8
)

grid_path = PIPELINE_DIR / "augmentation_examples.png"
plt.savefig(str(grid_path), dpi=300, bbox_inches="tight", facecolor=fig2.get_facecolor())
plt.close(fig2)

print(f"  [CREATED] Saved grid visualization figure: {grid_path}")

print("\n" + "=" * 70)
print("  AUGMENTATION VISUALIZATION COMPLETE")
print("  Both publication-quality figures ready for paper inclusion.")
print("=" * 70 + "\n")
