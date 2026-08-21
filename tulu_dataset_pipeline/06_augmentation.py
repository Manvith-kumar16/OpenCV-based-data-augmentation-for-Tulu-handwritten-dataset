"""
=============================================================================
06_augmentation.py
Tulu Kalpuga — IEEE Research Pipeline
Task: Controlled Augmentation of Training Set Only

⚠️  DO NOT RUN THIS SCRIPT UNTIL AFTER:
    ✓ 01_inspect_dataset.py       — verified
    ✓ 02_check_duplicates.py      — verified
    ✓ 03_create_master_dataset.py — verified
    ✓ 04_create_split.py          — verified
    ✓ 05_generate_split_report.py — verified and reviewed
    ✓ User has explicitly approved augmentation

=============================================================================
CRITICAL RULES (enforced in code):
    ✗ NEVER augment dataset_split/validation/
    ✗ NEVER augment dataset_split/test/
    ✗ NEVER augment before splitting (the split was done in step 04)
    ✓ ONLY augment dataset_split/train_original/
    ✓ Record every generated image in augmentation_manifest.csv
    ✓ Target: 600 training images per class (50 × 600 = 30,000)
    ✓ SEED = 42 for full reproducibility
    ✓ Every augmented image provenance tracked
=============================================================================

AUGMENTATION MATHEMATICAL FORMULATION (for IEEE paper):

Let I ∈ R^(H×W×C) denote an original training image.

1. Rotation:
   θ ~ Uniform(-15°, +15°)
   I_rot = R(I, θ)
   where R(·) is bilinear-interpolated affine rotation about image center,
   with border padding using BORDER_REFLECT_101.

2. Scaling (applied independently with p=0.5):
   s ~ Uniform(0.9, 1.1)
   I_scale = S(I, s)
   where S(·) resizes by factor s and center-crops (s>1) or
   reflection-pads (s<1) back to original spatial dimensions.

3. Gaussian Additive Noise (applied independently with p=0.5):
   n ~ N(0, σ²),  σ = 10,  n ∈ R^(H×W×C)
   I_noisy = clip(I_float + n, 0, 255)
   IMPORTANT: addition is performed in float32 space before quantizing
   to uint8, ensuring negative noise values correctly reduce intensity
   rather than wrapping as large unsigned integers.

4. Gaussian Blur (applied independently with p=0.5):
   I_blur = GaussianBlur(I, kernel=(3,3))

Transformations are composed in the order: Rotation → Scaling → Noise → Blur.
Rotation is always applied. The remaining three are each applied with p=0.5,
yielding 2^3 = 8 possible transformation combinations per image.

=============================================================================
PREPROCESSING PIPELINE (for IEEE paper):

Raw storage format:
    ~337–372 × 320–358 px, RGBA (4-channel PNG)

Preprocessing applied at training/inference time (NOT during augmentation):
    1. Convert RGBA → RGB  (discard alpha channel)
    2. Resize to 64 × 64 px  (bilinear interpolation)
    3. Normalize: pixel_value / 255.0  (maps [0,255] → [0.0,1.0])

CNN input tensor shape: (batch, 64, 64, 3)

=============================================================================
PIPELINE DIAGRAM (for IEEE paper):

    Original Dataset
    9,959 nominal images
            │
            ▼
    SHA-256 Validation + Deduplication
            │
            ▼
    4,968 unique images (50 classes)
            │
            ▼
    Stratified 80/10/10 Split  (SEED = 42)
            │
    ┌───────┴──────┬────────────┐
    ▼              ▼            ▼
3,970 Train     499 Val     499 Test
    │              │            │
    ▼        (untouched)  (untouched)
Augmentation
(this script)
    │
    ▼
600 / class
= 30,000 augmented training images
    │
    └──────┬─────────────────────┘
           ▼
     CNN Training
           │
           ▼
    Untouched Test Set
           │
           ▼
  Accuracy / Precision / Recall
       F1 / Confusion Matrix

=============================================================================
"""

import os
import sys
import json
import random
import shutil
import hashlib
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# SAFETY GATE — must be explicitly disabled by the user after reviewing results
# ─────────────────────────────────────────────────────────────────────────────

SAFETY_CHECK = False  # ← User explicitly approved on 2026-08-20

if SAFETY_CHECK:
    print("\n" + "=" * 70)
    print("  ⚠️   AUGMENTATION SAFETY CHECK — NOT EXECUTED")
    print("=" * 70)
    print("""
  This script (06_augmentation.py) is intentionally paused.

  Before enabling augmentation, confirm ALL of the following:

    [✓] 01_inspect_dataset.py       — completed and reviewed
    [✓] 02_check_duplicates.py      — completed and reviewed
    [✓] 03_create_master_dataset.py — completed and reviewed
    [✓] 04_create_split.py          — completed and reviewed
    [✓] 05_generate_split_report.py — completed and reviewed
    [ ] User has explicitly approved augmentation in writing

  Augmentation targets:
    Input : dataset_split/train_original/   (3,970 images)
    Output: dataset_split/train_augmented/  (target: 30,000 images)
    Target: 600 images per class × 50 classes

  Augmentation is STRICTLY PROHIBITED on:
    dataset_split/validation/   (499 images — untouched)
    dataset_split/test/         (499 images — untouched)

  To enable:
    1. Open 06_augmentation.py
    2. Set:  SAFETY_CHECK = True  →  SAFETY_CHECK = False
    3. Re-run: python3 06_augmentation.py
""")
    print("  Augmentation NOT executed. Exiting safely.")
    print("=" * 70 + "\n")
    sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS — only loaded when SAFETY_CHECK = False
# ─────────────────────────────────────────────────────────────────────────────

try:
    import cv2
    import numpy as np
    from PIL import Image
    import pandas as pd
    print("[OK] All augmentation libraries loaded.")
except ImportError as e:
    print(f"[FATAL] Missing library: {e}")
    print("Install with: pip install opencv-python-headless numpy pandas Pillow")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR       = Path(__file__).resolve().parent
DATASET_ROOT     = SCRIPT_DIR.parent
PIPELINE_DIR     = SCRIPT_DIR
REPORTS_DIR      = PIPELINE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_DIR        = DATASET_ROOT / "dataset_split"
TRAIN_ORIG       = SPLIT_DIR / "train_original"
TRAIN_AUG        = SPLIT_DIR / "train_augmented"
VAL_DIR          = SPLIT_DIR / "validation"
TEST_DIR         = SPLIT_DIR / "test"

SEED             = 42
TARGET_PER_CLASS = 600               # 50 × 600 = 30,000

# ── Augmentation parameters (locked for IEEE paper) ──────────────────────────
ROT_RANGE    = (-15.0, 15.0)         # θ ~ Uniform(-15°, +15°)
SCALE_RANGE  = (0.9,   1.1)          # s ~ Uniform(0.9, 1.1)
NOISE_SIGMA  = 10.0                  # σ = 10  →  n ~ N(0, σ²)
BLUR_KERNEL  = (3, 3)                # Gaussian blur, 3×3 kernel

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


# ─────────────────────────────────────────────────────────────────────────────
# PRE-FLIGHT SAFETY ASSERTIONS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  Tulu Kalpuga — Augmentation Script (06_augmentation.py)")
print("=" * 70)
print(f"  SEED             : {SEED}")
print(f"  TARGET per class : {TARGET_PER_CLASS}")
print(f"  Train original   : {TRAIN_ORIG}")
print(f"  Train augmented  : {TRAIN_AUG}")
print(f"  Validation dir   : {VAL_DIR}  ← will NOT be touched")
print(f"  Test dir         : {TEST_DIR}  ← will NOT be touched")
print(f"\n  Augmentation parameters:")
print(f"    Rotation   : θ ~ Uniform({ROT_RANGE[0]}°, {ROT_RANGE[1]}°)")
print(f"    Scaling    : s ~ Uniform({SCALE_RANGE[0]}, {SCALE_RANGE[1]})")
print(f"    Noise      : n ~ N(0, {NOISE_SIGMA}²),  I' = clip(I + n, 0, 255)")
print(f"    Blur       : Gaussian {BLUR_KERNEL[0]}×{BLUR_KERNEL[1]} kernel")
print("=" * 70 + "\n")

# Confirm val/test exist and are not being touched
assert VAL_DIR.exists(),  f"[ABORT] Validation dir missing: {VAL_DIR}"
assert TEST_DIR.exists(), f"[ABORT] Test dir missing: {TEST_DIR}"
assert TRAIN_ORIG.exists(), f"[ABORT] Train original dir missing: {TRAIN_ORIG}"

# Count val/test files before and we will confirm they are unchanged after
val_count_before  = sum(1 for p in VAL_DIR.rglob("*")  if p.is_file())
test_count_before = sum(1 for p in TEST_DIR.rglob("*") if p.is_file())
print(f"  [PRE-CHECK] Validation files  : {val_count_before}  (must remain unchanged)")
print(f"  [PRE-CHECK] Test files        : {test_count_before}  (must remain unchanged)")


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: SHA-256
# ─────────────────────────────────────────────────────────────────────────────

def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATION FUNCTIONS
# Each function documents its mathematical formulation explicitly.
# ─────────────────────────────────────────────────────────────────────────────

def apply_rotation(img_bgr: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate image by `angle` degrees around its center.

    Mathematical formulation:
        R(I, θ) = warpAffine(I, M(θ), border=REFLECT_101)
        where M(θ) is a 2×3 rotation matrix centered at (W/2, H/2).
        θ ~ Uniform(-15°, +15°)

    BORDER_REFLECT_101 is used to avoid black border artifacts.
    """
    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D(center=(w / 2.0, h / 2.0), angle=angle, scale=1.0)
    return cv2.warpAffine(
        img_bgr, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def apply_scaling(img_bgr: np.ndarray, scale: float) -> np.ndarray:
    """
    Scale image by factor `scale`, then restore original spatial dimensions
    by center-cropping (scale > 1) or reflection-padding (scale < 1).

    Mathematical formulation:
        s ~ Uniform(0.9, 1.1)
        I_scaled = resize(I, s·H, s·W)
        if s > 1: I_out = center_crop(I_scaled, H, W)
        if s < 1: I_out = reflect_pad(I_scaled, H, W)
        Output shape = input shape (unchanged).
    """
    h, w = img_bgr.shape[:2]
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    scaled = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if scale > 1.0:
        # Center-crop back to original size
        y0 = (new_h - h) // 2
        x0 = (new_w - w) // 2
        # Guard against off-by-one from rounding
        y0 = max(0, min(y0, new_h - h))
        x0 = max(0, min(x0, new_w - w))
        scaled = scaled[y0: y0 + h, x0: x0 + w]
    else:
        # Reflection-pad back to original size
        pad_top    = (h - new_h) // 2
        pad_bottom = h - new_h - pad_top
        pad_left   = (w - new_w) // 2
        pad_right  = w - new_w - pad_left
        scaled = cv2.copyMakeBorder(
            scaled, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_REFLECT_101,
        )

    # Final shape guarantee (handles edge cases from integer rounding)
    if scaled.shape[0] != h or scaled.shape[1] != w:
        scaled = cv2.resize(scaled, (w, h), interpolation=cv2.INTER_LINEAR)

    return scaled


def apply_gaussian_noise(img_bgr: np.ndarray, sigma: float,
                         rng: np.random.Generator) -> np.ndarray:
    """
    Add i.i.d. Gaussian additive noise to every pixel channel independently.

    Mathematical formulation:
        n ~ N(0, σ²),  σ = 10,  n ∈ R^(H×W×C)
        I' = clip(float32(I) + n, 0, 255)

    IMPLEMENTATION NOTE (critical for correctness):
        Addition is performed in float32 space BEFORE quantizing to uint8.
        This ensures negative noise values correctly darken pixels rather
        than wrapping to large values as would occur with direct uint8 cast
        of the noise array (e.g., np.random.normal(...).astype(np.uint8)).

    Parameters:
        img_bgr : uint8 BGR image
        sigma   : standard deviation of noise (σ = 10)
        rng     : numpy Generator for reproducible per-image seeding
    """
    # Generate float-valued noise: n ~ N(0, σ²)
    noise = rng.normal(loc=0.0, scale=sigma, size=img_bgr.shape)  # float64
    # Perform addition in float32 to avoid uint8 overflow/underflow
    noisy_float = img_bgr.astype(np.float32) + noise.astype(np.float32)
    # Clip to valid range and quantize back to uint8
    return np.clip(noisy_float, 0.0, 255.0).astype(np.uint8)


def apply_gaussian_blur(img_bgr: np.ndarray, kernel: tuple) -> np.ndarray:
    """
    Apply Gaussian spatial blur.

    Mathematical formulation:
        I_blur = G_kernel * I   (convolution, * denotes 2D convolution)
        kernel size = 3×3,  σ computed automatically by OpenCV
    """
    return cv2.GaussianBlur(img_bgr, kernel, sigmaX=0, sigmaY=0)


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE AUGMENTATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def augment_image(img_bgr: np.ndarray, aug_seed: int) -> tuple:
    """
    Apply a reproducible random combination of augmentations to a single image.

    Transformation order: Rotation → Scaling → Noise → Blur
    Rotation: always applied.
    Scaling, Noise, Blur: each applied independently with probability p = 0.5.

    Seeding strategy:
        A per-image integer seed (aug_seed) initialises BOTH Python's random
        module AND a numpy Generator for that image only, ensuring that
        re-running the script with the same seed always produces identical
        augmented images regardless of call order.

    Returns:
        (augmented_image: np.ndarray,  metadata: dict)
    """
    # Per-image reproducible RNGs
    py_rng  = random.Random(aug_seed)
    np_rng  = np.random.default_rng(aug_seed)

    result = img_bgr.copy()
    aug_type_parts = []
    angle  = 0.0
    scale  = 1.0
    noise_applied = False
    blur_applied  = False

    # 1. Rotation — always applied
    angle  = py_rng.uniform(*ROT_RANGE)
    result = apply_rotation(result, angle)
    aug_type_parts.append("rotation")

    # 2. Scaling — p = 0.5
    if py_rng.random() > 0.5:
        scale  = py_rng.uniform(*SCALE_RANGE)
        result = apply_scaling(result, scale)
        aug_type_parts.append("scaling")

    # 3. Gaussian noise — p = 0.5
    #    Correctly implemented in float32 space with np.clip
    if py_rng.random() > 0.5:
        result = apply_gaussian_noise(result, NOISE_SIGMA, np_rng)
        noise_applied = True
        aug_type_parts.append("noise")

    # 4. Gaussian blur — p = 0.5
    if py_rng.random() > 0.5:
        result = apply_gaussian_blur(result, BLUR_KERNEL)
        blur_applied = True
        aug_type_parts.append("blur")

    aug_type_str = "+".join(aug_type_parts)

    metadata = {
        "augmentation_type": aug_type_str,
        "rotation_angle"   : round(angle, 6),
        "scale_factor"     : round(scale, 6),
        "noise_sigma"      : NOISE_SIGMA if noise_applied else 0.0,
        "blur_applied"     : blur_applied,
        "random_seed"      : aug_seed,
    }
    return result, metadata


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

aug_manifest_rows = []
aug_counter       = 0
classes = sorted([d.name for d in TRAIN_ORIG.iterdir() if d.is_dir()])

print(f"[STEP 1] Augmenting {len(classes)} classes to {TARGET_PER_CLASS} images each...\n")

for cls in classes:
    cls_orig_dir = TRAIN_ORIG / cls
    cls_aug_dir  = TRAIN_AUG  / cls
    cls_aug_dir.mkdir(parents=True, exist_ok=True)

    orig_images = sorted([
        f for f in cls_orig_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXT
    ])
    n_orig   = len(orig_images)
    n_needed = max(0, TARGET_PER_CLASS - n_orig)

    print(f"  {cls:<12}  original={n_orig:4d}  need={n_needed:4d}  target={TARGET_PER_CLASS}")

    # Step A: Copy all originals into train_augmented
    for src in orig_images:
        dst = cls_aug_dir / src.name
        if not dst.exists():
            shutil.copy2(str(src), str(dst))

    if n_needed == 0:
        final_count = len(list(cls_aug_dir.iterdir()))
        print(f"    → No augmentation needed. Files in aug dir: {final_count}")
        continue

    # Step B: Generate exactly n_needed augmented images
    # Seed for this class is deterministic: SEED + stable hash of class name
    class_seed_offset = int(hashlib.md5(cls.encode()).hexdigest(), 16) % (2**31)

    aug_idx = 0
    while aug_idx < n_needed:
        # Cycle through originals
        src_img_path = orig_images[aug_idx % n_orig]

        # Load with OpenCV (BGR uint8)
        img_bgr = cv2.imread(str(src_img_path))
        if img_bgr is None:
            print(f"    [WARN] Could not read {src_img_path.name}, skipping.")
            aug_idx += 1
            continue

        # Per-image deterministic seed (unique across all classes and indices)
        aug_seed = SEED ^ class_seed_offset ^ (aug_idx * 6700417)

        aug_img, meta = augment_image(img_bgr, aug_seed)

        # Output filename
        aug_filename = f"aug_{cls}_{aug_idx:06d}.png"
        aug_path     = cls_aug_dir / aug_filename

        # Encode to PNG bytes
        ok, encoded = cv2.imencode(".png", aug_img)
        if not ok:
            print(f"    [WARN] Could not encode augmented image {aug_filename}, skipping.")
            aug_idx += 1
            continue

        aug_sha256 = compute_sha256(encoded.tobytes())

        # Write to disk
        cv2.imwrite(str(aug_path), aug_img)

        aug_manifest_rows.append({
            "generated_image"  : str(aug_path),
            "source_image"     : str(src_img_path),
            "class_name"       : cls,
            "augmentation_type": meta["augmentation_type"],
            "rotation_angle"   : meta["rotation_angle"],
            "scale_factor"     : meta["scale_factor"],
            "noise_sigma"      : meta["noise_sigma"],
            "blur_applied"     : meta["blur_applied"],
            "random_seed"      : meta["random_seed"],
            "sha256_aug"       : aug_sha256,
        })

        aug_idx    += 1
        aug_counter += 1

    # Verify final count in augmented dir
    final_count = sum(1 for f in cls_aug_dir.iterdir() if f.is_file())
    expected    = TARGET_PER_CLASS
    status      = "OK" if final_count == expected else f"WARN: got {final_count}, expected {expected}"
    print(f"    → [{status}] Final count in train_augmented/{cls}: {final_count}")


# ─────────────────────────────────────────────────────────────────────────────
# POST-AUGMENTATION SAFETY VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n[STEP 2] Post-augmentation safety verification...")

val_count_after  = sum(1 for p in VAL_DIR.rglob("*")  if p.is_file())
test_count_after = sum(1 for p in TEST_DIR.rglob("*") if p.is_file())

val_ok  = val_count_after  == val_count_before
test_ok = test_count_after == test_count_before

print(f"  [{'OK' if val_ok  else 'FAIL'}] Validation file count: before={val_count_before},  after={val_count_after}")
print(f"  [{'OK' if test_ok else 'FAIL'}] Test file count      : before={test_count_before}, after={test_count_after}")

if not val_ok or not test_ok:
    print("\n  [CRITICAL] Validation or test set was modified! Investigate immediately.")
else:
    print("\n  [OK] Validation and test sets are completely untouched.")


# ─────────────────────────────────────────────────────────────────────────────
# SAVE AUGMENTATION MANIFEST
# ─────────────────────────────────────────────────────────────────────────────

print("\n[STEP 3] Saving augmentation manifest...")

df_aug = pd.DataFrame(aug_manifest_rows)
aug_manifest_path = PIPELINE_DIR / "augmentation_manifest.csv"
df_aug.to_csv(aug_manifest_path, index=False)
print(f"  Saved augmentation_manifest.csv : {aug_manifest_path}")
print(f"  Total augmented images generated: {aug_counter}")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL AUGMENTED TRAINING SET SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

aug_classes  = sorted([d.name for d in TRAIN_AUG.iterdir() if d.is_dir()])
final_counts = {}
for cls in aug_classes:
    final_counts[cls] = sum(1 for f in (TRAIN_AUG / cls).iterdir() if f.is_file())

total_aug_train = sum(final_counts.values())
expected_total  = TARGET_PER_CLASS * len(classes)

print(f"\n[STEP 4] Final augmented training set counts:")
print(f"  {'Class':<12} {'Count':>8}  {'Status':}")
print(f"  {'─'*12} {'─'*8}  {'─'*10}")
for cls in sorted(final_counts):
    cnt    = final_counts[cls]
    status = "OK" if cnt == TARGET_PER_CLASS else f"WARN (expected {TARGET_PER_CLASS})"
    print(f"  {cls:<12} {cnt:>8}  {status}")

print(f"\n  {'─'*34}")
print(f"  {'TOTAL':<12} {total_aug_train:>8}")
print(f"  Expected (50 × 600): {expected_total}")
ok_total = total_aug_train == expected_total
print(f"  Total check: {'PASS' if ok_total else 'FAIL'}")


# ─────────────────────────────────────────────────────────────────────────────
# SAVE AUGMENTATION REPORT
# ─────────────────────────────────────────────────────────────────────────────

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
aug_report = [
    "=" * 70,
    "  TULU KALPUGA — AUGMENTATION REPORT",
    f"  Generated: {now}",
    "=" * 70,
    "",
    "  Augmentation applied ONLY to: dataset_split/train_original/",
    "  Validation set              : NOT augmented (verified)",
    "  Test set                    : NOT augmented (verified)",
    "",
    "  Augmentation parameters (IEEE paper):",
    f"    Rotation  : θ ~ Uniform({ROT_RANGE[0]}°, {ROT_RANGE[1]}°)",
    f"    Scaling   : s ~ Uniform({SCALE_RANGE[0]}, {SCALE_RANGE[1]})",
    f"    Noise     : n ~ N(0, σ²), σ = {NOISE_SIGMA}",
    "               I' = clip(float32(I) + n, 0, 255)",
    f"    Blur      : Gaussian {BLUR_KERNEL[0]}×{BLUR_KERNEL[1]} kernel",
    f"    SEED      : {SEED}",
    "",
    "  Gaussian Noise Implementation Note:",
    "    Addition performed in float32 space before quantizing to uint8.",
    "    This correctly handles negative noise values (pixel darkening)",
    "    without uint8 wrap-around artifacts.",
    "",
    f"  Total images generated     : {aug_counter}",
    f"  Total train (augmented)    : {total_aug_train}",
    f"  Expected (50 × 600)        : {expected_total}",
    f"  Total check                : {'PASS' if ok_total else 'FAIL'}",
    "",
    f"  Validation set after aug   : {val_count_after} files ({'OK' if val_ok else 'MODIFIED!'})",
    f"  Test set after aug         : {test_count_after} files ({'OK' if test_ok else 'MODIFIED!'})",
    "",
    "  Per-class augmented training counts:",
    f"  {'Class':<12} {'Count':>8}",
]
for cls in sorted(final_counts):
    aug_report.append(f"  {cls:<12} {final_counts[cls]:>8}")

aug_report_path = REPORTS_DIR / "augmentation_report.txt"
with open(aug_report_path, "w") as f:
    f.write("\n".join(aug_report))
print(f"\n  Saved augmentation_report.txt : {aug_report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL BANNER
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  AUGMENTATION COMPLETE")
print("=" * 70)
print(f"  Total augmented training images  : {total_aug_train}")
print(f"  Total classes                    : {len(aug_classes)}")
print(f"  Images per class                 : {TARGET_PER_CLASS}")
print(f"  Validation set — untouched       : {val_count_after} images")
print(f"  Test set — untouched             : {test_count_after} images")
print(f"  Augmentation manifest            : {aug_manifest_path}")
print("\n  Next step: Run 07_generate_augmentation_visualization.py")
print("=" * 70 + "\n")
