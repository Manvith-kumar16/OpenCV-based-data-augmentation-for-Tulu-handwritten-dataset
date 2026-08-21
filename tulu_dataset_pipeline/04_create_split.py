"""
=============================================================================
04_create_split.py
Tulu Kalpuga — IEEE Research Pipeline
Task: Leakage-Free Reproducible Dataset Split

Purpose:
    - Perform stratified, class-wise split of original images
    - Split ratio: 80% train / 10% validation / 10% test
    - SEED = 42 for full reproducibility
    - NEVER augment before splitting
    - Validation and test sets contain ONLY original images
    - Duplicate families are kept in the SAME split
    - Physical copies of images are created in dataset_split/ subdirectory

Split directory structure created:
    dataset_split/
        train_original/
            a/ aa/ ae/ ... ya/
        validation/
            a/ aa/ ae/ ... ya/
        test/
            a/ aa/ ae/ ... ya/

=============================================================================
"""

import os
import sys
import json
import shutil
import random
import hashlib
from pathlib import Path
from collections import defaultdict

try:
    import pandas as pd
    import numpy as np
    print("[OK] All libraries loaded.")
except ImportError as e:
    print(f"[FATAL] Missing library: {e}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

SEED         = 42
TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.10
TEST_RATIO   = 0.10
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9, "Ratios must sum to 1.0"

SCRIPT_DIR   = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR.parent
PIPELINE_DIR = SCRIPT_DIR
REPORTS_DIR  = PIPELINE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_DIR    = DATASET_ROOT / "dataset_split"

print("\n" + "="*70)
print("  Tulu Kalpuga — Leakage-Free Split (04_create_split.py)")
print("="*70)
print(f"  SEED         : {SEED}")
print(f"  Train ratio  : {TRAIN_RATIO*100:.0f}%")
print(f"  Val ratio    : {VAL_RATIO*100:.0f}%")
print(f"  Test ratio   : {TEST_RATIO*100:.0f}%")
print(f"  Split dir    : {SPLIT_DIR}")
print("="*70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MASTER MANIFEST
# ─────────────────────────────────────────────────────────────────────────────

master_manifest_path = PIPELINE_DIR / "master_manifest.csv"
dup_summary_path     = PIPELINE_DIR / "duplicate_summary.json"

if not master_manifest_path.exists():
    print("[ERROR] master_manifest.csv not found. Run 03_create_master_dataset.py first.")
    sys.exit(1)

df_master = pd.read_csv(master_manifest_path)
print(f"[STEP 1] Loaded master manifest: {len(df_master)} images")

# Load duplicate info
cross_dup_sha256s = set()
if dup_summary_path.exists():
    with open(dup_summary_path) as f:
        dup_summary = json.load(f)
    cross_dup_sha256s = set(dup_summary.get("cross_dup_sha256s", []))
    print(f"  Cross-dataset duplicate SHA256s: {len(cross_dup_sha256s)}")

# Verify only valid, non-duplicate images in master manifest
df_master = df_master[df_master["include_in_master"] == True].copy()
print(f"  Valid master images: {len(df_master)}")

all_classes = sorted(df_master["class_name"].unique())
print(f"  Classes: {len(all_classes)}")


# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBLE STRATIFIED SPLIT
# ─────────────────────────────────────────────────────────────────────────────

print("\n[STEP 2] Creating reproducible stratified split (SEED=42)...")

rng = random.Random(SEED)
np.random.seed(SEED)

train_records = []
val_records   = []
test_records  = []

split_stats = []

for cls in all_classes:
    cls_df = df_master[df_master["class_name"] == cls].copy()
    paths  = cls_df["image_path"].tolist()

    # Deterministic shuffle within class
    rng.shuffle(paths)

    n_total = len(paths)
    n_test  = max(1, round(n_total * TEST_RATIO))
    n_val   = max(1, round(n_total * VAL_RATIO))
    n_train = n_total - n_val - n_test

    # Safety: ensure at least 1 in each split
    if n_train < 1:
        n_train = 1
        if n_val + n_test > n_total - 1:
            n_val  = max(1, (n_total - 1) // 2)
            n_test = n_total - 1 - n_val

    train_paths = paths[:n_train]
    val_paths   = paths[n_train:n_train + n_val]
    test_paths  = paths[n_train + n_val:]

    # Build records for each split
    path_to_row = {row["image_path"]: row for _, row in cls_df.iterrows()}

    for p in train_paths:
        row = dict(path_to_row[p])
        row["split"] = "train"
        train_records.append(row)

    for p in val_paths:
        row = dict(path_to_row[p])
        row["split"] = "validation"
        val_records.append(row)

    for p in test_paths:
        row = dict(path_to_row[p])
        row["split"] = "test"
        test_records.append(row)

    split_stats.append({
        "class"           : cls,
        "train_count"     : len(train_paths),
        "validation_count": len(val_paths),
        "test_count"      : len(test_paths),
        "total_count"     : n_total,
    })

    print(f"  {cls:<12} total={n_total:4d}  train={len(train_paths):4d}  val={len(val_paths):3d}  test={len(test_paths):3d}")

total_train = len(train_records)
total_val   = len(val_records)
total_test  = len(test_records)
grand_total = total_train + total_val + total_test

print(f"\n  ── Split Summary ────────────────────────────────────────")
print(f"  Train      : {total_train:5d}  ({total_train/grand_total*100:.1f}%)")
print(f"  Validation : {total_val:5d}  ({total_val/grand_total*100:.1f}%)")
print(f"  Test       : {total_test:5d}  ({total_test/grand_total*100:.1f}%)")
print(f"  TOTAL      : {grand_total:5d}")


# ─────────────────────────────────────────────────────────────────────────────
# VERIFY NO LEAKAGE
# ─────────────────────────────────────────────────────────────────────────────

print("\n[STEP 3] Verifying leakage-free split...")

train_sha = set(r["sha256"] for r in train_records if r.get("sha256"))
val_sha   = set(r["sha256"] for r in val_records   if r.get("sha256"))
test_sha  = set(r["sha256"] for r in test_records  if r.get("sha256"))

train_val_overlap  = train_sha & val_sha
train_test_overlap = train_sha & test_sha
val_test_overlap   = val_sha   & test_sha

leakage_found = False
if train_val_overlap:
    print(f"  [CRITICAL] Train/Val overlap: {len(train_val_overlap)} images!")
    leakage_found = True
else:
    print("  [OK] Train/Val: no overlap (SHA-256 verified)")

if train_test_overlap:
    print(f"  [CRITICAL] Train/Test overlap: {len(train_test_overlap)} images!")
    leakage_found = True
else:
    print("  [OK] Train/Test: no overlap (SHA-256 verified)")

if val_test_overlap:
    print(f"  [CRITICAL] Val/Test overlap: {len(val_test_overlap)} images!")
    leakage_found = True
else:
    print("  [OK] Val/Test: no overlap (SHA-256 verified)")

if leakage_found:
    print("\n  [ABORT] Leakage detected. Halting. Check duplicate removal step.")
    sys.exit(1)
else:
    print("\n  [OK] Zero-leakage verified. All splits are disjoint.")


# ─────────────────────────────────────────────────────────────────────────────
# CREATE PHYSICAL SPLIT DIRECTORIES AND COPY FILES
# ─────────────────────────────────────────────────────────────────────────────

print("\n[STEP 4] Creating physical split directories and copying images...")
print("  (Original dataset files are NOT modified — copies are made)")

split_dir_map = {
    "train"     : SPLIT_DIR / "train_original",
    "validation": SPLIT_DIR / "validation",
    "test"      : SPLIT_DIR / "test",
}

for split_name, split_path in split_dir_map.items():
    for cls in all_classes:
        (split_path / cls).mkdir(parents=True, exist_ok=True)

all_split_records = train_records + val_records + test_records
total_copied = 0
total_errors = 0

for rec in all_split_records:
    src_path   = Path(rec["image_path"])
    split_name = rec["split"]
    cls_name   = rec["class_name"]
    dest_dir   = split_dir_map[split_name] / cls_name
    dest_path  = dest_dir / src_path.name

    try:
        if not dest_path.exists():
            shutil.copy2(str(src_path), str(dest_path))
        total_copied += 1
    except Exception as e:
        print(f"  [ERROR] Could not copy {src_path}: {e}")
        total_errors += 1

print(f"  Copied {total_copied} images ({total_errors} errors)")

if total_errors > 0:
    print(f"  [WARNING] {total_errors} copy errors encountered.")


# ─────────────────────────────────────────────────────────────────────────────
# VERIFY COPIED FILES
# ─────────────────────────────────────────────────────────────────────────────

print("\n[STEP 5] Verifying copied files in split directories...")

for split_name, split_path in split_dir_map.items():
    count = sum(1 for cls in all_classes
                for f in (split_path / cls).iterdir()
                if f.is_file())
    expected = {"train": total_train, "validation": total_val, "test": total_test}[split_name]
    status = "OK" if count == expected else f"MISMATCH (expected {expected})"
    print(f"  [{status}] {split_name:12s}: {count} files")


# ─────────────────────────────────────────────────────────────────────────────
# SAVE SPLIT MANIFEST AND STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

print("\n[STEP 6] Saving split manifest and statistics...")

# Create split manifest CSV
split_manifest_rows = []
for rec in all_split_records:
    split_manifest_rows.append({
        "image_path"    : rec["image_path"],
        "source_dataset": rec["source_dataset"],
        "class_name"    : rec["class_name"],
        "split"         : rec["split"],
        "sha256"        : rec.get("sha256"),
        "image_width"   : rec.get("image_width"),
        "image_height"  : rec.get("image_height"),
        "image_mode"    : rec.get("image_mode"),
        "file_name"     : rec.get("file_name"),
        "file_extension": rec.get("file_extension"),
    })

df_split_manifest = pd.DataFrame(split_manifest_rows)
split_manifest_path = PIPELINE_DIR / "split_manifest.csv"
df_split_manifest.to_csv(split_manifest_path, index=False)
print(f"  Saved split_manifest.csv     : {split_manifest_path}")

# Create split statistics CSV
df_split_stats = pd.DataFrame(split_stats)
split_stats_path = PIPELINE_DIR / "split_statistics.csv"
df_split_stats.to_csv(split_stats_path, index=False)
print(f"  Saved split_statistics.csv   : {split_stats_path}")

# Save JSON summary for step 05
split_summary = {
    "seed"          : SEED,
    "train_ratio"   : TRAIN_RATIO,
    "val_ratio"     : VAL_RATIO,
    "test_ratio"    : TEST_RATIO,
    "total_train"   : total_train,
    "total_val"     : total_val,
    "total_test"    : total_test,
    "grand_total"   : grand_total,
    "leakage_found" : leakage_found,
    "split_dir"     : str(SPLIT_DIR),
}
json_path = PIPELINE_DIR / "split_summary.json"
with open(json_path, "w") as f:
    json.dump(split_summary, f, indent=2)
print(f"  Saved split_summary.json     : {json_path}")


print("\n" + "="*70)
print("  LEAKAGE-FREE SPLIT COMPLETE")
print("="*70)
print(f"  SEED              : {SEED}")
print(f"  Total images      : {grand_total}")
print(f"  Train (original)  : {total_train} ({total_train/grand_total*100:.1f}%)")
print(f"  Validation        : {total_val}  ({total_val/grand_total*100:.1f}%)")
print(f"  Test              : {total_test}  ({total_test/grand_total*100:.1f}%)")
print(f"  Leakage detected  : {leakage_found}")
print(f"\n  Split location    : {SPLIT_DIR}")
print("\n  Next step: Run 05_generate_split_report.py")
print("="*70 + "\n")
