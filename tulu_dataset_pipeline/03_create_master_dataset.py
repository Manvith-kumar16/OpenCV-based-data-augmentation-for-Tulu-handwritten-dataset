"""
=============================================================================
03_create_master_dataset.py
Tulu Kalpuga — IEEE Research Pipeline
Task: Create Logical Master Dataset with Provenance Tracking

Purpose:
    - Merge both dataset directories into a logical master view
    - Preserve full provenance (source dataset, original path)
    - Handle duplicates: keep ONE copy, mark it, record provenance
    - Create dataset_manifest.csv with all metadata
    - DO NOT copy or modify original files

IMPORTANT:
    - No files are moved or copied. Only a manifest CSV is created.
    - The "master dataset" is a logical view over the original directories.
    - Physical split copies will be created in Step 04.
=============================================================================
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from collections import defaultdict

try:
    from PIL import Image
    import pandas as pd
    import numpy as np
    print("[OK] All libraries loaded.")
except ImportError as e:
    print(f"[FATAL] Missing library: {e}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR.parent
PIPELINE_DIR = SCRIPT_DIR
REPORTS_DIR  = PIPELINE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_V2   = DATASET_ROOT / "Tulu-Dtatset-V2"
DATASET_ORIG = DATASET_ROOT / "tulu-dataset"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

print("\n" + "="*70)
print("  Tulu Kalpuga — Master Dataset Creation (03_create_master_dataset.py)")
print("="*70)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DUPLICATE SUMMARY FROM STEP 02
# ─────────────────────────────────────────────────────────────────────────────

dup_summary_path = PIPELINE_DIR / "duplicate_summary.json"
hash_manifest_path = PIPELINE_DIR / "hash_manifest.csv"

if not dup_summary_path.exists():
    print("[ERROR] duplicate_summary.json not found. Run 02_check_duplicates.py first.")
    sys.exit(1)

with open(dup_summary_path) as f:
    dup_summary = json.load(f)

cross_dup_sha256s = set(dup_summary.get("cross_dup_sha256s", []))

print(f"\n  Cross-dataset duplicate SHA256s to handle: {len(cross_dup_sha256s)}")
print(f"  Safe to combine: {dup_summary.get('safe_to_combine', 'unknown')}")

# Load hash manifest if available (from step 02)
if hash_manifest_path.exists():
    df_hashes = pd.read_csv(hash_manifest_path)
    print(f"  Loaded hash manifest: {len(df_hashes)} records")
else:
    print("[WARN] hash_manifest.csv not found. Will re-compute hashes.")
    df_hashes = None


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def get_image_metadata(filepath: Path) -> dict:
    try:
        with Image.open(filepath) as img:
            img.load()
            return {
                "image_width" : img.width,
                "image_height": img.height,
                "image_mode"  : img.mode,
                "is_valid"    : True,
            }
    except Exception as e:
        return {
            "image_width" : None,
            "image_height": None,
            "image_mode"  : None,
            "is_valid"    : False,
            "error"       : str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# BUILD MASTER DATASET MANIFEST
# ─────────────────────────────────────────────────────────────────────────────

print("\n[STEP 1] Building master dataset manifest...")

# If hash manifest exists from step 02, use it; otherwise scan from scratch
if df_hashes is not None:
    # Build lookup from path -> sha256
    hash_lookup = dict(zip(df_hashes["path"], df_hashes["sha256"]))
else:
    hash_lookup = {}

manifest_rows = []
seen_sha256 = {}   # sha256 -> first record (for dedup tracking)

datasets_to_scan = [
    (DATASET_V2,   "Tulu-Dtatset-V2"),
    (DATASET_ORIG, "tulu-dataset"),
]

total_seen = 0
total_skipped_dup = 0
total_invalid = 0

for dataset_path, dataset_name in datasets_to_scan:
    print(f"\n  Scanning {dataset_name}...")
    class_dirs = sorted([
        d for d in dataset_path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    for cls_dir in class_dirs:
        cls_name = cls_dir.name
        for fpath in sorted(cls_dir.iterdir()):
            if not fpath.is_file():
                continue
            ext = fpath.suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            total_seen += 1

            # Get SHA256 from cached manifest or compute now
            sha256 = hash_lookup.get(str(fpath)) or compute_sha256(fpath)

            is_cross_dup = sha256 in cross_dup_sha256s

            # For cross-dataset duplicates: keep the V2 copy as canonical
            if is_cross_dup:
                if sha256 in seen_sha256:
                    # This is the duplicate copy — record it but mark as duplicate
                    canonical_path = seen_sha256[sha256]["image_path"]
                    total_skipped_dup += 1
                    manifest_rows.append({
                        "image_path"     : str(fpath),
                        "source_dataset" : dataset_name,
                        "class_name"     : cls_name,
                        "file_name"      : fpath.name,
                        "file_extension" : ext,
                        "sha256"         : sha256,
                        "image_width"    : None,
                        "image_height"   : None,
                        "image_mode"     : None,
                        "is_valid"       : True,
                        "is_duplicate"   : True,
                        "canonical_path" : canonical_path,
                        "include_in_master": False,
                    })
                    continue
                else:
                    seen_sha256[sha256] = {"image_path": str(fpath), "dataset": dataset_name}
            elif sha256 and sha256 in seen_sha256:
                # Same-dataset internal duplicate
                total_skipped_dup += 1
                canonical_path = seen_sha256[sha256]["image_path"]
                manifest_rows.append({
                    "image_path"     : str(fpath),
                    "source_dataset" : dataset_name,
                    "class_name"     : cls_name,
                    "file_name"      : fpath.name,
                    "file_extension" : ext,
                    "sha256"         : sha256,
                    "image_width"    : None,
                    "image_height"   : None,
                    "image_mode"     : None,
                    "is_valid"       : True,
                    "is_duplicate"   : True,
                    "canonical_path" : canonical_path,
                    "include_in_master": False,
                })
                continue
            else:
                if sha256:
                    seen_sha256[sha256] = {"image_path": str(fpath), "dataset": dataset_name}

            # Get image metadata for non-duplicate valid images
            meta = get_image_metadata(fpath)

            if not meta["is_valid"]:
                total_invalid += 1
                manifest_rows.append({
                    "image_path"     : str(fpath),
                    "source_dataset" : dataset_name,
                    "class_name"     : cls_name,
                    "file_name"      : fpath.name,
                    "file_extension" : ext,
                    "sha256"         : sha256,
                    "image_width"    : None,
                    "image_height"   : None,
                    "image_mode"     : None,
                    "is_valid"       : False,
                    "is_duplicate"   : False,
                    "canonical_path" : None,
                    "include_in_master": False,
                })
                continue

            manifest_rows.append({
                "image_path"       : str(fpath),
                "source_dataset"   : dataset_name,
                "class_name"       : cls_name,
                "file_name"        : fpath.name,
                "file_extension"   : ext,
                "sha256"           : sha256,
                "image_width"      : meta["image_width"],
                "image_height"     : meta["image_height"],
                "image_mode"       : meta["image_mode"],
                "is_valid"         : True,
                "is_duplicate"     : False,
                "canonical_path"   : None,
                "include_in_master": True,
            })

print(f"\n  Total files scanned      : {total_seen}")
print(f"  Invalid images           : {total_invalid}")
print(f"  Duplicate images skipped : {total_skipped_dup}")


# ─────────────────────────────────────────────────────────────────────────────
# MASTER DATASET STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

df_manifest = pd.DataFrame(manifest_rows)
df_master   = df_manifest[df_manifest["include_in_master"] == True].copy()

master_total = len(df_master)
print(f"\n  Master dataset size (unique valid images): {master_total}")

# Per-class count
class_counts = df_master.groupby(["class_name", "source_dataset"]).size().unstack(fill_value=0)

print("\n[STEP 2] Per-class master dataset summary:")
print(f"\n  {'Class':<12} {'V2':>6} {'Orig':>6} {'Total':>8}")
print(f"  {'─'*12} {'─'*6} {'─'*6} {'─'*8}")

for cls in sorted(df_master["class_name"].unique()):
    v2_c   = len(df_master[(df_master["class_name"]==cls) & (df_master["source_dataset"]=="Tulu-Dtatset-V2")])
    orig_c = len(df_master[(df_master["class_name"]==cls) & (df_master["source_dataset"]=="tulu-dataset")])
    print(f"  {cls:<12} {v2_c:>6} {orig_c:>6} {v2_c+orig_c:>8}")

# Overall stats
combined_per_class = df_master.groupby("class_name").size()
print(f"\n  Total classes            : {len(combined_per_class)}")
print(f"  Total images (master)    : {master_total}")
print(f"  Min per class            : {combined_per_class.min()}")
print(f"  Max per class            : {combined_per_class.max()}")
print(f"  Mean per class           : {combined_per_class.mean():.2f}")
print(f"  Std dev per class        : {combined_per_class.std():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

print("\n[STEP 3] Saving manifest and report...")

manifest_csv_path = PIPELINE_DIR / "dataset_manifest.csv"
df_manifest.to_csv(manifest_csv_path, index=False)
print(f"  Saved dataset_manifest.csv : {manifest_csv_path}")

# Save master-only manifest (cleaner for downstream)
master_csv_path = PIPELINE_DIR / "master_manifest.csv"
df_master.to_csv(master_csv_path, index=False)
print(f"  Saved master_manifest.csv  : {master_csv_path}")

# Save per-class count table
per_class_csv_path = PIPELINE_DIR / "master_class_counts.csv"
df_class = pd.DataFrame({
    "class_name": sorted(df_master["class_name"].unique()),
    "v2_count"  : [len(df_master[(df_master["class_name"]==c) & (df_master["source_dataset"]=="Tulu-Dtatset-V2")]) for c in sorted(df_master["class_name"].unique())],
    "orig_count": [len(df_master[(df_master["class_name"]==c) & (df_master["source_dataset"]=="tulu-dataset")]) for c in sorted(df_master["class_name"].unique())],
    "combined"  : [len(df_master[df_master["class_name"]==c]) for c in sorted(df_master["class_name"].unique())],
})
df_class.to_csv(per_class_csv_path, index=False)
print(f"  Saved master_class_counts.csv : {per_class_csv_path}")

# Save JSON summary for step 04
master_summary = {
    "total_master_images": master_total,
    "total_classes"      : int(len(combined_per_class)),
    "min_per_class"      : int(combined_per_class.min()),
    "max_per_class"      : int(combined_per_class.max()),
    "mean_per_class"     : float(combined_per_class.mean()),
    "std_per_class"      : float(combined_per_class.std()),
    "total_duplicates_removed": total_skipped_dup,
    "total_invalid"      : total_invalid,
}
json_path = PIPELINE_DIR / "master_summary.json"
with open(json_path, "w") as f:
    json.dump(master_summary, f, indent=2)
print(f"  Saved master_summary.json  : {json_path}")

print("\n" + "="*70)
print("  MASTER DATASET MANIFEST COMPLETE")
print("="*70)
print(f"  Master dataset size    : {master_total} images")
print(f"  Classes                : {len(combined_per_class)}")
print(f"  Duplicates removed     : {total_skipped_dup}")
print(f"  Invalid images removed : {total_invalid}")
print("\n  NOTE: Original files have NOT been moved or modified.")
print("  The master manifest is a logical view over the original directories.")
print("\n  Next step: Run 04_create_split.py")
print("="*70 + "\n")
