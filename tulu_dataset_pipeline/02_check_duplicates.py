"""
=============================================================================
02_check_duplicates.py
Tulu Kalpuga — IEEE Research Pipeline
Task: Duplicate and Near-Duplicate Detection

Purpose:
    - Compute SHA-256 hash for every image in both datasets
    - Identify exact duplicates across and within datasets
    - Compute perceptual hashes (pHash, dHash) for near-duplicate detection
    - Report: duplicate count, percentage, affected classes
    - CRITICAL: Prevents data leakage before the split step

IMPORTANT: This script is READ-ONLY. It does NOT modify the original dataset.
=============================================================================
"""

import os
import sys
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import traceback

try:
    from PIL import Image
    import imagehash
    import pandas as pd
    import numpy as np
    print("[OK] All libraries loaded.")
except ImportError as e:
    print(f"[FATAL] Missing library: {e}")
    print("Install with: pip install Pillow imagehash pandas numpy")
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

# Near-duplicate threshold (Hamming distance on pHash)
# pHash is 64-bit; distance <= 10 is typically "near-duplicate"
PHASH_THRESHOLD = 10
DHASH_THRESHOLD = 10

print("\n" + "="*70)
print("  Tulu Kalpuga — Duplicate Detection (02_check_duplicates.py)")
print("="*70)
print(f"  pHash near-duplicate threshold : Hamming <= {PHASH_THRESHOLD}")
print(f"  dHash near-duplicate threshold : Hamming <= {DHASH_THRESHOLD}")
print("="*70 + "\n")


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


def compute_phash(filepath: Path):
    try:
        with Image.open(filepath) as img:
            img.load()
            return imagehash.phash(img)
    except Exception:
        return None


def compute_dhash(filepath: Path):
    try:
        with Image.open(filepath) as img:
            img.load()
            return imagehash.dhash(img)
    except Exception:
        return None


def collect_all_images(dataset_path: Path, dataset_name: str) -> list:
    """Walk a dataset directory and collect all image records."""
    records = []
    class_dirs = sorted([
        d for d in dataset_path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    for cls_dir in class_dirs:
        for fpath in sorted(cls_dir.iterdir()):
            if fpath.is_file() and fpath.suffix.lower() in SUPPORTED_EXTENSIONS:
                records.append({
                    "path"          : str(fpath),
                    "filepath"      : fpath,
                    "filename"      : fpath.name,
                    "class_name"    : cls_dir.name,
                    "source_dataset": dataset_name,
                    "extension"     : fpath.suffix.lower(),
                })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# COLLECT ALL IMAGES
# ─────────────────────────────────────────────────────────────────────────────

print("[STEP 1] Collecting all image paths from both datasets...")
records_v2   = collect_all_images(DATASET_V2,   "Tulu-Dtatset-V2")
records_orig = collect_all_images(DATASET_ORIG,  "tulu-dataset")
all_records  = records_v2 + records_orig

print(f"  Images in Tulu-Dtatset-V2 : {len(records_v2)}")
print(f"  Images in tulu-dataset    : {len(records_orig)}")
print(f"  Total images              : {len(all_records)}")


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE SHA-256 HASHES
# ─────────────────────────────────────────────────────────────────────────────

print("\n[STEP 2] Computing SHA-256 hashes for all images (exact duplicate check)...")
print("  This may take a few minutes...")

for i, rec in enumerate(all_records):
    rec["sha256"] = compute_sha256(rec["filepath"])
    if (i + 1) % 500 == 0:
        print(f"  Hashed {i+1}/{len(all_records)} images...")

print(f"  Done. Hashed {len(all_records)} images.")


# ─────────────────────────────────────────────────────────────────────────────
# EXACT DUPLICATE DETECTION (SHA-256)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[STEP 3] Identifying exact duplicates by SHA-256...")

hash_groups = defaultdict(list)
for rec in all_records:
    if rec["sha256"]:
        hash_groups[rec["sha256"]].append(rec)

exact_dup_groups = {h: recs for h, recs in hash_groups.items() if len(recs) > 1}
exact_dup_count  = sum(len(recs) - 1 for recs in exact_dup_groups.values())

# Classify duplicates by where they occur
within_v2_dups   = []  # both copies in V2
within_orig_dups = []  # both copies in Orig
cross_dups       = []  # copies span BOTH datasets

for h, recs in exact_dup_groups.items():
    sources = set(r["source_dataset"] for r in recs)
    if sources == {"Tulu-Dtatset-V2"}:
        within_v2_dups.append((h, recs))
    elif sources == {"tulu-dataset"}:
        within_orig_dups.append((h, recs))
    else:
        cross_dups.append((h, recs))

total_images = len(all_records)
dup_pct = (exact_dup_count / total_images * 100) if total_images > 0 else 0

print(f"\n  ── Exact Duplicate Results (SHA-256) ──────────────────────")
print(f"  Total duplicate images (extra copies) : {exact_dup_count}")
print(f"  Duplicate percentage                  : {dup_pct:.2f}%")
print(f"  Duplicate groups                      : {len(exact_dup_groups)}")
print(f"  Within Tulu-Dtatset-V2                : {len(within_v2_dups)} groups")
print(f"  Within tulu-dataset                   : {len(within_orig_dups)} groups")
print(f"  CROSS-dataset duplicates              : {len(cross_dups)} groups")

# Show cross-dataset duplicates — CRITICAL for leakage
if cross_dups:
    print(f"\n  [CRITICAL] {len(cross_dups)} cross-dataset duplicate groups found!")
    print("  These images appear in BOTH datasets and must be handled carefully.")
    cross_dup_classes = defaultdict(int)
    for h, recs in cross_dups:
        for r in recs:
            cross_dup_classes[r["class_name"]] += 1
    print(f"  Classes affected:")
    for cls, cnt in sorted(cross_dup_classes.items()):
        print(f"    {cls}: {cnt} duplicate instances")
else:
    print("\n  [OK] No cross-dataset exact duplicates detected.")

if within_v2_dups:
    print(f"\n  [WARN] {len(within_v2_dups)} internal duplicate groups within Tulu-Dtatset-V2")
if within_orig_dups:
    print(f"\n  [WARN] {len(within_orig_dups)} internal duplicate groups within tulu-dataset")


# ─────────────────────────────────────────────────────────────────────────────
# PERCEPTUAL HASH (pHash + dHash) — NEAR-DUPLICATE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

print("\n[STEP 4] Computing perceptual hashes (pHash, dHash) for near-duplicate detection...")
print(f"  Near-duplicate threshold: Hamming distance <= {PHASH_THRESHOLD}")
print("  This may take several minutes for ~10,000 images...")

for i, rec in enumerate(all_records):
    rec["phash"] = compute_phash(rec["filepath"])
    rec["dhash"] = compute_dhash(rec["filepath"])
    if (i + 1) % 500 == 0:
        print(f"  Computed perceptual hashes: {i+1}/{len(all_records)}")

print(f"  Done. Computed perceptual hashes for {len(all_records)} images.")


# ─────────────────────────────────────────────────────────────────────────────
# NEAR-DUPLICATE DETECTION (Cross-dataset only — most critical for leakage)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[STEP 5] Near-duplicate detection across datasets (cross-dataset only)...")
print("  Comparing every V2 image against every Orig image per class...")
print("  (Same-class only to reduce computation; cross-class near-dups are less critical)")

near_dup_pairs     = []
near_dup_cross_cnt = 0

# Group by class for efficiency
v2_by_class   = defaultdict(list)
orig_by_class = defaultdict(list)

for r in records_v2:
    v2_by_class[r["class_name"]].append(r)
for r in records_orig:
    orig_by_class[r["class_name"]].append(r)

all_classes = sorted(set(list(v2_by_class.keys()) + list(orig_by_class.keys())))

for cls in all_classes:
    v2_cls   = v2_by_class.get(cls, [])
    orig_cls = orig_by_class.get(cls, [])
    for rv2 in v2_cls:
        for rorig in orig_cls:
            if rv2.get("phash") is None or rorig.get("phash") is None:
                continue
            # Only flag if NOT already an exact duplicate
            if rv2["sha256"] == rorig["sha256"]:
                continue
            dist = rv2["phash"] - rorig["phash"]
            if dist <= PHASH_THRESHOLD:
                near_dup_pairs.append({
                    "class"         : cls,
                    "v2_path"       : rv2["path"],
                    "orig_path"     : rorig["path"],
                    "phash_distance": dist,
                    "v2_sha256"     : rv2["sha256"],
                    "orig_sha256"   : rorig["sha256"],
                })
                near_dup_cross_cnt += 1

near_dup_pct = (near_dup_cross_cnt / total_images * 100) if total_images > 0 else 0
near_dup_classes = list(set(p["class"] for p in near_dup_pairs))

print(f"\n  ── Near-Duplicate Results (cross-dataset, pHash) ──────────")
print(f"  Near-duplicate pairs found  : {near_dup_cross_cnt}")
print(f"  Near-duplicate % of total   : {near_dup_pct:.2f}%")
print(f"  Classes with near-dups      : {len(near_dup_classes)}")
if near_dup_classes:
    for cls in sorted(near_dup_classes):
        cnt = sum(1 for p in near_dup_pairs if p["class"] == cls)
        print(f"    {cls}: {cnt} near-duplicate pairs")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY AND RECOMMENDATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  [STEP 6] Summary and Recommendation")
print("="*70)

total_problematic = exact_dup_count + near_dup_cross_cnt

if len(cross_dups) == 0 and near_dup_cross_cnt == 0:
    recommendation = (
        "SAFE TO COMBINE: No cross-dataset exact or near-duplicates detected.\n"
        "  The two datasets appear to be complementary (independent samples).\n"
        "  Proceed to create master dataset."
    )
elif len(cross_dups) > 0 and near_dup_cross_cnt == 0:
    recommendation = (
        f"CAUTION: {len(cross_dups)} exact cross-dataset duplicate groups found.\n"
        "  These must be deduplicated BEFORE combining.\n"
        "  In the split step, keep all copies of a duplicate in ONE split only."
    )
else:
    recommendation = (
        f"WARNING: {len(cross_dups)} exact + {near_dup_cross_cnt} near-duplicate cross-dataset pairs.\n"
        "  Significant overlap between datasets detected.\n"
        "  All duplicates must be tracked. Only ONE copy per duplicate family\n"
        "  should be used in the final dataset, or duplicates must be assigned\n"
        "  to the same split to prevent leakage."
    )

print(f"\n  RECOMMENDATION: {recommendation}")


# ─────────────────────────────────────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

print("\n[STEP 7] Saving duplicate detection results...")

# Save all hash data
df_all = pd.DataFrame([
    {
        "path"          : r["path"],
        "filename"      : r["filename"],
        "class_name"    : r["class_name"],
        "source_dataset": r["source_dataset"],
        "extension"     : r["extension"],
        "sha256"        : r["sha256"],
        "phash"         : str(r["phash"]) if r.get("phash") is not None else None,
        "dhash"         : str(r["dhash"]) if r.get("dhash") is not None else None,
    }
    for r in all_records
])
hash_manifest_path = PIPELINE_DIR / "hash_manifest.csv"
df_all.to_csv(hash_manifest_path, index=False)
print(f"  Saved hash manifest   : {hash_manifest_path}")

# Save exact duplicate groups
if exact_dup_groups:
    dup_rows = []
    for h, recs in exact_dup_groups.items():
        sources = set(r["source_dataset"] for r in recs)
        dup_type = "cross-dataset" if len(sources) > 1 else f"within-{list(sources)[0]}"
        for r in recs:
            dup_rows.append({
                "sha256"     : h,
                "dup_type"   : dup_type,
                "path"       : r["path"],
                "class_name" : r["class_name"],
                "source"     : r["source_dataset"],
            })
    df_dups = pd.DataFrame(dup_rows)
    dup_csv_path = PIPELINE_DIR / "exact_duplicates.csv"
    df_dups.to_csv(dup_csv_path, index=False)
    print(f"  Saved exact dups      : {dup_csv_path}")

# Save near-duplicate pairs
if near_dup_pairs:
    df_near = pd.DataFrame(near_dup_pairs)
    near_dup_csv_path = PIPELINE_DIR / "near_duplicates.csv"
    df_near.to_csv(near_dup_csv_path, index=False)
    print(f"  Saved near dups       : {near_dup_csv_path}")

# Save duplicate report
report_lines = [
    "="*70,
    "  TULU KALPUGA — DUPLICATE DETECTION REPORT",
    "  Generated by: 02_check_duplicates.py",
    "="*70,
    "",
    f"  Total images examined       : {total_images}",
    f"  From Tulu-Dtatset-V2        : {len(records_v2)}",
    f"  From tulu-dataset           : {len(records_orig)}",
    "",
    "  ── Exact Duplicates (SHA-256) ──",
    f"  Duplicate groups            : {len(exact_dup_groups)}",
    f"  Duplicate images (extras)   : {exact_dup_count}",
    f"  Duplicate percentage        : {dup_pct:.2f}%",
    f"  Within V2 duplicate groups  : {len(within_v2_dups)}",
    f"  Within Orig duplicate groups: {len(within_orig_dups)}",
    f"  Cross-dataset dup groups    : {len(cross_dups)}",
    "",
    "  ── Near-Duplicates (pHash) ──",
    f"  pHash threshold (Hamming)   : <= {PHASH_THRESHOLD}",
    f"  Near-duplicate pairs        : {near_dup_cross_cnt}",
    f"  Near-duplicate percentage   : {near_dup_pct:.2f}%",
    f"  Classes affected            : {sorted(near_dup_classes)}",
    "",
    "  ── Recommendation ──",
    f"  {recommendation}",
    "",
]

if cross_dups:
    report_lines.append("  ── Cross-Dataset Exact Duplicate Details ──")
    for h, recs in cross_dups[:50]:
        report_lines.append(f"  SHA256: {h[:16]}...")
        for r in recs:
            report_lines.append(f"    [{r['source_dataset']}] {r['class_name']}/{r['filename']}")
    report_lines.append("")

if near_dup_pairs:
    report_lines.append("  ── Near-Duplicate Details (first 50 pairs) ──")
    for pair in near_dup_pairs[:50]:
        report_lines.append(
            f"  Class: {pair['class']} | "
            f"V2: {Path(pair['v2_path']).name} | "
            f"Orig: {Path(pair['orig_path']).name} | "
            f"pHash dist: {pair['phash_distance']}"
        )

report_text = "\n".join(report_lines)
report_path = REPORTS_DIR / "duplicate_report.txt"
with open(report_path, "w") as f:
    f.write(report_text)
print(f"  Saved duplicate report : {report_path}")

# Save JSON for downstream scripts
dup_summary = {
    "total_images"       : total_images,
    "exact_dup_groups"   : len(exact_dup_groups),
    "exact_dup_count"    : exact_dup_count,
    "dup_pct"            : dup_pct,
    "within_v2_dups"     : len(within_v2_dups),
    "within_orig_dups"   : len(within_orig_dups),
    "cross_dup_groups"   : len(cross_dups),
    "near_dup_pairs"     : near_dup_cross_cnt,
    "near_dup_pct"       : near_dup_pct,
    "near_dup_classes"   : sorted(near_dup_classes),
    "safe_to_combine"    : (len(cross_dups) == 0 and near_dup_cross_cnt == 0),
    "cross_dup_sha256s"  : [h for h, _ in cross_dups],
}
json_path = PIPELINE_DIR / "duplicate_summary.json"
with open(json_path, "w") as f:
    json.dump(dup_summary, f, indent=2)
print(f"  Saved duplicate JSON   : {json_path}")


print("\n" + "="*70)
print("  DUPLICATE DETECTION COMPLETE")
print("="*70)
print(f"  Exact duplicates (extra copies) : {exact_dup_count}")
print(f"  Near-duplicate pairs (cross-ds) : {near_dup_cross_cnt}")
print(f"  Safe to combine datasets        : {dup_summary['safe_to_combine']}")
print("\n  Next step: Run 03_create_master_dataset.py")
print("="*70 + "\n")
