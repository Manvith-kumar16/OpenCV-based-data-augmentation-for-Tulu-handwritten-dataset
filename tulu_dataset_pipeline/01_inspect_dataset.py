"""
=============================================================================
01_inspect_dataset.py
Tulu Kalpuga — IEEE Research Pipeline
Task: Dataset Inspection and Verification

Purpose:
    - Locate both dataset directories
    - Verify class count and class name consistency
    - Count total files and valid image files per class
    - Detect unsupported, corrupted or unreadable images
    - Report image dimensions, modes, and extensions
    - Output summary statistics for IEEE paper

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
    from PIL import Image, UnidentifiedImageError
    print("[OK] Pillow available")
except ImportError:
    print("[FATAL] Pillow not installed. Run: pip install Pillow")
    sys.exit(1)

try:
    import numpy as np
    print("[OK] NumPy available")
except ImportError:
    print("[FATAL] NumPy not installed. Run: pip install numpy")
    sys.exit(1)

try:
    import pandas as pd
    print("[OK] Pandas available")
except ImportError:
    print("[FATAL] Pandas not installed. Run: pip install pandas")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR.parent          # "Tulu Dataset/"
PIPELINE_DIR = SCRIPT_DIR                  # "tulu_dataset_pipeline/"
REPORTS_DIR  = PIPELINE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_V2   = DATASET_ROOT / "Tulu-Dtatset-V2"
DATASET_ORIG = DATASET_ROOT / "tulu-dataset"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}

# Expected reference counts (from paper — will be verified, NOT assumed)
EXPECTED_V2_TOTAL    = 4979
EXPECTED_ORIG_TOTAL  = 4980
EXPECTED_COMBINED    = 9959
EXPECTED_CLASSES     = 50

print("\n" + "="*70)
print("  Tulu Kalpuga — Dataset Inspection Script (01_inspect_dataset.py)")
print("="*70)
print(f"  Dataset V2   : {DATASET_V2}")
print(f"  Dataset Orig : {DATASET_ORIG}")
print(f"  Reports Dir  : {REPORTS_DIR}")
print("="*70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "ERROR"


def inspect_image(filepath: Path) -> dict:
    """
    Attempt to open and verify an image.
    Returns a dict with validity, dimensions, mode, and any error.
    """
    result = {
        "path"       : str(filepath),
        "filename"   : filepath.name,
        "extension"  : filepath.suffix.lower(),
        "file_size"  : None,
        "is_valid"   : False,
        "width"      : None,
        "height"     : None,
        "mode"       : None,
        "sha256"     : None,
        "error"      : None,
    }
    try:
        result["file_size"] = filepath.stat().st_size
    except Exception as e:
        result["error"] = f"stat error: {e}"
        return result

    # Phase 1: verify() to detect corruption
    try:
        with Image.open(filepath) as img:
            img.verify()   # raises on corrupt JPEG/PNG etc.
    except UnidentifiedImageError:
        result["error"] = "unidentified image format"
        return result
    except Exception as e:
        result["error"] = f"verify failed: {e}"
        return result

    # Phase 2: reopen to read metadata (verify() closes the file pointer)
    try:
        with Image.open(filepath) as img:
            img.load()     # force full decode
            result["width"]  = img.width
            result["height"] = img.height
            result["mode"]   = img.mode
            result["is_valid"] = True
    except Exception as e:
        result["error"] = f"load failed: {e}"
        return result

    result["sha256"] = compute_sha256(filepath)
    return result


def inspect_dataset(dataset_path: Path, dataset_name: str) -> dict:
    """
    Fully inspect a dataset directory.
    Returns structured results dict.
    """
    print(f"\n{'─'*60}")
    print(f"  Inspecting: {dataset_name}")
    print(f"  Path      : {dataset_path}")
    print(f"{'─'*60}")

    if not dataset_path.exists():
        print(f"  [ERROR] Directory does not exist: {dataset_path}")
        return {"error": f"Directory not found: {dataset_path}"}

    # Enumerate class directories
    class_dirs = sorted([
        d for d in dataset_path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    class_names = [d.name for d in class_dirs]

    print(f"  Classes found    : {len(class_dirs)}")

    per_class_stats = {}
    all_records     = []
    total_files     = 0
    total_valid     = 0
    total_unsupported = 0
    total_corrupted = 0

    for cls_dir in class_dirs:
        cls_name = cls_dir.name
        all_files = sorted([f for f in cls_dir.iterdir() if f.is_file()])
        total_files += len(all_files)

        valid_records   = []
        unsupported     = []
        corrupted       = []

        for fpath in all_files:
            ext = fpath.suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                unsupported.append(str(fpath))
                continue

            rec = inspect_image(fpath)
            rec["class_name"]      = cls_name
            rec["source_dataset"]  = dataset_name

            if rec["is_valid"]:
                valid_records.append(rec)
                total_valid += 1
            else:
                corrupted.append(rec)
                total_corrupted += 1

            all_records.append(rec)

        total_unsupported += len(unsupported)

        # Collect dimension statistics for valid images
        widths  = [r["width"]  for r in valid_records if r["width"]  is not None]
        heights = [r["height"] for r in valid_records if r["height"] is not None]
        modes   = list(set(r["mode"] for r in valid_records if r["mode"] is not None))

        per_class_stats[cls_name] = {
            "total_files"     : len(all_files),
            "valid_count"     : len(valid_records),
            "unsupported"     : len(unsupported),
            "corrupted"       : len(corrupted),
            "min_width"       : min(widths)  if widths  else None,
            "max_width"       : max(widths)  if widths  else None,
            "min_height"      : min(heights) if heights else None,
            "max_height"      : max(heights) if heights else None,
            "image_modes"     : modes,
            "unsupported_files": unsupported,
            "corrupted_files" : [r["path"] for r in corrupted],
        }

        status = "OK" if len(corrupted) == 0 and len(unsupported) == 0 else "WARN"
        print(f"    [{status}] {cls_name:10s} | files: {len(all_files):4d} | valid: {len(valid_records):4d} "
              f"| corrupt: {len(corrupted)} | unsupported: {len(unsupported)}")

    valid_counts = [per_class_stats[c]["valid_count"] for c in class_names]

    summary = {
        "dataset_name"       : dataset_name,
        "dataset_path"       : str(dataset_path),
        "num_classes"        : len(class_dirs),
        "class_names"        : class_names,
        "total_files"        : total_files,
        "total_valid_images" : total_valid,
        "total_unsupported"  : total_unsupported,
        "total_corrupted"    : total_corrupted,
        "min_per_class"      : int(min(valid_counts)) if valid_counts else 0,
        "max_per_class"      : int(max(valid_counts)) if valid_counts else 0,
        "mean_per_class"     : float(np.mean(valid_counts)) if valid_counts else 0,
        "std_per_class"      : float(np.std(valid_counts))  if valid_counts else 0,
        "per_class"          : per_class_stats,
        "all_records"        : all_records,
    }

    print(f"\n  ── Summary: {dataset_name} ──────────────────────────────")
    print(f"  Total class dirs       : {summary['num_classes']}")
    print(f"  Total files found      : {summary['total_files']}")
    print(f"  Total valid images     : {summary['total_valid_images']}")
    print(f"  Total unsupported      : {summary['total_unsupported']}")
    print(f"  Total corrupted        : {summary['total_corrupted']}")
    print(f"  Min images per class   : {summary['min_per_class']}")
    print(f"  Max images per class   : {summary['max_per_class']}")
    print(f"  Mean images per class  : {summary['mean_per_class']:.2f}")
    print(f"  Std dev images/class   : {summary['std_per_class']:.4f}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# MAIN INSPECTION
# ─────────────────────────────────────────────────────────────────────────────

print("\n[STEP 1] Inspecting Tulu-Dtatset-V2 ...")
results_v2 = inspect_dataset(DATASET_V2, "Tulu-Dtatset-V2")

print("\n[STEP 2] Inspecting tulu-dataset ...")
results_orig = inspect_dataset(DATASET_ORIG, "tulu-dataset")


# ─────────────────────────────────────────────────────────────────────────────
# CLASS CONSISTENCY CHECK
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  [STEP 3] Class Consistency Check")
print("="*70)

classes_v2   = set(results_v2.get("class_names", []))
classes_orig = set(results_orig.get("class_names", []))

only_in_v2   = classes_v2   - classes_orig
only_in_orig = classes_orig - classes_v2
common       = classes_v2   & classes_orig

print(f"  Classes in Tulu-Dtatset-V2    : {len(classes_v2)}")
print(f"  Classes in tulu-dataset       : {len(classes_orig)}")
print(f"  Common classes                : {len(common)}")
print(f"  Only in Tulu-Dtatset-V2       : {sorted(only_in_v2)  if only_in_v2  else 'None'}")
print(f"  Only in tulu-dataset          : {sorted(only_in_orig) if only_in_orig else 'None'}")

if only_in_v2 or only_in_orig:
    print("\n  [WARNING] Class mismatch detected between datasets!")
else:
    print("\n  [OK] Both datasets have identical class sets.")


# ─────────────────────────────────────────────────────────────────────────────
# CLASS COUNT COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  [STEP 4] Per-Class Image Count Comparison")
print("="*70)

all_classes = sorted(common)

rows = []
for cls in all_classes:
    v2_count   = results_v2.get("per_class",   {}).get(cls, {}).get("valid_count", 0)
    orig_count = results_orig.get("per_class", {}).get(cls, {}).get("valid_count", 0)
    combined   = v2_count + orig_count
    rows.append({
        "class"    : cls,
        "v2_count" : v2_count,
        "orig_count": orig_count,
        "combined" : combined,
    })

df_counts = pd.DataFrame(rows)
df_counts.set_index("class", inplace=True)

print(f"\n  {'Class':<12} {'V2':>6} {'Orig':>6} {'Combined':>10}")
print(f"  {'─'*12} {'─'*6} {'─'*6} {'─'*10}")
for cls, row in df_counts.iterrows():
    flag = " ← MIN" if row["combined"] < 180 else ""
    print(f"  {cls:<12} {row['v2_count']:>6} {row['orig_count']:>6} {row['combined']:>10}{flag}")

total_v2   = df_counts["v2_count"].sum()
total_orig = df_counts["orig_count"].sum()
total_comb = df_counts["combined"].sum()

print(f"\n  {'TOTAL':<12} {total_v2:>6} {total_orig:>6} {total_comb:>10}")
print(f"\n  Min combined per class : {df_counts['combined'].min()}")
print(f"  Max combined per class : {df_counts['combined'].max()}")
print(f"  Mean combined per class: {df_counts['combined'].mean():.2f}")
print(f"  Std combined per class : {df_counts['combined'].std():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# VERIFY AGAINST PAPER NUMBERS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  [STEP 5] Verification Against Reported Paper Numbers")
print("="*70)

def check_number(label, expected, actual):
    match = "MATCH" if expected == actual else "MISMATCH"
    icon  = "✓" if expected == actual else "✗"
    print(f"  [{icon}] {label}")
    print(f"       Old Paper Value : {expected}")
    print(f"       Actual Verified : {actual}")
    if expected != actual:
        diff = actual - expected
        print(f"       Difference      : {diff:+d}")
        print(f"       [ACTION NEEDED] Update paper with actual verified value.")
    print()

check_number("Total images in Tulu-Dtatset-V2", EXPECTED_V2_TOTAL,   results_v2.get("total_valid_images", 0))
check_number("Total images in tulu-dataset",    EXPECTED_ORIG_TOTAL,  results_orig.get("total_valid_images", 0))
check_number("Combined total images",           EXPECTED_COMBINED,
             results_v2.get("total_valid_images", 0) + results_orig.get("total_valid_images", 0))
check_number("Number of classes (V2)",          EXPECTED_CLASSES,     results_v2.get("num_classes", 0))
check_number("Number of classes (Orig)",        EXPECTED_CLASSES,     results_orig.get("num_classes", 0))


# ─────────────────────────────────────────────────────────────────────────────
# CORRUPTED IMAGE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

all_records_combined = (
    results_v2.get("all_records",   []) +
    results_orig.get("all_records", [])
)

corrupted_records = [r for r in all_records_combined if not r["is_valid"] and r.get("extension") in SUPPORTED_EXTENSIONS]
unsupported_records = [r for r in all_records_combined if r.get("extension") not in SUPPORTED_EXTENSIONS]

print("="*70)
print(f"  [STEP 6] Corrupted / Unreadable Image Summary")
print("="*70)
if corrupted_records:
    print(f"  CORRUPTED images: {len(corrupted_records)}")
    for r in corrupted_records:
        print(f"    - {r['path']} | Error: {r['error']}")
else:
    print("  [OK] No corrupted images detected.")

print(f"\n  Unsupported file types: {len(unsupported_records)}")
if unsupported_records:
    for r in unsupported_records:
        print(f"    - {r['path']}")


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE DIMENSION STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  [STEP 7] Image Dimension Statistics")
print("="*70)

valid_records = [r for r in all_records_combined if r["is_valid"]]
widths  = [r["width"]  for r in valid_records]
heights = [r["height"] for r in valid_records]
modes   = list(set(r["mode"] for r in valid_records if r["mode"]))

print(f"  Width  — min: {min(widths)}, max: {max(widths)}, mean: {np.mean(widths):.1f}")
print(f"  Height — min: {min(heights)}, max: {max(heights)}, mean: {np.mean(heights):.1f}")
print(f"  Unique image modes: {modes}")

# Check uniform 64x64
non_standard = [(r["path"], r["width"], r["height"]) for r in valid_records
                if r["width"] != 64 or r["height"] != 64]
if non_standard:
    print(f"\n  [WARN] {len(non_standard)} images are NOT 64×64:")
    for path, w, h in non_standard[:20]:
        print(f"    {path}  ({w}×{h})")
    if len(non_standard) > 20:
        print(f"    ... and {len(non_standard)-20} more")
else:
    print("  [OK] All images are 64×64 px.")


# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  [STEP 8] Saving Results")
print("="*70)

# Save per-class count table as CSV
count_csv_path = PIPELINE_DIR.parent / "tulu_dataset_pipeline" / "inspection_class_counts.csv"
df_counts.to_csv(count_csv_path)
print(f"  Saved class counts : {count_csv_path}")

# Save all records as CSV (inspection manifest)
df_records = pd.DataFrame(valid_records)
inspection_manifest_path = PIPELINE_DIR / "inspection_manifest.csv"
if not df_records.empty:
    df_records.to_csv(inspection_manifest_path, index=False)
    print(f"  Saved inspection manifest : {inspection_manifest_path}")

# Save text report
report_lines = []
report_lines.append("="*70)
report_lines.append("  TULU KALPUGA — DATASET INSPECTION REPORT")
report_lines.append("  Generated by: 01_inspect_dataset.py")
report_lines.append("="*70)
report_lines.append("")
report_lines.append(f"  Dataset V2   : {DATASET_V2}")
report_lines.append(f"  Dataset Orig : {DATASET_ORIG}")
report_lines.append("")
report_lines.append("  ── Dataset V2 Summary ──")
report_lines.append(f"  Classes               : {results_v2.get('num_classes', 'N/A')}")
report_lines.append(f"  Total files           : {results_v2.get('total_files', 'N/A')}")
report_lines.append(f"  Valid images          : {results_v2.get('total_valid_images', 'N/A')}")
report_lines.append(f"  Corrupted             : {results_v2.get('total_corrupted', 'N/A')}")
report_lines.append(f"  Unsupported           : {results_v2.get('total_unsupported', 'N/A')}")
report_lines.append(f"  Min/Max/Mean per class: {results_v2.get('min_per_class')}/{results_v2.get('max_per_class')}/{results_v2.get('mean_per_class'):.2f}")
report_lines.append("")
report_lines.append("  ── Dataset Orig Summary ──")
report_lines.append(f"  Classes               : {results_orig.get('num_classes', 'N/A')}")
report_lines.append(f"  Total files           : {results_orig.get('total_files', 'N/A')}")
report_lines.append(f"  Valid images          : {results_orig.get('total_valid_images', 'N/A')}")
report_lines.append(f"  Corrupted             : {results_orig.get('total_corrupted', 'N/A')}")
report_lines.append(f"  Unsupported           : {results_orig.get('total_unsupported', 'N/A')}")
report_lines.append(f"  Min/Max/Mean per class: {results_orig.get('min_per_class')}/{results_orig.get('max_per_class')}/{results_orig.get('mean_per_class'):.2f}")
report_lines.append("")
report_lines.append("  ── Combined ──")
report_lines.append(f"  Total valid images    : {results_v2.get('total_valid_images',0) + results_orig.get('total_valid_images',0)}")
report_lines.append(f"  Total classes         : {len(all_classes)}")
report_lines.append(f"  Min combined/class    : {df_counts['combined'].min()}")
report_lines.append(f"  Max combined/class    : {df_counts['combined'].max()}")
report_lines.append(f"  Mean combined/class   : {df_counts['combined'].mean():.2f}")
report_lines.append("")
report_lines.append("  ── Verification Against Paper ──")
actual_v2   = results_v2.get("total_valid_images", 0)
actual_orig = results_orig.get("total_valid_images", 0)
actual_comb = actual_v2 + actual_orig
report_lines.append(f"  Paper claimed V2   : {EXPECTED_V2_TOTAL}  | Actual: {actual_v2}  | {'MATCH' if actual_v2==EXPECTED_V2_TOTAL else 'MISMATCH'}")
report_lines.append(f"  Paper claimed Orig : {EXPECTED_ORIG_TOTAL} | Actual: {actual_orig} | {'MATCH' if actual_orig==EXPECTED_ORIG_TOTAL else 'MISMATCH'}")
report_lines.append(f"  Paper claimed Total: {EXPECTED_COMBINED}  | Actual: {actual_comb} | {'MATCH' if actual_comb==EXPECTED_COMBINED else 'MISMATCH'}")
report_lines.append("")
report_lines.append("  ── Image Dimensions ──")
report_lines.append(f"  Width  — min: {min(widths)}, max: {max(widths)}, mean: {np.mean(widths):.1f}")
report_lines.append(f"  Height — min: {min(heights)}, max: {max(heights)}, mean: {np.mean(heights):.1f}")
report_lines.append(f"  Modes  : {modes}")
report_lines.append(f"  Non-64x64 images: {len(non_standard)}")
report_lines.append("")
report_lines.append("  ── Per-Class Counts ──")
report_lines.append(f"  {'Class':<12} {'V2':>6} {'Orig':>6} {'Combined':>10}")
for cls in all_classes:
    row = df_counts.loc[cls]
    report_lines.append(f"  {cls:<12} {row['v2_count']:>6} {row['orig_count']:>6} {row['combined']:>10}")

report_text = "\n".join(report_lines)
report_path = REPORTS_DIR / "inspection_report.txt"
with open(report_path, "w") as f:
    f.write(report_text)
print(f"  Saved inspection report : {report_path}")

# Save JSON summary (for use by subsequent scripts)
summary_for_json = {
    "v2": {k: v for k, v in results_v2.items() if k != "all_records"},
    "orig": {k: v for k, v in results_orig.items() if k != "all_records"},
    "combined_valid": actual_comb,
    "combined_classes": len(all_classes),
}
json_path = PIPELINE_DIR / "inspection_summary.json"
with open(json_path, "w") as f:
    json.dump(summary_for_json, f, indent=2)
print(f"  Saved inspection JSON  : {json_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL STATUS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  INSPECTION COMPLETE")
print("="*70)
print(f"  Total valid images found    : {actual_comb}")
print(f"  Total classes               : {len(all_classes)}")
print(f"  Corrupted images            : {len(corrupted_records)}")
print(f"  Unsupported file types      : {len(unsupported_records)}")
print("\n  Next step: Run 02_check_duplicates.py")
print("="*70 + "\n")
