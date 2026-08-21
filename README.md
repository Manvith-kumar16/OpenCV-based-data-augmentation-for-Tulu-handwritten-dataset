# OpenCV-based-data-augmentation-for-Tulu-handwritten-dataset

A robust data pipeline for processing, deduplicating, augmenting, and splitting the Tulu handwritten dataset using OpenCV. 

## 🚀 Overview

This repository contains a comprehensive data engineering and augmentation pipeline specifically designed for handwritten characters in the Tulu script. It transforms raw image data into a machine-learning-ready dataset by ensuring data quality (removing duplicates), generating reproducible train/val/test splits, and enriching the dataset using robust computer vision augmentations via OpenCV.

## 🛠️ Pipeline Architecture

The pipeline is organized into a sequential set of 8 Python scripts within the `tulu_dataset_pipeline` directory:

1. **`01_inspect_dataset.py`**
   - Scans the initial raw dataset.
   - Generates manifests and class-wise counts to identify imbalances or data irregularities.

2. **`02_check_duplicates.py`**
   - Utilizes perceptual hashing and pixel-level comparison to identify both exact and near-duplicates.
   - Produces reports highlighting redundant data to ensure dataset integrity.

3. **`03_create_master_dataset.py`**
   - Cleans the raw data by applying the deduplication logic.
   - Consolidates all unique, high-quality images into a unified master dataset.

4. **`04_create_split.py`**
   - Stratifies and splits the master dataset into reproducible `train`, `validation`, and `test` subsets.
   - Maintains class distribution ratios across all subsets.

5. **`05_generate_split_report.py`**
   - Validates the splits.
   - Generates statistical summaries and manifests (JSON/CSV) to document the exact composition of each subset.

6. **`06_augmentation.py`**
   - The core OpenCV augmentation engine. 
   - Dynamically applies transformations such as:
     - **Rotation:** Simulates varied handwriting slants.
     - **Scaling:** Mimics different writing sizes and distances.
     - **Gaussian Noise:** Enhances model robustness against sensor noise.
     - **Gaussian Blur:** Simulates out-of-focus or lower-resolution captures.
   - Applied selectively to the training split to artificially expand the dataset without leaking augmented data into validation/test sets.

7. **`07_generate_augmentation_visualization.py`**
   - Generates visual grids (e.g., `augmentation_examples.png`) to inspect the applied augmentations qualitatively.
   - Useful for verifying that augmentations maintain character legibility.

8. **`08_verify_dataset_and_report.py`**
   - The final verification step.
   - Confirms dataset integrity, checks for corrupted files, and produces the final dataset summaries.

## 📊 Reports & Outputs

The pipeline generates comprehensive telemetry during execution, saved in the `reports/` directory and as root-level CSV/JSON manifests:
- **Manifests:** `dataset_manifest.csv`, `split_manifest.csv`, `augmentation_manifest.csv`
- **Summaries:** `duplicate_summary.json`, `split_summary.json`
- **Visuals:** `augmentation_pipeline_flow.png`, `augmentation_examples.png`

## 💻 Requirements & Usage

*   Python 3.8+
*   OpenCV (`opencv-python`)
*   NumPy
*   Pandas

To run the pipeline, execute the scripts sequentially from `01` to `08`. Note: The large dataset folders and zip files are intentionally omitted from this repository via `.gitignore` to comply with Git size limits.

---
*Created as part of the Tulu handwritten dataset research and augmentation project.*
