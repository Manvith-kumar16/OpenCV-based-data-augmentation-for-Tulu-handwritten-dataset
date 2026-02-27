# OpenCV-based Data Augmentation for Tulu Handwritten Dataset

A Python-based tool for augmenting handwritten Tulu character images using OpenCV, designed to balance and expand datasets for machine learning applications.

## 📋 Overview

This project applies various image augmentation techniques to increase the size of a Tulu handwritten character dataset. It ensures each alphabet class has a consistent number of samples (default: 600 images per class) by synthetically generating new training data from existing samples.

## ✨ Features

- **Random Rotation**: Rotates images by ±15 degrees to simulate natural handwriting variations
- **Random Scaling**: Scales images between 0.9x to 1.1x magnification
- **Noise Addition**: Adds Gaussian noise to simulate image degradation and improve model robustness
- **Gaussian Blur**: Randomly applies blur to some images for additional variation
- **Batch Processing**: Automatically processes all alphabet classes in the dataset
- **Progress Tracking**: Uses tqdm for real-time progress visualization

## 🛠️ Requirements

- Python 3.6+
- OpenCV (`cv2`)
- NumPy
- tqdm

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/Manvith-kumar16/OpenCV-based-data-augmentation-for-Tulu-handwritten-dataset.git
cd OpenCV-based-data-augmentation-for-Tulu-handwritten-dataset
```

2. Install dependencies:
```bash
pip install opencv-python numpy tqdm
```

## 🚀 Usage

1. **Prepare your dataset** in the following structure:
```
Dataset/
├── alphabet_1/
│   ├── image_1.png
│   ├── image_2.png
│   └── ...
├── alphabet_2/
│   ├── image_1.png
│   └── ...
└── ...
```

2. **Update the dataset path** in `augmentation.py`:
```python
DATASET_PATH = r"path\to\your\Dataset"
```

3. **Set target image count** (optional):
```python
TARGET_COUNT = 600   # Number of images per alphabet class
```

4. **Run the augmentation script**:
```bash
python augmentation.py
```

## 📊 Output

- Augmented images are saved alongside original images with the naming format: `aug_{count}.png`
- Console output shows:
  - Current number of images per alphabet
  - Progress updates for each alphabet class
  - Final completion message

## 🔄 Augmentation Techniques

The script applies the following transformations to create diverse training samples:

| Technique | Parameters | Purpose |
|-----------|-----------|---------|
| Rotation | ±15° | Simulate natural handwriting angle variations |
| Scaling | 0.9x - 1.1x | Simulate pen pressure and stroke width variations |
| Gaussian Noise | σ=10 | Improve robustness to image noise |
| Blur | 3×3 kernel | Simulate camera focus variations |

## 📝 Configuration

Modify these variables in `augmentation.py` to customize the augmentation:

```python
DATASET_PATH = # Path to dataset
TARGET_COUNT = 600   # Target number of images per alphabet class
```

## 🎯 Use Cases

- Training handwritten character recognition models
- Creating balanced datasets for machine learning
- Improving model generalization through data augmentation
- Preprocessing Tulu script recognition datasets

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues and pull requests.

## 📧 Contact

For questions or suggestions, please open an issue on the GitHub repository.

---

**Note**: The Tulu script is a South Indian script used to write the Tulu language spoken in Karnataka, India.
