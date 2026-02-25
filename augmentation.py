import os
import cv2
import numpy as np
from tqdm import tqdm
import random

DATASET_PATH = r"D:\MINI PROJECT\Datasets\Dataset2\final_dataset"

TARGET_COUNT = 600   # final images per alphabet


def augment_image(img):
    h, w = img.shape[:2]

    # Random rotation
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Random scaling
    scale = random.uniform(0.9, 1.1)
    img = cv2.resize(img, None, fx=scale, fy=scale)
    img = cv2.resize(img, (w, h))

    # Random noise
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)

    # Random blur
    if random.random() > 0.5:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


for alphabet in os.listdir(DATASET_PATH):

    folder_path = os.path.join(DATASET_PATH, alphabet)
    images = os.listdir(folder_path)

    print(f"Processing {alphabet} | Current = {len(images)}")

    count = len(images)
    idx = 0

    while count < TARGET_COUNT:

        img_name = images[idx % len(images)]
        img_path = os.path.join(folder_path, img_name)

        img = cv2.imread(img_path)

        aug_img = augment_image(img)

        new_name = f"aug_{count}.png"
        cv2.imwrite(os.path.join(folder_path, new_name), aug_img)

        count += 1
        idx += 1

    print(f"✅ {alphabet} → {count} images")


print("\n🎉 DATA AUGMENTATION COMPLETE!")
