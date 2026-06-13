"""Synthetic Chest Radiograph Generator for Anomaly Detection.

This module models lung phantoms and overlays custom anatomical structures
and pathological abnormalities (Pneumonia, Tumor, COVID-19) for model testing.
"""

import argparse
import os
import random
from typing import Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm


def create_lung_phantom(
    width: int = 256, height: int = 256, anomaly_type: Optional[str] = None
) -> Image.Image:
    """Generates a synthetic chest radiograph (lung phantom) image.

    Args:
        width: Targeted width of the generated image.
        height: Targeted height of the generated image.
        anomaly_type: Type of anomaly to simulate. Supported options:
            'pneumonia', 'tumor', 'covid', or None (normal).

    Returns:
        A PIL Image in grayscale ('L' mode) containing the generated radiograph.
    """
    # 1. Background (Dark)
    img = Image.new("L", (width, height), color=10)
    draw = ImageDraw.Draw(img)

    # 2. Thorax outline (Rounded rectangle-ish ellipse)
    thorax_bbox = (width * 0.1, height * 0.1, width * 0.9, height * 0.9)
    draw.ellipse(thorax_bbox, fill=30)

    # 3. Lungs (Elliptical, dark zones)
    # Left Lung
    left_lung_bbox = (width * 0.2, height * 0.2, width * 0.45, height * 0.8)
    draw.ellipse(left_lung_bbox, fill=5)

    # Right Lung
    right_lung_bbox = (width * 0.55, height * 0.2, width * 0.8, height * 0.8)
    draw.ellipse(right_lung_bbox, fill=5)

    # 4. Ribs (Arcs drawn over lungs)
    for i in range(5):
        y = height * (0.25 + i * 0.1)
        # Left ribs
        draw.arc(
            (width * 0.15, y - 20, width * 0.45, y + 20),
            start=180,
            end=360,
            fill=60,
            width=3,
        )
        # Right ribs
        draw.arc(
            (width * 0.55, y - 20, width * 0.85, y + 20),
            start=180,
            end=360,
            fill=60,
            width=3,
        )

    # 5. Spine (Central column shadow)
    draw.line(
        (width * 0.5, height * 0.15, width * 0.5, height * 0.85), fill=80, width=8
    )

    # 6. Heart (Shadow on left lung - actually right side of image)
    heart_bbox = (width * 0.45, height * 0.5, width * 0.65, height * 0.75)
    draw.ellipse(heart_bbox, fill=40)

    # Add imaging sensor noise/texture
    img_np = np.array(img)
    noise = np.random.normal(0, 5, img_np.shape)
    img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_np)

    # 7. Pathology Synthesis
    if anomaly_type:
        draw = ImageDraw.Draw(img)
        if anomaly_type == "pneumonia":
            # Blurry white patch infiltration
            for _ in range(random.randint(1, 3)):
                x = random.randint(int(width * 0.2), int(width * 0.8))
                y = random.randint(int(height * 0.3), int(height * 0.7))
                r = random.randint(10, 30)

                # Draw a blurry white patch
                patch = Image.new("L", (width, height), 0)
                p_draw = ImageDraw.Draw(patch)
                p_draw.ellipse((x - r, y - r, x + r, y + r), fill=150)
                patch = patch.filter(ImageFilter.GaussianBlur(radius=10))
                img = Image.composite(patch, img, patch)

        elif anomaly_type == "tumor":
            # Solid circular mass
            x = random.randint(int(width * 0.2), int(width * 0.8))
            y = random.randint(int(height * 0.3), int(height * 0.7))
            r = random.randint(5, 15)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=200)

        elif anomaly_type == "covid":
            # Ground-glass opacity haze over lung segments
            mask = Image.new("L", (width, height), 0)
            m_draw = ImageDraw.Draw(mask)
            m_draw.ellipse(left_lung_bbox, fill=100)
            m_draw.ellipse(right_lung_bbox, fill=100)
            mask = mask.filter(ImageFilter.GaussianBlur(radius=20))

            haze = Image.new("L", (width, height), 100)
            img = Image.composite(haze, img, mask)

    return img


def generate_dataset(base_dir: str) -> None:
    """Generates a structured medical imaging dataset for training and testing.

    Args:
        base_dir: Root directory path where generated partitions will be stored.
    """
    train_dir = os.path.join(base_dir, "train", "good")
    test_good_dir = os.path.join(base_dir, "test", "good")
    test_defective_dir = os.path.join(base_dir, "test", "defective")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_good_dir, exist_ok=True)
    os.makedirs(test_defective_dir, exist_ok=True)

    print("Generating Training Data (Normal)...")
    for i in tqdm(range(100)):
        img = create_lung_phantom()
        img.save(os.path.join(train_dir, f"normal_{i:03d}.png"))

    print("Generating Test Data (Normal)...")
    for i in tqdm(range(20)):
        img = create_lung_phantom()
        img.save(os.path.join(test_good_dir, f"test_normal_{i:03d}.png"))

    print("Generating Test Data (Anomalous)...")
    anomalies = ["pneumonia", "tumor", "covid"]
    for i in tqdm(range(15)):
        anomaly = anomalies[i % 3]
        img = create_lung_phantom(anomaly_type=anomaly)
        img.save(os.path.join(test_defective_dir, f"test_{anomaly}_{i:03d}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic lung radiograph dataset"
    )
    parser.add_argument("--output_dir", type=str, default="data")
    args = parser.parse_args()

    generate_dataset(args.output_dir)

