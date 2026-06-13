"""Evaluation and Inference Pipeline for the RADAR Anomaly Detector.

This script supports batch inference over a test partition and interactive
single-image diagnostic prediction with anomaly contour localization.
"""

import argparse
import os
import sys
import time
from typing import Optional, Dict, Any

# Inject workspace root into sys.path to allow execution from within subdirectories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.patches import Circle
from PIL import Image
from skimage import measure
from torchvision import transforms

from models.radar_model import RADAR
from utils.data_loader import get_data_loaders


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Converts a normalized PyTorch tensor back to a displayable NumPy image.

    Args:
        tensor: Image tensor of shape [C, H, W] normalized with ResNet statistics.

    Returns:
        De-normalized RGB NumPy array of shape [H, W, C] clipped to [0, 1].
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img


def run_single_inference(
    model: RADAR,
    image_path: str,
    transform: transforms.Compose,
    device: torch.device,
    config: Dict[str, Any],
) -> None:
    """Executes prediction and contour visualization on a single image.

    Args:
        model: Loaded RADAR evaluation model.
        image_path: Path to the target medical radiograph.
        transform: Image preprocessing transformations.
        device: PyTorch hardware acceleration backend.
        config: Configurations dictionary.
    """
    print(f"Processing single image: {image_path}")
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return

    try:
        orig_pil = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    image_tensor = transform(orig_pil).unsqueeze(0).to(device)

    # Perform inference and record latency
    start_time = time.time()
    with torch.no_grad():
        anomaly_map = model(image_tensor)
    end_time = time.time()
    inference_time_ms = (end_time - start_time) * 1000
    print(f"Inference Time: {inference_time_ms:.2f} ms")

    heatmap = anomaly_map[0, 0].cpu().numpy()
    orig_img_np = denormalize(image_tensor[0])

    # Overlay visualization setup
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(orig_img_np)
    ax.imshow(heatmap, cmap="jet", alpha=0.4, vmin=0, vmax=1)

    # Extract contour boundaries on scoring maps above detection threshold
    threshold = 0.05
    binary_map = heatmap > threshold
    contours = measure.find_contours(binary_map, 0.5)

    print(f"Max Anomaly Score: {np.max(heatmap):.4f}")
    print(f"Found {len(contours)} anomalous regions.")

    if len(contours) == 0:
        print("No anomalies detected above threshold.")

    for i, contour in enumerate(contours):
        # skimage contour coordinates returned as (row, col) -> (y, x)
        y = contour[:, 0]
        x = contour[:, 1]

        y_min, y_max = y.min(), y.max()
        x_min, x_max = x.min(), x.max()

        center_y = (y_min + y_max) / 2
        center_x = (x_min + x_max) / 2

        # Compute padded bounding circle radius
        height_c = y_max - y_min
        width_c = x_max - x_min
        radius = max(height_c, width_c) / 2 * 1.5

        # Filter out minor noise elements
        if radius < 5:
            continue

        print(
            f"  Region {i+1}: Center=({center_x:.1f}, {center_y:.1f}), Radius={radius:.1f}"
        )

        circ = Circle(
            (center_x, center_y),
            radius,
            fill=False,
            color="red",
            linewidth=4,
            linestyle="-",
        )
        ax.add_patch(circ)

    ax.axis("off")

    # Save outputs
    output_dir = os.path.join(config["output"]["results_dir"], "single_inference")
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(image_path)
    save_name = f"{os.path.splitext(filename)[0]}_result.png"
    save_path = os.path.join(output_dir, save_name)

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Result successfully saved to: {save_path}")


def run_batch_inference(
    model: RADAR, test_loader: Any, device: torch.device, config: Dict[str, Any]
) -> None:
    """Runs batch predictions across the test set and calculates metrics.

    Args:
        model: Loaded RADAR evaluation model.
        test_loader: PyTorch DataLoader for the test set.
        device: PyTorch execution backend.
        config: Configuration parameters.
    """
    output_dir = os.path.join(config["output"]["results_dir"], "visualizations")
    os.makedirs(output_dir, exist_ok=True)

    latencies = []
    print("Starting Batch Inference...")

    with torch.no_grad():
        for i, (image, label, path) in enumerate(test_loader):
            image = image.to(device)

            # Execution timing
            start_time = time.time()
            anomaly_map = model(image)
            end_time = time.time()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            # Generate multi-panel plots for anomalies and subset of normal images
            is_anomaly = label.item() == 1
            filename = os.path.basename(path[0])

            if i < 10 or is_anomaly:
                orig_img = denormalize(image[0])
                heatmap = anomaly_map[0, 0].cpu().numpy()

                # Generate binary mask using default confidence threshold
                threshold = 0.5
                mask = (heatmap > threshold).astype(np.float32)
                score = np.max(heatmap)

                # 1x4 Panel visual layout
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))

                axes[0].imshow(orig_img)
                axes[0].set_title(
                    f"Original\nLabel: {'Anomaly' if is_anomaly else 'Normal'}"
                )
                axes[0].axis("off")

                axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
                axes[1].set_title(f"Anomaly Map\nScore: {score:.3f}")
                axes[1].axis("off")

                axes[2].imshow(mask, cmap="gray")
                axes[2].set_title("Binary Mask")
                axes[2].axis("off")

                axes[3].imshow(orig_img)
                axes[3].imshow(heatmap, cmap="jet", alpha=0.4)
                axes[3].set_title(f"Overlay\nTime: {latency_ms:.1f}ms")
                axes[3].axis("off")

                plt.tight_layout()
                save_name = f"{'ANOMALY' if is_anomaly else 'NORMAL'}_{filename}"
                plt.savefig(os.path.join(output_dir, save_name))
                plt.close()

    avg_time = np.mean(latencies)
    print(f"\nAverage Inference Time: {avg_time:.2f} ms")
    print(f"Speedup vs Traditional Diffusion (2500ms): {2500/avg_time:.1f}x")
    print(f"Throughput: {1000/avg_time:.1f} FPS")


def inference(config_path: str, image_path: Optional[str] = None) -> None:
    """Initializes configurations, loads model weights, and runs inference.

    Args:
        config_path: Path to the YAML configuration file.
        image_path: Optional path to a single radiograph image. If omitted,
            triggers interactive CLI mode.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Device backend selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using execution device: {device}")

    # Load Model structure and parameters
    model = RADAR(config).to(device)
    model_path = os.path.join(config["output"]["results_dir"], "radar_model.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}. Train model first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Image transformations
    input_size = config["model"]["input_size"]
    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    # Interactive CLI routing
    if not image_path:
        print("\n" + "=" * 50)
        print("INTERACTIVE RADAR INFERENCE ENGINE")
        print("Please drag & drop or type a chest radiograph image path:")
        print("(Press Ctrl+C to terminate)")
        print("=" * 50)
        try:
            image_path = input("Image Path: ").strip()
            # Standardize path string cleanups from terminal interactions
            image_path = image_path.replace("'", "").replace('"', "").strip()
        except KeyboardInterrupt:
            print("\nExiting CLI program...")
            return

    if image_path:
        run_single_inference(model, image_path, transform, device, config)
    else:
        _, test_loader = get_data_loaders(config)
        run_batch_inference(model, test_loader, device, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RADAR medical image anomaly detection inference"
    )
    parser.add_argument(
        "--config", type=str, default="configs/radar_config.yaml"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        help="Path to single input image for localized prediction",
    )
    args = parser.parse_args()

    inference(args.config, args.image_path)

