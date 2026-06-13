"""Self-Supervised Training Pipeline for the RADAR Anomaly Detector.

This script trains the feature aggregator and scoring head parameters of RADAR
on normal chest radiographs to map healthy feature regions to zero anomaly maps.
"""

import argparse
import os
import sys

# Inject workspace root into sys.path to allow execution from within subdirectories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.optim as optim
import yaml

from models.radar_model import RADAR
from utils.data_loader import get_data_loaders


def train(config_path: str) -> None:
    """Executes the self-supervised training loop.

    Args:
        config_path: File path to the YAML configuration file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Determine hardware acceleration backend
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using execution device: {device}")

    # Initialize DataLoader
    train_loader, _ = get_data_loaders(config)

    # Instantiate model and send to device
    model = RADAR(config).to(device)

    # Optimizer (Aggregator and Scoring Head parameters only; backbone is frozen)
    optimizer = optim.Adam(
        list(model.aggregator.parameters()) + list(model.head.parameters()),
        lr=config["training"]["learning_rate"],
    )

    # Training Loop
    model.train()
    epochs = config["training"]["epochs"]

    print("Starting RADAR Self-Supervised Training...")
    for epoch in range(epochs):
        total_loss = 0.0
        for images, _, _ in train_loader:
            images = images.to(device)

            # Forward pass
            anomaly_map = model(images)

            # Loss objective: push predicted anomaly scores to 0 for healthy images
            loss = torch.mean(anomaly_map**2)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Output status report
        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | Mean Loss: {avg_loss:.6f}")

    # Save trained parameters
    os.makedirs(config["output"]["results_dir"], exist_ok=True)
    save_path = os.path.join(config["output"]["results_dir"], "radar_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model checkpoint successfully saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RADAR model on normal images"
    )
    parser.add_argument(
        "--config", type=str, default="configs/radar_config.yaml"
    )
    args = parser.parse_args()

    train(args.config)

