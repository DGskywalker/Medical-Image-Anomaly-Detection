"""Custom DataLoader and Dataset utilities for RADAR medical anomaly detection.

This module provides class definitions and helpers to load normal
images for model training and mixed normal/defective images for evaluation.
"""

import os
from typing import Any, Dict, Tuple
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class MedicalImageDataset(Dataset):
    """Custom dataset mapping lung radiograph images and their binary labels."""

    def __init__(
        self, root_dir: str, transform: Any = None, is_train: bool = True
    ) -> None:
        """Initializes dataset paths and labels by parsing directories.

        Args:
            root_dir: Target directory path containing images or subdirectories.
            transform: Optional torchvision transformations to apply to images.
            is_train: Boolean to set whether the dataset is for training
                (reads all normal files) or testing (reads subdirectories 'good' and 'defective').
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []  # 0 for normal, 1 for anomaly

        if is_train:
            # Training expects only normal images in the training folder
            if os.path.exists(root_dir):
                for fname in os.listdir(root_dir):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.image_paths.append(os.path.join(root_dir, fname))
                        self.labels.append(0)
        else:
            # Testing partition checks distinct 'good' and 'defective' subdirectories
            good_dir = os.path.join(root_dir, "good")
            defective_dir = os.path.join(root_dir, "defective")

            if os.path.exists(good_dir):
                for fname in os.listdir(good_dir):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.image_paths.append(os.path.join(good_dir, fname))
                        self.labels.append(0)

            if os.path.exists(defective_dir):
                for fname in os.listdir(defective_dir):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.image_paths.append(os.path.join(defective_dir, fname))
                        self.labels.append(1)

    def __len__(self) -> int:
        """Returns the total number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Any, int, str]:
        """Loads and processes the image at the given index.

        Args:
            idx: Index of image element to retrieve.

        Returns:
            A tuple of (transformed_image_tensor, integer_label, image_filepath).
        """
        img_path = self.image_paths[idx]
        # Open as RGB (even if grayscale) to match backbone pretraining expectations
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, img_path


def get_data_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Generates PyTorch DataLoader loaders for both training and testing datasets.

    Args:
        config: Configuration dictionary specifying input sizes, directories, batch sizes.

    Returns:
        A tuple containing (train_loader, test_loader).
    """
    input_size = config["model"]["input_size"]
    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            # ResNet standard normalization parameters
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    train_dataset = MedicalImageDataset(
        root_dir=config["data"]["train_dir"], transform=transform, is_train=True
    )

    test_dataset = MedicalImageDataset(
        root_dir=config["data"]["test_dir"], transform=transform, is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Inference evaluated image by image
        shuffle=False,
        num_workers=0,
    )

    return train_loader, test_loader

