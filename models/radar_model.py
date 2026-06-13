"""RADAR Model Architecture for Single-Step Medical Anomaly Detection.

This module defines the PyTorch architecture for the RADAR model, including
a frozen feature extractor backbone, a multi-scale feature aggregator,
and an anomaly scoring projection head.
"""

from typing import Dict, List, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SimpleFeatureExtractor(nn.Module):
    """Frozen ResNet18 feature extractor targeting intermediate layer representations."""

    def __init__(self, backbone_name: str = "resnet18") -> None:
        """Initializes the feature extractor and freezes backbone parameters.

        Args:
            backbone_name: Name of the torchvision backbone model. Supported: 'resnet18'.
        """
        super().__init__()
        if backbone_name == "resnet18":
            # Using weights instead of pretrained=True to avoid deprecation warnings
            weights = models.ResNet18_Weights.DEFAULT
            backbone = models.resnet18(weights=weights)
            
            # Extract features from intermediate layers
            self.layer1 = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
            )
            self.layer2 = backbone.layer2
            self.layer3 = backbone.layer3
        else:
            raise NotImplementedError(f"Backbone {backbone_name} not implemented")

        # Freeze backbone parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extracts multi-scale features from intermediate ResNet layers.

        Args:
            x: Input image tensor of shape [B, C, H, W].

        Returns:
            A list containing intermediate feature maps from layers 1, 2, and 3.
        """
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return [f1, f2, f3]


class FeatureAggregator(nn.Module):
    """Aggregates multi-scale intermediate features into a unified dense representation."""

    def __init__(self, in_channels_list: List[int], out_channels: int = 128) -> None:
        """Initializes 1x1 convolutions for mapping features to a common channel depth.

        Args:
            in_channels_list: List of input channel dimensions for each feature scale.
            out_channels: Targeted channel dimension for each scale before concatenation.
        """
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels_list]
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Projects, spatially interpolates, and concatenates multi-scale features.

        Args:
            features: List of multi-scale feature maps.

        Returns:
            Concatenated feature map of shape [B, len(features) * out_channels, H_f1, W_f1].
        """
        target_size = features[0].shape[2:]
        processed_features = []
        for i, f in enumerate(features):
            f_conv = self.convs[i](f)
            if f_conv.shape[2:] != target_size:
                f_conv = F.interpolate(
                    f_conv, size=target_size, mode="bilinear", align_corners=False
                )
            processed_features.append(f_conv)

        # Concatenate along the channel dimension
        return torch.cat(processed_features, dim=1)


class AnomalyScoringHead(nn.Module):
    """Parametric projection head mapping aggregated features to a pixel-wise anomaly score."""

    def __init__(self, in_channels: int) -> None:
        """Initializes the projection head convolution.

        Args:
            in_channels: Integrated channel dimensions of the aggregated features.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies 1x1 convolution and Sigmoid to predict a spatial anomaly probability.

        Args:
            x: Input aggregated feature tensor.

        Returns:
            An anomaly probability map of shape [B, 1, H, W] with values in [0, 1].
        """
        return self.sigmoid(self.conv(x))


class RADAR(nn.Module):
    """RADAR architecture for single-step medical anomaly detection."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initializes the RADAR components according to model configurations.

        Args:
            config: Configuration dictionary loaded from config files.
        """
        super().__init__()
        self.feature_extractor = SimpleFeatureExtractor(config["model"]["backbone"])

        # ResNet18 layer channels: layer1=64, layer2=128, layer3=256
        self.aggregator = FeatureAggregator([64, 128, 256])

        # Aggregated channels = 128 * 3 = 384
        self.head = AnomalyScoringHead(384)

        # Note: A true Memory Bank stores normal features during training.
        # For this simplified "Single-Step" real-time implementation, we train
        # the 'head' to learn the distribution of normal features directly
        # (acting as a parametric memory, similar to student-teacher models).

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executes the forward pass to produce a full-resolution anomaly heatmap.

        Args:
            x: Input radiograph tensor of shape [B, C, H, W].

        Returns:
            Full-resolution spatial anomaly probability map of shape [B, 1, H, W].
        """
        features = self.feature_extractor(x)
        aggregated = self.aggregator(features)
        anomaly_map = self.head(aggregated)

        # Upsample anomaly map to match original input resolution
        anomaly_map = F.interpolate(
            anomaly_map, size=x.shape[2:], mode="bilinear", align_corners=False
        )

        return anomaly_map

