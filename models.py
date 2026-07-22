"""
Module: models.py

This module defines an LRCN (Long-term Recurrent Convolutional Network)
for video classification. A pretrained ResNet extracts spatial features
from each frame, a bidirectional LSTM models temporal relationships, and
a learned attention layer combines information across the video.
"""

import torch
from torch import nn
from torchvision import models


class Identity(nn.Module):
    """Return an input tensor without modifying it."""

    def forward(self, input_tensor):
        """
        Return the input unchanged.

        Args:
            input_tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The unchanged input tensor.
        """
        return input_tensor


class LRCN(nn.Module):
    """
    Long-term Recurrent Convolutional Network for video classification.

    The model consists of:

    1. A pretrained ResNet frame-feature extractor.
    2. A bidirectional LSTM for temporal modeling.
    3. A learned temporal-attention layer.
    4. Dropout and a fully connected classification layer.

    Args:
        hidden_size (int):
            Number of features in each LSTM direction.
        n_layers (int):
            Number of stacked LSTM layers.
        dropout_rate (float):
            Dropout probability used by the LSTM and classifier.
        n_classes (int):
            Number of output classes.
        pretrained (bool, optional):
            Whether to use pretrained ImageNet weights.
        cnn_model (str, optional):
            ResNet backbone name.
    """

    def __init__(
        self,
        hidden_size,
        n_layers,
        dropout_rate,
        n_classes,
        pretrained=True,
        cnn_model="resnet34",
    ):
        super().__init__()

        base_cnn = self._create_backbone(
            cnn_model=cnn_model,
            pretrained=pretrained,
        )

        num_features = base_cnn.fc.in_features

        # Remove the original ImageNet classifier so the backbone returns
        # frame-level feature vectors.
        base_cnn.fc = Identity()
        self.base_model = base_cnn

        self.rnn = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_rate if n_layers > 1 else 0.0,
            bidirectional=True,
        )

        # Produces one importance score for every time step.
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        self.dropout = nn.Dropout(dropout_rate)

        # The bidirectional LSTM concatenates forward and backward features.
        self.fc = nn.Linear(
            hidden_size * 2,
            n_classes,
        )

    @staticmethod
    def _create_backbone(cnn_model, pretrained):
        """
        Construct the selected ResNet backbone.

        Args:
            cnn_model (str): ResNet architecture name.
            pretrained (bool): Whether to load ImageNet weights.

        Returns:
            torchvision.models.ResNet: Selected ResNet model.

        Raises:
            ValueError: If the requested backbone is unsupported.
        """
        backbone_options = {
            "resnet18": (
                models.resnet18,
                models.ResNet18_Weights.DEFAULT,
            ),
            "resnet34": (
                models.resnet34,
                models.ResNet34_Weights.DEFAULT,
            ),
            "resnet50": (
                models.resnet50,
                models.ResNet50_Weights.DEFAULT,
            ),
            "resnet101": (
                models.resnet101,
                models.ResNet101_Weights.DEFAULT,
            ),
            "resnet152": (
                models.resnet152,
                models.ResNet152_Weights.DEFAULT,
            ),
        }

        if cnn_model not in backbone_options:
            supported_models = ", ".join(backbone_options)

            raise ValueError(
                f"Unsupported CNN backbone '{cnn_model}'. "
                f"Choose one of: {supported_models}."
            )

        model_constructor, default_weights = backbone_options[cnn_model]
        weights = default_weights if pretrained else None

        return model_constructor(weights=weights)

    def forward(self, input_tensor):
        """
        Perform an LRCN forward pass.

        Args:
            input_tensor (torch.Tensor):
                Video tensor shaped
                (batch_size, time_steps, channels, height, width).

        Returns:
            torch.Tensor:
                Classification logits shaped
                (batch_size, n_classes).
        """
        batch_size, time_steps, channels, height, width = (
            input_tensor.shape
        )

        # Process all video frames through ResNet in one operation.
        frames = input_tensor.reshape(
            batch_size * time_steps,
            channels,
            height,
            width,
        )

        frame_features = self.base_model(frames)

        # Restore the video sequence structure.
        frame_features = frame_features.reshape(
            batch_size,
            time_steps,
            -1,
        )

        # Each time step contains concatenated forward and backward features.
        rnn_output, _ = self.rnn(frame_features)

        # Calculate a learned importance score for every frame.
        attention_scores = self.attention(
            rnn_output
        ).squeeze(-1)

        attention_weights = torch.softmax(
            attention_scores,
            dim=1,
        ).unsqueeze(-1)

        # Weighted combination of all LSTM time-step outputs.
        video_features = torch.sum(
            rnn_output * attention_weights,
            dim=1,
        )

        video_features = self.dropout(video_features)

        return self.fc(video_features)
    
class SlowFusion3D(nn.Module):
    """
    ResNet spatial feature extractor followed by 3D convolutions.

    ResNet processes each frame through layer3 while preserving its
    spatial feature map. The 3D convolution block then learns joint
    temporal and spatial features before classification.
    """

    def __init__(
        self,
        n_classes,
        dropout_rate=0.4,
        pretrained=True,
        cnn_model="resnet50",
    ):
        super().__init__()

        # Reuse the ResNet-construction function from LRCN.
        base_cnn = LRCN._create_backbone(
            cnn_model=cnn_model,
            pretrained=pretrained,
        )

        # layer3 has half as many output channels as the complete
        # ResNet feature vector:
        # ResNet-50: 1024
        # ResNet-34: 256
        layer3_channels = (
            base_cnn.fc.in_features // 2
        )

        self.base_model = base_cnn

        # Reduce the large ResNet feature map before the expensive
        # three-dimensional convolutions.
        self.channel_projection = nn.Sequential(
            nn.Conv3d(
                in_channels=layer3_channels,
                out_channels=64,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=64,
            ),
            nn.GELU(),
        )

        # Joint convolution over:
        # time × height × width.
        self.spatiotemporal_block = nn.Sequential(
            nn.Conv3d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3, 3),
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=128,
            ),
            nn.GELU(),

            # Preserve time while reducing spatial dimensions.
            nn.MaxPool3d(
                kernel_size=(1, 2, 2),
            ),

            nn.Conv3d(
                in_channels=128,
                out_channels=256,
                kernel_size=(3, 3, 3),
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=16,
                num_channels=256,
            ),
            nn.GELU(),

            # Produce one feature vector for the full video.
            nn.AdaptiveAvgPool3d(
                output_size=(1, 1, 1),
            ),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(
                256,
                n_classes,
            ),
        )

    def forward(self, input_tensor):
        """
        Perform a slow-fusion forward pass.

        Args:
            input_tensor:
                Video tensor shaped
                (batch, time, channels, height, width).

        Returns:
            Classification logits shaped
            (batch, n_classes).
        """
        (
            batch_size,
            time_steps,
            channels,
            height,
            width,
        ) = input_tensor.shape

        # Process all frames with ResNet in one operation.
        frames = input_tensor.reshape(
            batch_size * time_steps,
            channels,
            height,
            width,
        )

        # Run ResNet only through layer3. Do not use layer4,
        # global average pooling, or the ImageNet classifier.
        features = self.base_model.conv1(frames)
        features = self.base_model.bn1(features)
        features = self.base_model.relu(features)
        features = self.base_model.maxpool(features)
        features = self.base_model.layer1(features)
        features = self.base_model.layer2(features)
        features = self.base_model.layer3(features)

        (
            _,
            feature_channels,
            feature_height,
            feature_width,
        ) = features.shape

        # Restore the temporal dimension.
        features = features.reshape(
            batch_size,
            time_steps,
            feature_channels,
            feature_height,
            feature_width,
        )

        # Conv3d expects:
        # (batch, channels, time, height, width)
        features = features.permute(
            0,
            2,
            1,
            3,
            4,
        ).contiguous()

        features = self.channel_projection(
            features
        )
        features = self.spatiotemporal_block(
            features
        )

        return self.classifier(features)