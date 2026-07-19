"""Neural-network architectures for video action recognition."""

import torch
from torch import nn
from torchvision import models


class Identity(nn.Module):
    """Return an input tensor without modifying it."""

    def forward(self, input_tensor):
        return input_tensor


def _create_resnet_backbone(cnn_model, pretrained):
    """Construct a supported ResNet backbone."""
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
        supported = ", ".join(backbone_options)
        raise ValueError(
            f"Unsupported CNN backbone '{cnn_model}'. "
            f"Choose one of: {supported}."
        )

    constructor, default_weights = backbone_options[cnn_model]
    weights = default_weights if pretrained else None
    return constructor(weights=weights)


class LRCN(nn.Module):
    """ResNet, bidirectional LSTM, and learned temporal attention."""

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

        base_cnn = _create_resnet_backbone(
            cnn_model=cnn_model,
            pretrained=pretrained,
        )
        num_features = base_cnn.fc.in_features
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
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, input_tensor):
        batch_size, time_steps, channels, height, width = (
            input_tensor.shape
        )

        frames = input_tensor.reshape(
            batch_size * time_steps,
            channels,
            height,
            width,
        )
        frame_features = self.base_model(frames)
        frame_features = frame_features.reshape(
            batch_size,
            time_steps,
            -1,
        )

        rnn_output, _ = self.rnn(frame_features)
        attention_scores = self.attention(rnn_output).squeeze(-1)
        attention_weights = torch.softmax(
            attention_scores,
            dim=1,
        ).unsqueeze(-1)

        video_features = torch.sum(
            rnn_output * attention_weights,
            dim=1,
        )
        return self.fc(self.dropout(video_features))


class MotionTransformerClassifier(nn.Module):
    """
    ResNet frame encoder with explicit motion features and a Transformer.

    Each frame is encoded independently by ResNet. Consecutive feature
    differences form a motion stream. Appearance and motion tokens are fused,
    supplied with learned positional embeddings, and processed by a temporal
    Transformer encoder.
    """

    def __init__(
        self,
        n_classes,
        pretrained=True,
        cnn_model="resnet50",
        transformer_dim=256,
        transformer_heads=4,
        transformer_layers=2,
        transformer_ff_dim=1024,
        dropout_rate=0.25,
        max_frames=16,
    ):
        super().__init__()

        if transformer_dim % transformer_heads != 0:
            raise ValueError(
                "transformer_dim must be divisible by transformer_heads."
            )

        base_cnn = _create_resnet_backbone(
            cnn_model=cnn_model,
            pretrained=pretrained,
        )
        num_features = base_cnn.fc.in_features
        base_cnn.fc = Identity()

        self.base_model = base_cnn
        self.max_frames = max_frames

        self.appearance_projection = nn.Sequential(
            nn.Linear(num_features, transformer_dim),
            nn.LayerNorm(transformer_dim),
            nn.GELU(),
        )
        self.motion_projection = nn.Sequential(
            nn.Linear(num_features, transformer_dim),
            nn.LayerNorm(transformer_dim),
            nn.GELU(),
        )
        self.feature_fusion = nn.Sequential(
            nn.Linear(transformer_dim * 2, transformer_dim),
            nn.LayerNorm(transformer_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.class_token = nn.Parameter(
            torch.zeros(1, 1, transformer_dim)
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, max_frames + 1, transformer_dim)
        )
        self.embedding_dropout = nn.Dropout(dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(transformer_dim),
            enable_nested_tensor=False,

        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(transformer_dim, n_classes),
        )

        nn.init.trunc_normal_(
            self.class_token,
            std=0.02,
        )
        nn.init.trunc_normal_(
            self.position_embedding,
            std=0.02,
        )

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: Tensor shaped
                (batch, time, channels, height, width).
        """
        batch_size, time_steps, channels, height, width = (
            input_tensor.shape
        )

        if time_steps > self.max_frames:
            raise ValueError(
                f"Received {time_steps} frames, but max_frames is "
                f"{self.max_frames}."
            )

        frames = input_tensor.reshape(
            batch_size * time_steps,
            channels,
            height,
            width,
        )
        frame_features = self.base_model(frames)
        frame_features = frame_features.reshape(
            batch_size,
            time_steps,
            -1,
        )

        first_motion = torch.zeros_like(
            frame_features[:, :1, :]
        )
        later_motion = (
            frame_features[:, 1:, :]
            - frame_features[:, :-1, :]
        )
        motion_features = torch.cat(
            [first_motion, later_motion],
            dim=1,
        )

        appearance_tokens = self.appearance_projection(
            frame_features
        )
        motion_tokens = self.motion_projection(
            motion_features
        )

        temporal_tokens = self.feature_fusion(
            torch.cat(
                [appearance_tokens, motion_tokens],
                dim=-1,
            )
        )

        class_token = self.class_token.expand(
            batch_size,
            -1,
            -1,
        )
        temporal_tokens = torch.cat(
            [class_token, temporal_tokens],
            dim=1,
        )

        temporal_tokens = (
            temporal_tokens
            + self.position_embedding[:, : time_steps + 1, :]
        )
        temporal_tokens = self.embedding_dropout(
            temporal_tokens
        )

        encoded_tokens = self.temporal_encoder(
            temporal_tokens
        )
        video_features = encoded_tokens[:, 0, :]

        return self.classifier(video_features)
