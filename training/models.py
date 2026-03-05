from typing import Protocol, cast

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision.models import ResNet18_Weights, resnet18


class _SupportsForwardHidden(Protocol):
    def forward_hidden(
        self, faces: torch.Tensor, pose: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor: ...


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism to aggregate LSTM/GRU outputs over time.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

    def forward(
        self, lstm_outputs: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates attention weights and weighted sum of hidden states.
        """
        scores = self.attn(lstm_outputs).squeeze(-1)  # [B, T]

        max_len = scores.size(1)
        mask = (
            torch.arange(max_len, device=scores.device)[None, :]
            < lengths.to(scores.device)[:, None]
        )
        scores[~mask] = -1e9

        weights = torch.softmax(scores, dim=1)

        context = torch.sum(lstm_outputs * weights.unsqueeze(-1), dim=1)
        return context


def build_resnet_cnn(
    input_channels: int, freeze_all: bool = False
) -> tuple[nn.Module, int]:
    """
    Builds a ResNet model that can process images with an arbitrary number of
    channels. The final fully connected layer is replaced with an identity to
    extract feature vectors.

    Args:
        input_channels (int): Number of input channels.
        freeze_all (bool): Flag to freeze whole resnet.

    Returns:
        tuple[nn.Module, int]: A tuple containing the CNN model and the feature
        dimension.
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    if input_channels == 2:
        model.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

    feature_size = model.fc.in_features
    model.fc = nn.Identity()  # type: ignore

    for name, param in model.named_parameters():
        if freeze_all:
            param.requires_grad = False
        elif "layer4" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model, feature_size


class TinyMLPEncoder(nn.Module):
    """
    Baseline model using a tiny CNN, frame averaging, and an MLP classifier.
    """

    def __init__(self, hidden_size: int = 64, num_classes: int = 3) -> None:
        super().__init__()
        # Tiny CNN
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.feature_dim = 128

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.num_classes = num_classes
        self.classifier = nn.Linear(hidden_size, num_classes)

        self.output_dim = hidden_size

    def forward(
        self, batch_padded: torch.Tensor, pose: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for supervised classification.
        """
        device = batch_padded.device
        B, T, C, H, W = batch_padded.shape
        flat = batch_padded.view(B * T, C, H, W).to(device)
        feats = self.backbone(flat).view(B, T, -1)
        avg_feats = feats.mean(dim=1)

        hidden = self.mlp(avg_feats)
        logits = self.classifier(hidden)
        return logits

    def forward_hidden(
        self, batch_padded: torch.Tensor, pose: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for self-supervised learning; returns hidden representations.
        """
        device = batch_padded.device
        B, T, C, H, W = batch_padded.shape
        flat = batch_padded.view(B * T, C, H, W).to(device)
        feats = self.backbone(flat).view(B, T, -1)
        avg_feats = feats.mean(dim=1)

        hidden = self.mlp(avg_feats)
        return hidden


class FastGRU(nn.Module):
    """ResNet-18 backbone + GRU + temporal attention (face only)."""

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone, self.feature_dim = build_resnet_cnn(
            3, freeze_all=freeze_backbone
        )
        self.gru = nn.GRU(
            self.feature_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=0.2 if num_layers > 1 else 0,
            batch_first=True,
        )
        self.attention = TemporalAttention(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.output_dim = hidden_size
        self.num_classes = num_classes

    def _encode(self, faces: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = faces.shape
        feats = self.backbone(faces.view(B * T, C, H, W)).view(B, T, -1)
        packed = pack_padded_sequence(
            feats, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        return self.attention(out, lengths)

    def forward(
        self, faces: torch.Tensor, pose: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        return self.classifier(self._encode(faces, lengths))

    def forward_hidden(
        self, faces: torch.Tensor, pose: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        return self._encode(faces, lengths)


class FeatureAggregatingLSTM(nn.Module):
    """
    CNN + LSTM: Extract frame features and model temporal dynamics using LSTM.
    """

    def __init__(
        self, hidden_size: int = 64, num_layers: int = 1, num_classes: int = 3
    ) -> None:
        super().__init__()
        self.backbone, self.feature_dim = build_resnet_cnn(3)
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.output_dim = hidden_size

    def forward(
        self, batch_padded: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for supervised classification.
        """
        device = batch_padded.device
        B, T, C, H, W = batch_padded.shape

        batch_flat = batch_padded.view(B * T, C, H, W).to(device)
        features_flat = self.backbone(batch_flat)
        features_flat = features_flat.view(B, T, -1)

        packed = pack_padded_sequence(
            features_flat, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        _, (hn, _) = self.lstm(packed)
        last_hidden = hn[-1]
        logits = self.classifier(last_hidden)
        return logits

    def forward_hidden(
        self, batch_padded: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for self-supervised learning; returns the final LSTM hidden state.
        """
        device = batch_padded.device
        B, T, C, H, W = batch_padded.shape

        batch_flat = batch_padded.view(B * T, C, H, W).to(device)
        features_flat = self.backbone(batch_flat)
        features_flat = features_flat.view(B, T, -1)

        packed = pack_padded_sequence(
            features_flat, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hn, _) = self.lstm(packed)
        last_hidden = hn[-1]
        return last_hidden


class PoseGRU(nn.Module):
    """
    Encodes a sequence of upper-body keypoints with GRU + temporal attention.

    Input keypoints: [B, T, 17, 3]  (x_norm, y_norm, confidence)
    Only upper-body keypoints (indices 0-10) are used; low-confidence
    points are zeroed before projection.
    """

    # Nose, eyes, ears, shoulders, elbows, wrists
    UPPER_BODY_IDX: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 1,
        conf_threshold: float = 0.3,
    ) -> None:
        super().__init__()
        n_kp = len(self.UPPER_BODY_IDX)
        self.conf_threshold = conf_threshold

        self.input_proj = nn.Sequential(
            nn.Linear(n_kp * 3, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            128,
            hidden_size,
            num_layers=num_layers,
            dropout=0.1 if num_layers > 1 else 0,
            batch_first=True,
        )
        self.attention = TemporalAttention(hidden_size)
        self.output_dim = hidden_size

    def forward(self, pose: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Select upper-body keypoints
        x = pose[:, :, self.UPPER_BODY_IDX, :]  # [B, T, 11, 3]
        B, T, N, C = x.shape

        # Zero low-confidence keypoints
        conf = x[:, :, :, 2]
        low_conf = conf < self.conf_threshold
        x = x.clone()
        x[low_conf] = 0.0

        x = x.reshape(B, T, N * C)
        x = self.input_proj(x)

        # Clamp lengths to actual sequence length
        lengths = torch.clamp(lengths, max=T)

        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        return self.attention(out, lengths)


class DualStreamEncoder(nn.Module):
    """
    Late-fusion model combining a face stream (FastGRU) and a pose stream
    (PoseGRU).
    """

    def __init__(
        self,
        face_hidden: int = 128,
        pose_hidden: int = 64,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.face_encoder = FastGRU(hidden_size=face_hidden)
        self.pose_encoder = PoseGRU(hidden_size=pose_hidden)

        fusion_dim = face_hidden + pose_hidden
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(fusion_dim // 2, num_classes)
        self.output_dim = fusion_dim // 2
        self.num_classes = num_classes

    def forward_hidden(
        self, faces: torch.Tensor, pose: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Return fused representation before the classifier."""
        f = self.face_encoder.forward_hidden(faces, pose, lengths)
        p = self.pose_encoder(pose, lengths)
        return self.fusion(torch.cat([f, p], dim=1))

    def forward(
        self, faces: torch.Tensor, pose: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        return self.classifier(self.forward_hidden(faces, pose, lengths))


class SimCLRProjectionWrapper(nn.Module):
    """
    Wraps any dual-stream encoder with a MLP projection head for SimCLR.
    """

    def __init__(
        self,
        encoder: nn.Module,
        encoder_output_dim: int,
        projection_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(encoder_output_dim, encoder_output_dim),
            nn.ReLU(),
            nn.Linear(encoder_output_dim, projection_dim),
        )

    def forward(
        self, faces: torch.Tensor, pose: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        enc = cast(_SupportsForwardHidden, self.encoder)
        h = enc.forward_hidden(faces, pose, lengths)
        return self.projector(h)
