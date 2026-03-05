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
    """Temporal attention mechanism to aggregate GRU outputs over time."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, gru_outputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        scores = self.attn(gru_outputs).squeeze(-1)  # [B, T]
        max_len = scores.size(1)
        mask = (
            torch.arange(max_len, device=scores.device)[None, :]
            < lengths.to(scores.device)[:, None]
        )
        scores[~mask] = -1e9
        weights = torch.softmax(scores, dim=1)
        return torch.sum(gru_outputs * weights.unsqueeze(-1), dim=1)


def build_resnet_cnn(freeze_all: bool = False) -> tuple[nn.Module, int]:
    """ResNet-18 backbone with FC replaced by Identity. Only layer4 is trainable."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    feature_size: int = model.fc.in_features
    model.fc = nn.Identity()  # type: ignore[assignment]

    for name, param in model.named_parameters():
        if freeze_all:
            param.requires_grad = False
        else:
            param.requires_grad = "layer4" in name

    return model, feature_size


class FastGRU(nn.Module):
    """ResNet-18 backbone → GRU → temporal attention (face stream only)."""

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone, self.feature_dim = build_resnet_cnn(freeze_all=freeze_backbone)
        self.gru = nn.GRU(
            self.feature_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=0.2 if num_layers > 1 else 0.0,
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


class PoseGRU(nn.Module):
    """
    Encodes a sequence of upper-body keypoints with GRU + temporal attention.

    Input: [B, T, 17, 3]  (x_norm, y_norm, confidence)
    Upper-body keypoints (indices 0-10) only; low-confidence points zeroed.
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
            dropout=0.1 if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.attention = TemporalAttention(hidden_size)
        self.output_dim = hidden_size

    def forward(self, pose: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = pose[:, :, self.UPPER_BODY_IDX, :]  # [B, T, 11, 3]
        B, T, N, C = x.shape

        conf = x[:, :, :, 2]
        x = x.clone()
        x[conf < self.conf_threshold] = 0.0
        x = x.reshape(B, T, N * C)
        x = self.input_proj(x)

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
    (PoseGRU). Supports freezing the ResNet backbone via freeze_backbone.
    """

    def __init__(
        self,
        face_hidden: int = 128,
        pose_hidden: int = 64,
        num_classes: int = 3,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.face_encoder = FastGRU(
            hidden_size=face_hidden,
            freeze_backbone=freeze_backbone,
        )
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
        f = self.face_encoder.forward_hidden(faces, pose, lengths)
        p = self.pose_encoder(pose, lengths)
        return self.fusion(torch.cat([f, p], dim=1))

    def forward(
        self, faces: torch.Tensor, pose: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        return self.classifier(self.forward_hidden(faces, pose, lengths))


class SimCLRProjectionWrapper(nn.Module):
    """Wraps any encoder with an MLP projection head for SimCLR."""

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
