from typing import Protocol, cast

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision.models import ResNet18_Weights, resnet18


class _SupportsForwardHidden(Protocol):
    def forward_hidden(
        self, x: torch.Tensor, lengths: torch.Tensor
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


def build_resnet_cnn(input_channels: int) -> tuple[nn.Module, int]:
    """
    Builds a ResNet model that can process images with an arbitrary number of
    channels. The final fully connected layer is replaced with an identity to
    extract feature vectors.

    Args:
        input_channels (int): Number of input channels (e.g., 3 for RGB, 2 for
        optical flow).

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
        if "layer4" in name:
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
        self.image_extractor = nn.Sequential(
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

    def forward(self, batch_padded: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for supervised classification.
        """
        device = batch_padded.device
        B, T, C, H, W = batch_padded.shape
        flat = batch_padded.view(B * T, C, H, W).to(device)
        feats = self.image_extractor(flat).view(B, T, -1)
        avg_feats = feats.mean(dim=1)

        hidden = self.mlp(avg_feats)
        logits = self.classifier(hidden)
        return logits

    def forward_hidden(
        self, batch_padded: torch.Tensor, _: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for self-supervised learning; returns hidden representations.
        """
        device = batch_padded.device
        B, T, C, H, W = batch_padded.shape
        flat = batch_padded.view(B * T, C, H, W).to(device)
        feats = self.image_extractor(flat).view(B, T, -1)
        avg_feats = feats.mean(dim=1)

        hidden = self.mlp(avg_feats)
        return hidden


class FastGRU(nn.Module):
    """
    Sequential model with a ResNet backbone, GRU, and temporal attention.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
    ) -> None:
        super().__init__()

        self.image_extractor, self.feature_dim = build_resnet_cnn(3)

        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2 if num_layers > 1 else 0,
            batch_first=True,
        )
        self.attention = TemporalAttention(hidden_size)
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
        features_flat = self.image_extractor(batch_flat)
        features_flat = features_flat.view(B, T, -1)

        packed = pack_padded_sequence(
            features_flat, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_out, _ = self.gru(packed)
        outputs, _ = pad_packed_sequence(packed_out, batch_first=True)
        context = self.attention(outputs, lengths)
        logits = self.classifier(context)
        return logits

    def forward_hidden(
        self, batch_padded: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for self-supervised learning;
            returns features before the classifier.
        """
        device = batch_padded.device
        B, T, C, H, W = batch_padded.shape

        batch_flat = batch_padded.view(B * T, C, H, W).to(device)
        features_flat = self.image_extractor(batch_flat)
        features = features_flat.view(B, T, -1)

        packed = pack_padded_sequence(
            features, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)

        outputs, _ = pad_packed_sequence(packed_out, batch_first=True)

        context = self.attention(outputs, lengths)
        return context


class FeatureAggregatingLSTM(nn.Module):
    """
    CNN + LSTM: Extract frame features and model temporal dynamics using LSTM.
    """

    def __init__(
        self, hidden_size: int = 64, num_layers: int = 1, num_classes: int = 3
    ) -> None:
        super().__init__()
        self.image_extractor, self.feature_dim = build_resnet_cnn(3)
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.num_classes = num_classes
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self, batch_padded: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for supervised classification.
        """
        device = batch_padded.device
        B, T, C, H, W = batch_padded.shape

        batch_flat = batch_padded.view(B * T, C, H, W).to(device)
        features_flat = self.image_extractor(batch_flat)
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
        features_flat = self.image_extractor(batch_flat)
        features_flat = features_flat.view(B, T, -1)

        packed = pack_padded_sequence(
            features_flat, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hn, _) = self.lstm(packed)
        last_hidden = hn[-1]
        return last_hidden


class SimCLRProjectionWrapper(nn.Module):
    """
    Wrap any encoder with a projection head for SimCLR contrastive learning.
    """

    def __init__(
        self, encoder: nn.Module, encoder_output_dim: int, projection_dim: int = 256
    ) -> None:
        super().__init__()
        self.encoder = encoder

        self.projector = nn.Sequential(
            nn.Linear(encoder_output_dim, encoder_output_dim),
            nn.ReLU(),
            nn.Linear(encoder_output_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Extracts hidden features from the encoder
            and passes them through the projector head.
        """
        encoder = cast(_SupportsForwardHidden, self.encoder)
        h = encoder.forward_hidden(x, lengths)
        z = self.projector(h)
        return z
