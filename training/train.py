import argparse
import csv
import random
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Optional

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import ResNet18_Weights, resnet18
from faces_frames_dataset import FacesFramesSSLDataset, FacesFramesSupervisedDataset, SimCLRDataset
from transforms import VideoSimCLRTransform
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomResizedCrop
from tqdm import tqdm

# ==========================================
# Utility
# ==========================================


# Check for GPU availability and configure device
def _get_device() -> torch.device:
    """Return the available device (cuda if available else cpu)."""
    use_cuda = torch.cuda.is_available()
    return torch.device("cuda:0" if use_cuda else "cpu")


def _default_paths(frame_skip: int) -> tuple[Path, Path, Path, Path]:
    """Return default (img_dir, csv_file, weights_dir, logs_dir) paths."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = project_root / "data"  # Main data folder
    img_dir = (
        data_dir / "faces" / f"frame_skip_{frame_skip}"
    )  # Directory containing image sequences
    csv_file = data_dir / "labels.csv"  # CSV file with labels for each sequence
    weights_dir = data_dir / "weights"  # Directory to save model weights to
    logs_dir = project_root / "logs"  # Directory to save training logs to
    return img_dir, csv_file, weights_dir, logs_dir


def ssl_collate_fn(batch):
    v1_list = [item[0] for item in batch]
    v2_list = [item[1] for item in batch]
    lengths = torch.tensor([v.shape[0] for v in v1_list], dtype=torch.long)

    v1_padded = pad_sequence(v1_list, batch_first=True)
    v2_padded = pad_sequence(v2_list, batch_first=True)

    return v1_padded, v2_padded, lengths

def supervised_collate_fn(batch):
    faces_list = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    lengths = torch.tensor([f.shape[0] for f in faces_list], dtype=torch.long)

    faces_padded = pad_sequence(faces_list, batch_first=True)
    return faces_padded, labels, lengths

# ==========================================
# Models
# ==========================================

def build_resnet_cnn(input_channels: int) -> tuple[nn.Module, int]:
    """
    Builds a ResNet model that can process images with an arbitrary number of channels.
    The final fully connected layer is replaced with an identity to \
        extract feature vectors.

    Args:
        input_channels: number of input channels (e.g., 3 for RGB, 2 for optical flow)

    Returns:
        model: CNN feature extractor
        feature_size: dimensionality of the extracted feature vector
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    if input_channels == 2:
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    feature_size = model.fc.in_features
    model.fc = nn.Identity()

    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model, feature_size

class TinyMLPEncoder(nn.Module):
    """
    Baseline: Tiny CNN + frame averaging + MLP
    """

    def __init__(self, hidden_size=64, num_classes=3):
        super().__init__()
        # Tiny CNN
        self.image_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
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

    def forward(self, batch_video_list: List[torch.Tensor], lengths: torch.Tensor):
        device = _get_device()

        padded = pad_sequence(batch_video_list, batch_first=True)

        B, T, C, H, W = padded.shape
        flat = padded.view(B*T, C, H, W).to(device)
        feats = self.image_extractor(flat).view(B, T, -1)
        avg_feats = feats.mean(dim=1)

        hidden = self.mlp(avg_feats)
        logits = self.classifier(hidden)
        return logits

    def forward_hidden(self, batch_video_list: List[torch.Tensor], lengths: torch.Tensor):
        """Forward for SSL: returns hidden representation without classifier."""
        device = _get_device()

        padded = pad_sequence(batch_video_list, batch_first=True)

        B, T, C, H, W = padded.shape
        flat = padded.view(B*T, C, H, W).to(device)
        feats = self.image_extractor(flat).view(B, T, -1)
        avg_feats = feats.mean(dim=1)

        hidden = self.mlp(avg_feats)
        return hidden


class FeatureAggregatingLSTM(nn.Module):
    """
    CNN + LSTM: Extract frame features and model temporal dynamics.
    """

    def __init__(self, hidden_size=64, num_layers=1, num_classes=3):
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

    def forward(self, batch_padded: List[torch.Tensor], lengths: torch.Tensor):
        """
        Standard forward for supervised training: returns logits.
        """
        device = _get_device()
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

    def forward_hidden(self, batch_padded: List[torch.Tensor], lengths: torch.Tensor):
        """
        Forward for SSL: returns LSTM hidden state without classifier.
        """
        device = _get_device()
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
    """Wrap any encoder with a projection head for SimCLR."""
    def __init__(self, encoder: nn.Module, encoder_output_dim: int, projection_dim=256):
        super().__init__()
        self.encoder = encoder
        
        self.projector = nn.Sequential(
            nn.Linear(encoder_output_dim, encoder_output_dim),
            nn.ReLU(),
            nn.Linear(encoder_output_dim, projection_dim),
        )

    def forward(self, x, lengths):
        h = self.encoder.forward_hidden(x, lengths)
        z = self.projector(h)
        return z
    
# ==========================================
# Training
# ==========================================

def stratified_split(
    dataset: FacesFramesSupervisedDataset,
    fractions: tuple[float, float, float],
) -> tuple[list[int], list[int], list[int]]:
    """
    Splits the dataset into stratified train, validation, and test sets.

    Args:
        dataset: the full dataset to split
        fractions: list of fractions for each subset (train, val, test), \
            in our case (0.8, 0.1, 0.1)

    Returns:
        train: List of indices for training
        val: List of indices for validation
        test: List of indices for testing
    """
    indices = (
        [],  # negative
        [],  # neutral
        [],  # positive
    )
    for i in range(len(dataset)):
        _, _, label = dataset[i]
        if label is not None:
            indices[label].append(i)

    train_indices, val_indices, test_indices = [], [], []
    cumulative_fractions = [fractions[0], fractions[0] + fractions[1]]
    for i in range(3):
        random.shuffle(indices[i])
        train_indices.extend(
            indices[i][: int(cumulative_fractions[0] * len(indices[i]))]
        )
        val_indices.extend(
            indices[i][
                int(cumulative_fractions[0] * len(indices[i])) : int(
                    cumulative_fractions[1] * len(indices[i])
                )
            ]
        )
        test_indices.extend(
            indices[i][int(cumulative_fractions[1] * len(indices[i])) :]
        )

    return train_indices, val_indices, test_indices

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1, z2):
        batch_size = z1.size(0)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # [2*B, D]
        z = torch.cat([z1, z2], dim=0)
        
        # [2*B, 2*B]
        sim_matrix = torch.mm(z, z.T) / self.temperature

        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)
        
        labels = torch.arange(batch_size, device=z1.device)
        labels = torch.cat([labels + batch_size, labels])
        
        return F.cross_entropy(sim_matrix, labels)

def train_ssl(args, device, img_dir, weights_dir):
    print(f"--- STARTING SSL PRETRAINING (SimCLR) with {args.encoder} ---")
    base_ds = FacesFramesSSLDataset(img_dir)
    transform = VideoSimCLRTransform(size=224)
    ssl_ds = SimCLRDataset(base_ds, transform)
    loader = DataLoader(ssl_ds, batch_size=args.batch_size, shuffle=True, collate_fn=ssl_collate_fn)
    
    # Choose encoder
    if args.encoder == 'baseline':
        encoder = TinyMLPEncoder(hidden_size=64).to(device)
    elif args.encoder == 'resnet_lstm':
        encoder = FeatureAggregatingLSTM(hidden_size=64).to(device)
    
    model = SimCLRProjectionWrapper(encoder, encoder_output_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    nt_xent_loss = NTXentLoss()
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(loader, desc=f"SSL Epoch {epoch+1}", leave=True)
        for v1, v2, lengths in pbar:
            optimizer.zero_grad()
            
            z1 = model(v1, lengths)
            z2 = model(v2, lengths)
            
            loss = nt_xent_loss(z1, z2)
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix(loss=loss.item())
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{args.epochs} - Avg Loss: {avg_loss:.4f}")
    
    save_path = weights_dir / f"ssl_backbone_{args.encoder}.pt"
    torch.save(encoder.state_dict(), save_path)
    print(f"SSL Backbone saved to {save_path}")

def train_supervised(args, device, img_dir, csv_file, weights_dir):
    print(f"--- STARTING SUPERVISED TRAINING with {args.encoder} ---")
    full_ds = FacesFramesSupervisedDataset(csv_file, img_dir)
    train_idx, val_idx, test_idx = stratified_split(full_ds, (0.8, 0.1, 0.1))

    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    val_ds = torch.utils.data.Subset(full_ds, val_idx)
    test_ds = torch.utils.data.Subset(full_ds, test_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=supervised_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=supervised_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=supervised_collate_fn)

    # Choose encoder
    if args.encoder == 'baseline':
        model = TinyMLPEncoder(hidden_size=64).to(device)
    elif args.encoder == 'resnet_lstm':
        model = FeatureAggregatingLSTM(hidden_size=64).to(device)
    
    if args.load_ssl:
        ssl_path = weights_dir / f"ssl_backbone_{args.encoder}.pt"
        if ssl_path.exists():
            model.load_state_dict(torch.load(ssl_path, map_location=device))
            print(f"Loaded SSL weights from {ssl_path}!")
        else:
            print(f"Warning: SSL weights not found at {ssl_path}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        correct, total = 0, 0
        for faces, labels, lengths in tqdm(train_loader, desc=f"Supervised Epoch {epoch+1}"):
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(faces, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
        train_acc = correct / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for faces, labels, lengths in val_loader:
                labels = labels.to(device)
                logits = model(faces, lengths)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")


def sequence_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes cross-entropy loss while ignoring padded positions.

    Args:
        logits: tensor [B, num_classes]
        targets: tensor [B] with true class indices

    Returns:
        scalar average loss over valid time steps
    """
    return F.cross_entropy(logits, targets)


def sequence_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    """
    Computes accuracy only for non-padded elements.

    Args:
        logits: tensor [B, num_classes]
        targets: tensor [B]
    Returns:
        tuple with (correct_count, total_count)
    """
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)
        correct = int((preds == targets).sum().item())
        total = int(targets.shape[0])
        return correct, total

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["ssl", "supervised"], required=True)
    parser.add_argument("--encoder", type=str, choices=["baseline", "resnet_lstm"], default="resnet_lstm",
                        help="Encoder architecture: tiny_mlp (simplest), resnet_lstm (ResNet+LSTM)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--load-ssl", action="store_true")
    parser.add_argument("--frame-skip", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_dir, csv_file, weights_dir, logs_dir = _default_paths(args.frame_skip)

    if args.mode == "ssl":
        train_ssl(args, device, img_dir, weights_dir)
    else:
        train_supervised(args, device, img_dir, csv_file, weights_dir)
