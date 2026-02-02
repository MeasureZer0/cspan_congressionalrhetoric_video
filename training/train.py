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
    return v1_list, v2_list, lengths

def supervised_collate_fn(batch):
    faces = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    lengths = torch.tensor([f.shape[0] for f in faces], dtype=torch.long)
    return faces, labels, lengths

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
    return model, feature_size


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

    def forward(self, batch_video_list: List[torch.Tensor], lengths: torch.Tensor):
        """
        Standard forward for supervised training: returns logits.
        """
        device = _get_device()
        features = [self.image_extractor(seq.to(device)) for seq in batch_video_list]
        padded = pad_sequence(features, batch_first=True)
        packed = pack_padded_sequence(
            padded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hn, _) = self.lstm(packed)
        last_hidden = hn[-1]
        logits = self.classifier(last_hidden)
        return logits

    def forward_hidden(self, batch_video_list: List[torch.Tensor], lengths: torch.Tensor):
        """
        Forward for SSL: returns LSTM hidden state without classifier.
        """
        device = _get_device()
        features = [self.image_extractor(seq.to(device)) for seq in batch_video_list]
        padded = pad_sequence(features, batch_first=True)
        packed = pack_padded_sequence(
            padded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hn, _) = self.lstm(packed)
        last_hidden = hn[-1]
        return last_hidden


class SimCLRProjectionWrapper(nn.Module):
    """Wrap LSTM encoder with a projection head for SimCLR."""

    def __init__(self, encoder: FeatureAggregatingLSTM, projection_dim=256):
        super().__init__()
        self.encoder = encoder
        lstm_out_dim = encoder.lstm.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim),
            nn.ReLU(),
            nn.Linear(lstm_out_dim, projection_dim),
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
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # [2*B, D]
        z = torch.cat([z1, z2], dim=0)
        
        # [2*B, 2*B]
        sim_matrix = torch.mm(z, z.T) / self.temperature
        
        batch_size = z1.size(0)
        labels = torch.arange(batch_size, device=z1.device)
        labels = torch.cat([labels + batch_size, labels])
        
        return F.cross_entropy(sim_matrix, labels)

def train_ssl(args, device, img_dir, weights_dir):
    print("--- STARTING SSL PRETRAINING (SimCLR) ---")
    base_ds = FacesFramesSSLDataset(img_dir)
    transform = VideoSimCLRTransform(size=224)
    ssl_ds = SimCLRDataset(base_ds, transform)
    loader = DataLoader(ssl_ds, batch_size=args.batch_size, shuffle=True, collate_fn=ssl_collate_fn)
    encoder = FeatureAggregatingLSTM().to(device)
    model = SimCLRProjectionWrapper(encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    nt_xent_loss = NTXentLoss()
    total_loss = 0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(loader, desc=f"SSL Epoch {epoch+1}", leave=True)
        for v1, v2, lengths in pbar:
            optimizer.zero_grad()
            
            z1 = model(v1, lengths)
            z2 = model(v2, lengths)
            
            loss = nt_xent_loss(z1, z2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
    
    save_path = weights_dir / "ssl_backbone.pt"
    torch.save(encoder.state_dict(), save_path)
    print(f"SSL Backbone saved to {save_path}")

def train_supervised(args, device, img_dir, csv_file, weights_dir):
    print("--- STARTING SUPERVISED TRAINING ---")
    full_ds = FacesFramesSupervisedDataset(csv_file, img_dir)
    train_idx, val_idx, test_idx = stratified_split(full_ds)

    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    val_ds = torch.utils.data.Subset(full_ds, val_idx)
    test_ds = torch.utils.data.Subset(full_ds, test_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=supervised_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=supervised_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=supervised_collate_fn)

    model = FeatureAggregatingLSTM().to(device)
    if args.load_ssl:
        ssl_path = weights_dir / "ssl_backbone.pt"
        if ssl_path.exists():
            model.load_state_dict(torch.load(ssl_path, map_location=device))
            print("Loaded SSL weights!")

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
        logits: tensor [B, um_classes]
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


def evaluate(
    model: nn.Module, dataloader: DataLoader, device: torch.device, desc: str = "Eval"
) -> tuple[float, float]:
    """
    Evaluates the model on the provided dataset.

    Args:
        model: trained model
        dataloader: DataLoader to evaluate on
        device: computation device
        desc: progress bar description string

    Returns:
        avg_loss: average masked loss
        avg_acc: average masked accuracy
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch_image, batch_flow, labels, lengths in tqdm(dataloader, desc=desc):
            labels = labels.to(device)
            logits = model(batch_image, batch_flow, lengths)
            loss = sequence_loss(logits, labels)
            batch_correct, batch_total = sequence_accuracy(logits, labels)
            total_loss += loss.item() * batch_total
            total_correct += batch_correct
            total_samples += batch_total

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["ssl", "supervised"], required=True)
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
