import argparse
import csv
import random
import os
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Subset
from torchvision.models import ResNet18_Weights, resnet18
from faces_frames_dataset import FacesFramesSSLDataset, FacesFramesSupervisedDataset, SimCLRDataset
from transforms import VideoSimCLRTransform
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix

# ==========================================
# Utility
# ==========================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

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
        data_dir / "self-supervised"
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

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.verbose:
                print(f"Initial validation loss: {val_loss:.4f}")
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(f"Validation loss improved: {self.best_loss:.4f} → {val_loss:.4f}")
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop
    
class MemoryBank(nn.Module):
    def __init__(self, size: int, dim: int):
        super().__init__()
        self.size = size
        self.dim = dim
        
        init = F.normalize(torch.randn(size, dim), dim=1)
        
        self.register_buffer("bank", init)
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("is_full", torch.zeros(1, dtype=torch.bool))

    @torch.no_grad()
    def enqueue(self, z: torch.Tensor) -> None:
        z = F.normalize(z.detach(), dim=1)
        
        batch_size = z.shape[0]
        ptr = self.ptr.item()
        
        if batch_size > self.size:
            z = z[-self.size:]
            batch_size = self.size
            
        if ptr + batch_size <= self.size:
            self.bank[ptr : ptr + batch_size] = z
        else:
            tail = self.size - ptr
            self.bank[ptr:] = z[:tail]
            self.bank[: batch_size - tail] = z[tail:]

        new_ptr = (ptr + batch_size) % self.size
        self.ptr[0] = new_ptr
        
        if new_ptr < ptr or (ptr + batch_size) >= self.size:
            self.is_full[0] = True

    def get(self) -> torch.Tensor:
        if self.is_full.item():
            return self.bank.clone()
        return self.bank[: self.ptr.item()].clone()

    def __len__(self) -> int:
        return self.size if self.is_full.item() else self.ptr.item()

    def __repr__(self) -> str:
        return (f"MemoryBank(size={self.size}, dim={self.dim}, "
                f"filled={len(self)}/{self.size})")

# ==========================================
# Models
# ==========================================

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, lstm_outputs, lengths):
        scores = self.attn(lstm_outputs).squeeze(-1)  # [B, T]

        max_len = scores.size(1)
        mask = torch.arange(max_len, device = scores.device)[None, :] < lengths.to(scores.device)[:, None]
        scores[~mask] = -1e9

        weights = torch.softmax(scores, dim=1)

        context = torch.sum(lstm_outputs * weights.unsqueeze(-1), dim=1)
        return context

class EfficientCNN(nn.Module):
    
    def __init__(self, hidden_size=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d(1)
        )
        self.feature_dim = hidden_size
        
    def forward(self, x):
        return self.features(x).squeeze(-1).squeeze(-1)

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
        if "layer4" in name:
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

    def forward(self, batch_padded: torch.Tensor, lengths: torch.Tensor):
        device = batch_padded.device

        padded = pad_sequence(batch_padded, batch_first=True)

        B, T, C, H, W = padded.shape
        flat = padded.view(B*T, C, H, W).to(device)
        feats = self.image_extractor(flat).view(B, T, -1)
        avg_feats = feats.mean(dim=1)

        hidden = self.mlp(avg_feats)
        logits = self.classifier(hidden)
        return logits

    def forward_hidden(self, batch_padded: torch.Tensor, lengths: torch.Tensor):
        """Forward for SSL: returns hidden representation without classifier."""
        device = batch_padded.device

        padded = pad_sequence(batch_padded, batch_first=True)

        B, T, C, H, W = padded.shape
        flat = padded.view(B*T, C, H, W).to(device)
        feats = self.image_extractor(flat).view(B, T, -1)
        avg_feats = feats.mean(dim=1)

        hidden = self.mlp(avg_feats)
        return hidden

class FastGRU(nn.Module):
    
    def __init__(self, hidden_size=128, num_layers=2, num_classes=3, use_efficient_cnn=True):
        super().__init__()
        
        if use_efficient_cnn:
            self.image_extractor = EfficientCNN(hidden_size=128)
            self.feature_dim = 128
        else:
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

    def forward(self, batch_padded: torch.Tensor, lengths: torch.Tensor):
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

    def forward_hidden(self, batch_padded: torch.Tensor, lengths: torch.Tensor):
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

    def forward_hidden(self, batch_padded: List[torch.Tensor], lengths: torch.Tensor):
        """
        Forward for SSL: returns LSTM hidden state without classifier.
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
        _, label = dataset[i]
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
        z1 = F.normalize(z1.float(), dim=1)
        z2 = F.normalize(z2.float(), dim=1)

        z = torch.cat([z1, z2], dim=0)                        
        sim_matrix = torch.mm(z, z.T) / self.temperature     

        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

        labels = torch.arange(batch_size, device=z1.device)
        labels = torch.cat([labels + batch_size, labels])

        return F.cross_entropy(sim_matrix, labels)


class NTXentLossWithMemoryBank(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, memory_bank) -> torch.Tensor:
        B = z1.size(0)
        z1 = F.normalize(z1.float(), dim=1)
        z2 = F.normalize(z2.float(), dim=1)

        pos_sim = (z1 * z2).sum(dim=1, keepdim=True) / self.temperature

        inbatch_sim = torch.mm(z1, z2.T) / self.temperature
        diag_mask = torch.eye(B, device=z1.device, dtype=torch.bool)
        inbatch_sim = inbatch_sim.masked_fill(diag_mask, float('-inf'))

        bank = memory_bank.get()
        bank_sim = torch.mm(z1, bank.T) / self.temperature

        logits = torch.cat([pos_sim, inbatch_sim, bank_sim], dim=1)
        labels = torch.zeros(B, dtype=torch.long, device=z1.device)
        loss = F.cross_entropy(logits, labels)

        memory_bank.enqueue(z2)
        return loss
    
def _build_encoder(args, device):
    if args.encoder == 'baseline':
        encoder = TinyMLPEncoder(hidden_size=64).to(device)
        encoder_dim = 64
    elif args.encoder == 'fast_gru':
        encoder = FastGRU(hidden_size=128, use_efficient_cnn=False).to(device)
        encoder_dim = 128
    elif args.encoder == 'resnet_lstm':
        encoder = FeatureAggregatingLSTM(hidden_size=64).to(device)
        encoder_dim = 64
    else:
        raise ValueError(f"Unknown encoder: {args.encoder}")
    return encoder, encoder_dim

def _build_optimizer(model, args):
    if args.encoder == 'fast_gru':
        return torch.optim.Adam([
            {"params": model.image_extractor.parameters(), "lr": 1e-3},
            {"params": model.gru.parameters(),             "lr": 1e-3},
            {"params": model.attention.parameters(),       "lr": 1e-3},
            {"params": model.classifier.parameters(),      "lr": 1e-3},
        ])
    elif args.encoder == 'resnet_lstm':
        return torch.optim.Adam([
            {"params": model.image_extractor.parameters(), "lr": 1e-5},
            {"params": model.lstm.parameters(),            "lr": 1e-4},
            {"params": model.classifier.parameters(),      "lr": 5e-4},
        ])
    else:  # baseline
        return torch.optim.Adam(model.parameters(), lr=1e-4)

def train_ssl(args, device, img_dir, weights_dir):
    print(f"--- STARTING SSL PRETRAINING (SimCLR) with {args.encoder} ---")
    base_ds = FacesFramesSSLDataset(img_dir)

    if args.subset and args.subset < len(base_ds):
        indices = random.sample(range(len(base_ds)), args.subset)
        base_ds = Subset(base_ds, indices)
        print(f"Using subset of {args.subset} samples for SSL training")

    transform = VideoSimCLRTransform(size=128)
    ssl_ds = SimCLRDataset(base_ds, transform)
    loader = DataLoader(ssl_ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=ssl_collate_fn, pin_memory=True)

    encoder, encoder_dim = _build_encoder(args, device)
    projection_dim = 256

    if args.encoder == 'fast_gru':
        for name, param in encoder.image_extractor.named_parameters():
            if "layer3" in name or "layer4" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    model = SimCLRProjectionWrapper(encoder, encoder_output_dim=encoder_dim,
                                    projection_dim=projection_dim).to(device)
    optimizer = torch.optim.Adam([
        {"params": encoder.image_extractor.parameters(), "lr": 1e-5},
        {"params": encoder.gru.parameters(),             "lr": 1e-4},
        {"params": model.projector.parameters(),         "lr": 1e-4},
    ])

    if args.use_memory_bank:
        memory_bank = MemoryBank(size=args.bank_size, dim=projection_dim).to(device)
        criterion = NTXentLossWithMemoryBank(temperature=args.temperature)
        print(f"Using Memory Bank | size={args.bank_size}, dim={projection_dim}, temperature={args.temperature}")
    else:
        criterion = NTXentLoss(temperature=args.temperature)
        memory_bank = None
        print(f"Using standard NTXentLoss | temperature={args.temperature}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    print(f"Batch size: {args.batch_size} | Initial LR: 1e-4")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        pbar = tqdm(loader, desc=f"SSL Epoch {epoch+1}", leave=True)

        for v1, v2, lengths in pbar:
            v1 = v1.to(device)
            v2 = v2.to(device)

            optimizer.zero_grad()

            if args.use_memory_bank:
                z1 = model(v1, lengths)
                z2 = model(v2, lengths)
                loss = criterion(z1, z2, memory_bank)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            else:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    z1 = model(v1, lengths)
                    z2 = model(v2, lengths)
                loss = criterion(z1, z2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            postfix = {"loss": f"{loss.item():.4f}"}
            if args.use_memory_bank:
                postfix["bank"] = f"{len(memory_bank)}/{args.bank_size}"
            pbar.set_postfix(**postfix)

        avg_loss = epoch_loss / num_batches
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.epochs} - Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        scheduler.step()

    save_path = weights_dir / f"ssl_backbone_{args.encoder}.pt"
    torch.save(encoder.state_dict(), save_path)
    print(f"SSL Backbone saved to {save_path}")

def train_supervised(args, device, img_dir, csv_file, weights_dir, logs_dir):
    print(f"--- STARTING SUPERVISED TRAINING with {args.encoder} ---")

    full_ds = FacesFramesSupervisedDataset(csv_file, img_dir)
    train_idx, val_idx, test_idx = stratified_split(full_ds, (0.8, 0.1, 0.1))

    train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=args.batch_size,
                              shuffle=True, collate_fn=supervised_collate_fn, pin_memory=True)
    val_loader   = DataLoader(Subset(full_ds, val_idx),   batch_size=args.batch_size,
                              collate_fn=supervised_collate_fn, pin_memory=True)
    test_loader  = DataLoader(Subset(full_ds, test_idx),  batch_size=args.batch_size,
                              collate_fn=supervised_collate_fn, pin_memory=True)

    model, _ = _build_encoder(args, device)

    if args.load_ssl:
        ssl_path = weights_dir / f"ssl_backbone_{args.encoder}.pt"
        if ssl_path.exists():
            model.load_state_dict(torch.load(ssl_path, map_location=device), strict=False)
            print(f"Loaded SSL weights from {ssl_path}")
        else:
            print(f"Warning: SSL weights not found at {ssl_path}")

    optimizer = _build_optimizer(model, args)

    class_counts = np.bincount([label for _, label in full_ds])
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    weights = weights / weights.sum()
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=15, min_delta=0.0001)

    best_val_acc = 0.0
    best_model_path = weights_dir / f"best_{args.encoder}_supervised.pt"

    logs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = logs_dir / f"supervised_{args.encoder}_log.csv"

    csv_file_handle = open(csv_path, mode='w', newline='')
    writer = csv.writer(csv_file_handle)
    writer.writerow([
        "epoch", "train_loss", "val_loss",
        "train_acc", "val_acc", "val_f1_macro", "val_conf_matrix", "lr"
    ])

    try:
        for epoch in range(args.epochs):
            # ==================== TRAINING ====================
            model.train()
            train_loss = 0.0
            correct, total = 0, 0

            pbar = tqdm(train_loader, desc=f"Supervised Epoch {epoch+1}")
            for faces, labels, lengths in pbar:
                faces  = faces.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = model(faces, lengths)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                pbar.set_postfix(loss=f"{loss.item():.4f}",
                                 acc=f"{correct/total:.4f}" if total > 0 else "0")

            train_acc = correct / total
            avg_train_loss = train_loss / len(train_loader)

            # ==================== VALIDATION ====================
            model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for faces, labels, lengths in val_loader:
                    faces  = faces.to(device)
                    labels = labels.to(device)
                    logits = model(faces, lengths)
                    val_loss += criterion(logits, labels).item()
                    preds = logits.argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
            val_f1  = f1_score(all_labels, all_preds, average='macro')
            val_cm  = confusion_matrix(all_labels, all_preds)
            val_cm_str = ";".join([",".join(map(str, row)) for row in val_cm])
            current_lr = optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val   Loss: {avg_val_loss:.4f}, Val   Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            print(f"  LR: {current_lr:.6f}")

            writer.writerow([
                epoch + 1,
                f"{avg_train_loss:.6f}",
                f"{avg_val_loss:.6f}",
                f"{train_acc:.6f}",
                f"{val_acc:.6f}",
                f"{val_f1:.6f}",
                val_cm_str,
                f"{current_lr:.8f}",
            ])
            csv_file_handle.flush()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                print(f"  ✓ Saved best model (val_acc: {val_acc:.4f})")

            scheduler.step()

            if early_stopping(avg_val_loss):
                print("Early stopping triggered!")
                break

    finally:
        csv_file_handle.close()
        print(f"\n✓ Training logs saved to {csv_path}")

    # ==================== TEST ====================
    print("\n--- FINAL TEST EVALUATION ---")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    test_preds, test_labels_list = [], []
    with torch.no_grad():
        for faces, labels, lengths in test_loader:
            faces  = faces.to(device)
            labels = labels.to(device)
            preds  = model(faces, lengths).argmax(dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels_list.extend(labels.cpu().numpy())

    test_acc = np.mean(np.array(test_preds) == np.array(test_labels_list))
    test_f1  = f1_score(test_labels_list, test_preds, average='macro')
    test_cm  = confusion_matrix(test_labels_list, test_preds)

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1-macro: {test_f1:.4f}")
    print(f"Confusion Matrix:\n{test_cm}")

    test_results_path = logs_dir / f"test_results_{args.encoder}.txt"
    with open(test_results_path, 'w') as f:
        f.write(f"Model: {args.encoder}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test F1-macro: {test_f1:.4f}\n")
        f.write(f"Test Confusion Matrix:\n{test_cm}\n")
    print(f"Test results saved to {test_results_path}")

if __name__ == "__main__":
    set_seed(37)
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    type=str, choices=["ssl", "supervised"], required=True)
    parser.add_argument("--encoder", type=str, choices=["baseline", "fast_gru", "resnet_lstm"],
                        default="fast_gru")
    parser.add_argument("--epochs",          type=int,   default=10)
    parser.add_argument("--batch-size",      type=int,   default=4)
    parser.add_argument("--load-ssl",        action="store_true")
    parser.add_argument("--frame-skip",      type=int,   default=15)
    parser.add_argument("--subset",          type=int,   default=None)
    parser.add_argument("--use-memory-bank", action="store_true")
    parser.add_argument("--bank-size",       type=int,   default=32768)
    parser.add_argument("--temperature",     type=float, default=0.5)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    img_dir, csv_file, weights_dir, logs_dir = _default_paths(args.frame_skip)
    weights_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "ssl":
        train_ssl(args, device, img_dir, weights_dir)
    else:
        train_supervised(args, device, img_dir, csv_file, weights_dir, logs_dir)
