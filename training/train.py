"""
Training script for self-supervised (SimCLR) and supervised learning
on skeleton pose sequences extracted from videos via YOLO.
"""

import argparse
import csv
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Subset

from poses_dataset import (
    PosesSSLDataset,
    PosesSupervisedDataset,
    SimCLRDataset,
    PoseAugmentation
)
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


def _get_device() -> torch.device:
    use_cuda = torch.cuda.is_available()
    return torch.device("cuda:0" if use_cuda else "cpu")


def _default_paths(frame_skip: int) -> tuple[Path, Path, Path, Path]:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = project_root / "data"
    img_dir = data_dir / "pose_features"
    csv_file = data_dir / "labels.csv"
    weights_dir = data_dir / "weights"
    logs_dir = project_root / "logs"
    return img_dir, csv_file, weights_dir, logs_dir


def ssl_collate_fn(batch):
    v1_list = [item[0] for item in batch]
    v2_list = [item[1] for item in batch]
    lengths = torch.tensor([v.shape[0] for v in v1_list], dtype=torch.long)

    v1_padded = pad_sequence(v1_list, batch_first=True)
    v2_padded = pad_sequence(v2_list, batch_first=True)

    return v1_padded, v2_padded, lengths


def supervised_collate_fn(batch):
    poses_list = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    lengths = torch.tensor([f.shape[0] for f in poses_list], dtype=torch.long)

    poses_padded = pad_sequence(poses_list, batch_first=True)
    return poses_padded, labels, lengths


class EarlyStopping:
    """
    Stops training when validation loss stops improving.
    """
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
    """
    Stores past batch representations to increase the effective
    batch size for Contrastive Learning without VRAM overhead.
    """
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
        scores = self.attn(lstm_outputs).squeeze(-1)
        max_len = scores.size(1)
        mask = torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]
        scores[~mask] = -1e9
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(lstm_outputs * weights.unsqueeze(-1), dim=1)
        return context


class PoseGRU(nn.Module):
    """
    Sequence model for pose tracking. 
    Processes extracted YOLO joints via linear layers, then applies GRU/LSTM.
    """
    def __init__(self, input_dim=51, hidden_size=128, num_layers=2, num_classes=3, GRU=True):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        self.feature_dim = 128
        
        if GRU:
            self.recc = nn.GRU(
                input_size=self.feature_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=0.2 if num_layers > 1 else 0,
                batch_first=True,
            )
        else:
            self.recc = nn.LSTM(
                input_size=self.feature_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=0.2 if num_layers > 1 else 0,
                batch_first=True,
            )
            
        self.attention = TemporalAttention(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, batch_padded: torch.Tensor, lengths: torch.Tensor):
        features = self.feature_extractor(batch_padded)
        packed = pack_padded_sequence(
            features, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_out, _ = self.recc(packed)
        outputs, _ = pad_packed_sequence(packed_out, batch_first=True)
        
        context = self.attention(outputs, lengths)
        logits = self.classifier(context)
        return logits

    def forward_hidden(self, batch_padded: torch.Tensor, lengths: torch.Tensor):
        """
        Required for SSL SimCLR projection head.
        Bypasses the final classifier layer.
        """
        features = self.feature_extractor(batch_padded)
        packed = pack_padded_sequence(
            features, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.recc(packed)
        outputs, _ = pad_packed_sequence(packed_out, batch_first=True)
        context = self.attention(outputs, lengths)
        return context


class SimCLRProjectionWrapper(nn.Module):
    """
    SimCLR wrapper attaching a non-linear projection head (MLP) 
    to the base encoder. Contrastive learning loss is calculated 
    on the outputs of this projection head.
    """
    def __init__(self, encoder: nn.Module, encoder_output_dim: int, projection_dim: int):
        super().__init__()
        self.encoder = encoder
        
        self.projector = nn.Sequential(
            nn.Linear(encoder_output_dim, encoder_output_dim),
            nn.ReLU(),
            nn.Linear(encoder_output_dim, projection_dim)
        )

    def forward(self, batch_padded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        representation = self.encoder.forward_hidden(batch_padded, lengths)
        projection = self.projector(representation)
        return projection


# ==========================================
# Training
# ==========================================

def stratified_split(
    dataset: PosesSupervisedDataset,
    fractions: tuple[float, float, float],
) -> tuple[list[int], list[int], list[int]]:
    """
    Splits the dataset into stratified train, validation, and test sets.
    """
    indices = ([], [], [])
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
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        z = torch.cat([z1, z2], dim=0)
        sim_matrix = torch.mm(z, z.T) / self.temperature

        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, torch.finfo(sim_matrix.dtype).min)
        
        labels = torch.arange(batch_size, device=z1.device)
        labels = torch.cat([labels + batch_size, labels])
        
        return F.cross_entropy(sim_matrix, labels)
    

class NTXentLossWithMemoryBank(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, memory_bank) -> torch.Tensor:
        B = z1.size(0)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        pos_sim = (z1 * z2).sum(dim=1, keepdim=True) / self.temperature
        inbatch_sim = torch.mm(z1, z2.T) / self.temperature
        
        diag_mask = torch.eye(B, device=z1.device, dtype=torch.bool)
        inbatch_sim = inbatch_sim.masked_fill(
            diag_mask, torch.finfo(inbatch_sim.dtype).min
        )

        bank = memory_bank.get()
        bank_sim = torch.mm(z1, bank.T) / self.temperature

        logits = torch.cat([pos_sim, inbatch_sim, bank_sim], dim=1)
        labels = torch.zeros(B, dtype=torch.long, device=z1.device)
        loss = F.cross_entropy(logits, labels)

        memory_bank.enqueue(z2)
        return loss


def train_ssl(args, device, img_dir, weights_dir):
    print(f"--- STARTING SSL PRETRAINING (SimCLR) with {args.encoder} ---")
    base_ds = PosesSSLDataset(img_dir)

    """
    Automatic inference of the feature dimension size
    """
    first_sample = base_ds[0]
    sample_tensor = first_sample[0] if isinstance(first_sample, tuple) else first_sample
    detected_dim = sample_tensor.shape[1]
    args.input_dim = detected_dim

    if args.subset and args.subset < len(base_ds):
        indicies = random.sample(range(len(base_ds)), args.subset)
        base_ds = Subset(base_ds, indicies)

    transform = PoseAugmentation(noise_std=0.01)
    ssl_ds = SimCLRDataset(base_ds, transform)
    loader = DataLoader(ssl_ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=ssl_collate_fn, pin_memory=True)

    if args.encoder == 'pose_gru':
        encoder = PoseGRU(input_dim=args.input_dim, hidden_size=128).to(device)
        encoder_dim = 128
    else:
        raise ValueError(f"Unknown encoder: {args.encoder}")
        
    projection_dim = 256
    model = SimCLRProjectionWrapper(encoder, encoder_output_dim=encoder_dim,
                                    projection_dim=projection_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    if args.use_memory_bank:
        memory_bank = MemoryBank(size=args.bank_size, dim=projection_dim).to(device)
        criterion = NTXentLossWithMemoryBank(temperature=args.temperature)
    else:
        criterion = NTXentLoss(temperature=args.temperature)
        memory_bank = None

    accumulation_steps = args.accumulation_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        pbar = tqdm(loader, desc=f"SSL Epoch {epoch+1}", leave=True)
        optimizer.zero_grad()
        
        for i, (v1, v2, lengths) in enumerate(pbar):
            v1 = v1.to(device)
            v2 = v2.to(device)

            z1 = model(v1, lengths)
            z2 = model(v2, lengths)

            if args.use_memory_bank:
                loss = criterion(z1, z2, memory_bank)
            else:
                loss = criterion(z1, z2)
                
            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * accumulation_steps
            num_batches += 1
            
            postfix = {"loss": f"{loss.item() * accumulation_steps:.4f}"}
            if args.use_memory_bank:
                postfix["bank"] = f"{len(memory_bank)}/{args.bank_size}"
            pbar.set_postfix(**postfix)
        
        if (len(loader) % accumulation_steps) != 0:
            optimizer.step()
            optimizer.zero_grad()
    
    scheduler.step()
    
    save_path = weights_dir / f"ssl_backbone_{args.encoder}.pt"
    torch.save(encoder.state_dict(), save_path)
    print(f"SSL Backbone saved to {save_path}")


def train_supervised(args, device, img_dir, csv_file, weights_dir, logs_dir):
    print(f"--- STARTING SUPERVISED TRAINING with {args.encoder} ---")
    
    full_ds = PosesSupervisedDataset(csv_file, img_dir)
    
    """
    Automatic inference of the feature dimension size
    """
    sample_tensor, _ = full_ds[0]
    detected_dim = sample_tensor.shape[1]
    args.input_dim = detected_dim

    train_idx, val_idx, test_idx = stratified_split(full_ds, (0.8, 0.1, 0.1))

    train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=args.batch_size,
                              shuffle=True, collate_fn=supervised_collate_fn, pin_memory=True)
    val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=args.batch_size,
                            collate_fn=supervised_collate_fn, pin_memory=True)
    test_loader = DataLoader(Subset(full_ds, test_idx), batch_size=args.batch_size,
                             collate_fn=supervised_collate_fn, pin_memory=True)

    if args.encoder == 'pose_gru':
        model = PoseGRU(input_dim=args.input_dim, hidden_size=128).to(device)
    else:
        raise ValueError(f"Unknown encoder: {args.encoder}")
    
    if args.load_ssl:
        ssl_path = weights_dir / f"ssl_backbone_{args.encoder}.pt"
        if ssl_path.exists():
            model.load_state_dict(torch.load(ssl_path, map_location=device), strict=False)
            print(f"Loaded SSL weights from {ssl_path}")

    optimizer = torch.optim.Adam([
        {"params": model.feature_extractor.parameters(), "lr": 1e-5},
        {"params": model.recc.parameters(), "lr": 1e-4},
        {"params": model.attention.parameters(), "lr": 5e-4},
        {"params": model.classifier.parameters(), "lr": 5e-4},
    ])

    class_counts = np.bincount([label for _, label in full_ds])
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    weights = weights / weights.sum()
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    
    accumulation_steps = args.accumulation_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=10, min_delta=0.0001)
    
    best_val_acc = 0.0
    best_model_path = weights_dir / f"best_{args.encoder}_supervised.pt"
    
    logs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = logs_dir / f"supervised_{args.encoder}_log.csv"
    
    csv_file_handle = open(csv_path, mode='w', newline='')
    writer = csv.writer(csv_file_handle)
    writer.writerow([
        "epoch", "train_loss", "val_loss",
        "train_acc", "val_acc", "val_f1_macro", "val_conf_matrix"
    ])
    
    try:
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            correct, total = 0, 0
            optimizer.zero_grad()
            
            pbar = tqdm(train_loader, desc=f"Supervised Epoch {epoch+1}")
            for i, (poses, labels, lengths) in enumerate(pbar):
                labels = labels.to(device)
                poses = poses.to(device)
                lengths = lengths.to(device)
                
                logits = model(poses, lengths)
                loss = criterion(logits, labels) / accumulation_steps
                loss.backward()
                
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                train_loss += loss.item() * accumulation_steps
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
            if (len(train_loader) % accumulation_steps) != 0:
                optimizer.step()
                optimizer.zero_grad()

            train_acc = correct / total
            avg_train_loss = train_loss / len(train_loader)
            
            model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []
            
            with torch.no_grad():
                for poses, labels, lengths in val_loader:
                    poses = poses.to(device)
                    labels = labels.to(device)
                    lengths = lengths.to(device)
                    
                    logits = model(poses, lengths)
                    loss = criterion(logits, labels)
                    
                    val_loss += loss.item()
                    preds = logits.argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
            val_f1 = f1_score(all_labels, all_preds, average='macro')
            val_cm = confusion_matrix(all_labels, all_preds)
            val_cm_str = ";".join([",".join(map(str, row)) for row in val_cm])
            
            current_lr = optimizer.param_groups[0]['lr']
            
            writer.writerow([
                epoch+1, f"{avg_train_loss:.6f}", f"{avg_val_loss:.6f}",
                f"{train_acc:.6f}", f"{val_acc:.6f}", f"{val_f1:.6f}",
                val_cm_str, f"{current_lr:.8f}"
            ])
            csv_file_handle.flush()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
            
            scheduler.step()
            
            if early_stopping(avg_val_loss):
                print("Early stopping triggered!")
                break
                
    finally:
        csv_file_handle.close()
    
    print("\n--- FINAL TEST EVALUATION ---")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    test_preds, test_labels_list = [], []
    
    with torch.no_grad():
        for poses, labels, lengths in test_loader:
            poses = poses.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            logits = model(poses, lengths)
            preds = logits.argmax(dim=1)
            
            test_preds.extend(preds.cpu().numpy())
            test_labels_list.extend(labels.cpu().numpy())
    
    test_acc = np.mean(np.array(test_preds) == np.array(test_labels_list))
    test_f1 = f1_score(test_labels_list, test_preds, average='macro')
    test_cm = confusion_matrix(test_labels_list, test_preds)
    
    test_results_path = logs_dir / f"test_results_{args.encoder}.txt"
    with open(test_results_path, 'w') as f:
        f.write(f"Model: {args.encoder}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test F1-macro: {test_f1:.4f}\n")
        f.write(f"Test Confusion Matrix:\n{test_cm}\n")


if __name__ == "__main__":
    set_seed(37)
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["ssl", "supervised"], required=True)
    parser.add_argument("--encoder", type=str, choices=["pose_gru"], default="pose_gru")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--load-ssl", action="store_true")
    parser.add_argument("--frame-skip", type=int, default=15)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--use-memory-bank", action="store_true")
    parser.add_argument("--bank-size", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.5)
    
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