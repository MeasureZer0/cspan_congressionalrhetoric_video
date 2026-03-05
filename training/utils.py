import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from .faces_frames_dataset import FacesFramesSupervisedDataset


def set_seed(seed: int = 42) -> None:
    """Sets the seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def default_paths() -> tuple[Path, Path, Path, Path]:
    """Return default (img_dir, csv_file, weights_dir, logs_dir) paths."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = project_root / "data"
    img_dir = data_dir / "processed/frame_skip_30"
    csv_file = data_dir / "labels.csv"
    weights_dir = data_dir / "weights"
    logs_dir = project_root / "logs"
    return img_dir, csv_file, weights_dir, logs_dir


def ssl_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate for SimCLR SSL.

    Each item in batch: (face_v1, face_v2, pose_v1, pose_v2)

    Returns
    -------
    face_v1, face_v2 : padded  [B, T, 3, H, W]
    pose_v1, pose_v2 : padded  [B, T, 17, 3]
    lengths          : [B]  – based on face_v1 sequence lengths
    """
    fv1_list = [item[0] for item in batch]
    fv2_list = [item[1] for item in batch]
    pv1_list = [item[2] for item in batch]
    pv2_list = [item[3] for item in batch]

    lengths = torch.tensor([v.shape[0] for v in fv1_list], dtype=torch.long)

    fv1 = pad_sequence(fv1_list, batch_first=True)
    fv2 = pad_sequence(fv2_list, batch_first=True)
    pv1 = pad_sequence(pv1_list, batch_first=True)
    pv2 = pad_sequence(pv2_list, batch_first=True)

    return fv1, fv2, pv1, pv2, lengths


def supervised_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate for supervised training.

    Each item in batch: (faces, pose, label)
    """
    faces_list = [item[0] for item in batch]
    pose_list = [item[1] for item in batch]
    labels = torch.stack([item[2] for item in batch])
    lengths = torch.tensor([f.shape[0] for f in faces_list], dtype=torch.long)

    faces_padded = pad_sequence(faces_list, batch_first=True)
    pose_padded = pad_sequence(pose_list, batch_first=True)

    return faces_padded, pose_padded, labels, lengths


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        verbose: bool = False,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss: float | None = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(f"Loss improved {self.best_loss:.4f} → {val_loss:.4f}")
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


def stratified_split(
    dataset: FacesFramesSupervisedDataset,
    fractions: tuple[float, float, float],
) -> tuple[list[int], list[int], list[int]]:
    """
    Stratified train / val / test split.
    """
    class_indices: list[list[int]] = [[], [], []]
    for i in range(len(dataset)):
        _, _, label = dataset[i]
        class_indices[int(label.item())].append(i)

    train_idx, val_idx, test_idx = [], [], []
    cum = [fractions[0], fractions[0] + fractions[1]]

    for cls_indices in class_indices:
        random.shuffle(cls_indices)
        n = len(cls_indices)
        n_tr = int(cum[0] * n)
        n_va = int(cum[1] * n)
        train_idx.extend(cls_indices[:n_tr])
        val_idx.extend(cls_indices[n_tr:n_va])
        test_idx.extend(cls_indices[n_va:])

    return train_idx, val_idx, test_idx
