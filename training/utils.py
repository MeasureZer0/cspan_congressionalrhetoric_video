import random
from pathlib import Path

import numpy as np
import torch
from faces_frames_dataset import FacesFramesSupervisedDataset
from torch.nn.utils.rnn import pad_sequence


def set_seed(seed: int = 42) -> None:
    """Sets the seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def _default_paths() -> tuple[Path, Path, Path, Path]:
    """Return default (img_dir, csv_file, weights_dir, logs_dir) paths."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = project_root / "data"  # Main data folder
    img_dir = data_dir / "self-supervised"  # Directory containing image sequences
    csv_file = data_dir / "labels.csv"  # CSV file with labels for each sequence
    weights_dir = data_dir / "weights"  # Directory to save model weights to
    logs_dir = project_root / "logs"  # Directory to save training logs to
    return img_dir, csv_file, weights_dir, logs_dir


def ssl_collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for SimCLR SSL training.
    Pairs two augmented views of the same video and pads
        them to the max length in batch.
    """
    v1_list = [item[0] for item in batch]
    v2_list = [item[1] for item in batch]
    lengths = torch.tensor([v.shape[0] for v in v1_list], dtype=torch.long)

    v1_padded = pad_sequence(v1_list, batch_first=True)
    v2_padded = pad_sequence(v2_list, batch_first=True)

    return v1_padded, v2_padded, lengths


def supervised_collate_fn(
    batch: list,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for supervised training.
    Pads sequences to the max length in batch and stacks labels.
    """
    faces_list = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    lengths = torch.tensor([f.shape[0] for f in faces_list], dtype=torch.long)

    faces_padded = pad_sequence(faces_list, batch_first=True)
    return faces_padded, labels, lengths


class EarlyStopping:
    """
    Early stopping to terminate training when validation loss stops improving.
    """

    def __init__(
        self, patience: int = 5, min_delta: float = 0.001, verbose: bool = False
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
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
                print(
                    f"Validation loss improved: {self.best_loss:.4f} → {val_loss:.4f}"
                )
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


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
        tuple: (train_indices, val_indices, test_indices)
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
