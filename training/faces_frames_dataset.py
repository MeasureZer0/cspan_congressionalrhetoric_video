import os
from pathlib import Path
from typing import Any, Callable, Optional, Literal

import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np


class FacesFramesSSLDataset(Dataset):
    """
    PyTorch Dataset for loading preprocessed face tensors
    from .pt files without labels.

    Used as a base dataset for SSL methods.
    """
    def __init__(self, img_dir: Path):
        """
        Args:
            img_dir (Path): Directory containing *_faces.pt and *_flows.pt files.
        """
        self.img_dir = Path(img_dir)
        self.samples = self._build_sample()

    def _build_sample(self):
        """
        Scan directory and collect all available sample stems.

        Returns:
            list[str]: List of file stems without suffixes.
        """
        files = sorted(self.img_dir.glob("*_faces.pt"))
        return [f.stem.replace("_faces", "") for f in files]

    def __len__(self):
        """
        Returns:
            int: Number of available samples.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load a single sample (faces tensor).

        Args:
            idx (int): Sample index.

        Returns:
            tuple[Tensor, Tensor]: faces tensor.
        """
        stem = self.samples[idx]

        face_path = self.img_dir / f"{stem}_faces.pt"

        try:
            faces = torch.load(face_path)
        except Exception as e:
            print(f"Corrupted file: {face_path}")
            return self.__getitem__((idx + 1) % len(self))

        return faces


class FacesFramesSupervisedDataset(Dataset):
    """
    PyTorch Dataset for supervised training.

    Loads face tensors and corresponding labels from CSV.
    Returns (faces, label).
    """

    def __init__(self, csv_file: Path, img_dir: Path):
        """
        Args:
            csv_file (Path): CSV file containing labels.
            img_dir (Path): Directory with preprocessed tensors.
        """
        self.img_dir = Path(img_dir)
        self.csv = pd.read_csv(csv_file)

        self.classes = {"negative": 0, "neutral": 1, "positive": 2}
        self.samples = self._build_index()

    def _build_index(self):
        """
        Build index of valid samples based on CSV and existing files.

        Returns:
            list[tuple[str, str]]: List of (stem, label_string).
        """
        samples = []
        for i in range(len(self.csv)):
            video_name = str(self.csv.iloc[i, 0])
            label_str = str(self.csv.iloc[i, 1]).strip()

            stem = Path(video_name).stem
            if (self.img_dir / f"{stem}_faces.pt").exists():
                samples.append((stem, label_str))
        return samples

    def __len__(self):
        """
        Returns:
            int: Number of labeled samples.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load tensors and label for supervised learning.

        Args:
            idx (int): Sample index.

        Returns:
            tuple[Tensor, Tensor, Tensor]: faces, flows and label tensor.
        """
        stem, label_str = self.samples[idx]

        faces = torch.load(self.img_dir / f"{stem}_faces.pt")

        label = torch.tensor(self.classes[label_str], dtype=torch.long)

        return faces, label

class SimCLRDataset(Dataset):
    """
    Wrapper Dataset for SimCLR ssl.

    Takes a base dataset and returns two augmented views
    of the same sample: (view1, view2).
    """

    def __init__(self, base_dataset: Dataset, transform):
        """
        Args:
            base_dataset (Dataset): Dataset returning tensors without labels.
            transform (Callable): Augmentation transform.
        """
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        """
        Returns:
            int: Number of samples in base dataset.
        """
        return len(self.base)

    def __getitem__(self, idx):
        """
        Generate two augmented views of the same sample.

        Args:
            idx (int): Sample index.

        Returns:
            tuple[Tensor, Tensor]: Two augmented tensors.
        """
        faces = self.base[idx]

        v1 = self.transform(faces)
        v2 = self.transform(faces)

        return v1, v2