"""
PyTorch Datasets for loading face tensors + pose keypoints from .pt files.
Supports SSL (unlabelled) and supervised tasks.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

# ---------------------------------------------------------------------------
# SSL base dataset
# ---------------------------------------------------------------------------


class FacesFramesSSLDataset(Dataset):
    """
    Loads preprocessed face tensors and pose keypoints without labels.
    Used as the base dataset for self-supervised methods.
    """

    def __init__(
        self,
        img_dir: Path,
        min_frames: int = 8,
        max_frames: int = 120,
    ) -> None:
        self.img_dir = Path(img_dir)
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.samples = self._build_sample()

    def _build_sample(self) -> list[str]:
        files = sorted(self.img_dir.glob("*_faces.pt"))
        return [f.stem.replace("_faces", "") for f in files]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        faces : torch.Tensor  [T, 3, H, W]
        pose  : torch.Tensor  [T, 17, 3]   (x_norm, y_norm, confidence)
        """
        stem = self.samples[idx]

        try:
            faces = torch.load(self.img_dir / f"{stem}_faces.pt")
        except Exception:
            print(f"Corrupted file: {stem}_faces.pt")
            return self.__getitem__((idx + 1) % len(self))

        pose_path = self.img_dir / f"{stem}_pose.pt"
        if pose_path.exists():
            try:
                pose = torch.load(pose_path)
            except Exception:
                pose = torch.zeros(faces.shape[0], 17, 3)
        else:
            pose = torch.zeros(faces.shape[0], 17, 3)

        # Align lengths
        min_t = min(faces.shape[0], pose.shape[0])
        faces = faces[:min_t]
        pose = pose[:min_t]

        # Pad short sequences
        n = faces.shape[0]
        if n < self.min_frames:
            repeat = int(np.ceil(self.min_frames / n))
            faces = faces.repeat(repeat, 1, 1, 1)[: self.min_frames]
            pose = pose.repeat(repeat, 1, 1)[: self.min_frames]

        # Truncate long sequences
        if faces.shape[0] > self.max_frames:
            faces = faces[: self.max_frames]
            pose = pose[: self.max_frames]

        return faces, pose


# ---------------------------------------------------------------------------
# Supervised dataset
# ---------------------------------------------------------------------------


class FacesFramesSupervisedDataset(Dataset):
    """
    Loads face tensors, pose keypoints, and integer labels from a CSV.
    """

    def __init__(self, csv_file: Path, img_dir: Path) -> None:
        self.img_dir = Path(img_dir)
        self.csv = pd.read_csv(csv_file)
        self.classes = {"negative": 0, "neutral": 1, "positive": 2}
        self.samples = self._build_index()

    def _build_index(self) -> list[tuple[str, str]]:
        samples = []
        for i in range(len(self.csv)):
            video_name = str(self.csv.iloc[i, 0])
            label_str = str(self.csv.iloc[i, 1]).strip()
            stem = Path(video_name).stem
            if (self.img_dir / f"{stem}_faces.pt").exists():
                samples.append((stem, label_str))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        faces : torch.Tensor  [T, 3, H, W]
        pose  : torch.Tensor  [T, 17, 3]
        label : torch.Tensor  scalar long
        """
        stem, label_str = self.samples[idx]

        faces = torch.load(self.img_dir / f"{stem}_faces.pt")
        pose_path = self.img_dir / f"{stem}_pose.pt"
        pose = (
            torch.load(pose_path)
            if pose_path.exists()
            else torch.zeros(faces.shape[0], 17, 3)
        )

        # Align face / pose lengths
        min_t = min(faces.shape[0], pose.shape[0])
        faces = faces[:min_t]
        pose = pose[:min_t]

        label = torch.tensor(self.classes[label_str], dtype=torch.long)
        return faces, pose, label


# ---------------------------------------------------------------------------
# SimCLR wrapper
# ---------------------------------------------------------------------------


class SimCLRDataset(Dataset):
    """
    Wraps a base SSL dataset and returns two independently-augmented views
    of each sample for contrastive pre-training.
    """

    def __init__(self, base: FacesFramesSSLDataset, face_transform: Callable, pose_transform: Callable) -> None:
        self.base = base
        self.face_transform = face_transform
        self.pose_transform = pose_transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        face_v1, face_v2 : two independently augmented face sequences  [T, 3, H, W]
        pose_v1, pose_v2 : two independently augmented pose sequences  [T, 17, 3]
        """
        sample = self.base[idx]
        faces: torch.Tensor = sample[0]
        pose: torch.Tensor = sample[1]
        return (
            self.face_transform(faces),
            self.face_transform(faces),
            self.pose_transform(pose),
            self.pose_transform(pose),
        )
