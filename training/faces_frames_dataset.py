"""PyTorch datasets for face tensor + pose keypoint sequences.

Supports both self-supervised (SSL) and supervised training workflows.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FacesFramesSSLDataset(Dataset):
    """Load preprocessed face tensors and pose keypoints without labels."""

    def __init__(
        self,
        img_dir: Path,
        min_frames: int = 8,
        max_frames: int = 120,
    ) -> None:
        self.img_dir = Path(img_dir)
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.samples = self._build_sample_list()

    def _build_sample_list(self) -> list[str]:
        return [
            f.stem.replace("_faces", "")
            for f in sorted(self.img_dir.glob("*_faces.pt"))
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (faces [T,3,H,W], pose [T,17,3])."""
        stem = self.samples[idx]

        try:
            faces: torch.Tensor = torch.load(
                self.img_dir / f"{stem}_faces.pt", weights_only=True
            )
        except Exception:
            print(f"[warn] Corrupted file: {stem}_faces.pt — skipping")
            return self.__getitem__((idx + 1) % len(self))

        pose_path = self.img_dir / f"{stem}_pose.pt"
        try:
            pose: torch.Tensor = torch.load(pose_path, weights_only=True)
        except Exception:
            pose = torch.zeros(faces.shape[0], 17, 3)

        # Align temporal lengths
        t = min(faces.shape[0], pose.shape[0])
        faces, pose = faces[:t], pose[:t]

        # Pad sequences that are too short
        if t < self.min_frames:
            reps = int(np.ceil(self.min_frames / t))
            faces = faces.repeat(reps, 1, 1, 1)[: self.min_frames]
            pose = pose.repeat(reps, 1, 1)[: self.min_frames]

        # Truncate sequences that are too long
        faces = faces[: self.max_frames]
        pose = pose[: self.max_frames]

        return faces, pose


class FacesFramesSupervisedDataset(Dataset):
    """Load face tensors, pose keypoints, and integer labels from a CSV.

    Parameters
    ----------
    csv_file:
        CSV with columns ``[video_name, label]``.
    img_dir:
        Directory containing ``*_faces.pt`` and ``*_pose.pt`` tensors.
    transform:
        Augmentation applied to the face tensor ``[T, 3, H, W]``.
        Pass only for the training split; leave ``None`` for val/test.
    aug_multiplier:
        How many augmented copies of each training sample appear per epoch.
        Has no effect when *transform* is ``None``.
    """

    _CLASSES = {"negative": 0, "neutral": 1, "positive": 2}

    def __init__(
        self,
        csv_file: Path,
        img_dir: Path,
        transform: Optional[Callable] = None,
        aug_multiplier: int = 1,
    ) -> None:
        self.img_dir = Path(img_dir)
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.aug_multiplier = aug_multiplier if transform is not None else 1
        self.samples = self._build_index()

    def _build_index(self) -> list[tuple[str, str]]:
        base = [
            (Path(str(self.df.iloc[i, 0])).stem, str(self.df.iloc[i, 1]).strip())
            for i in range(len(self.df))
            if (self.img_dir / f"{Path(str(self.df.iloc[i, 0])).stem}_faces.pt").exists()
        ]
        return base * self.aug_multiplier

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stem, label_str = self.samples[idx]

        faces: torch.Tensor = torch.load(
            self.img_dir / f"{stem}_faces.pt", weights_only=True
        )

        pose_path = self.img_dir / f"{stem}_pose.pt"
        pose: torch.Tensor = (
            torch.load(pose_path, weights_only=True)
            if pose_path.exists()
            else torch.zeros(faces.shape[0], 17, 3)
        )

        t = min(faces.shape[0], pose.shape[0])
        faces, pose = faces[:t], pose[:t]

        if self.transform is not None:
            faces = self.transform(faces)

        label = torch.tensor(self._CLASSES[label_str], dtype=torch.long)
        return faces, pose, label


class SimCLRDataset(Dataset):
    """Wrap a base SSL dataset and return two *independently* augmented views.

    Both the face and pose transforms are applied twice with different random
    seeds so that view-1 and view-2 are genuinely distinct augmentations of
    the same sample, which is required for contrastive pre-training.
    """

    def __init__(
        self,
        base: FacesFramesSSLDataset,
        face_transform: Callable,
        pose_transform: Callable,
    ) -> None:
        self.base = base
        self.face_transform = face_transform
        self.pose_transform = pose_transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (face_v1, face_v2, pose_v1, pose_v2) — two independent views."""
        faces, pose = self.base[idx]

        # Call transforms separately so each gets different random parameters
        face_v1 = self.face_transform(faces)
        face_v2 = self.face_transform(faces)
        pose_v1 = self.pose_transform(pose)
        pose_v2 = self.pose_transform(pose)

        return face_v1, face_v2, pose_v1, pose_v2