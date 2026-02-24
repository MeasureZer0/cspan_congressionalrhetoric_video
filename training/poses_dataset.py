import os
from pathlib import Path
from typing import Any, Callable, Optional, Literal

import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class PosesSSLDataset(Dataset):
    """
    Dataset for loading POSE data from .npy files.
    Used for SSL training without labels.
    """
    def __init__(self, img_dir: Path, min_frames: int = 15, max_frames: int = 120):
        self.img_dir = Path(img_dir)
        self.samples = self._build_sample()
        self.min_frames = min_frames
        self.max_frames = max_frames

    def _build_sample(self):
        """
        Scanning directory for .npy YOLO outputs
        """
        files = sorted(self.img_dir.glob("*.npy"))
        return [f.stem for f in files]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stem = self.samples[idx]
        pose_path = self.img_dir / f"{stem}.npy"

        try:
            """
            Loading numpy array of shape [Frames, Features]
            and converting to PyTorch Tensor
            """
            pose_data = np.load(pose_path)
            pose_tensor = torch.from_numpy(pose_data).float()
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self))

        n_frames = pose_tensor.shape[0]

        """
        Handling sequence length variations:
        Trimming if too long, repeating if too short
        """
        if n_frames > self.max_frames:
            pose_tensor = pose_tensor[:self.max_frames]

        elif n_frames < self.min_frames:
            if n_frames == 0:
                return self.__getitem__((idx + 1) % len(self))
                
            repeat_factor = int(np.ceil(self.min_frames / n_frames))
            pose_tensor = pose_tensor.repeat(repeat_factor, 1)
            pose_tensor = pose_tensor[:self.min_frames]

        return pose_tensor


class PosesSupervisedDataset(Dataset):
    """
    Dataset for supervised training.
    Loads pairs of (pose_tensor, label).
    """
    def __init__(self, csv_file: Path, img_dir: Path):
        self.img_dir = Path(img_dir)
        self.csv = pd.read_csv(csv_file)
        self.classes = {"negative": 0, "neutral": 1, "positive": 2}
        self.samples = self._build_index()

    def _build_index(self):
        """
        Creating an index of existing .npy files
        that match the CSV labels.
        """
        samples = []
        for i in range(len(self.csv)):
            video_name = str(self.csv.iloc[i, 0])
            label_str = str(self.csv.iloc[i, 1]).strip()

            stem = Path(video_name).stem
            
            if (self.img_dir / f"{stem}.npy").exists():
                samples.append((stem, label_str))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stem, label_str = self.samples[idx]

        pose_path = self.img_dir / f"{stem}.npy"
        pose_data = np.load(pose_path)
        pose_tensor = torch.from_numpy(pose_data).float()

        label = torch.tensor(self.classes[label_str], dtype=torch.long)

        return pose_tensor, label


class SimCLRDataset(Dataset):
    """
    Wrapper for SimCLR.
    Returns two augmented versions of the same sample.
    """
    def __init__(self, base_dataset: Dataset, transform=None):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        pose = self.base[idx]

        if self.transform:
            v1 = self.transform(pose)
            v2 = self.transform(pose)
        else:
            v1 = pose
            v2 = pose

        return v1, v2


class PoseAugmentation:
    """
    Custom augmentation for YOLO skeleton coordinates.
    Adds Gaussian noise and randomly drops joints.
    """
    def __init__(self, noise_std=0.01, dropout_prob=0.1):
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob

    def __call__(self, x):
        aug_x = x.clone()
        
        # Gaussian noise (jitter)
        noise = torch.randn_like(aug_x) * self.noise_std
        aug_x += noise
        
        # Random joint dropout
        if self.dropout_prob > 0:
            mask = torch.rand_like(aug_x) > self.dropout_prob
            aug_x = aug_x * mask
            
        return aug_x