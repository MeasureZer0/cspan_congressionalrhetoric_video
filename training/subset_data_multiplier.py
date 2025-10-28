"""
Subset-aware data multiplier that prevents validation data leakage.

This ensures that only training samples are multiplied, keeping validation
data completely separate.
"""

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from video_augmentation import VideoAugmentation


class SubsetDataMultiplier(Dataset):
    """
    Data multiplier that works with a specific subset of indices.

    This prevents validation data leakage by only multiplying the samples
    corresponding to training indices.
    """

    def __init__(
        self,
        csv_file: Path,
        img_dir: Path,
        train_indices: List[int],
        multiplier: int = 2,
        augmentation_strength: str = "standard",
        include_original: bool = True,
    ) -> None:
        """
        Args:
            csv_file: Path to CSV with labels
            img_dir: Directory with face/flow tensors
            train_indices: List of indices to use for training (prevents val leakage)
            multiplier: How many versions of each training sample
            augmentation_strength: "light", "standard", or "heavy"
            include_original: Whether to include non-augmented versions
        """
        self.csv_file = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.train_indices = train_indices
        self.multiplier = multiplier
        self.include_original = include_original

        # Mapping string labels to integer classes
        self.classes = {"negative": 0, "neutral": 1, "positive": 2}

        # Create augmentation
        if augmentation_strength == "light":
            self.augmentation = VideoAugmentation(
                rotation_degrees=5.0,
                brightness=0.05,
                contrast=0.05,
                saturation=0.05,
                hue=0.02,
                probability=1.0,
            )
        elif augmentation_strength == "standard":
            self.augmentation = VideoAugmentation(
                rotation_degrees=10.0,
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05,
                probability=1.0,
            )
        else:  # heavy
            self.augmentation = VideoAugmentation(
                rotation_degrees=15.0,
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                probability=1.0,
            )

        # Calculate total size based on training indices only
        num_train_samples = len(train_indices)
        if include_original:
            self.total_size = num_train_samples * multiplier
        else:
            self.total_size = num_train_samples * (multiplier - 1)

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a training sample - either original or augmented version."""

        # Map global index to training sample and version
        train_sample_idx = idx // self.multiplier
        version = idx % self.multiplier

        # Get the actual dataset index from training indices
        actual_dataset_idx = self.train_indices[train_sample_idx]

        # Load data using the actual dataset index
        stem = Path(str(self.csv_file.iloc[actual_dataset_idx, 0])).stem
        face_path = self.img_dir / f"{stem}_faces.pt"
        flow_path = self.img_dir / f"{stem}_flows.pt"

        faces = torch.load(face_path)
        flows = torch.load(flow_path)

        # Get label
        label_str = str(self.csv_file.iloc[actual_dataset_idx, 1]).strip()
        label = torch.tensor(self.classes[label_str], dtype=torch.long)

        # Decide if this should be original or augmented
        if self.include_original and version == 0:
            # Return original
            return faces, flows, label
        else:
            # Return augmented version
            faces_aug, flows_aug = self.augmentation(faces, flows)
            return faces_aug, flows_aug, label
