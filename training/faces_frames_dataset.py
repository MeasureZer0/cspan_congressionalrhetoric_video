import os
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

# Define transform types
Transform = Optional[Callable[[Any], Any]]


class FacesFramesDataset(Dataset):
    """
    PyTorch Dataset for loading preprocessed face tensors from .pt files
    and their corresponding labels from a CSV file.

    Supports loading both original and augmented versions (with '_aug' suffix).
    When augmented versions exist, they are treated as separate training samples.
    """

    def __init__(
        self,
        csv_file: Path,
        img_dir: Path,
        transform: Transform = None,
        target_transform: Transform = None,
        include_augmented: bool = False,
    ) -> None:
        """
        Args:
            csv_file (Path): Path to the CSV file with labels.
            img_dir (Path): Directory with preprocessed .pt face tensors.
            transform (callable, optional): Optional transform to apply to face tensors.
            target_transform (callable, optional): Optional transform applied to labels.
            include_augmented (bool): Whether to include augmented versions (_aug files)
                as additional training samples. Default is False.
        """
        # Load CSV with labels
        self.csv_file = pd.read_csv(csv_file)

        # Directory containing preprocessed face tensors
        self.img_dir = img_dir

        # Optional transformations for input and target
        self.transform = transform
        self.target_transform = target_transform

        # Mapping string labels to integer classes
        # Needed because models expect numeric labels
        self.classes = {"negative": 0, "neutral": 1, "positive": 2}

        # Build index of available samples (original + augmented)
        self.samples = self._build_sample_index(include_augmented)

    def _build_sample_index(self, include_augmented: bool) -> list[tuple[str, str]]:
        """
        Build an index of all available samples.

        Returns:
            List of tuples (stem, label_str) for each available sample.
            If include_augmented is True, includes both original and '_aug' versions.
        """
        samples = []

        for idx in range(len(self.csv_file)):
            video_name = str(self.csv_file.iloc[idx, 0])
            label_str = str(self.csv_file.iloc[idx, 1]).strip()
            stem = Path(video_name).stem

            # Check if original files exist
            face_path = os.path.join(self.img_dir, f"{stem}_faces.pt")
            flow_path = os.path.join(self.img_dir, f"{stem}_flows.pt")

            if os.path.exists(face_path) and os.path.exists(flow_path):
                samples.append((stem, label_str))

            # Check if augmented files exist
            if include_augmented:
                aug_face_path = os.path.join(self.img_dir, f"{stem}_aug_faces.pt")
                aug_flow_path = os.path.join(self.img_dir, f"{stem}_aug_flows.pt")

                if os.path.exists(aug_face_path) and os.path.exists(aug_flow_path):
                    samples.append((f"{stem}_aug", label_str))

        return samples

    def __len__(self) -> int:
        # Return the total number of samples including augmented versions
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load a single sample (faces tensor and label) given an index.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: faces tensor, flows tensor,
                and label tensor
        """
        # Get stem and label from the sample index
        stem, label_str = self.samples[idx]

        # Construct the full path to the .pt files
        face_path = os.path.join(self.img_dir, f"{stem}_faces.pt")
        flow_path = os.path.join(self.img_dir, f"{stem}_flows.pt")

        # Load the preprocessed frames tensor
        faces = torch.load(face_path)
        flows = torch.load(flow_path)

        # Load and convert string label to tensor
        label = torch.tensor(self.classes[label_str], dtype=torch.long)

        # Apply optional transformations to faces tensor only
        if self.transform:
            faces = self.transform(faces)

        # Apply optional transformations to label tensor
        if self.target_transform:
            label = self.target_transform(label)

        # Return faces, flows and label tensors
        return faces, flows, label
