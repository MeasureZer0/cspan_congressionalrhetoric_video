import os
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

# Define transform type
Transform = Optional[Callable[[Any], Any]]

class FacesFramesDataset(Dataset):
    """
    PyTorch Dataset for loading preprocessed face tensors from .pt files
    and their corresponding labels from a CSV file.
    """

    def __init__(
        self,
        csv_file: Path,
        img_dir: Path,
        transform: Transform = None,
        target_transform: Transform = None,
    ) -> None:
        """
        Args:
            csv_file (Path): Path to the CSV file with labels.
            img_dir (Path): Directory with preprocessed .pt face tensors.
            transform (callable, optional): Optional transform to apply to face tensors.
            target_transform (callable, optional): Optional transform applied to labels.
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

    def __len__(self) -> int:
        # Return the total number of samples in the dataset
        return len(self.csv_file)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load a single sample (faces tensor and label) given an index.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: faces tensor and label tensor
        """
        # Extract the stem (filename without extension) from CSV
        stem = Path(str(self.csv_file.iloc[idx, 0])).stem

        # Construct the full path to the .pt file
        face_path = os.path.join(self.img_dir, f"{stem}_faces.pt")
        flow_path = os.path.join(self.img_dir, f"{stem}_flows.pt")

        # Load the preprocessed frames tensor
        faces = torch.load(face_path)
        flows = torch.load(flow_path)

        # Get the label string from CSV
        label_str = str(self.csv_file.iloc[idx, 1]).strip()

        # Load and convert string label to tensor
        label = torch.tensor(self.classes[label_str], dtype=torch.long)

        # Apply optional transformations to faces tensor
        if self.transform:
            faces = self.transform(faces)

        # Apply optional transformations to label tensor
        if self.target_transform:
            label = self.target_transform(label)

        # Return faces tensor and label tensor
        return faces, flows, label
