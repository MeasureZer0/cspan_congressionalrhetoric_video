import os
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

# Define transform type: accept plain callables (functions, lambdas,
# torchvision.transforms.Compose, etc.).
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
        self.csv_file = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.csv_file)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Any]:
        # {Video name without extension}_faces.pt
        stem = Path(str(self.csv_file.iloc[idx, 0])).stem
        face_path = os.path.join(self.img_dir, f"{stem}_faces.pt")

        # Load the frames tensor and label
        faces = torch.load(face_path)
        label = self.csv_file.iloc[idx, 1]

        # Apply transformations if any
        if self.transform:
            faces = self.transform(faces)
        if self.target_transform:
            label = self.target_transform(label)

        return faces, label
