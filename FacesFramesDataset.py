import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from pathlib import Path

class FacesFramesDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, target_transform=None):
        self.csv_file = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.csv_file)
        
    def __getitem__(self, idx):
        # nazwa wideo bez rozszerzenia, a potem _faces.pt
        stem = Path(self.csv_file.iloc[idx, 0]).stem
        face_path = os.path.join(self.img_dir, f"{stem}_faces.pt")
        
        # wczytaj tensor twarzy
        faces = torch.load(face_path)
        label = self.csv_file.iloc[idx, 1]
        
        if self.transform:
            faces = self.transform(faces)
        if self.target_transform:
            label = self.target_transform(label)
        
        return faces, label


# Paths
faces_dir = Path("data/processed_faces")
labels_path = Path("data/sample_video/labels.csv")

face_dataset = FacesFramesDataset(
    csv_file=labels_path,
    img_dir=faces_dir
)

# 🔍 Sprawdzenie działania datasetu
print(f"Liczba próbek: {len(face_dataset)}")

for i in range(min(3, len(face_dataset))):
    faces, label = face_dataset[i]
    print(f"Idx {i}: faces shape = {faces.shape}, label = {label}")