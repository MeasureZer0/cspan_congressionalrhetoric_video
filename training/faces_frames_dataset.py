import os
from pathlib import Path
from typing import Any, Callable, Optional, Literal

import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class FacesFramesSSLDataset(Dataset):
    """
    Dataset do ładowania danych POSE (szkielet) z plików .npy
    Używany do treningu SSL (bez etykiet).
    """
    def __init__(self, img_dir: Path, min_frames: int = 15, max_frames: int = 120):
        """
        Args:
            img_dir (Path): Katalog zawierający pliki .npy (np. data/pose_features).
        """
        self.img_dir = Path(img_dir)
        self.samples = self._build_sample()
        self.min_frames = min_frames
        self.max_frames = max_frames

    def _build_sample(self):
        """
        Skanuje katalog w poszukiwaniu plików .npy
        """
        # Szukamy plików .npy (dane z YOLO)
        files = sorted(self.img_dir.glob("*.npy"))
        return [f.stem for f in files]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stem = self.samples[idx]
        pose_path = self.img_dir / f"{stem}.npy"

        try:
            # Ładowanie numpy array [Frames, 34]
            pose_data = np.load(pose_path)
            # Konwersja na Tensor PyTorch
            pose_tensor = torch.from_numpy(pose_data).float()
        except Exception as e:
            print(f"Corrupted file: {pose_path}")
            return self.__getitem__((idx + 1) % len(self))

        n_frames = pose_tensor.shape[0]

        # Obsługa długości sekwencji (przycinanie lub powielanie)
        if n_frames > self.max_frames:
            pose_tensor = pose_tensor[:self.max_frames]

        elif n_frames < self.min_frames:
            if n_frames == 0: # Zabezpieczenie przed pustym plikiem
                return self.__getitem__((idx + 1) % len(self))
                
            repeat_factor = int(np.ceil(self.min_frames / n_frames))
            # Dla danych 2D (Time, Features) powtarzamy tylko w osi 0
            pose_tensor = pose_tensor.repeat(repeat_factor, 1)
            pose_tensor = pose_tensor[:self.min_frames]

        return pose_tensor


class FacesFramesSupervisedDataset(Dataset):
    """
    Dataset do treningu nadzorowanego (z etykietami z CSV).
    Ładuje pary: (pose_tensor, label).
    """

    def __init__(self, csv_file: Path, img_dir: Path):
        """
        Args:
            csv_file (Path): Ścieżka do pliku labels.csv.
            img_dir (Path): Katalog z plikami .npy.
        """
        self.img_dir = Path(img_dir)
        self.csv = pd.read_csv(csv_file)

        self.classes = {"negative": 0, "neutral": 1, "positive": 2}
        self.samples = self._build_index()

    def _build_index(self):
        """
        Tworzy indeks istniejących plików .npy pasujących do CSV.
        """
        samples = []
        for i in range(len(self.csv)):
            video_name = str(self.csv.iloc[i, 0])
            label_str = str(self.csv.iloc[i, 1]).strip()

            # Usuwamy rozszerzenie wideo, żeby dostać nazwę pliku
            stem = Path(video_name).stem
            
            # Sprawdzamy czy istnieje odpowiednik .npy
            if (self.img_dir / f"{stem}.npy").exists():
                samples.append((stem, label_str))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stem, label_str = self.samples[idx]

        # Ładowanie danych
        pose_path = self.img_dir / f"{stem}.npy"
        pose_data = np.load(pose_path)
        
        # Konwersja na tensor
        pose_tensor = torch.from_numpy(pose_data).float()

        # Konwersja etykiety
        label = torch.tensor(self.classes[label_str], dtype=torch.long)

        # Dataset zwraca (dane, label), collate_fn w train.py zajmie się paddingiem
        return pose_tensor, label


class SimCLRDataset(Dataset):
    """
    Wrapper dla SimCLR - zwraca dwie wersje tego samego przykładu.
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
            # Jeśli brak augmentacji, zwracamy oryginał (dla testów)
            v1 = pose
            v2 = pose

        return v1, v2


class PoseAugmentation:
    """
    Augmentacja dedykowana dla współrzędnych szkieletu (YOLO Pose).
    Zamiast obracać obrazek, dodajemy szum do liczb.
    """
    def __init__(self, noise_std=0.01, dropout_prob=0.1):
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob

    def __call__(self, x):
        # x shape: [Time, 34]
        
        # 1. Kopia tensora, żeby nie psuć oryginału
        aug_x = x.clone()
        
        # 2. Gaussian Noise (Jitter) - dodajemy małe losowe wartości
        noise = torch.randn_like(aug_x) * self.noise_std
        aug_x += noise
        
        # 3. Random Joint Dropout (opcjonalnie)
        # Zerujemy losowe punkty, żeby sieć nie polegała tylko na jednym stawie
        if self.dropout_prob > 0:
            mask = torch.rand_like(aug_x) > self.dropout_prob
            aug_x = aug_x * mask
            
        return aug_x