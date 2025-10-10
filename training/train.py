from pathlib import Path

import torch
from faces_frames_dataset import FacesFramesDataset
from torch.utils.data import DataLoader, random_split

# CUDA / device settings
use_cuda = torch.cuda.is_available()  # Check if GPU is available
device = torch.device("cuda:0" if use_cuda else "cpu")  # Set device to GPU if available

# Data paths
# Define paths to input data relative to this file
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
data_dir = project_root / "data"  # Main data folder
img_dir = Path(data_dir / "faces")  # Folder with face images
csv_file = Path(data_dir / "labels.csv")  # CSV file with labels

# DataLoader parameters
params = {
    "batch_size": 1,  # Batch size
    "shuffle": True,  # Shuffle the data
    "num_workers": 0,  # Number of subprocesses to use for data loading
}

# Create dataset
dataset = FacesFramesDataset(csv_file, img_dir)

# Split dataset: train, validation, test
dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)  # 80% for training
val_size = int(dataset_size * 0.1)  # 10% for validation
test_size = dataset_size - train_size - val_size  # remainder for testing

# Random split of the dataset
train, val, test = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders
training_generator = DataLoader(train, **params)
validation_generator = DataLoader(val, **params)
test_generator = DataLoader(test, **params)

# Iterate over batches
for face_batch, flow_batch, local_labels in training_generator:
    # Move batch to device
    face_batch = face_batch.to(device)
    flow_batch = flow_batch.to(device)
    local_labels = local_labels.to(device)

    print("Face batch size:", face_batch.shape)
    print("Flow batch size:", flow_batch.shape)
    print("Labels:", local_labels)
