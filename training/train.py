from pathlib import Path

import torch
import torch.nn.functional as F

from faces_frames_dataset import FacesFramesDataset
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torchvision.models import resnet18, ResNet18_Weights
from torch import nn
from tqdm import tqdm

# CUDA / device settings
use_cuda = torch.cuda.is_available()  # Check if GPU is available
device = torch.device("cuda:0" if use_cuda else "cpu")  # Set device to GPU if available

torch.manual_seed(2)

# Data paths
# Define paths to input data relative to this file
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
data_dir = project_root / "data"  # Main data folder
img_dir = Path(data_dir / "faces")  # Folder with face images
csv_file = Path(data_dir / "labels.csv")  # CSV file with labels

def collate_fn(batch):
    batch_image = [item[0].to(device) for item in batch]
    batch_flow = [item[1].to(device) for item in batch]
    labels = torch.stack([item[2] for item in batch]).to(device)
    lengths = torch.tensor([seq.shape[0] for seq in batch_image], dtype=torch.long)
    return batch_image, batch_flow, labels, lengths

# DataLoader parameters
params = {
    "batch_size": 2,  # Batch size
    "shuffle": True,  # Shuffle the data
    "num_workers": 0,  # Number of subprocesses to use for data loading
    "collate_fn": collate_fn
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

# CNN builder
def build_cnn(input_channels: int):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    feature_size = model.fc.in_features
    model.fc = nn.Identity()
    model.to(device)
    return model, feature_size

class FeatureAggregatingLSTM(nn.Module):
    def __init__(self, hidden_size = 128, num_layers = 5, num_classes=3):
        super().__init__()
        self.image_extractor, feature_size = build_cnn(3)
        self.flow_extractor, _  = build_cnn(2)
        self.lstm = nn.LSTM(2 * feature_size, hidden_size=hidden_size, num_layers= num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, batch_image, batch_flow, lengths):
        features = []
        for seq_image, seq_flow in zip(batch_image, batch_flow):
            seq_image = seq_image.to(device)
            seq_flow = seq_flow.to(device)
            image_features = self.image_extractor(seq_image)
            flow_features = self.flow_extractor(seq_flow)
            seq_features = torch.cat([image_features, flow_features], dim = 1)
            features.append(seq_features)
        
        padded = pad_sequence(features, batch_first=True)
        packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed)
        output_padded, _ = pad_packed_sequence(packed_output, batch_first=True)
        logits = self.fc(output_padded)
        print(logits.shape)
        return logits


def masked_loss(logits, targets, lengths):
    T = logits.shape[1]
    mask = torch.arange(T, device=logits.device).unsqueeze(0) < lengths.unsqueeze(1)
    loss = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    print(loss.shape, mask.shape)
    loss = loss * mask
    return loss.sum() / mask.sum()
      

model = FeatureAggregatingLSTM(num_classes=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(1):
    model.train()
    for batch_image, batch_flow, labels, lengths in tqdm(training_generator):
        optimizer.zero_grad()
        logits = model(batch_image, batch_flow, lengths)
        loss = masked_loss(logits, labels.unsqueeze(1).repeat(1, logits.shape[1]).to(device), lengths.to(device))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: loss = {loss:.4f}")