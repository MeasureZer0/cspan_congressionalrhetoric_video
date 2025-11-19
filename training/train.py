import argparse
import csv
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from faces_frames_dataset import FacesFramesDataset
from subset_data_multiplier import SubsetDataMultiplier
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Subset
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm


# Check for GPU availability and configure device
def _get_device() -> torch.device:
    """Return the available device (cuda if available else cpu)."""
    use_cuda = torch.cuda.is_available()
    return torch.device("cuda:0" if use_cuda else "cpu")


def _default_paths(frame_skip: int) -> tuple[Path, Path, Path, Path]:
    """Return default (img_dir, csv_file, weights_dir, logs_dir) paths."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = project_root / "data" / f"frame_skip_{frame_skip}"  # Main data folder
    img_dir = data_dir / "faces"  # Directory containing image sequences
    csv_file = data_dir / "labels.csv"  # CSV file with labels for each sequence
    weights_dir = data_dir / "weights"  # Directory to save model weights to
    logs_dir = project_root / "logs"  # Directory to save training logs to
    return img_dir, csv_file, weights_dir, logs_dir


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Custom collate function for variable-length video sequences.
    It ensures that sequences of different lengths are properly handled when batched.

    Args:
        batch: list of tuples (images_tensor, flow_tensor, label)

    Returns:
        batch_image: list of tensors, each of shape (seq_len, C, H, W)
        batch_flow: list of tensors, each of shape (seq_len, C, H, W)
        labels: tensor of shape (batch_size,)
        lengths: tensor containing sequence lengths for each batch element
    """
    # NOTE: do not move device resolution to module scope; resolve at call time
    batch_image = [item[0] for item in batch]
    batch_flow = [item[1] for item in batch]
    labels = torch.stack([item[2] for item in batch]).long()
    lengths = torch.tensor([seq.shape[0] for seq in batch_image], dtype=torch.long)
    return batch_image, batch_flow, labels, lengths


# DataLoader parameters
params = {
    "batch_size": 2,  # Batch size
    "shuffle": True,  # Randomize sequence order each epoch
    "num_workers": 0,  # Data loading in the main process
    "collate_fn": collate_fn,
}


def build_resnet_cnn(input_channels: int) -> tuple[nn.Module, int]:
    """
    Builds a ResNet model that can process images with an arbitrary number of channels.
    The final fully connected layer is replaced with an identity to \
        extract feature vectors.

    Args:
        input_channels: number of input channels (e.g., 3 for RGB, 2 for optical flow)

    Returns:
        model: CNN feature extractor
        feature_size: dimensionality of the extracted feature vector
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    if input_channels == 2:
        model.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

    for name, param in model.named_parameters():
        if not (
            (name.startswith("conv1") and input_channels == 2)
            or name.startswith("layer4")
        ):
            param.requires_grad = False

    feature_size = model.fc.in_features
    # remove classification layer
    model.fc = nn.Identity()  # type: ignore
    # model will be moved to the target device by the caller (e.g. model.to(device))
    return model, feature_size


def build_small_cnn(input_channels: int) -> tuple[nn.Module, int]:
    """
    Builds a small CNN model that can process images with an arbitrary number of channels.

    Args:
        input_channels: number of input channels (e.g., 3 for RGB, 2 for optical flow)

    Returns:
        model: CNN feature extractor
        feature_size: dimensionality of the extracted feature vector
    """
    model = nn.Sequential(
        nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
    )
    feature_size = 128
    return model, feature_size


class FeatureAggregatingLSTM(nn.Module):
    """
    A hybrid architecture combining CNN feature extraction and LSTM temporal modeling.
    Each frame and optical flow pair is processed through CNNs, concatenated, \
        and fed into LSTM.
    """

    def __init__(
        self,
        hidden_size: int = 8,
        num_layers: int = 1,
        num_classes: int = 3,
        cnn_type: str = "resnet",
    ) -> None:
        super().__init__()

        # Separate CNN extractors for RGB frames and optical flow
        if cnn_type == "resnet":
            print(f"CNN MODEL: {cnn_type}")
            self.image_extractor, feature_size = build_resnet_cnn(3)
            self.flow_extractor, _ = build_resnet_cnn(2)
        elif cnn_type == "small":
            print(f"CNN MODEL: {cnn_type}")
            self.image_extractor, feature_size = build_small_cnn(3)
            self.flow_extractor, _ = build_small_cnn(2)
        else:
            raise ValueError(f"Unknown cnn_type: {cnn_type}")

        # LSTM processes the concatenated feature vectors over time
        self.lstm = nn.LSTM(
            2 * feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.5 if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(0.4)

        # Final linear layer for classification at each time step
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        batch_image: list[torch.Tensor],
        batch_flow: list[torch.Tensor],
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the model. Extracts features per frame, concatenates them,
        and processes the temporal sequence using an LSTM.

        Args:
            batch_image: list of tensors (B elements, each [seq_len, 3, H, W])
            batch_flow: list of tensors (B elements, each [seq_len, 2, H, W])
            lengths: tensor with the actual lengths of each sequence in the batch

        Returns:
            logits: tensor of shape [B, num_classes]
        """
        features = []
        device = next(self.parameters()).device
        for seq_image, seq_flow in zip(batch_image, batch_flow, strict=False):
            # move inputs to the same device as the model parameters
            seq_image = seq_image.to(device)
            seq_flow = seq_flow.to(device)
            image_features = self.image_extractor(seq_image)  # [T, feat_dim]
            flow_features = self.flow_extractor(seq_flow)  # [T, feat_dim]
            seq_features = torch.cat([image_features, flow_features], dim=1)
            features.append(seq_features)

        # Pad sequences to the same length
        padded = pad_sequence(features, batch_first=True)
        # Pack sequences for LSTM to skip padded elements
        packed = pack_padded_sequence(
            padded, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed)

        # Unpack sequences back to padded representation
        output_padded, _ = pad_packed_sequence(packed_output, batch_first=True)
        avg_pool = torch.sum(output_padded, dim=1) / lengths.to(device).unsqueeze(1)

        avg_pool = self.dropout(avg_pool)

        logits = self.fc(avg_pool)
        return logits


def stratified_split(
    dataset: FacesFramesDataset,
    fractions: tuple[float, float, float],
) -> tuple[list[int], list[int], list[int]]:
    """
    Splits the dataset into stratified train, validation, and test sets.

    Args:
        dataset: the full dataset to split
        fractions: list of fractions for each subset (train, val, test), \
            in our case (0.8, 0.1, 0.1)

    Returns:
        train: List of indices for training
        val: List of indices for validation
        test: List of indices for testing
    """
    indices = (
        [],  # negative
        [],  # neutral
        [],  # positive
    )
    for i in range(len(dataset)):
        _, _, label = dataset[i]
        if label is not None:
            indices[label].append(i)

    train_indices, val_indices, test_indices = [], [], []
    cumulative_fractions = [fractions[0], fractions[0] + fractions[1]]
    for i in range(3):
        random.shuffle(indices[i])
        train_indices.extend(
            indices[i][: int(cumulative_fractions[0] * len(indices[i]))]
        )
        val_indices.extend(
            indices[i][
                int(cumulative_fractions[0] * len(indices[i])) : int(
                    cumulative_fractions[1] * len(indices[i])
                )
            ]
        )
        test_indices.extend(
            indices[i][int(cumulative_fractions[1] * len(indices[i])) :]
        )

    return train_indices, val_indices, test_indices


def run_training(
    *,
    frame_skip: int = 30,
    epochs: int = 10,
    batch_size: int = 2,
    data_multiplier: int = 1,
    augmentation_strength: str = "standard",
    output_csv: Path | str | None = None,
    cnn_type: str = "resnet",
) -> None:
    """
    Run the full training loop.

    Args:
        epochs: number of training epochs
        batch_size: batch size for training
        data_multiplier: how many versions of each video used for training
        augmentation_strength: "light", "standard", or "heavy"
        output_csv: path to output CSV file to append training results
    """

    assert 1 <= data_multiplier <= 2, "data_multiplier must be 1 or and 2"

    device = _get_device()
    torch.manual_seed(2)
    random.seed(2)

    # Resolve default paths if not provided
    img_dir, csv_file, weights_dir, logs_dir = _default_paths(frame_skip=frame_skip)

    params = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": 0,
        "collate_fn": collate_fn,
    }

    original_dataset = FacesFramesDataset(csv_file, img_dir)

    # Use stratified split: put around 60% train, 20% val, 20% test
    train_indices, val_indices, _ = stratified_split(original_dataset, (0.8, 0.2, 0.0))

    print(f"Original dataset: {len(original_dataset)} samples")
    print(f"Train split: {len(train_indices)} samples")

    if data_multiplier > 1:
        train_dataset = SubsetDataMultiplier(
            csv_file=csv_file,
            img_dir=img_dir,
            train_indices=train_indices,
            multiplier=data_multiplier,
            augmentation_strength=augmentation_strength,
        )
    else:
        train_dataset = Subset(original_dataset, train_indices)

    print(f"Training samples (after augmentation): {len(train_dataset)}")
    print(f"Validation samples: {len(val_indices)}")

    training_generator = DataLoader(train_dataset, **params)
    validation_generator = DataLoader(Subset(original_dataset, val_indices), **params)
    # test_generator = DataLoader(Subset(original_dataset, test_indices), **params)

    model = FeatureAggregatingLSTM(num_classes=3, cnn_type=cnn_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # Prepare output CSV path
    if output_csv is None:
        output_path = (
            logs_dir
            / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
    else:
        output_path = Path(output_csv)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # If file does not exist or is empty, write header
    if not output_path.exists() or output_path.stat().st_size == 0:
        with open(output_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "val_loss", "val_acc"])
            f.flush()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for batch_image, batch_flow, labels, lengths in tqdm(
            training_generator, desc=f"Epoch {epoch + 1}"
        ):
            optimizer.zero_grad()
            labels = labels.to(device)
            logits = model(batch_image, batch_flow, lengths)

            loss = sequence_loss(logits, labels)
            batch_correct, batch_total = sequence_accuracy(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += batch_correct
            total_samples += batch_total
        avg_train_loss = total_loss / total_samples
        avg_train_acc = total_correct / total_samples
        print(
            f"Epoch {epoch + 1}: train loss = {avg_train_loss:.4f}, "
            f"acc = {avg_train_acc:.4f}"
        )

        # Run validation phase after each epoch
        val_loss, val_acc = evaluate(
            model,
            validation_generator,
            device,
            desc="Validation",
        )
        scheduler.step(val_loss)
        print(f"Validation: loss = {val_loss:.4f}, acc = {val_acc * 100:.2f}%")

        # Append epoch validation loss to CSV and flush to disk
        with open(output_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{val_loss:.6f}", f"{val_acc:.6f}"])
            f.flush()

    # Final test evaluation after all epochs, right not we're not using it
    # test_loss, test_acc = evaluate(model, test_generator, device, desc="Test")
    # print(f"Test: loss = {test_loss:.4f}, acc = {test_acc * 100:.2f}%")

    # Save final model to disk
    final_model_path = weights_dir / f"final_model_epoch_{epochs}.pt"
    weights_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        model.state_dict(),
        final_model_path,
    )
    print(f"Final model saved: {final_model_path}")

    # Append final test loss to CSV and flush
    # with open(output_path, "a", newline="", encoding="utf-8") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["final", f"{test_loss:.6f}", f"{test_acc:.6f}"])
    #     f.flush()


def sequence_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes cross-entropy loss while ignoring padded positions.

    Args:
        logits: tensor [B, um_classes]
        targets: tensor [B] with true class indices

    Returns:
        scalar average loss over valid time steps
    """
    return F.cross_entropy(logits, targets)


def sequence_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes accuracy only for non-padded elements.

    Args:
        logits: tensor [B, num_classes]
        targets: tensor [B]
    Returns:
        scalar accuracy value over valid positions
    """
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)
        correct = (preds == targets).sum().item()
        total = targets.shape[0]
        return correct, total


def evaluate(
    model: nn.Module, dataloader: DataLoader, device: torch.device, desc: str = "Eval"
) -> tuple[float, float]:
    """
    Evaluates the model on the provided dataset.

    Args:
        model: trained model
        dataloader: DataLoader to evaluate on
        device: computation device
        desc: progress bar description string

    Returns:
        avg_loss: average masked loss
        avg_acc: average masked accuracy
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch_image, batch_flow, labels, lengths in tqdm(dataloader, desc=desc):
            labels = labels.to(device)
            logits = model(batch_image, batch_flow, lengths)
            loss = sequence_loss(logits, labels)
            batch_correct, batch_total = sequence_accuracy(logits, labels)
            total_loss += loss.item() * batch_total
            total_correct += batch_correct
            total_samples += batch_total

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Train the video classification model.")
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=30,
        help="the number of frames skipped during preprocessing.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--data-multiplier",
        type=int,
        default=1,
        help="How many versions of each video used for training (default: 1 = \
              no extra data).",
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        default="standard",
        choices=["light", "standard", "heavy"],
        help="Augmentation strength (default: standard).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help=(
            "Path to output CSV file to append training results. "
            "Default is logs/training_results_{datetime}.csv."
        ),
    )
    parser.add_argument(
        "--cnn-type",
        type=str,
        default="resnet",
        choices=["resnet", "small"],
        help="Type of CNN: 'resnet' (pretrained) or 'small' (custom lightweight CNN).",
    )
    args = parser.parse_args()

    run_training(
        frame_skip=args.frame_skip,
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_multiplier=args.data_multiplier,
        augmentation_strength=args.augmentation,
        output_csv=args.output_csv,
        cnn_type=args.cnn_type,
    )
