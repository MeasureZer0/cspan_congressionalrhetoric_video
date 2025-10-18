import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from faces_frames_dataset import FacesFramesDataset
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, random_split
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm


# Check for GPU availability and configure device
def _get_device() -> torch.device:
    """Return the available device (cuda if available else cpu)."""
    use_cuda = torch.cuda.is_available()
    return torch.device("cuda:0" if use_cuda else "cpu")


def _default_paths() -> tuple[Path, Path, Path, Path]:
    """Return default (img_dir, csv_file, weights_dir, logs_dir) paths."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = project_root / "data"  # Main data folder
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
    labels = torch.stack([item[2] for item in batch])
    lengths = torch.tensor([seq.shape[0] for seq in batch_image], dtype=torch.long)
    return batch_image, batch_flow, labels, lengths


# DataLoader parameters
params = {
    "batch_size": 2,  # Batch size
    "shuffle": True,  # Randomize sequence order each epoch
    "num_workers": 0,  # Data loading in the main process
    "collate_fn": collate_fn,
}


def build_cnn(input_channels: int) -> tuple[nn.Module, int]:
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
    model.conv1 = nn.Conv2d(
        input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    feature_size = model.fc.in_features
    # remove classification layer
    model.fc = nn.Identity()  # type: ignore
    # model will be moved to the target device by the caller (e.g. model.to(device))
    return model, feature_size


class FeatureAggregatingLSTM(nn.Module):
    """
    A hybrid architecture combining CNN feature extraction and LSTM temporal modeling.
    Each frame and optical flow pair is processed through CNNs, concatenated, \
        and fed into LSTM.
    """

    def __init__(
        self, hidden_size: int = 128, num_layers: int = 5, num_classes: int = 3
    ) -> None:
        super().__init__()
        # Separate ResNet extractors for RGB frames and optical flow
        self.image_extractor, feature_size = build_cnn(3)
        self.flow_extractor, _ = build_cnn(2)

        # LSTM processes the concatenated feature vectors over time
        self.lstm = nn.LSTM(
            2 * feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )

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
            logits: tensor of shape [B, T, num_classes]
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
        logits = self.fc(output_padded)
        return logits


def run_training(
    *,
    epochs: int = 10,
    batch_size: int = 2,
    output_csv: Path | str | None = None,
) -> None:
    """
    Run the full training loop.

    Args:
        epochs: number of training epochs
        batch_size: batch size for training
    """

    device = _get_device()
    torch.manual_seed(2)

    # Resolve default paths if not provided
    img_dir, csv_file, weights_dir, logs_dir = _default_paths()

    params = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": 0,
        "collate_fn": collate_fn,
    }

    dataset = FacesFramesDataset(csv_file, img_dir)
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)
    val_size = int(dataset_size * 0.1)
    test_size = dataset_size - train_size - val_size

    train, val, test = random_split(dataset, [train_size, val_size, test_size])

    training_generator = DataLoader(train, **params)
    validation_generator = DataLoader(val, **params)
    test_generator = DataLoader(test, **params)

    model = FeatureAggregatingLSTM(num_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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
            os.fsync(f.fileno())

    for epoch in range(epochs):
        model.train()
        total_loss, total_acc = 0.0, 0.0
        for batch_image, batch_flow, labels, lengths in tqdm(
            training_generator, desc=f"Epoch {epoch + 1}"
        ):
            optimizer.zero_grad()
            labels = labels.unsqueeze(1).repeat(1, max(lengths)).to(device)
            logits = model(batch_image, batch_flow, lengths)

            loss = masked_loss(logits, labels, lengths)
            acc = masked_accuracy(logits, labels, lengths)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc.item()

        avg_train_loss = total_loss / len(training_generator)
        avg_train_acc = total_acc / len(training_generator)
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
        print(f"Validation: loss = {val_loss:.4f}, acc = {val_acc * 100:.2f}%")

        # Append epoch validation loss to CSV and flush to disk
        with open(output_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{val_loss:.6f}", f"{val_acc:.6f}"])
            f.flush()

    # Final test evaluation after all epochs
    test_loss, test_acc = evaluate(model, test_generator, device, desc="Test")
    print(f"Test: loss = {test_loss:.4f}, acc = {test_acc * 100:.2f}%")

    # Save final model to disk
    final_model_path = weights_dir / f"final_model_epoch_{epochs}.pt"
    weights_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        model.state_dict(),
        final_model_path,
    )
    print(f"Final model saved: {final_model_path}")

    # Append final test loss to CSV and flush
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["final", f"{test_loss:.6f}", f"{test_acc:.6f}"])
        f.flush()


def masked_loss(
    logits: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    """
    Computes cross-entropy loss while ignoring padded positions.

    Args:
        logits: tensor [B, T, num_classes]
        targets: tensor [B, T] with true class indices
        lengths: tensor [B] with sequence lengths

    Returns:
        scalar average loss over valid time steps
    """
    T = logits.shape[1]
    # ensure mask is created on the same device as logits
    mask = torch.arange(T, device=logits.device).unsqueeze(0) < lengths.to(
        logits.device
    ).unsqueeze(1)
    loss = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    loss = loss * mask
    return loss.sum() / mask.sum()


def masked_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    """
    Computes accuracy only for non-padded elements.

    Args:
        logits: tensor [B, T, num_classes]
        targets: tensor [B, T]
        lengths: tensor [B]

    Returns:
        scalar accuracy value over valid positions
    """
    T = logits.shape[1]
    # ensure mask is created on the same device as logits
    mask = torch.arange(T, device=logits.device).unsqueeze(0) < lengths.to(
        logits.device
    ).unsqueeze(1)

    predicted_labels = torch.argmax(logits, dim=2)
    correct = (predicted_labels == targets) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    return accuracy


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
    total_loss, total_acc = 0.0, 0.0
    with torch.no_grad():
        for batch_image, batch_flow, labels, lengths in tqdm(dataloader, desc=desc):
            labels = labels.unsqueeze(1).repeat(1, max(lengths)).to(device)
            logits = model(batch_image, batch_flow, lengths)
            loss = masked_loss(logits, labels, lengths)
            acc = masked_accuracy(logits, labels, lengths)
            total_loss += loss.item()
            total_acc += acc.item()

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / len(dataloader)
    return avg_loss, avg_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Train the video classification model.")
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
        "--output-csv",
        type=str,
        default=None,
        help=(
            "Path to output CSV file to append training results. "
            "Default is logs/{datetime}.csv."
        ),
    )
    args = parser.parse_args()

    run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_csv=args.output_csv,
    )
