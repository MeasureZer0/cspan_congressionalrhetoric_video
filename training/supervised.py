"""Supervised fine-tuning and evaluation."""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .encoder import build_encoder
from .faces_frames_dataset import FacesFramesSupervisedDataset
from .optimizer import build_optimizer
from .transforms import VideoSimCLRTransform
from .utils import EarlyStopping, supervised_collate_fn


def train_supervised(
    args: argparse.Namespace,
    device: torch.device,
    img_dir: Path,
    csv_file: Path,
    weights_dir: Path,
    logs_dir: Path,
) -> None:
    """Supervised training loop with validation-based
    early stopping and test evaluation."""
    print(f"--- SUPERVISED TRAINING ({args.encoder}) ---")

    train_csv = csv_file.parent / "train.csv"
    val_csv = csv_file.parent / "val.csv"
    test_csv = csv_file.parent / "test.csv"

    train_transform = VideoSimCLRTransform(
        size=128,
        jitter_params=(0.3, 0.3, 0.3, 0.05),
        gray_p=0.1,
    )

    aug_multiplier = getattr(args, "aug_multiplier", 1)

    # Build datasets — keep a clean (no-augment) version for class-weight computation
    train_ds_base = FacesFramesSupervisedDataset(train_csv, img_dir)
    train_ds = FacesFramesSupervisedDataset(
        train_csv,
        img_dir,
        transform=train_transform,
        aug_multiplier=aug_multiplier,
    )
    val_ds = FacesFramesSupervisedDataset(val_csv, img_dir)
    test_ds = FacesFramesSupervisedDataset(test_csv, img_dir)

    n_orig = len(train_ds_base)
    print(
        f"Train: {len(train_ds)} ({n_orig} × {aug_multiplier}) | "
        f"Val: {len(val_ds)} | Test: {len(test_ds)}"
    )

    def _make_loader(
        dataset: FacesFramesSupervisedDataset, *, shuffle: bool
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=supervised_collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

    train_loader = _make_loader(train_ds, shuffle=True)
    val_loader = _make_loader(val_ds, shuffle=False)
    test_loader = _make_loader(test_ds, shuffle=False)

    model, _ = build_encoder(args, device)

    if args.load_ssl:
        ssl_path = weights_dir / f"ssl_backbone_{args.encoder}.pt"
        if ssl_path.exists():
            model.load_state_dict(
                torch.load(ssl_path, map_location=device, weights_only=True),
                strict=False,
            )
            print(f"Loaded SSL weights from {ssl_path}")
        else:
            print(f"[warn] SSL weights not found at {ssl_path} — training from scratch")

    optimizer = build_optimizer(model, args)

    # Inverse-frequency class weights computed from the clean training set
    class_counts = np.bincount(
        [int(label.item()) for _, _, label in train_ds_base],
        minlength=3,
    )
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    weights /= weights.sum()
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    early_stopping = EarlyStopping(patience=7, min_delta=1e-4)

    best_val_acc = 0.0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model_path = weights_dir / f"best_{args.encoder}_{timestamp}.pt"

    logs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = logs_dir / f"supervised_{args.encoder}.csv"

    with open(csv_path, mode="w", newline="") as log_fh:
        writer = csv.writer(log_fh)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "train_acc",
                "val_acc",
                "val_f1_macro",
                "val_conf_matrix",
                "lr",
            ]
        )

        try:
            for epoch in range(args.epochs):
                # ── Training ──────────────────────────────────────────────
                model.train()
                train_loss = 0.0
                correct = total = 0
                pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} train")

                for faces, pose, labels, lengths in pbar:
                    faces = faces.to(device, non_blocking=True)
                    pose = pose.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    optimizer.zero_grad()
                    logits = model(faces, pose, lengths)
                    loss = criterion(logits, labels)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    train_loss += loss.item()
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        acc=f"{correct / total:.4f}" if total else "—",
                    )

                train_acc = correct / total
                avg_train_loss = train_loss / len(train_loader)

                # ── Validation ────────────────────────────────────────────
                model.eval()
                val_loss = 0.0
                all_preds: list[int] = []
                all_labels: list[int] = []

                with torch.no_grad():
                    for faces, pose, labels, lengths in val_loader:
                        faces = faces.to(device)
                        pose = pose.to(device)
                        labels = labels.to(device)
                        logits = model(faces, pose, lengths)
                        val_loss += criterion(logits, labels).item()
                        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
                        all_labels.extend(labels.cpu().tolist())

                avg_val_loss = val_loss / len(val_loader)
                val_acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
                val_f1 = float(f1_score(all_labels, all_preds, average="macro"))
                val_cm = confusion_matrix(all_labels, all_preds)
                val_cm_str = ";".join(",".join(map(str, r)) for r in val_cm)
                current_lr = optimizer.param_groups[0]["lr"]

                print(
                    f"  train_loss={avg_train_loss:.4f}  train_acc={train_acc:.4f}  "
                    f"val_loss={avg_val_loss:.4f}  val_acc={val_acc:.4f}  "
                    f"val_f1={val_f1:.4f}  lr={current_lr:.2e}"
                )

                writer.writerow(
                    [
                        epoch + 1,
                        f"{avg_train_loss:.6f}",
                        f"{avg_val_loss:.6f}",
                        f"{train_acc:.6f}",
                        f"{val_acc:.6f}",
                        f"{val_f1:.6f}",
                        val_cm_str,
                        f"{current_lr:.8f}",
                    ]
                )
                log_fh.flush()

                scheduler.step()

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), best_model_path)
                    print(f"  ✓ Best model saved (val_acc={val_acc:.4f})")

                if early_stopping(avg_val_loss):
                    print("Early stopping triggered.")
                    break

        finally:
            print(f"\n✓ Training log → {csv_path}")

    # ── Test evaluation ───────────────────────────────────────────────────
    print("\n--- FINAL TEST EVALUATION ---")
    model.load_state_dict(
        torch.load(best_model_path, map_location=device, weights_only=True)
    )
    model.eval()

    test_preds: list[int] = []
    test_labels_list: list[int] = []

    with torch.no_grad():
        for faces, pose, labels, lengths in tqdm(test_loader, desc="Testing"):
            faces = faces.to(device)
            pose = pose.to(device)
            labels = labels.to(device)
            preds = model(faces, pose, lengths).argmax(dim=1)
            test_preds.extend(preds.cpu().tolist())
            test_labels_list.extend(labels.cpu().tolist())

    test_acc = float(np.mean(np.array(test_preds) == np.array(test_labels_list)))
    test_f1 = float(f1_score(test_labels_list, test_preds, average="macro"))
    test_cm = confusion_matrix(test_labels_list, test_preds)

    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Test F1-macro : {test_f1:.4f}")
    print(f"Confusion Matrix:\n{test_cm}")

    results_path = logs_dir / f"test_results_{args.encoder}.txt"
    results_path.write_text(
        f"Model      : {args.encoder}\n"
        f"Checkpoint : {best_model_path}\n"
        f"Accuracy   : {test_acc:.4f}\n"
        f"F1-macro   : {test_f1:.4f}\n"
        f"Confusion Matrix:\n{test_cm}\n"
    )
    print(f"Test results → {results_path}")
