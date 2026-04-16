"""Self-supervised SimCLR pre-training — dual stream (face + pose)."""

import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .encoder import build_encoder
from .faces_frames_dataset import FacesFramesSSLDataset, SimCLRDataset
from .losses import NTXentLoss
from .models import SimCLRProjectionWrapper
from .pose_transforms import PoseSimCLRTransform
from .transforms import VideoSimCLRTransform
from .utils import ssl_collate_fn


def train_ssl(
    args: argparse.Namespace,
    device: torch.device,
    img_dir: Path,
    weights_dir: Path,
) -> None:
    print(f"--- STARTING SSL PRETRAINING (SimCLR) with {args.encoder} ---")

    base_ds: FacesFramesSSLDataset | Subset = FacesFramesSSLDataset(img_dir)

    if args.subset and args.subset < len(base_ds):
        indices = random.sample(range(len(base_ds)), args.subset)
        base_ds = Subset(base_ds, indices)
        print(f"Using subset of {args.subset} samples")

    ssl_ds = SimCLRDataset(
        base_ds, VideoSimCLRTransform(size=128), PoseSimCLRTransform()
    )

    loader = DataLoader(
        ssl_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=ssl_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    encoder, encoder_dim = build_encoder(args, device)

    if args.encoder == "fast_gru":
        for name, param in encoder.backbone.named_parameters():
            param.requires_grad = not ("layer1" in name or "layer2" in name)
    elif args.encoder == "dual_stream":
        for name, param in encoder.face_encoder.backbone.named_parameters():
            param.requires_grad = not ("layer1" in name or "layer2" in name)

    model = SimCLRProjectionWrapper(
        encoder,
        encoder_output_dim=encoder_dim,
        projection_dim=256,
    ).to(device)

    param_groups: list[dict] = []

    if args.encoder == "fast_gru":
        param_groups += [
            {"params": encoder.backbone.parameters(), "lr": 1e-5},
            {"params": encoder.gru.parameters(), "lr": 1e-4},
            {"params": encoder.attention.parameters(), "lr": 1e-4},
        ]
    elif args.encoder == "dual_stream":
        param_groups += [
            {"params": encoder.face_encoder.backbone.parameters(), "lr": 1e-5},
            {"params": encoder.face_encoder.gru.parameters(), "lr": 1e-4},
            {"params": encoder.face_encoder.attention.parameters(), "lr": 1e-4},
            {"params": encoder.pose_encoder.parameters(), "lr": 1e-4},
            {"params": encoder.fusion.parameters(), "lr": 1e-4},
        ]

    param_groups.append({"params": model.projector.parameters(), "lr": 1e-4})
    optimizer = torch.optim.Adam(param_groups)

    criterion = NTXentLoss(temperature=args.temperature)
    print(f"NTXentLoss | temp={args.temperature}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    print(f"Batch size: {args.batch_size} | LR: {optimizer.param_groups[0]['lr']}")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        pbar = tqdm(loader, desc=f"SSL Epoch {epoch + 1}", leave=True)

        for fv1, fv2, pv1, pv2, lengths in pbar:
            fv1 = fv1.to(device, non_blocking=True)
            fv2 = fv2.to(device, non_blocking=True)
            pv1 = pv1.to(device, non_blocking=True)
            pv2 = pv2.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                z1 = model(fv1, pv1, lengths)
                z2 = model(fv2, pv2, lengths)
            loss = criterion(z1, z2)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / num_batches
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{args.epochs} – Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}"
        )
        scheduler.step()

    save_path = (
        weights_dir
        / f"ssl_backbone_{args.encoder}_bs{args.batch_size}_t{args.temperature}.pt"
    )
    torch.save(encoder.state_dict(), save_path)
    print(f"SSL backbone saved → {save_path}")
