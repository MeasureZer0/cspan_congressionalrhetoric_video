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
from .losses import NTXentLoss, NTXentLossWithMemoryBank
from .memory_bank import MemoryBank
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
    """
    SimCLR pre-training loop.

    Each batch contains two independently-augmented views of every sample:
      face_v1, face_v2  – from VideoSimCLRTransform
      pose_v1, pose_v2  – from PoseSimCLRTransform

    The encoder (DualStreamEncoder or FastGRU) maps each view to an
    embedding, the projection head maps to the contrastive space, and
    NTXentLoss pulls same-sample embeddings together.
    """
    print(f"--- STARTING SSL PRETRAINING (SimCLR) with {args.encoder} ---")

    base_ds = FacesFramesSSLDataset(img_dir)

    if args.subset and args.subset < len(base_ds):
        indices = random.sample(range(len(base_ds)), args.subset)
        base_ds: FacesFramesSSLDataset | Subset[tuple[torch.Tensor, torch.Tensor]]
        base_ds = Subset(base_ds, indices)
        print(f"Using subset of {args.subset} samples")

    face_transform = VideoSimCLRTransform(size=128)
    pose_transform = PoseSimCLRTransform()

    ssl_ds = SimCLRDataset(base_ds, face_transform, pose_transform)

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
    projection_dim = 256

    if args.encoder == "fast_gru":
        if hasattr(encoder, "backbone") and hasattr(
            encoder.backbone, "named_parameters"
        ):
            for name, param in encoder.backbone.named_parameters():
                param.requires_grad = not ("layer1" in name or "layer2" in name)

    elif args.encoder == "dual_stream":
        face_enc = getattr(encoder, "face_encoder", None)
        if (
            face_enc is not None
            and hasattr(face_enc, "backbone")
            and hasattr(face_enc.backbone, "named_parameters")
        ):
            for name, param in face_enc.backbone.named_parameters():
                param.requires_grad = not ("layer1" in name or "layer2" in name)

    model = SimCLRProjectionWrapper(
        encoder,
        encoder_output_dim=encoder_dim,
        projection_dim=projection_dim,
    ).to(device)

    # Per-component learning rates
    param_groups: list[dict] = []

    if args.encoder == "fast_gru":
        face_enc = encoder
        param_groups.append({"params": face_enc.backbone.parameters(), "lr": 1e-5})
        param_groups.append({"params": face_enc.gru.parameters(), "lr": 1e-4})
        param_groups.append({"params": face_enc.attention.parameters(), "lr": 1e-4})

    elif args.encoder == "dual_stream":
        face_enc = encoder.face_encoder
        param_groups.append({"params": face_enc.backbone.parameters(), "lr": 1e-5})
        param_groups.append({"params": face_enc.gru.parameters(), "lr": 1e-4})
        param_groups.append({"params": face_enc.attention.parameters(), "lr": 1e-4})
        param_groups.append({"params": encoder.pose_encoder.parameters(), "lr": 1e-4})
        param_groups.append({"params": encoder.fusion.parameters(), "lr": 1e-4})

    elif args.encoder == "baseline":
        param_groups.append({"params": encoder.parameters(), "lr": 1e-4})

    param_groups.append({"params": model.projector.parameters(), "lr": 1e-4})

    optimizer = torch.optim.Adam(param_groups)

    if args.use_memory_bank:
        memory_bank = MemoryBank(size=args.bank_size, dim=projection_dim).to(device)
        criterion = NTXentLossWithMemoryBank(temperature=args.temperature)
        print(f"Memory bank | size={args.bank_size} | temp={args.temperature}")
    else:
        criterion = NTXentLoss(temperature=args.temperature)
        memory_bank = None
        print(f"Standard NTXentLoss | temp={args.temperature}")

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
            if args.use_memory_bank:
                z1 = model(fv1, pv1, lengths)
                z2 = model(fv2, pv2, lengths)
                loss = criterion(z1, z2, memory_bank)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    z1 = model(fv1, pv1, lengths)
                    z2 = model(fv2, pv2, lengths)
                loss = criterion(z1, z2)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            postfix = {"loss": f"{loss.item():.4f}"}
            if args.use_memory_bank and memory_bank is not None:
                postfix["bank"] = f"{len(memory_bank)}/{args.bank_size}"
            pbar.set_postfix(postfix)

        avg_loss = epoch_loss / num_batches
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{args.epochs} – "
            f"Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}"
        )
        scheduler.step()

    # Save encoder backbone (without projection head)
    save_path = (
        weights_dir / f"ssl_backbone_{args.encoder}"
        f"_bs{args.batch_size}"
        f"_mb{args.use_memory_bank}"
        f"_t{args.temperature}.pt"
    )
    torch.save(encoder.state_dict(), save_path)
    print(f"SSL backbone saved → {save_path}")
