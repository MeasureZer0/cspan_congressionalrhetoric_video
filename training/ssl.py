import argparse
import random
from pathlib import Path

import torch
from faces_frames_dataset import FacesFramesSSLDataset, SimCLRDataset
from losses import NTXentLoss, NTXentLossWithMemoryBank
from memory_bank import MemoryBank
from models import SimCLRProjectionWrapper
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transforms import VideoSimCLRTransform

from .encoder import _build_encoder
from .utils import ssl_collate_fn


def train_ssl(
    args: argparse.Namespace, device: torch.device, img_dir: Path, weights_dir: Path
) -> None:
    """
    Main loop for self-supervised pre-training using SimCLR.
    """
    print(f"--- STARTING SSL PRETRAINING (SimCLR) with {args.encoder} ---")
    base_ds = FacesFramesSSLDataset(img_dir)

    if args.subset and args.subset < len(base_ds):
        indices = random.sample(range(len(base_ds)), args.subset)
        base_ds = Subset(base_ds, indices)
        print(f"Using subset of {args.subset} samples for SSL training")

    transform = VideoSimCLRTransform(size=128)
    ssl_ds = SimCLRDataset(base_ds, transform)

    loader = DataLoader(
        ssl_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=ssl_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    encoder, encoder_dim = _build_encoder(args, device)
    projection_dim = 256

    if args.encoder == "fast_gru":
        for name, param in encoder.image_extractor.named_parameters():
            if "layer3" in name or "layer4" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    model = SimCLRProjectionWrapper(
        encoder, encoder_output_dim=encoder_dim, projection_dim=projection_dim
    ).to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": encoder.image_extractor.parameters(), "lr": 1e-5},
            {"params": encoder.gru.parameters(), "lr": 1e-4},
            {"params": model.projector.parameters(), "lr": 1e-4},
        ]
    )

    if args.use_memory_bank:
        memory_bank = MemoryBank(size=args.bank_size, dim=projection_dim).to(device)
        criterion = NTXentLossWithMemoryBank(temperature=args.temperature)
        print(
            f"Using Memory Bank | size={args.bank_size}, \
                dim={projection_dim}, \
                    temperature={args.temperature}"
        )
    else:
        criterion = NTXentLoss(temperature=args.temperature)
        memory_bank = None
        print(f"Using standard NTXentLoss | temperature={args.temperature}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    print(f"Batch size: {args.batch_size} | Initial LR: 1e-4")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        pbar = tqdm(loader, desc=f"SSL Epoch {epoch + 1}", leave=True)

        for v1, v2, lengths in pbar:
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)

            optimizer.zero_grad()

            if args.use_memory_bank:
                z1 = model(v1, lengths)
                z2 = model(v2, lengths)
                loss = criterion(z1, z2, memory_bank)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            else:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    z1 = model(v1, lengths)
                    z2 = model(v2, lengths)
                loss = criterion(z1, z2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            postfix = {"loss": f"{loss.item():.4f}"}
            if args.use_memory_bank:
                postfix["bank"] = f"{len(memory_bank)}/{args.bank_size}"
            pbar.set_postfix(**postfix)

        avg_loss = epoch_loss / num_batches
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{args.epochs} - Avg Loss: {avg_loss:.4f}, \
                LR: {current_lr:.6f}"
        )
        scheduler.step()

    save_path = weights_dir / f"ssl_backbone_{args.encoder}.pt"
    torch.save(encoder.state_dict(), save_path)
    print(f"SSL Backbone saved to {save_path}")
