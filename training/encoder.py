import argparse

import torch

from .models import DualStreamEncoder, FastGRU


def build_encoder(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[FastGRU | DualStreamEncoder, int]:
    """Instantiate the encoder specified by args.encoder."""
    freeze_backbone = getattr(args, "freeze_backbone", False)

    if args.encoder == "fast_gru":
        encoder = FastGRU(hidden_size=128, freeze_backbone=freeze_backbone).to(device)
        encoder_dim = encoder.output_dim

    elif args.encoder == "dual_stream":
        encoder = DualStreamEncoder(
            face_hidden=128,
            pose_hidden=64,
            freeze_backbone=freeze_backbone,
        ).to(device)
        encoder_dim = encoder.output_dim

    else:
        raise ValueError(
            f"Unknown encoder: {args.encoder!r}. "
            "Choose one of: 'fast_gru', 'dual_stream'."
        )

    return encoder, encoder_dim
