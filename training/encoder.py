import argparse

import torch

from .models import DualStreamEncoder, FastGRU, TinyMLPEncoder


def build_encoder(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DualStreamEncoder | FastGRU | TinyMLPEncoder, int]:
    """
    Instantiate the encoder chosen on the command line.
    """
    if args.encoder == "baseline":
        encoder = TinyMLPEncoder(hidden_size=64).to(device)
        encoder_dim = 64

    elif args.encoder == "fast_gru":
        freeze_backbone = getattr(args, "freeze_backbone", False)
        encoder = FastGRU(hidden_size=128, freeze_backbone=freeze_backbone).to(device)
        encoder_dim = 128

    elif args.encoder == "dual_stream":
        encoder = DualStreamEncoder(
            face_hidden=128,
            pose_hidden=64,
        ).to(device)
        encoder_dim = encoder.output_dim

    else:
        raise ValueError(f"Unknown encoder: {args.encoder!r}")

    return encoder, encoder_dim
