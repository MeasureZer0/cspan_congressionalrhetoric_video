import argparse

import torch

from .models import FastGRU, TinyMLPEncoder


def build_encoder(
    args: argparse.Namespace, device: torch.device
) -> tuple[TinyMLPEncoder | FastGRU, int]:
    """
    Instantiates the chosen encoder model based on command line arguments.
    """
    if args.encoder == "baseline":
        encoder = TinyMLPEncoder(hidden_size=64).to(device)
        encoder_dim = 64
    elif args.encoder == "fast_gru":
        encoder = FastGRU(hidden_size=128).to(device)
        encoder_dim = 128
    else:
        raise ValueError(f"Unknown encoder: {args.encoder}")
    return encoder, encoder_dim
