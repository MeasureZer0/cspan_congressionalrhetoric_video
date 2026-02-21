import argparse

import torch
from models import FastGRU, TinyMLPEncoder
from torch import nn


def _build_encoder(
    args: argparse.Namespace, device: torch.device
) -> tuple[nn.Module, int]:
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
