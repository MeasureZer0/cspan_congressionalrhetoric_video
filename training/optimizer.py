import argparse

import torch
from torch import nn


def _build_optimizer(
    model: nn.Module, args: argparse.Namespace
) -> torch.optim.Optimizer:
    """
    Builds the Adam optimizer for the model, with
        specific LR settings for GRU components.
    """
    if args.encoder == "fast_gru":
        # Check if it's a SimCLR wrapper or the raw encoder
        encoder = model.encoder if hasattr(model, "encoder") else model
        return torch.optim.Adam(
            [
                {"params": encoder.image_extractor.parameters(), "lr": 1e-3},
                {"params": encoder.gru.parameters(), "lr": 1e-3},
                {"params": encoder.attention.parameters(), "lr": 1e-3},
                {
                    "params": model.classifier.parameters()
                    if hasattr(model, "classifier")
                    else [],
                    "lr": 1e-3,
                },
            ]
        )
    else:  # baseline
        return torch.optim.Adam(model.parameters(), lr=1e-4)
