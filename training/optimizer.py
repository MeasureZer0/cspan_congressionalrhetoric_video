import argparse
from typing import Protocol, cast

import torch
from torch import nn


class _FastGruLike(Protocol):
    image_extractor: nn.Module
    gru: nn.Module
    attention: nn.Module


def build_optimizer(
    model: nn.Module, args: argparse.Namespace
) -> torch.optim.Optimizer:
    """
    Builds the Adam optimizer for the model, with \
        specific LR settings for GRU components.
    """
    if args.encoder == "fast_gru":
        # Check if it's a SimCLR wrapper or the raw encoder
        encoder = cast(_FastGruLike, getattr(model, "encoder", model))
        classifier = getattr(model, "classifier", None)
        return torch.optim.Adam(
            [
                {"params": encoder.image_extractor.parameters(), "lr": 1e-5},
                {"params": encoder.gru.parameters(), "lr": 1e-5},
                {"params": encoder.attention.parameters(), "lr": 1e-4},
                {
                    "params": classifier.parameters()
                    if isinstance(classifier, nn.Module)
                    else [],
                    "lr": 1e-3,
                },
            ]
        )
    else:  # baseline
        return torch.optim.Adam(model.parameters(), lr=1e-4)
