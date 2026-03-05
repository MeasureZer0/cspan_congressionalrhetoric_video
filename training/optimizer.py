import argparse
from typing import Protocol, cast

import torch
from torch import nn


class _FastGruLike(Protocol):
    backbone: nn.Module
    gru: nn.Module
    attention: nn.Module


class _DualStreamLike(Protocol):
    face_encoder: nn.Module
    pose_encoder: nn.Module
    fusion: nn.Module
    classifier: nn.Module


def build_optimizer(
    model: nn.Module,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    """Build an Adam optimizer with per-component learning rates."""
    if args.encoder == "fast_gru":
        enc = cast(_FastGruLike, getattr(model, "encoder", model))
        classifier = getattr(model, "classifier", None)

        param_groups: list[dict] = [
            {"params": enc.backbone.parameters(), "lr": 1e-5},
            {"params": enc.gru.parameters(), "lr": 1e-4},
            {"params": enc.attention.parameters(), "lr": 1e-4},
        ]
        if classifier is not None:
            param_groups.append({"params": classifier.parameters(), "lr": 1e-3})

        return torch.optim.Adam(param_groups, weight_decay=1e-4)

    if args.encoder == "dual_stream":
        dual = cast(_DualStreamLike, getattr(model, "encoder", model))
        face_enc = cast(_FastGruLike, dual.face_encoder)

        param_groups = [
            {"params": face_enc.backbone.parameters(), "lr": 1e-5},
            {"params": face_enc.gru.parameters(), "lr": 5e-5},
            {"params": face_enc.attention.parameters(), "lr": 5e-5},
            {"params": dual.pose_encoder.parameters(), "lr": 5e-5},
            {"params": dual.fusion.parameters(), "lr": 1e-4},
            {"params": dual.classifier.parameters(), "lr": 1e-4},
        ]
        return torch.optim.Adam(param_groups, weight_decay=1e-4)

    raise ValueError(
        f"Unknown encoder: {args.encoder!r}. Choose one of: 'fast_gru', 'dual_stream'."
    )
