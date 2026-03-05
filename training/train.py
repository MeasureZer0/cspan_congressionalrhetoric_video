import argparse

import torch

from .ssl import train_ssl
from .supervised import train_supervised
from .utils import default_paths, set_seed

"""
Training script for self-supervised (SimCLR) and supervised learning
on face tensor sequences extracted from videos.
"""


if __name__ == "__main__":
    set_seed(37)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, choices=["ssl", "supervised"], required=True
    )
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["fast_gru", "dual_stream"],
        default="dual_stream",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--load-ssl", action="store_true")
    parser.add_argument("--frame-skip", type=int, default=30)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--use-memory-bank", action="store_true")
    parser.add_argument("--bank-size", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument(
        "--freeze-backbone", action="store_true", help="Freeze ResNet backbone"
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    img_dir, csv_file, weights_dir, logs_dir = default_paths()
    weights_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "ssl":
        train_ssl(args, device, img_dir, weights_dir)
    else:
        train_supervised(args, device, img_dir, csv_file, weights_dir, logs_dir)
