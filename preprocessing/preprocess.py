"""Main preprocessing script with argument parsing."""

import argparse
from pathlib import Path

from config import PreprocessingConfig
from crop_faces import process_videos_in_parallel


def parse_args() -> PreprocessingConfig:
    """Parse command line arguments and merge with default configuration."""
    parser = argparse.ArgumentParser(
        description="Process videos to extract face and pose tensors."
    )

    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--label-file", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--frame-skip", type=int)
    parser.add_argument(
        "--size", type=int, nargs=2, help="Target size for face tensors (height width)."
    )
    parser.add_argument("--margin", type=float)
    parser.add_argument("--crop-width-ratio", type=float)
    parser.add_argument("--purge", action="store_true")
    parser.add_argument("--max-workers", type=int)

    args = parser.parse_args()
    overrides = {k: v for k, v in vars(args).items() if v is not None}

    if "size" in overrides and isinstance(overrides["size"], list):
        overrides["size"] = tuple(overrides["size"])

    return PreprocessingConfig.load(**overrides)


if __name__ == "__main__":
    config = parse_args()
    print("Preprocessing configuration:")
    print(config)
    process_videos_in_parallel(config=config)
