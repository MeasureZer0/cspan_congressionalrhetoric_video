"""Main preprocessing entry point."""

import argparse
import sys
from pathlib import Path

# Allow running as a script from the project root: python preprocessing/preprocess.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PreprocessingConfig  # noqa: E402
from crop_faces import process_videos_in_parallel  # noqa: E402


def parse_args() -> PreprocessingConfig:
    parser = argparse.ArgumentParser(
        description="Extract face and pose tensors from raw videos."
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--label-file", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--frame-skip", type=int)
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        help="Target face tensor size (height width)",
    )
    parser.add_argument("--margin", type=float)
    parser.add_argument("--crop-width-ratio", type=float)
    parser.add_argument(
        "--purge", action="store_true", help="Overwrite existing output tensors"
    )
    parser.add_argument("--max-workers", type=int)

    args = parser.parse_args()
    overrides = {k: v for k, v in vars(args).items() if v is not None}

    if "size" in overrides and isinstance(overrides["size"], list):
        overrides["size"] = tuple(overrides["size"])

    return PreprocessingConfig.load(**overrides)


if __name__ == "__main__":
    config = parse_args()
    print("Preprocessing configuration:\n" + str(config))
    process_videos_in_parallel(config=config)
