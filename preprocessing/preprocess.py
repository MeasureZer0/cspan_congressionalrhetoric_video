"""Main preprocessing script with argument parsing."""

import argparse
from dataclasses import fields
from pathlib import Path

from config import AugmentationConfig, PreprocessingConfig
from crop_faces import process_videos_in_parallel


def parse_args() -> PreprocessingConfig:
    """Parse command line arguments and merge with default configuration."""
    parser = argparse.ArgumentParser(
        description="Process videos to extract face tensors with optional augmentation."
    )

    # Input/output arguments
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Path to the directory containing video files.",
    )
    parser.add_argument(
        "--label-file",
        type=Path,
        help="Path to the CSV file containing video labels.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Path to the directory where output tensors will be saved.",
    )

    # Processing parameters
    parser.add_argument(
        "--frame-skip",
        type=int,
        help="Save only every N-th frame.",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        help="Target size for face tensors (height width).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        help="Margin to add around detected faces.",
    )
    parser.add_argument(
        "--crop-width-ratio",
        type=float,
        help="Width ratio for center cropping before face detection.",
    )

    # Execution options
    parser.add_argument(
        "--purge",
        action="store_true",
        help="Purge existing output files before processing.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum number of parallel workers.",
    )

    # Augmentation arguments
    parser.add_argument(
        "--use-augmentation",
        action="store_true",
        help="Enable data augmentation.",
    )
    parser.add_argument(
        "--rotation-degrees",
        type=float,
        help="Maximum degrees for random rotation.",
    )
    parser.add_argument(
        "--brightness",
        type=float,
        help="Brightness jitter factor.",
    )
    parser.add_argument(
        "--contrast",
        type=float,
        help="Contrast jitter factor.",
    )
    parser.add_argument(
        "--saturation",
        type=float,
        help="Saturation jitter factor.",
    )
    parser.add_argument(
        "--hue",
        type=float,
        help="Hue jitter factor.",
    )

    args = parser.parse_args()

    # Create top-level overrides (filtering None values)
    # Note: store_true arguments (purge, use_augmentation) will be False if not set.
    # Since config defaults are also False, this is safe.
    overrides = {k: v for k, v in vars(args).items() if v is not None}

    # Convert size from list to tuple if present in overrides
    if "size" in overrides and isinstance(overrides["size"], list):
        overrides["size"] = tuple(overrides["size"])

    # Load base config with top-level overrides
    config = PreprocessingConfig.load(**overrides)

    # Apply augmentation overrides manually since they are nested
    # Map --use-augmentation to config.augmentation.enabled
    if args.use_augmentation:
        config.augmentation.enabled = True

    # Identify and apply other augmentation fields dynamically
    aug_fields = {f.name for f in fields(AugmentationConfig)}
    for field_name in aug_fields:
        if field_name == "enabled":
            continue
        # Check if this field exists in args and was set (not None)
        val = overrides.get(field_name)
        if val is not None:
            setattr(config.augmentation, field_name, val)

    return config


if __name__ == "__main__":
    config = parse_args()

    print("Preprocessing configuration:")
    print(config)

    process_videos_in_parallel(config=config)
