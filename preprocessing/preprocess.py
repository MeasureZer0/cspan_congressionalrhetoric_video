"""Main preprocessing script with argument parsing."""

import argparse
from pathlib import Path

from config import AugmentationConfig, PreprocessingConfig
from crop_faces import process_videos_in_parallel


def parse_args() -> PreprocessingConfig:
    """Parse command line arguments into PreprocessingConfig."""
    # Create default config
    default_config = PreprocessingConfig()

    parser = argparse.ArgumentParser(
        description="Process videos to extract face tensors with optional augmentation."
    )

    # Input/output arguments
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_config.data_dir,
        help="Path to the directory containing video files.",
    )
    parser.add_argument(
        "--label-file",
        type=Path,
        default=default_config.label_file,
        help="Path to the CSV file containing video labels.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_config.out_dir,
        help="Path to the directory where output tensors will be saved.",
    )

    # Processing parameters
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=default_config.frame_skip,
        help="Save only every N-th frame.",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=default_config.size,
        help="Target size for face tensors (height width).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=default_config.margin,
        help="Margin to add around detected faces.",
    )
    parser.add_argument(
        "--crop-width-ratio",
        type=float,
        default=default_config.crop_width_ratio,
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
        default=default_config.max_workers,
        help="Maximum number of parallel workers. Defaults to CPU count.",
    )

    # Augmentation arguments
    parser.add_argument(
        "--use-augmentation",
        action="store_true",
        help="Enable data augmentation during preprocessing.",
    )
    parser.add_argument(
        "--rotation-degrees",
        type=float,
        default=default_config.augmentation.rotation_degrees,
        help="Maximum degrees for random rotation in augmentation.",
    )
    parser.add_argument(
        "--brightness",
        type=float,
        default=default_config.augmentation.brightness,
        help="Brightness jitter factor for augmentation.",
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=default_config.augmentation.contrast,
        help="Contrast jitter factor for augmentation.",
    )
    parser.add_argument(
        "--saturation",
        type=float,
        default=default_config.augmentation.saturation,
        help="Saturation jitter factor for augmentation.",
    )
    parser.add_argument(
        "--hue",
        type=float,
        default=default_config.augmentation.hue,
        help="Hue jitter factor for augmentation.",
    )
    parser.add_argument(
        "--augmentation-probability",
        type=float,
        default=default_config.augmentation.probability,
        help="Probability of applying augmentation to each video.",
    )

    args = parser.parse_args()

    # Convert size from list to tuple if needed
    if isinstance(args.size, list):
        args.size = tuple(args.size)

    # Build config from args
    config = PreprocessingConfig(
        data_dir=args.data_dir,
        label_file=args.label_file,
        out_dir=args.out_dir,
        frame_skip=args.frame_skip,
        size=args.size,
        margin=args.margin,
        crop_width_ratio=args.crop_width_ratio,
        purge=args.purge,
        max_workers=args.max_workers,
        augmentation=AugmentationConfig(
            enabled=args.use_augmentation,
            rotation_degrees=args.rotation_degrees,
            brightness=args.brightness,
            contrast=args.contrast,
            saturation=args.saturation,
            hue=args.hue,
            probability=args.augmentation_probability,
        ),
    )

    return config


if __name__ == "__main__":
    config = parse_args()

    # Validate paths
    assert config.label_file.exists(), f"Label file not found: {config.label_file}"
    assert config.data_dir.exists(), f"Data directory not found: {config.data_dir}"

    print("Configuration:")
    print(f"  Data dir: {config.data_dir}")
    print(f"  Label file: {config.label_file}")
    print(f"  Output dir: {config.out_dir}")
    print(f"  Frame skip: {config.frame_skip}")
    print(f"  Size: {config.size}")
    print(f"  Margin: {config.margin}")
    print(f"  Crop width ratio: {config.crop_width_ratio}")
    print(f"  Augmentation: {config.augmentation.enabled}")
    if config.augmentation.enabled:
        print(f"    Probability: {config.augmentation.probability}")
        print(f"    Rotation: ±{config.augmentation.rotation_degrees}°")
        print(f"    Brightness: ±{config.augmentation.brightness}")
        print(f"    Contrast: ±{config.augmentation.contrast}")
        print(f"    Saturation: ±{config.augmentation.saturation}")
        print(f"    Hue: ±{config.augmentation.hue}")

    process_videos_in_parallel(config=config)
