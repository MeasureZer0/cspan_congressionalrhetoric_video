#!/usr/bin/env python3

import argparse
import random
import shutil
import sys
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move or copy random video files from one folder to another."
    )

    parser.add_argument("source", type=Path, help="Source directory")
    parser.add_argument("destination", type=Path, help="Destination directory")

    parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=1000,
        help="Number of random files to move (default: 1000)",
    )

    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Search subdirectories recursively",
    )

    parser.add_argument("--copy", action="store_true", help="Copy instead of move")

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files if they already exist in destination",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be moved without actually doing it",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.source.is_dir():
        print("Source directory does not exist.", file=sys.stderr)
        sys.exit(1)

    if not args.destination.is_dir():
        print("Destination directory does not exist.", file=sys.stderr)
        sys.exit(1)

    # Choose search mode
    iterator = args.source.rglob("*") if args.recursive else args.source.glob("*")

    video_files = [
        f for f in iterator if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if not video_files:
        print("No video files found.")
        return

    selected_files = random.sample(video_files, min(args.count, len(video_files)))

    operation = shutil.copy2 if args.copy else shutil.move
    action_word = "Copying" if args.copy else "Moving"

    processed = 0

    for file_path in selected_files:
        destination_path = args.destination / file_path.name

        if destination_path.exists() and not args.overwrite:
            print(f"Skipping (exists): {file_path.name}")
            continue

        print(f"{action_word}: {file_path.name}")

        if not args.dry_run:
            operation(str(file_path), str(destination_path))

        processed += 1

    print(f"\nDone. {action_word} {processed} files.")


if __name__ == "__main__":
    main()
