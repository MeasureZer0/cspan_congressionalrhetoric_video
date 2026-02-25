#!/usr/bin/env python3

import argparse
import csv
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
        default=100,
        help="Target number of videos in destination after processing (default: 100)",
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

    parser.add_argument(
        "--exclude-csv",
        type=Path,
        help="CSV file with a 'filename' column; listed videos will \
            not be moved to destination",
    )

    parser.add_argument(
        "--restore-excluded",
        action="store_true",
        help="Move excluded videos from destination back to source before \
            balancing destination count",
    )

    return parser.parse_args()


def load_excluded_filenames(csv_path: Path | None) -> set[str]:
    if csv_path is None:
        return set()

    if not csv_path.is_file():
        print(f"Exclude CSV does not exist: {csv_path}", file=sys.stderr)
        sys.exit(1)

    excluded: set[str] = set()

    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "filename" not in reader.fieldnames:
                print(
                    "Exclude CSV must contain a 'filename' column.",
                    file=sys.stderr,
                )
                sys.exit(1)

            for row in reader:
                filename = (row.get("filename") or "").strip()
                if filename:
                    excluded.add(Path(filename).name)
    except OSError as exc:
        print(f"Failed to read exclude CSV: {exc}", file=sys.stderr)
        sys.exit(1)

    return excluded


def list_videos(directory: Path, recursive: bool) -> list[Path]:
    iterator = directory.rglob("*") if recursive else directory.glob("*")
    return [f for f in iterator if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS]


def move_or_copy(src: Path, dst: Path, *, copy_mode: bool, dry_run: bool) -> None:
    action = shutil.copy2 if copy_mode else shutil.move
    if not dry_run:
        action(str(src), str(dst))


def main() -> None:
    args = parse_args()

    if not args.source.is_dir():
        print("Source directory does not exist.", file=sys.stderr)
        sys.exit(1)

    if not args.destination.is_dir():
        print("Destination directory does not exist.", file=sys.stderr)
        sys.exit(1)

    if args.count < 0:
        print("--count must be >= 0", file=sys.stderr)
        sys.exit(1)

    excluded_filenames = load_excluded_filenames(args.exclude_csv)
    operation_is_copy = args.copy
    action_word = "Copying" if operation_is_copy else "Moving"

    restored = 0
    restored_skipped = 0

    if args.restore_excluded and excluded_filenames:
        destination_videos = list_videos(args.destination, recursive=False)
        excluded_in_destination = [
            f for f in destination_videos if f.name in excluded_filenames
        ]

        for file_path in excluded_in_destination:
            source_path = args.source / file_path.name

            if source_path.exists() and not args.overwrite:
                print(f"Skipping restore (exists in source): {file_path.name}")
                restored_skipped += 1
                continue

            print(f"Restoring to source: {file_path.name}")
            move_or_copy(file_path, source_path, copy_mode=False, dry_run=args.dry_run)
            restored += 1

    destination_videos = list_videos(args.destination, recursive=False)
    destination_names = {f.name for f in destination_videos}
    current_count = len(destination_videos)

    if current_count >= args.count:
        print(
            f"Destination already has {current_count} videos (target: {args.count}). \
                No additional files needed."
        )
        if args.restore_excluded and excluded_filenames:
            print(f"Restore summary: restored={restored}, skipped={restored_skipped}")
        return

    source_videos = list_videos(args.source, recursive=args.recursive)

    eligible_source_videos = [
        f
        for f in source_videos
        if f.name not in excluded_filenames and f.name not in destination_names
    ]

    if not eligible_source_videos:
        print("No eligible source videos found to fill destination.")
        if args.restore_excluded and excluded_filenames:
            print(f"Restore summary: restored={restored}, skipped={restored_skipped}")
        return

    needed = args.count - current_count
    selected_files = random.sample(
        eligible_source_videos, min(needed, len(eligible_source_videos))
    )

    processed = 0
    skipped_exists = 0

    for file_path in selected_files:
        destination_path = args.destination / file_path.name

        if destination_path.exists() and not args.overwrite:
            print(f"Skipping (exists): {file_path.name}")
            skipped_exists += 1
            continue

        print(f"{action_word}: {file_path.name}")

        move_or_copy(
            file_path,
            destination_path,
            copy_mode=operation_is_copy,
            dry_run=args.dry_run,
        )

        processed += 1

    final_destination_count = current_count + processed

    print(f"\nDone. {action_word} {processed} files.")
    if args.restore_excluded and excluded_filenames:
        print(f"Restore summary: restored={restored}, skipped={restored_skipped}")
    if excluded_filenames:
        print(f"Excluded filenames loaded: {len(excluded_filenames)}")
    if skipped_exists:
        print(f"Skipped due to existing destination file: {skipped_exists}")
    print(f"Destination videos before: {current_count}")
    print(f"Destination target: {args.count}")
    print(f"Destination videos after (expected): {final_destination_count}")
    if final_destination_count < args.count:
        print(
            "Warning: Could not reach target count because there were not enough \
                eligible source videos."
        )


if __name__ == "__main__":
    main()
