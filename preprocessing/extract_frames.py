"""Frame extraction from video files."""

import cv2
from numpy import ndarray

_FPS_FALLBACK = 30
_MAX_SKIP_SECONDS = 5


def extract_frames(
    path: str,
    frame_skip: int = 30,
    skip_start_ratio: float = 0.1,
    skip_end_ratio: float = 0.1,
    max_frames: int = 120,
) -> list[ndarray]:
    """Extract up to *max_frames* frames from a video, sampling every *frame_skip* frames.

    Parameters
    ----------
    path:
        Path to the video file.
    frame_skip:
        Keep one frame every *frame_skip* frames.
    skip_start_ratio:
        Fraction of the video to skip at the start (capped at 5 s).
    skip_end_ratio:
        Fraction of the video to skip at the end (capped at 5 s).
    max_frames:
        Hard cap on the number of returned frames.

    Returns
    -------
    list[ndarray]
        Extracted BGR frames.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[error] Cannot open video: {path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or _FPS_FALLBACK
    max_skip_frames = int(_MAX_SKIP_SECONDS * fps)

    skip_start = min(int(total_frames * skip_start_ratio), max_skip_frames)
    skip_end = min(int(total_frames * skip_end_ratio), max_skip_frames)
    last_valid = total_frames - skip_end

    frames: list[ndarray] = []
    counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if counter >= last_valid:
            break

        if counter >= skip_start and (counter - skip_start) % frame_skip == 0:
            frames.append(frame)
            if len(frames) >= max_frames:
                break

        counter += 1

    cap.release()
    print(f"[frames] {len(frames)} extracted from {path}")
    return frames