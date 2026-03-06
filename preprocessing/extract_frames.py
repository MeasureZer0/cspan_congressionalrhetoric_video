import cv2
from numpy import ndarray


def extract_frames(
    path: str,
    frame_skip: int = 30,
    skip_start_ratio: float = 0.1,
    skip_end_ratio: float = 0.1,
) -> list[ndarray]:
    """
    Extract frames from a video file.

    Parameters
    ----------
    path : str
        Path to the video file.
    frame_skip: int, default=30
        Save only every N-th frame.
    skip_start_ratio: float, default=0.1
        Ratio of frames to skip at the start of the video. Maximum 5 seconds
        at 30 fps.
    skip_end_ratio: float, default=0.1
        Ratio of frames to skip at the end of the video. Maximum 5 seconds
        at 30 fps.

    Returns
    -------
    frames : list
        A list of extracted frames.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {path}; skipping.")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    skip_start = min(int(total_frames * skip_start_ratio), 5 * 30)
    skip_end = min(int(total_frames * skip_end_ratio), 5 * 30)

    frames: list[ndarray] = []
    counter = 0

    max_frames = 120

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if counter < skip_start or counter >= total_frames - skip_end:
            counter += 1
            continue

        if counter % frame_skip == 0:
            frames.append(frame)

        # Stop capturing if max_frames is reached
        if len(frames) >= max_frames:
            break

        counter += 1

    cap.release()

    print(f"Frames captured from {path}: {len(frames)}")
    return frames
