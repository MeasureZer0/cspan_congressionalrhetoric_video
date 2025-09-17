"""Video Preprocessing Module"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2  # OpenCV for video processing
import face_recognition
import numpy as np  # NumPy for storing frames as arrays
import pandas as pd  # Pandas for reading CSV labels
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Define paths to input data:
# - data_dir: directory containing videos and labels
# - label_file: CSV file with labels (video filename + label)
data_dir = Path("data/sample_video")
label_file = Path(data_dir / "labels.csv")


def extract_frames(cap: cv2.VideoCapture, frame_skip: int = 10) -> list:
    """
    Extract frames from a VideoCapture object.

    Parameters
    ----------
    cap: cv2.VideoCapture
        OpenCV VideoCapture object for the video file.
    frame_skip: int, default=10
        Save only every N-th frame (sampling frequency).

    Returns
    -------
    frames : list
        A list of extracted frames (each frame as a NumPy array).
    """
    frames = []
    count = 0  # frame counter
    while True:
        # - cap.read() returns:
        #     ret: Boolean indicating if a frame was read successfully
        #     frame: the actual frame image
        ret, frame = cap.read()

        # If no frame is returned:
        # - End of the video has been reached, OR
        # - An error occurred while reading
        if not ret:
            print("End of video or error occurred.")
            break

        # Store only every N-th frame
        if count % frame_skip == 0:
            frames.append(frame)

        # Debugging visualization:
        # Uncomment the code below to display video frames while processing
        # cv2.imshow("Video Frame", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        # Increment frame counter
        count += 1

    # Cleanup: release system resources after processing
    # - cap.release(): closes the video file
    # - cv2.destroyAllWindows():
    #   closes any OpenCV windows (uncomment during visualization)
    cap.release()
    # cv2.destroyAllWindows()

    # Print how many frames were captured
    print(f"Frames captured: {len(frames)}")
    return frames


def _rgb(frame: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR frame to RGB for face_recognition / PIL."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def detect_largest_face(
    frame: np.ndarray, detection_size: Optional[int] = 640
) -> Optional[Tuple[int, int, int, int]]:
    """Detect faces and return the largest bounding box.

    For speed we optionally resize the image so its largest side equals
    ``detection_size`` before running face detection, then map coordinates
    back to the original frame.

    Returns (top, right, bottom, left) or None if no face found.
    """
    rgb = _rgb(frame)
    h, w = rgb.shape[:2]

    # Optionally resize for faster detection
    if detection_size is not None and max(h, w) > detection_size:
        scale = detection_size / max(h, w)
        small_w = max(1, int(round(w * scale)))
        small_h = max(1, int(round(h * scale)))
        small = cv2.resize(rgb, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        locations = face_recognition.face_locations(small, model="hog")
        if not locations:
            return None

        # choose largest in small coords and map back
        areas = [(loc[2] - loc[0]) * (loc[1] - loc[3]) for loc in locations]
        largest_idx = int(np.argmax(areas))
        t, r, b, left = locations[largest_idx]
        # map back to original coordinates
        inv_scale = 1.0 / scale
        return (
            int(round(t * inv_scale)),
            int(round(r * inv_scale)),
            int(round(b * inv_scale)),
            int(round(left * inv_scale)),
        )

    # no resizing
    locations = face_recognition.face_locations(rgb, model="hog")
    if not locations:
        return None
    areas = [(loc[2] - loc[0]) * (loc[1] - loc[3]) for loc in locations]
    largest_idx = int(np.argmax(areas))
    return locations[largest_idx]


def crop_face(
    frame: np.ndarray,
    location: Optional[Tuple[int, int, int, int]],
    margin: Optional[float],
) -> Optional[np.ndarray]:
    """Crop a face from frame given a (top, right, bottom, left) tuple.

    Returns a numpy array (H, W, C) in uint8 RGB order.
    """
    if location is None:
        return None
    top, right, bottom, left = location
    rgb = _rgb(frame)
    # clamp coordinates
    h, w = rgb.shape[:2]

    # optionally expand box by margin (fraction of box size). margin may be
    # a float (fraction) or an absolute pixel value (int); interpret >0 and
    # <1 as fraction.
    if margin:
        box_h = bottom - top
        box_w = right - left
        if margin < 1.0:
            pad_h = int(round(box_h * margin))
            pad_w = int(round(box_w * margin))
        else:
            pad_h = int(round(margin))
            pad_w = int(round(margin))
        top -= pad_h
        bottom += pad_h
        left -= pad_w
        right += pad_w

    top = max(0, top)
    left = max(0, left)
    bottom = min(h, bottom)
    right = min(w, right)
    if bottom <= top or right <= left:
        return None
    return rgb[top:bottom, left:right]


def frames_to_face_tensors(
    frames: List[np.ndarray],
    size: tuple = (224, 224),
    detection_size: Optional[int] = 640,
    margin: Optional[float] = 0.0,
) -> torch.Tensor:
    """Detect largest face in each frame, crop, resize and convert to a tensor.

    Parameters
    - frames: list of BGR frames (as read by OpenCV)
    - size: (H, W) target size for resizing

    Returns
    - tensor of shape (N, C, H, W) where N is number of frames where a face was found.
      Returns an empty tensor with shape (0, 3, H, W) if no faces detected.
    """
    if not frames:
        return torch.empty((0, 3, size[0], size[1]), dtype=torch.float32)

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.ToTensor(),
            # normalization can be added if needed
        ]
    )

    tensors = []
    for f in frames:
        loc = detect_largest_face(f, detection_size=detection_size)
        face = crop_face(f, loc, margin=margin)
        if face is None:
            # skip frames without a detected face
            continue
        # transform expects HWC in uint8
        try:
            t = transform(face)
        except Exception:
            # fallback: convert with PIL.Image
            pil = Image.fromarray(face)
            t = transforms.ToTensor()(pil)
        tensors.append(t)

    if not tensors:
        return torch.empty((0, 3, size[0], size[1]), dtype=torch.float32)

    return torch.stack(tensors)


def save_tensor(tensor: torch.Tensor, out_path: str) -> None:
    """Save tensor to disk using torch.save. Creates parent dir if needed."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, str(out))


def process_and_save_all(
    data_dir: Path,
    label_file: Path,
    out_dir: Path,
    size: tuple = (224, 224),
    detection_size: Optional[int] = 640,
    margin: float = 0.0,
) -> List[str]:
    """Load videos, extract faces, convert to tensors and save per-video tensors.

    Returns a list of output file paths (one per video that produced faces).
    """
    outputs: List[str] = []
    df = pd.read_csv(label_file)

    # Use tqdm to show a progress bar over videos
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
        video_name = row.iloc[0]
        # prepare paths
        video_path = os.path.join(data_dir, video_name)
        stem = Path(video_name).stem
        out_path = Path(out_dir) / f"{stem}_faces.pt"

        # skip if already processed
        if out_path.exists():
            print(f"Skipping {video_name}: output already exists -> {out_path}")
            outputs.append(str(out_path))
            continue

        # open video and extract frames (per-video memory)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_name}; skipping.")
            continue

        print(f"Processing video {idx}: {video_name}")
        frames = extract_frames(cap)

        tensor = frames_to_face_tensors(
            frames, size=size, detection_size=detection_size, margin=margin
        )

        if tensor.numel() == 0:
            print(f"No faces found for {video_name}; skipping save.")
            continue

        # ensure output directory exists and save
        save_tensor(tensor, str(out_path))
        outputs.append(str(out_path))
        print(f"Saved {tensor.shape} for {video_name} -> {out_path}")

    return outputs


if __name__ == "__main__":
    process_and_save_all(
        data_dir,
        label_file,
        out_dir=Path("data/processed_faces"),
        size=(224, 224),
        detection_size=640,
        margin=0.2,
    )
