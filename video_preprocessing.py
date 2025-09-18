"""Video Preprocessing Module"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2  # OpenCV for video processing
import numpy as np  # NumPy for storing frames as arrays
import pandas as pd  # Pandas for reading CSV labels
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from preprocessing.extract_frames import extract_frames

# Define paths to input data:
# - data_dir: directory containing videos and labels
# - label_file: CSV file with labels (video filename + label)
data_dir = Path("data/sample_video")
label_file = Path(data_dir / "labels.csv")
models_dir = Path("models")

# Initialise YuNet face detector
face_detector = cv2.FaceDetectorYN.create(
    model=str(models_dir / "face_detection_yunet_2023mar.onnx"),
    config="",
    input_size=(320, 320),
    score_threshold=0.9,
    nms_threshold=0.3,
    top_k=5000,
)


def _rgb(frame: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR frame to RGB for face_recognition / PIL."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def detect_speakers_face(
    frame: np.ndarray, detection_size: Optional[int] = 640
) -> Optional[Tuple[int, int, int, int]]:
    """
    Convert OpenCV BGR frame to RGB for face_recognition / PIL.
    Args:
        frame (np.ndarray): Input frame in BGR format.
    Returns:
        np.ndarray: Frame in RGB format.
    """
    rgb = _rgb(frame)
    h, w = rgb.shape[:2]

    # Optionally resize for faster detection
    scale = 1.0
    if detection_size is not None and max(h, w) > detection_size:
        scale = detection_size / max(h, w)
        small_w = max(1, int(round(w * scale)))
        small_h = max(1, int(round(h * scale)))
        small = cv2.resize(rgb, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        face_detector.setInputSize((small_w, small_h))
        _, faces = face_detector.detect(small)
    else:
        face_detector.setInputSize((w, h))
        _, faces = face_detector.detect(rgb)

    if faces is None or len(faces) == 0:
        return None

    # Convert detected faces to bounding boxes
    locations = []
    for face in faces:
        # Ensure face is an array, not a scalar
        if isinstance(face, (np.ndarray, list)):
            box = np.array(face[0:4]).astype(np.uint32)
            top = int(round(box[1]))
            right = int(round(box[0] + box[2]))
            bottom = int(round(box[1] + box[3]))
            left = int(round(box[0]))
            locations.append((top, right, bottom, left))
    if not locations:
        return None

    best_idx = 0
    if len(locations) > 1:
        # Choose the face closest to the image center
        img_center = np.array(
            [
                h * scale / 2 if scale != 1.0 else h / 2,
                w * scale / 2 if scale != 1.0 else w / 2,
            ]
        )

        min_dist = float("inf")
        best_idx = 0
        for i, loc in enumerate(locations):
            t, r, b, left = loc
            # Compute face center
            face_center = np.array(
                [
                    (t + b) / 2,
                    (left + r) / 2,
                ]
            )
            dist = np.linalg.norm(face_center - img_center)

            if dist < min_dist:
                min_dist = dist
                best_idx = i

    # Map back to original coordinates if resized
    t, r, b, left = locations[best_idx]
    if scale != 1.0:
        inv_scale = 1.0 / scale
        return (
            int(round(t * inv_scale)),
            int(round(r * inv_scale)),
            int(round(b * inv_scale)),
            int(round(left * inv_scale)),
        )
    else:
        return (t, r, b, left)


def crop_face(
    frame: np.ndarray,
    location: Optional[Tuple[int, int, int, int]],
    margin: Optional[float],
) -> Optional[np.ndarray]:
    """
    Crop a face from frame given a (top, right, bottom, left) tuple.
    Optionally expand the bounding box by a margin.
    Args:
        frame (np.ndarray): Input frame in BGR format.
        location (tuple or None): (top, right, bottom, left) bounding box.
        margin (float or None): Margin to add around the box.
    Returns:
        np.ndarray or None: Cropped face in RGB, or None if invalid.
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
    """
    Detect the face closest to horizontal center \
    in each frame, crop, resize, and convert to a tensor.
    Args:
        frames (List[np.ndarray]): List of BGR frames (OpenCV).
        size (tuple): Target size (H, W) for resizing.
        detection_size (int or None): Resize largest side to this before detection.
        margin (float or None): Margin to add around detected face box.
    Returns:
        torch.Tensor: Tensor of shape (N, C, H, W) for N frames with detected faces.
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
        loc = detect_speakers_face(f, detection_size=detection_size)
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
    """
    Save tensor to disk using torch.save. Creates parent dir if needed.
    Args:
        tensor (torch.Tensor): Tensor to save.
        out_path (str): Output file path.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, str(out))


def process_and_save_all(
    data_dir: Path,
    label_file: Path,
    out_dir: Path,
    frame_skip: int = 10,
    size: tuple = (224, 224),
    detection_size: Optional[int] = 640,
    margin: float = 0.0,
    purge: bool = False,
) -> List[str]:
    """
    Load videos, extract faces, convert to tensors and save per-video tensors.
    Args:
        data_dir (Path): Directory containing video files.
        label_file (Path): CSV file with video labels.
        out_dir (Path): Directory to save output tensors.
        frame_skip (int): Save only every N-th frame.
        size (tuple): Target size for face tensors.
        detection_size (int or None): Size for face detection.
        margin (float): Margin to add around detected faces.
        purge (bool): If True, overwrite existing outputs.
    Returns:
        List[str]: List of output file paths (one per video that produced faces).
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

        # skip if not purging and already processed
        if not purge and out_path.exists():
            print(f"Skipping {video_name}: output already exists -> {out_path}")
            outputs.append(str(out_path))
            continue
        print(f"Processing video {idx}: {video_name}")
        frames = extract_frames(path=video_path, frame_skip=frame_skip)
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
    parser = argparse.ArgumentParser(
        description="Process videos to extract face tensors."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/sample_video"),
        help="Path to the directory containing video files.",
    )
    parser.add_argument(
        "--label_file",
        type=Path,
        default=Path("data/sample_video/labels.csv"),
        help="Path to the CSV file containing video labels.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/processed_faces"),
        help="Path to the directory where output tensors will be saved.",
    )
    parser.add_argument(
        "--frame_skip", type=int, default=10, help="Save only every N-th frame."
    )
    parser.add_argument(
        "--size", type=tuple, default=(224, 224), help="Target size for face tensors."
    )
    parser.add_argument(
        "--detection_size", type=int, default=640, help="Size for face detection."
    )
    parser.add_argument(
        "--margin", type=float, default=0.1, help="Margin to add around detected faces."
    )
    parser.add_argument(
        "--purge",
        action="store_true",
        help="Purge existing output files before processing.",
    )

    args = parser.parse_args()

    process_and_save_all(
        data_dir=args.data_dir,
        label_file=args.label_file,
        out_dir=args.out_dir,
        frame_skip=args.frame_skip,
        size=args.size,
        detection_size=args.detection_size,
        margin=args.margin,
        purge=args.purge,
    )
