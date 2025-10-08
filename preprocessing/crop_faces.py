"""Video Preprocessing Module"""

import argparse
import concurrent.futures
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2  # OpenCV for video processing
import numpy as np  # NumPy for storing frames as arrays
import pandas as pd  # Pandas for reading CSV labels
import torch
from extract_frames import extract_frames
from raft_optical_flow import get_optical_flow_between_frames
from torchvision import transforms
from tqdm import tqdm

# Define paths to input data relative to this file
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
data_dir = project_root / "data"
raw_videos_dir = data_dir / "raw_videos"
label_file = data_dir / "labels.csv"
models_dir = data_dir / "weights"

# Assert all required files exist
assert (label_file).exists(), f"Label file not found: {label_file}"
assert raw_videos_dir.exists(), f"Raw videos directory not found: {raw_videos_dir}"

assert (models_dir / "face_detection_yunet_2023mar.onnx").exists(), (
    f"Model file not found: {models_dir / 'face_detection_yunet_2023mar.onnx'}. "
    "Please run scripts/download-weights.py to download the model weights."
)

# Initialise YuNet face detector
face_detector = cv2.FaceDetectorYN.create(
    model=str(models_dir / "face_detection_yunet_2023mar.onnx"),
    config="",
    input_size=(768, 576),
    score_threshold=0.9,
    nms_threshold=0.3,
    top_k=5000,
)


def _rgb(frame: np.ndarray) -> np.ndarray:
    """
    Convert OpenCV BGR frame to RGB for face_recognition / PIL.
    Args:
        frame (np.ndarray): Input frame in BGR format.
    Returns:
        np.ndarray: Frame in RGB format.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def detect_speakers_face(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect the most central face in the given video frame using YuNet.

    Args:
        frame (np.ndarray): Input frame in BGR format.
            for detection (resized for speed). Default = 640.
            If None, detection runs on the original resolution.

    Returns:
        Optional[Tuple[int, int, int, int]]: Bounding box of the detected face in
            (top, right, bottom, left) format.
            Returns None if no face is found.
    """
    # Convert from BGR to RGB
    rgb = _rgb(frame)
    h, w = rgb.shape[:2]
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
        img_center = np.array([h / 2, w / 2])

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

    t, r, b, left = locations[best_idx]
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


def frames_to_faces_and_optical_flows(
    frames: List[np.ndarray],
    size: tuple = (224, 224),
    margin: Optional[float] = 0.0,
    crop_width_ratio: float = 0.5,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Detect the face closest to horizontal center \
    in each frame, crop, resize, and convert to a tensor.
    We also compute optical flow between frames.
    Args:
        frames (List[np.ndarray]): List of BGR frames (OpenCV).
        size (tuple): Target size (H, W) for resizing.
        margin (float or None): Margin to add around detected face box.
    Returns:
        List[torch.Tensor]: Tensor of shape (N, C, H, W) for N frames \
            with detected faces.
        List[torch.Tensor]: List of optical flow tensors between consecutive frames.
    """
    if not frames:
        return [torch.empty((0, 3, size[0], size[1]), dtype=torch.float32)], []

    face_tensors = []
    optical_flows = []

    prev_face = (None)
    
    for f in frames:
        _, w = f.shape[:2]
        new_w = int(w * crop_width_ratio)
        start = (w - new_w) // 2
        f_cropped = f[:, start : start + new_w, :]

        loc = detect_speakers_face(f_cropped)
        face = crop_face(f_cropped, loc, margin=margin)
        if face is None:
            # skip frames without a detected face
            continue

        # Resize the face and convert directly to tensor
        face_resized = cv2.resize(face, size)
        face_chw = torch.from_numpy(face_resized.transpose(2, 0, 1)).float() / 255.0
        face_tensors.append(face_chw)
        # Then calculate optical flow
        optical_flow = get_optical_flow_between_frames(
            prev_face if prev_face is not None else face_chw, face_chw
        )
        prev_face = face_chw

        optical_flows.append(optical_flow)

    return face_tensors, optical_flows


def stack_tensors(
    tensors: List[torch.Tensor], size: tuple = (224, 224)
) -> torch.Tensor:
    """
    Stack a list of tensors into a single tensor along the first dimension.
    Args:
        tensors (List[torch.Tensor]): List of tensors to stack.
    Returns:
        torch.Tensor: Stacked tensor.
    """
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


def process_single_video(
    video_path: str,
    out_path_base: Path,
    frame_skip: int,
    size: tuple,
    margin: float,
    purge: bool,
) -> Optional[str]:
    """
    Worker function: process one video, extract faces, save tensor.
    Returns output path if successful, None otherwise.
    """
    out_path_faces = Path(f"{out_path_base}_faces.pt")
    out_path_flows = Path(f"{out_path_base}_flows.pt")
    if not purge and (out_path_faces.exists() and out_path_flows.exists()):
        print(f"Skipping {video_path}: output already exists -> {out_path_base}")
        return str(out_path_base)

    frames = extract_frames(path=video_path, frame_skip=frame_skip)
    tensors, optical_flows = frames_to_faces_and_optical_flows(
        frames, size=size, margin=margin
    )
    faces_tensor = stack_tensors(tensors, size=size)
    flows_tensor = stack_tensors(optical_flows, size=size)
    if faces_tensor.numel() == 0:
        print(f"No faces found for {video_path}; skipping save.")
        return None

    save_tensor(faces_tensor, str(out_path_faces))
    print(f"Saved {faces_tensor.shape} for {video_path} -> {out_path_faces}")
    if flows_tensor.numel() == 0:
        print(f"No optical flows found for {video_path}; skipping save.")
        return None
    save_tensor(flows_tensor, str(out_path_flows))
    print(f"Saved {flows_tensor.shape} for {video_path} -> {out_path_flows}")
    return str(out_path_base)


def process_videos_in_parallel(
    data_dir: Path,
    label_file: Path,
    out_dir: Path,
    frame_skip: int = 30,
    size: tuple = (224, 224),
    margin: float = 0.0,
    purge: bool = False,
) -> list[str]:
    """
    Process all videos in parallel, each worker processes one video.
    Shows a progress bar that updates as each worker finishes.
    """
    df = pd.read_csv(label_file)
    jobs = []
    for _, row in df.iterrows():
        video_name = row.iloc[0]
        video_path = os.path.join(data_dir, video_name)
        stem = Path(video_name).stem
        out_path_base = Path(out_dir) / str(stem)
        jobs.append((video_path, out_path_base, frame_skip, size, margin, purge))

    # Use as many workers as CPU cores if possible, but not more than jobs
    max_workers = min(os.cpu_count() or 4, len(jobs))
    print(f"Processing {len(jobs)} videos with {max_workers} workers...")
    outputs: List[str] = []
    # Use ProcessPoolExecutor to distribute the jobs to all workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Start all the jobs, executor.submit returns a future
        futures = [executor.submit(process_single_video, *job) for job in jobs]
        with tqdm(total=len(futures), desc="Processing videos") as pbar:
            # The as_completed iterator yields futures as they complete
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    outputs.append(result)
                # Update the progress bar upon each completion
                pbar.update(1)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process videos to extract face tensors."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=raw_videos_dir,
        help="Path to the directory containing video files.",
    )
    parser.add_argument(
        "--label_file",
        type=Path,
        default=label_file,
        help="Path to the CSV file containing video labels.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=data_dir / "faces",
        help="Path to the directory where output tensors will be saved.",
    )
    parser.add_argument(
        "--frame_skip", type=int, default=30, help="Save only every N-th frame."
    )
    parser.add_argument(
        "--size", type=tuple, default=(224, 224), help="Target size for face tensors."
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

    process_videos_in_parallel(
        data_dir=args.data_dir,
        label_file=args.label_file,
        out_dir=args.out_dir,
        frame_skip=args.frame_skip,
        size=args.size,
        margin=args.margin,
        purge=args.purge,
    )
