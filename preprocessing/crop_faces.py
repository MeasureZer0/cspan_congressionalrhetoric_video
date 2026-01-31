"""Video Preprocessing Module"""

import concurrent.futures
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2  # OpenCV for video processing
import numpy as np  # NumPy for storing frames as arrays
import pandas as pd  # Pandas for reading CSV labels
import torch
from config import FaceDetectionConfig, PreprocessingConfig
from extract_frames import extract_frames
from frame_augmentation import FrameAugmentation
from tqdm import tqdm

# Load face detection config
face_config = None

# Process-local face detector (initialized lazily in each worker)
_face_detector = None


def _get_face_detector() -> cv2.FaceDetectorYN:
    """Get or create face detector for current process.

    This lazy initialization ensures each worker process gets its own
    detector instance, avoiding pickling issues with ProcessPoolExecutor.
    """

    global _face_detector
    if _face_detector is None:
        global face_config
        if face_config is None:
            face_config = FaceDetectionConfig()

        _face_detector = cv2.FaceDetectorYN.create(
            model=str(face_config.model_path),
            config="",
            input_size=face_config.input_size,
            score_threshold=face_config.score_threshold,
            nms_threshold=face_config.nms_threshold,
            top_k=face_config.top_k,
        )
    return _face_detector


def _rgb(frame: np.ndarray) -> np.ndarray:
    """
    Convert OpenCV BGR frame to RGB for face_recognition / PIL.
    Args:
        frame (np.ndarray): Input frame in BGR format.
    Returns:
        np.ndarray: Frame in RGB format.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def detect_speakers_face(
    frame: np.ndarray, detector: cv2.FaceDetectorYN
) -> Optional[Tuple[int, int, int, int]]:
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
    # Get detector for current process

    # Convert from BGR to RGB
    rgb = _rgb(frame)
    h, w = rgb.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(rgb)
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


def detect_and_crop_faces(
    frames: List[np.ndarray],
    detector: cv2.FaceDetectorYN,
    size: tuple = (224, 224),
    margin: Optional[float] = 0.0,
    crop_width_ratio: float = 0.5,
) -> List[np.ndarray]:
    """
    Detect the face closest to horizontal center in each frame, crop, and resize.

    Args:
        frames (List[np.ndarray]): List of BGR frames (OpenCV).
        detector (cv2.FaceDetectorYN): Face detector instance.
        size (tuple): Target size (H, W) for resizing.
        margin (float or None): Margin to add around detected face box.
        crop_width_ratio (float): Ratio of width to crop from center.

    Returns:
        List[np.ndarray]: List of cropped and resized face images (RGB).
    """
    if not frames:
        return []

    face_crops = []

    for f in frames:
        _, w = f.shape[:2]
        new_w = int(w * crop_width_ratio)
        start = (w - new_w) // 2
        f_cropped = f[:, start : start + new_w, :]

        loc = detect_speakers_face(f_cropped, detector=detector)
        face = crop_face(f_cropped, loc, margin=margin)
        if face is None:
            # skip frames without a detected face
            continue

        # Resize the face
        face_resized = cv2.resize(face, size)
        face_crops.append(face_resized)

    return face_crops


def convert_faces_to_tensors(
    faces: List[np.ndarray],
    augmenter: Optional[FrameAugmentation] = None,
) -> List[torch.Tensor]:
    """
    Convert list of numpy face images to tensors, optionally applying augmentation.

    Args:
        faces (List[np.ndarray]): List of RGB face images (numpy).
        augmenter (Optional[FrameAugmentation]): Augmenter to apply.

    Returns:
        List[torch.Tensor]: List of tensors (C, H, W).
    """
    if not faces:
        return []

    # Initialize augmentation parameters once for the entire sequence
    if augmenter is not None:
        augmenter.initialize_sequence_params()

    face_tensors = []
    for face in faces:
        # Apply augmentation if provided
        if augmenter is not None:
            face = augmenter.augment_numpy_frame(face)

        # Convert to tensor
        face_chw = torch.from_numpy(face.transpose(2, 0, 1)).float() / 255.0
        face_tensors.append(face_chw)

    return face_tensors


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
    crop_width_ratio: float,
    purge: bool,
    augmenter: Optional[FrameAugmentation] = None,
) -> Optional[str]:
    """
    Worker function: process one video, extract faces, save tensor(s).

    If augmenter is provided, saves both original (base path) and
    augmented (base + '_aug') versions.

    Returns base output path if successful, None otherwise.
    """
    out_path_orig = Path(f"{out_path_base}_faces.pt")
    out_path_aug = Path(f"{out_path_base}_aug_faces.pt") if augmenter else None

    # Check existence
    orig_exists = out_path_orig.exists()
    aug_exists = (
        out_path_aug.exists() if out_path_aug else True
    )  # If no aug needed, "exists" is True

    if not purge and orig_exists and aug_exists:
        print(f"Skipping {video_path}: outputs already exist.")
        return str(out_path_base)

    # If we need to process, we do it once
    frames = extract_frames(path=video_path, frame_skip=frame_skip)
    detector = _get_face_detector()

    # 1. Detect and crop
    face_crops = detect_and_crop_faces(
        frames,
        detector=detector,
        size=size,
        margin=margin,
        crop_width_ratio=crop_width_ratio,
    )

    if not face_crops:
        print(f"No faces found for {video_path}; skipping save.")
        return None

    # 2. Save Original if needed
    if purge or not orig_exists:
        tensors_orig = convert_faces_to_tensors(face_crops, augmenter=None)
        faces_tensor_orig = stack_tensors(tensors_orig, size=size)
        save_tensor(faces_tensor_orig, str(out_path_orig))
        print(f"Saved {faces_tensor_orig.shape} -> {out_path_orig}")

    # 3. Save Augmented if needed
    if augmenter and (purge or not aug_exists):
        tensors_aug = convert_faces_to_tensors(face_crops, augmenter=augmenter)
        faces_tensor_aug = stack_tensors(tensors_aug, size=size)
        save_tensor(faces_tensor_aug, str(out_path_aug))
        print(f"Saved {faces_tensor_aug.shape} (Augmented) -> {out_path_aug}")

    return str(out_path_base)


def process_videos_in_parallel(
    config: PreprocessingConfig,
) -> list[str]:
    """
    Process all videos in parallel, each worker processes one video.
    Shows a progress bar that updates as each worker finishes.
    """
    random.seed(2025)  # For reproducibility

    df = pd.read_csv(config.label_file)

    # Create augmenter if augmentation is enabled
    augmenter = None
    if config.augmentation.enabled:
        augmenter = FrameAugmentation(config=config.augmentation)
        print("Augmentation enabled - will create both original and augmented versions")

    jobs = []
    for _, row in df.iterrows():
        video_name = row.iloc[0]
        video_path = os.path.join(config.data_dir, video_name)
        stem = Path(video_name).stem
        out_path_base = Path(config.out_dir) / str(stem)

        # Single job per video handles both original and augmented
        jobs.append(
            (
                video_path,
                out_path_base,
                config.frame_skip,
                config.size,
                config.margin,
                config.crop_width_ratio,
                config.purge,
                augmenter,
            )
        )

    # Use as many workers as CPU cores if possible, but not more than jobs
    if config.max_workers is None:
        max_workers = min(os.cpu_count() or 4, len(jobs))
    else:
        max_workers = min(config.max_workers, len(jobs))

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
