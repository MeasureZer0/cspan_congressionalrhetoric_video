"""Video Preprocessing Module — faces + pose keypoints."""

import concurrent.futures
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from config import FaceDetectionConfig, PreprocessingConfig
from extract_frames import extract_frames
from extract_pose import extract_pose_from_frames
from tqdm import tqdm

face_config = None
_face_detector = None


def _get_face_detector() -> cv2.FaceDetectorYN:
    """Get or create face detector for current process (lazy, per-worker)."""
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
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def detect_speakers_face(
    frame: np.ndarray, detector: cv2.FaceDetectorYN
) -> Optional[Tuple[int, int, int, int]]:
    """Detect the most central face in the given frame using YuNet."""
    rgb = _rgb(frame)
    h, w = rgb.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(rgb)
    if faces is None or len(faces) == 0:
        return None

    locations = []
    for face in faces:
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
        img_center = np.array([h / 2, w / 2])
        min_dist = float("inf")
        for i, (t, r, b, left) in enumerate(locations):
            face_center = np.array([(t + b) / 2, (left + r) / 2])
            dist = np.linalg.norm(face_center - img_center)
            if dist < min_dist:
                min_dist = dist
                best_idx = i

    return locations[best_idx]


def crop_face(
    frame: np.ndarray,
    location: Optional[Tuple[int, int, int, int]],
    margin: Optional[float],
) -> Optional[np.ndarray]:
    """Crop a face from frame given a (top, right, bottom, left) bounding box."""
    if location is None:
        return None
    top, right, bottom, left = location
    rgb = _rgb(frame)
    h, w = rgb.shape[:2]

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
) -> Tuple[List[np.ndarray], List[int]]:
    """Detect, crop and resize the most central face in each frame."""
    if not frames:
        return [], []

    face_crops: List[np.ndarray] = []
    valid_indices: List[int] = []

    for idx, f in enumerate(frames):
        _, w = f.shape[:2]
        new_w = int(w * crop_width_ratio)
        start = (w - new_w) // 2
        f_cropped = f[:, start : start + new_w, :]

        loc = detect_speakers_face(f_cropped, detector=detector)
        face = crop_face(f_cropped, loc, margin=margin)
        if face is None:
            continue

        face_crops.append(cv2.resize(face, size))
        valid_indices.append(idx)

    return face_crops, valid_indices


def faces_to_tensor(faces: List[np.ndarray]) -> torch.Tensor:
    """Convert a list of RGB numpy face images to a float tensor (N, 3, H, W)."""
    if not faces:
        return torch.empty((0, 3, 0, 0), dtype=torch.float32)
    tensors = [
        torch.from_numpy(face.transpose(2, 0, 1)).float() / 255.0 for face in faces
    ]
    return torch.stack(tensors)


def save_tensor(tensor: torch.Tensor, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, str(out_path))


def process_single_video(
    video_path: str,
    out_path_base: Path,
    frame_skip: int,
    size: tuple,
    margin: float,
    crop_width_ratio: float,
    purge: bool,
) -> Optional[str]:
    """
    Worker function: extract faces + pose keypoints from one video, save tensors.
    Returns the base output path on success, None on failure.
    """
    out_faces = Path(f"{out_path_base}_faces.pt")
    out_pose = Path(f"{out_path_base}_pose.pt")

    if not purge and out_faces.exists() and out_pose.exists():
        print(f"Skipping {video_path}: outputs already exist.")
        return str(out_path_base)

    frames = extract_frames(path=video_path, frame_skip=frame_skip)
    if not frames:
        print(f"No frames extracted from {video_path}; skipping.")
        return None

    detector = _get_face_detector()
    face_crops, valid_indices = detect_and_crop_faces(
        frames,
        detector=detector,
        size=size,
        margin=margin,
        crop_width_ratio=crop_width_ratio,
    )

    if not face_crops:
        print(f"No faces found in {video_path}; skipping.")
        return None

    valid_frames = [frames[i] for i in valid_indices]
    pose_tensor = extract_pose_from_frames(
        valid_frames, crop_width_ratio=crop_width_ratio
    )
    faces_tensor = faces_to_tensor(face_crops)

    save_tensor(faces_tensor, out_faces)
    save_tensor(pose_tensor, out_pose)
    print(
        f"Saved {faces_tensor.shape} faces, {pose_tensor.shape} pose -> {out_path_base}"
    )

    return str(out_path_base)


def process_videos_in_parallel(config: PreprocessingConfig) -> list[str]:
    """Process all videos in parallel, one worker per video."""
    df = pd.read_csv(config.label_file)

    jobs = []
    for _, row in df.iterrows():
        video_name = row.iloc[0]
        video_path = os.path.join(config.data_dir, video_name)
        stem = Path(video_name).stem
        out_path_base = Path(config.out_dir) / str(stem)
        jobs.append(
            (
                video_path,
                out_path_base,
                config.frame_skip,
                config.size,
                config.margin,
                config.crop_width_ratio,
                config.purge,
            )
        )

    max_workers = min(config.max_workers or os.cpu_count() or 4, len(jobs))
    print(f"Processing {len(jobs)} videos with {max_workers} workers...")

    outputs: list[str] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_video, *job) for job in jobs]
        with tqdm(total=len(futures), desc="Processing videos") as pbar:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    outputs.append(result)
                pbar.update(1)

    return outputs
