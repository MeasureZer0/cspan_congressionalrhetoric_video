"""Video preprocessing — face crops + pose keypoints."""

import concurrent.futures
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .config import FaceDetectionConfig, PreprocessingConfig
from .extract_frames import extract_frames
from .extract_pose import extract_pose_from_frames

# Per-process lazy state (safe with ProcessPoolExecutor)
_face_config: Optional[FaceDetectionConfig] = None
_face_detector: Optional[cv2.FaceDetectorYN] = None


def _get_face_detector() -> cv2.FaceDetectorYN:
    """Return a process-local face detector, creating it on first call."""
    global _face_detector, _face_config
    if _face_detector is None:
        if _face_config is None:
            _face_config = FaceDetectionConfig()
        _face_detector = cv2.FaceDetectorYN.create(
            model=str(_face_config.model_path),
            config="",
            input_size=_face_config.input_size,
            score_threshold=_face_config.score_threshold,
            nms_threshold=_face_config.nms_threshold,
            top_k=_face_config.top_k,
        )
    return _face_detector


def _bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def detect_speakers_face(
    frame: np.ndarray,
    detector: cv2.FaceDetectorYN,
) -> Optional[Tuple[int, int, int, int]]:
    """Return (top, right, bottom, left) for the most central face, or None."""
    rgb = _bgr_to_rgb(frame)
    h, w = rgb.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(rgb)
    if faces is None or len(faces) == 0:
        return None

    locations: List[Tuple[int, int, int, int]] = []
    for face in faces:
        box = np.asarray(face[:4], dtype=np.uint32)
        top = int(box[1])
        left = int(box[0])
        bottom = int(box[1] + box[3])
        right = int(box[0] + box[2])
        locations.append((top, right, bottom, left))

    if not locations:
        return None

    if len(locations) == 1:
        return locations[0]

    img_centre = np.array([h / 2.0, w / 2.0])
    best_idx = min(
        range(len(locations)),
        key=lambda i: np.linalg.norm(
            np.array(
                [
                    (locations[i][0] + locations[i][2]) / 2.0,
                    (locations[i][3] + locations[i][1]) / 2.0,
                ]
            )
            - img_centre
        ),
    )
    return locations[best_idx]


def crop_face(
    frame: np.ndarray,
    location: Optional[Tuple[int, int, int, int]],
    margin: float,
) -> Optional[np.ndarray]:
    """Crop a face from *frame* given (top, right, bottom, left),
    with optional margin."""
    if location is None:
        return None

    top, right, bottom, left = location
    rgb = _bgr_to_rgb(frame)
    h, w = rgb.shape[:2]

    if margin > 0:
        box_h = bottom - top
        box_w = right - left
        if margin < 1.0:
            pad_h = int(round(box_h * margin))
            pad_w = int(round(box_w * margin))
        else:
            pad_h = int(round(margin))
            pad_w = int(round(margin))
        top = max(0, top - pad_h)
        bottom = min(h, bottom + pad_h)
        left = max(0, left - pad_w)
        right = min(w, right + pad_w)

    if bottom <= top or right <= left:
        return None

    return rgb[top:bottom, left:right]


def detect_and_crop_faces(
    frames: List[np.ndarray],
    detector: cv2.FaceDetectorYN,
    size: Tuple[int, int] = (224, 224),
    margin: float = 0.0,
    crop_width_ratio: float = 0.5,
) -> Tuple[List[np.ndarray], List[int]]:
    """Detect, crop, and resize the most central face in each frame.

    Returns the cropped face images and the indices of frames where a face
    was found (so callers can align pose tensors).
    """
    face_crops: List[np.ndarray] = []
    valid_indices: List[int] = []

    for idx, frame in enumerate(frames):
        _, w = frame.shape[:2]
        new_w = int(w * crop_width_ratio)
        start = (w - new_w) // 2
        centre_crop = frame[:, start : start + new_w]

        loc = detect_speakers_face(centre_crop, detector)
        face = crop_face(centre_crop, loc, margin=margin)
        if face is None:
            continue

        face_crops.append(cv2.resize(face, size[::-1]))  # cv2 expects (W, H)
        valid_indices.append(idx)

    return face_crops, valid_indices


def faces_to_tensor(faces: List[np.ndarray]) -> torch.Tensor:
    """Convert a list of RGB uint8 images to a float32 tensor [N, 3, H, W]."""
    if not faces:
        return torch.empty((0, 3, 0, 0), dtype=torch.float32)
    tensors = [
        torch.from_numpy(face.transpose(2, 0, 1)).float().div_(255.0) for face in faces
    ]
    return torch.stack(tensors)


def _save_tensor(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)


def process_single_video(
    video_path: str,
    out_path_base: Path,
    frame_skip: int,
    size: Tuple[int, int],
    margin: float,
    crop_width_ratio: float,
    purge: bool,
) -> Optional[str]:
    """Extract faces + pose from one video and save tensors.

    Returns the base output path string on success, or None on failure.
    This function is designed to run inside a ProcessPoolExecutor worker.
    """
    out_faces = Path(f"{out_path_base}_faces.pt")
    out_pose = Path(f"{out_path_base}_pose.pt")

    if not purge and out_faces.exists() and out_pose.exists():
        return str(out_path_base)

    frames = extract_frames(path=video_path, frame_skip=frame_skip)
    if not frames:
        print(f"[skip] No frames extracted: {video_path}")
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
        print(f"[skip] No faces detected: {video_path}")
        return None

    valid_frames = [frames[i] for i in valid_indices]
    pose_tensor = extract_pose_from_frames(
        valid_frames, crop_width_ratio=crop_width_ratio
    )
    faces_tensor = faces_to_tensor(face_crops)

    _save_tensor(faces_tensor, out_faces)
    _save_tensor(pose_tensor, out_pose)
    print(f"[ok] faces={faces_tensor.shape} pose={pose_tensor.shape} → {out_path_base}")

    return str(out_path_base)


def process_videos_in_parallel(config: PreprocessingConfig) -> List[str]:
    """Process all videos listed in config.label_file in parallel."""
    df = pd.read_csv(config.label_file)

    jobs = [
        (
            str(config.data_dir / str(row.iloc[0])),
            Path(config.out_dir) / Path(str(row.iloc[0])).stem,  # type: ignore[arg-type]
            config.frame_skip,
            config.size,
            config.margin,
            config.crop_width_ratio,
            config.purge,
        )
        for _, row in df.iterrows()
    ]

    max_workers = min(config.max_workers or os.cpu_count() or 4, len(jobs))
    print(f"Processing {len(jobs)} videos with {max_workers} workers…")

    outputs: List[str] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_video, *job): job for job in jobs}
        with tqdm(total=len(futures), desc="Preprocessing") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    job = futures[future]
                    print(f"[error] {job[0]}: {exc}")
                    result = None
                if result is not None:
                    outputs.append(result)
                pbar.update(1)

    print(f"\nDone. {len(outputs)}/{len(jobs)} videos processed successfully.")
    return outputs
