"""
Pose keypoint extraction using YOLOv26-Pose.

Produces tensors of shape [T, 17, 3] (x_norm, y_norm, confidence)
aligned with the face frames extracted by crop_faces.py.
"""

from pathlib import Path

import numpy as np
import torch

_pose_model = None


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
YOLO_WEIGHTS = PROJECT_ROOT / "data" / "weights" / "yolo26m-pose.pt"


def _get_pose_model():  # noqa: ANN202
    """Lazily load YOLOv8-nano pose model (one instance per process)."""
    global _pose_model
    if _pose_model is None:
        from ultralytics import YOLO

        _pose_model = YOLO(str(YOLO_WEIGHTS))
    return _pose_model


def _extract_best_person(results, frame_h: int, frame_w: int) -> torch.Tensor:  # noqa: ANN001
    """
    From a YOLO result pick the person whose bounding-box centre is closest
    to the horizontal centre of the frame.
    """
    kp_out = torch.zeros(17, 3)

    if not results or results[0].keypoints is None:
        return kp_out

    kps = results[0].keypoints
    if kps.xy.shape[0] == 0:
        return kp_out

    # Choose the person closest to the frame centre
    best_idx = 0
    if kps.xy.shape[0] > 1:
        frame_cx = frame_w / 2.0
        frame_cy = frame_h / 2.0
        boxes = results[0].boxes.xywh.cpu().numpy()
        dists = np.linalg.norm(boxes[:, :2] - np.array([frame_cx, frame_cy]), axis=1)
        best_idx = int(np.argmin(dists))

    xy = kps.xy[best_idx].cpu().numpy()
    conf = kps.conf[best_idx].cpu().numpy()

    # Normalise to [0, 1]
    xy_norm = xy / np.array([frame_w, frame_h], dtype=np.float32)
    xy_norm = np.clip(xy_norm, 0.0, 1.0)

    kp_out[:, :2] = torch.from_numpy(xy_norm)
    kp_out[:, 2] = torch.from_numpy(conf.astype(np.float32))
    return kp_out


def extract_pose_from_frames(
    frames: list,
    crop_width_ratio: float = 0.5,
    conf_threshold: float = 0.25,
) -> torch.Tensor:
    """
    Run YOLOv26-Pose on every frame and return a keypoint tensor.
    """
    if not frames:
        return torch.zeros((0, 17, 3))

    model = _get_pose_model()
    all_kp: list[torch.Tensor] = []

    for frame in frames:
        h, w = frame.shape[:2]
        new_w = int(w * crop_width_ratio)
        start = (w - new_w) // 2
        frame_crop = frame[:, start : start + new_w, :]
        ch, cw = frame_crop.shape[:2]

        results = model(
            frame_crop,
            verbose=False,
            conf=conf_threshold,
        )
        kp = _extract_best_person(results, ch, cw)
        all_kp.append(kp)

    return torch.stack(all_kp)  # [T, 17, 3]
