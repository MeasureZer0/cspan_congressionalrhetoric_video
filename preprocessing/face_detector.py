"""
Face detection utilities using facenet-pytorch's MTCNN.

This module exposes a module-level `mtcnn` detector (created on import)
and helper functions to detect the most central face in a single frame or
in a batch of frames. Frames are expected as OpenCV BGR numpy arrays.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import List, Optional, Tuple

import numpy as np
import torch

from facenet_pytorch import MTCNN

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# keep_all=True so detect() returns all boxes per image
# We'll pick the most central one
mtcnn = MTCNN(keep_all=True, post_process=True, device=_device)


def _choose_central_box(
    boxes: np.ndarray, image_shape: tuple
) -> Optional[Tuple[int, int, int, int]]:
    """Choose the box closest to the image center.

    Args:
            boxes: array of shape (N, 4) with (x1, y1, x2, y2)
            image_shape: (h, w, ...)

    Returns:
            (top, right, bottom, left) or None
    """
    if boxes is None or len(boxes) == 0:
        return None
    h, w = image_shape[:2]
    img_center = np.array([w / 2.0, h / 2.0])  # x, y
    best_idx = 0
    min_dist = float("inf")
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        face_center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
        dist = np.linalg.norm(face_center - img_center)
        if dist < min_dist:
            min_dist = dist
            best_idx = i
    x1, y1, x2, y2 = boxes[best_idx]
    left = int(round(x1))
    top = int(round(y1))
    right = int(round(x2))
    bottom = int(round(y2))
    return (top, right, bottom, left)


def detect_speakers_face(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Detect faces in a single BGR OpenCV frame and return the central face box.

    Returns a tuple in (top, right, bottom, left) order to match the rest of
    the preprocessing code in this repository. Returns None if no face found.
    """
    if frame is None:
        return None
    # Convert BGR -> RGB (mtcnn expects RGB)
    rgb = frame[..., ::-1]
    # mtcnn.detect can accept a single image and will return boxes or None
    boxes_result = mtcnn.detect(rgb)

    # mtcnn.detect may return (boxes, probs) or (boxes, probs, landmarks)
    if isinstance(boxes_result, tuple):
        boxes = boxes_result[0]
    else:
        boxes = boxes_result

    # boxes may be None or an array. For a single image it might be wrapped in
    # a list-like where element 0 contains the array. Handle both defensively.
    if boxes is None:
        return None
    if isinstance(boxes, (list, tuple)):
        if len(boxes) == 0 or boxes[0] is None:
            return None
        boxes_arr = np.asarray(boxes[0])
    else:
        boxes_arr = np.asarray(boxes)
    return _choose_central_box(boxes_arr, rgb.shape)


def detect_faces_batch(
    frames: List[np.ndarray],
    batch_size: int = 32,
) -> List[Optional[Tuple[int, int, int, int]]]:
    """Detect faces on a list of BGR frames using batching.

    Args:
            frames: list of OpenCV BGR numpy arrays.
            batch_size: how many images to feed into the detector at once.

    Returns:
            A list with the selected (top, right, bottom, left) box per frame or
            None for frames where no face was detected.
    """
    results: List[Optional[Tuple[int, int, int, int]]] = [None] * len(frames)
    if not frames:
        return results

    # Convert to RGB numpy arrays (mtcnn accepts list of numpy images)
    rgb_frames = [f[..., ::-1] if f is not None else None for f in frames]

    # Process in batches
    for i in range(0, len(rgb_frames), batch_size):
        batch = rgb_frames[i : i + batch_size]
        # Filter out None entries but keep indices
        idx_map = [j for j, img in enumerate(batch) if img is not None]
        imgs_to_pass = [batch[j] for j in idx_map]
        if not imgs_to_pass:
            continue
        boxes_result = mtcnn.detect(imgs_to_pass)
        # mtcnn.detect may return (boxes, probs) or (boxes, probs, landmarks)
        if isinstance(boxes_result, tuple):
            boxes_batch = boxes_result[0]
        else:
            boxes_batch = boxes_result
        # boxes_batch will be a list of arrays (one per image)
        for k, boxset in enumerate(boxes_batch):
            global_idx = i + idx_map[k]
            if boxset is None or len(boxset) == 0:
                results[global_idx] = None
            else:
                results[global_idx] = _choose_central_box(
                    np.asarray(boxset), frames[global_idx].shape
                )

    return results
