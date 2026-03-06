import random

import torch

# left/right pairs for horizontal flip
FLIP_PAIRS: list[tuple[int, int]] = [
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
]


class PoseSimCLRTransform:
    """
    Augmentation pipeline for pose landmark sequences.

    Applied independently to generate the two SimCLR views.

    Parameters
    ----------
    noise_std : float
        Gaussian noise std added to landmark coordinates.
    dropout_p : float
        Probability of zeroing each keypoint.
    scale_range : (float, float)
        Uniform scale range applied around the frame centre.
    flip_p : float
        Probability of a left-right flip.
    temporal_crop : float
        Minimum fraction of the sequence to keep.
    conf_threshold : float
        Keypoints below this confidence are treated as invisible.
    """

    def __init__(
        self,
        noise_std: float = 0.01,
        dropout_p: float = 0.15,
        scale_range: tuple[float, float] = (0.9, 1.1),
        flip_p: float = 0.5,
        temporal_crop: float = 0.8,
        conf_threshold: float = 0.3,
    ) -> None:
        self.noise_std = noise_std
        self.dropout_p = dropout_p
        self.scale_range = scale_range
        self.flip_p = flip_p
        self.temporal_crop = temporal_crop
        self.conf_threshold = conf_threshold

    def __call__(self, pose: torch.Tensor) -> torch.Tensor:
        pose = pose.clone()
        T = pose.shape[0]

        # Temporal crop
        min_t = max(1, int(T * self.temporal_crop))
        start = random.randint(0, max(0, T - min_t))
        pose = pose[start : start + min_t]

        # Horizontal flip
        if random.random() < self.flip_p:
            pose[:, :, 0] = 1.0 - pose[:, :, 0]  # flip x-coordinates
            # Swap left/right pairs
            for l_idx, r_idx in FLIP_PAIRS:
                pose[:, l_idx, :], pose[:, r_idx, :] = (
                    pose[:, r_idx, :].clone(),
                    pose[:, l_idx, :].clone(),
                )

        # Scale
        scale = random.uniform(*self.scale_range)
        pose[:, :, :2] = (pose[:, :, :2] - 0.5) * scale + 0.5
        pose[:, :, :2].clamp_(0.0, 1.0)

        # Gaussian noise
        visible = pose[:, :, 2] > self.conf_threshold  # [T, 17]
        noise = torch.randn_like(pose[:, :, :2]) * self.noise_std
        pose[:, :, :2] += noise * visible.unsqueeze(-1).float()
        pose[:, :, :2].clamp_(0.0, 1.0)

        # Random keypoint dropout
        drop = torch.rand(pose.shape[0], pose.shape[1]) < self.dropout_p
        pose[drop] = 0.0

        return pose
