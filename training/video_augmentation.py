"""
Data augmentation transforms for faces and optical flow tensors.

DEPRECATED: This module is deprecated in favor of preprocessing-time augmentation.
Augmentation should now be applied during preprocessing using the
`preprocessing/frame_augmentation.py` module. This ensures that optical flow
is computed on augmented frames, providing more realistic flow data.

This module is kept for backward compatibility with SubsetDataMultiplier,
but new code should use preprocessing-time augmentation instead.

This module provides custom transforms that can be applied to both face images
and optical flow data while maintaining consistency between them.
"""

import random
from typing import Tuple

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class VideoAugmentation:
    """
    Applies consistent augmentations to both face images and optical flow tensors.

    This class ensures that the same random transformations are applied to both
    the face images and optical flow data to maintain spatial consistency.
    """

    def __init__(
        self,
        rotation_degrees: float = 10.0,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        probability: float = 0.5,
    ) -> None:
        """
        Args:
            rotation_degrees: Maximum degrees for random rotation
            brightness: How much to jitter brightness. brightness_factor \
                is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            contrast: How much to jitter contrast. contrast_factor \
                is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            saturation: How much to jitter saturation. saturation_factor \
                is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            hue: How much to jitter hue. hue_factor is chosen uniformly from [-hue, hue]
            scale_range: Range of scale factors for resizing before cropping
            probability: Probability of applying augmentations
        """
        self.rotation_degrees = rotation_degrees
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.probability = probability

        # Color jitter transform (only for face images, not optical flow)
        # is defined in __call__ to ensure same params are used across frames

    def __call__(
        self, faces: torch.Tensor, flows: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentations to both faces and flows tensors.

        Args:
            faces: Face images tensor of shape (seq_len, C, H, W)
            flows: Optical flow tensor of shape (seq_len, C, H, W)

        Returns:
            Tuple of augmented (faces, flows) tensors
        """
        # Skip augmentation with some probability
        if random.random() > self.probability:
            return faces, flows

        seq_len, _, h, w = faces.shape

        angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)

        # Apply transforms to each frame
        augmented_faces = []
        augmented_flows = []

        jitter_fn = T.ColorJitter(
            self.brightness, self.contrast, self.saturation, self.hue
        )

        for i in range(seq_len):
            face_frame = faces[i]  # Shape: (C, H, W)
            flow_frame = flows[i]  # Shape: (C, H, W)

            face_frame = TF.rotate(
                face_frame, angle, interpolation=T.InterpolationMode.BILINEAR
            )
            flow_frame = TF.rotate(
                flow_frame, angle, interpolation=T.InterpolationMode.BILINEAR
            )

            # Apply color jitter only to face images (not to optical flow)
            face_frame = jitter_fn(face_frame)

            augmented_faces.append(face_frame)
            augmented_flows.append(flow_frame)

        # Stack back into tensors
        augmented_faces = torch.stack(augmented_faces)
        augmented_flows = torch.stack(augmented_flows)

        return augmented_faces, augmented_flows
