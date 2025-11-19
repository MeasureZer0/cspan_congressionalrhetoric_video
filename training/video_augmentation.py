"""
Data augmentation transforms for faces and optical flow tensors.

This module provides custom transforms that can be applied to both face images
and optical flow data while maintaining consistency between them.
"""

import random
from typing import Tuple

import torch
import torchvision.transforms as T


class VideoAugmentation:
    """
    Applies consistent augmentations to both face images and optical flow tensors.

    This class ensures that the same random transformations are applied to both
    the face images and optical flow data to maintain spatial consistency.
    """

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        probability: float = 0.5,
    ) -> None:
        """
        Args:
            brightness: How much to jitter brightness. brightness_factor \
                is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            contrast: How much to jitter contrast. contrast_factor \
                is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            saturation: How much to jitter saturation. saturation_factor \
                is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            hue: How much to jitter hue. hue_factor is chosen uniformly from [-hue, hue]
            crop_size: Size of the random crop (height, width). If None, \
                no cropping is applied
            scale_range: Range of scale factors for resizing before cropping
            probability: Probability of applying augmentations
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.probability = probability

        # Color jitter transform (only for face images, not optical flow)
        self.color_jitter = T.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

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

        # Apply transforms to each frame
        augmented_faces = []
        augmented_flows = []

        for i in range(seq_len):
            face_frame = faces[i]  # Shape: (C, H, W)
            flow_frame = flows[i]  # Shape: (C, H, W)

            # Apply color jitter only to face images (not to optical flow)
            face_frame = self.color_jitter(face_frame)

            augmented_faces.append(face_frame)
            augmented_flows.append(flow_frame)

        # Stack back into tensors
        augmented_faces = torch.stack(augmented_faces)
        augmented_flows = torch.stack(augmented_flows)

        return augmented_faces, augmented_flows
