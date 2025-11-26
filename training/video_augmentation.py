"""
Data augmentation transforms for faces and optical flow tensors.

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
        size: Tuple[int, int] = (224, 224),
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
            scale_range: Range of scale factors for resizing before cropping
            probability: Probability of applying augmentations
        """
        self.size = size
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

        seq_len, _, _, _ = faces.shape

        # Apply transforms to each frame
        augmented_faces = []
        augmented_flows = []

        # Draw jittering for whole video
        jitter_fn = T.ColorJitter(
            self.brightness, self.contrast, self.saturation, self.hue
        )


        # Get jittering params
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
        T.ColorJitter.get_params(jitter_fn.brightness, jitter_fn.contrast, jitter_fn.saturation, jitter_fn.hue)

        do_flip = random.random() < 0.5

        # Draw RandomResizedCrop params
        i, j, h, w = T.RandomResizedCrop.get_params(
            img=faces[0], scale=(0.8, 1.0), ratio=(0.9, 1.1)
        )

        for k in range(seq_len):
            face_frame = faces[k]  # Shape: (C, H, W)
            flow_frame = flows[k]  # Shape: (C, H, W)

            # Crop & Resize
            face_frame = TF.resized_crop(face_frame, i, j, h, w, self.size, antialias=True)
            flow_frame = TF.resized_crop(flow_frame, i, j, h, w, self.size, antialias=True)

            # Horizontal Flip
            if do_flip:
                face_frame = TF.hflip(face_frame)
                flow_frame = TF.hflip(flow_frame)
                
                flow_frame[0, :, :] *= -1

            # Apply color jitter to face images
            for fn_id in fn_idx:
                if fn_id == 0:
                    face_frame = TF.adjust_brightness(face_frame, brightness_factor)
                elif fn_id == 1:
                    face_frame = TF.adjust_contrast(face_frame, contrast_factor)
                elif fn_id == 2:
                    face_frame = TF.adjust_saturation(face_frame, saturation_factor)
                elif fn_id == 3:
                    face_frame = TF.adjust_hue(face_frame, hue_factor)

            augmented_faces.append(face_frame)
            augmented_flows.append(flow_frame)

        return torch.stack(augmented_faces), torch.stack(augmented_flows)