"""
Frame-level augmentation transforms for preprocessing.

This module provides augmentation transforms that are applied to individual frames
during preprocessing.
"""

import random
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from config import AugmentationConfig


class FrameAugmentation:
    """
    Applies augmentations to individual frames during preprocessing.

    This class ensures consistent augmentation across a video sequence, with
    the same parameters applied to all frames in the sequence.
    """

    def __init__(
        self,
        config: AugmentationConfig,
    ) -> None:
        """
        Args:
            config: AugmentationConfig object with augmentation parameters
        """
        self.rotation_degrees = config.rotation_degrees
        self.brightness = config.brightness
        self.contrast = config.contrast
        self.saturation = config.saturation
        self.hue = config.hue

        # These will be set once per sequence
        self.current_angle: Optional[float] = None
        self.brightness_factor: Optional[float] = None
        self.contrast_factor: Optional[float] = None
        self.saturation_factor: Optional[float] = None
        self.hue_factor: Optional[float] = None

    def initialize_sequence_params(self) -> None:
        """
        Initialize augmentation parameters for a new sequence.
        Should be called once per video sequence before processing frames.

        Always applies augmentation - samples random parameters once for the
        entire sequence.
        """
        # Sample random parameters once for the entire sequence
        self.current_angle = random.uniform(
            -self.rotation_degrees, self.rotation_degrees
        )

        # Sample color jitter parameters that will be applied to all frames
        self.brightness_factor = (
            random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            if self.brightness > 0
            else 1.0
        )
        self.contrast_factor = (
            random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            if self.contrast > 0
            else 1.0
        )
        self.saturation_factor = (
            random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            if self.saturation > 0
            else 1.0
        )
        self.hue_factor = random.uniform(-self.hue, self.hue) if self.hue > 0 else 0.0

    def augment_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations to a single frame using sequence-wide parameters.

        Args:
            frame: Frame tensor of shape (C, H, W), values in [0, 1]

        Returns:
            Augmented frame tensor of shape (C, H, W)
        """
        if self.current_angle is None:
            return frame

        # Apply rotation
        frame = TF.rotate(
            frame, self.current_angle, interpolation=T.InterpolationMode.BILINEAR
        )

        # Apply color jitter using the same parameters for all frames
        if self.brightness_factor is not None and self.brightness_factor != 1.0:
            frame = TF.adjust_brightness(frame, self.brightness_factor)
        if self.contrast_factor is not None and self.contrast_factor != 1.0:
            frame = TF.adjust_contrast(frame, self.contrast_factor)
        if self.saturation_factor is not None and self.saturation_factor != 1.0:
            frame = TF.adjust_saturation(frame, self.saturation_factor)
        if self.hue_factor is not None and self.hue_factor != 0.0:
            frame = TF.adjust_hue(frame, self.hue_factor)

        return frame

    def augment_numpy_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to a numpy frame (H, W, C) in RGB format.

        Args:
            frame: Numpy array of shape (H, W, C) with values in [0, 255]

        Returns:
            Augmented numpy array of shape (H, W, C) with values in [0, 255]
        """
        if self.current_angle is None:
            return frame

        # Convert to tensor
        frame_tensor = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0

        # Apply augmentation
        augmented_tensor = self.augment_frame(frame_tensor)

        # Convert back to numpy
        augmented_array = (augmented_tensor * 255.0).byte().permute(1, 2, 0).numpy()

        return augmented_array
