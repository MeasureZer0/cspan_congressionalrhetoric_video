"""
Data augmentation transforms for faces and optical flow tensors.

This module provides custom transforms that can be applied to both face images
and optical flow data while maintaining consistency between them.
"""

import random
from typing import Optional, Tuple

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
        rotation_degrees: float = 15.0,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        crop_size: Optional[Tuple[int, int]] = None,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        probability: float = 0.5,
    ) -> None:
        """
        Args:
            rotation_degrees: Range of degrees for random rotation \
                (-rotation_degrees, +rotation_degrees)
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
        self.rotation_degrees = rotation_degrees
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.crop_size = crop_size
        self.scale_range = scale_range
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

        seq_len, c_faces, h, w = faces.shape
        _, c_flows, _, _ = flows.shape

        # Generate random parameters for consistent application
        angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        scale = random.uniform(*self.scale_range)

        # For random crop, we need to determine
        # crop parameters if crop_size is specified
        crop_params = None
        if self.crop_size is not None:
            crop_h, crop_w = self.crop_size
            # Scale the image first
            new_h, new_w = int(h * scale), int(w * scale)

            # Make sure crop size is not larger than scaled image
            crop_h = min(crop_h, new_h)
            crop_w = min(crop_w, new_w)

            # Random crop position
            top = random.randint(0, max(0, new_h - crop_h))
            left = random.randint(0, max(0, new_w - crop_w))
            crop_params = (top, left, crop_h, crop_w, new_h, new_w)

        # Apply transforms to each frame
        augmented_faces = []
        augmented_flows = []

        for i in range(seq_len):
            face_frame = faces[i]  # Shape: (C, H, W)
            flow_frame = flows[i]  # Shape: (C, H, W)

            # Apply rotation to both faces and flows
            face_frame = TF.rotate(
                face_frame, angle, interpolation=TF.InterpolationMode.BILINEAR
            )
            flow_frame = TF.rotate(
                flow_frame, angle, interpolation=TF.InterpolationMode.BILINEAR
            )

            # Apply scaling and cropping if specified
            if crop_params is not None:
                top, left, crop_h, crop_w, new_h, new_w = crop_params

                # Resize first
                face_frame = TF.resize(
                    face_frame,
                    (new_h, new_w),
                    interpolation=TF.InterpolationMode.BILINEAR,
                )
                flow_frame = TF.resize(
                    flow_frame,
                    (new_h, new_w),
                    interpolation=TF.InterpolationMode.BILINEAR,
                )

                # Then crop
                face_frame = TF.crop(face_frame, top, left, crop_h, crop_w)
                flow_frame = TF.crop(flow_frame, top, left, crop_h, crop_w)

                # Resize back to original size
                face_frame = TF.resize(
                    face_frame, [h, w], interpolation=TF.InterpolationMode.BILINEAR
                )
                flow_frame = TF.resize(
                    flow_frame, [h, w], interpolation=TF.InterpolationMode.BILINEAR
                )

            # Apply color jitter only to face images (not to optical flow)
            face_frame = self.color_jitter(face_frame)

            augmented_faces.append(face_frame)
            augmented_flows.append(flow_frame)

        # Stack back into tensors
        augmented_faces = torch.stack(augmented_faces)
        augmented_flows = torch.stack(augmented_flows)

        return augmented_faces, augmented_flows


class SimpleVideoAugmentation:
    """
    Simplified augmentation that applies transforms independently to each frame.
    Use this for more randomness or if the consistent approach is too restrictive.
    """

    def __init__(
        self,
        rotation_degrees: float = 10.0,
        brightness: float = 0.1,
        contrast: float = 0.1,
        saturation: float = 0.1,
        hue: float = 0.05,
        probability: float = 0.3,
    ) -> None:
        self.rotation_degrees = rotation_degrees
        self.probability = probability

        self.color_jitter = T.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def __call__(
        self, faces: torch.Tensor, flows: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentations independently to each frame."""
        if random.random() > self.probability:
            return faces, flows

        seq_len = faces.shape[0]
        augmented_faces = []
        augmented_flows = []

        for i in range(seq_len):
            face_frame = faces[i]
            flow_frame = flows[i]

            # Random rotation for this frame
            if random.random() < 0.5:  # 50% chance of rotation per frame
                angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
                face_frame = TF.rotate(
                    face_frame, angle, interpolation=TF.InterpolationMode.BILINEAR
                )
                flow_frame = TF.rotate(
                    flow_frame, angle, interpolation=TF.InterpolationMode.BILINEAR
                )

            # Color jitter only for faces
            if random.random() < 0.7:  # 70% chance of color jitter per frame
                face_frame = self.color_jitter(face_frame)

            augmented_faces.append(face_frame)
            augmented_flows.append(flow_frame)

        return torch.stack(augmented_faces), torch.stack(augmented_flows)


class OpticalFlowAugmentation:
    """
    Special augmentation considerations for optical flow data.

    Note: Optical flow represents motion vectors, so some augmentations
    need special handling. For example, rotation changes the direction
    of flow vectors, and we should be careful with color-based transforms.
    """

    def __init__(
        self, rotation_degrees: float = 10.0, probability: float = 0.3
    ) -> None:
        self.rotation_degrees = rotation_degrees
        self.probability = probability

    def _rotate_flow_vectors(
        self, flow: torch.Tensor, angle_rad: float
    ) -> torch.Tensor:
        """
        Rotate optical flow vectors to account for image rotation.

        Args:
            flow: Optical flow tensor of shape (2, H, W) where first channel \
                is u (horizontal) and second channel is v (vertical)
            angle_rad: Rotation angle in radians

        Returns:
            Rotated flow tensor
        """
        if flow.shape[0] != 2:
            # If not a standard 2-channel flow, just return as-is
            return flow

        cos_a = torch.cos(torch.tensor(angle_rad))
        sin_a = torch.sin(torch.tensor(angle_rad))

        u = flow[0]  # horizontal component
        v = flow[1]  # vertical component

        # Rotate the flow vectors
        u_rot = cos_a * u - sin_a * v
        v_rot = sin_a * u + cos_a * v

        return torch.stack([u_rot, v_rot])

    def __call__(self, flows: torch.Tensor) -> torch.Tensor:
        """Apply optical flow specific augmentations."""
        if random.random() > self.probability:
            return flows

        seq_len, c, h, w = flows.shape
        augmented_flows = []

        for i in range(seq_len):
            flow_frame = flows[i]  # Shape: (C, H, W)

            if random.random() < 0.5:  # 50% chance of rotation
                angle_deg = random.uniform(
                    -self.rotation_degrees, self.rotation_degrees
                )
                angle_rad = torch.deg2rad(torch.tensor(angle_deg))

                # Rotate the image
                flow_frame = TF.rotate(
                    flow_frame, angle_deg, interpolation=TF.InterpolationMode.BILINEAR
                )

                # Rotate the flow vectors if this is a 2-channel flow
                flow_frame = self._rotate_flow_vectors(flow_frame, angle_rad.item())

            augmented_flows.append(flow_frame)

        return torch.stack(augmented_flows)
