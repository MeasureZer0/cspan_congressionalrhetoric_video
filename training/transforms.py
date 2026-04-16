"""Augmentation transforms for video sequences (SimCLR)."""

import random

import torch
import torchvision.transforms.functional as F
from torchvision.transforms import RandomResizedCrop


class VideoSimCLRTransform:
    """SimCLR augmentation pipeline for video tensors [T, C, H, W]."""

    def __init__(
        self,
        size: int = 224,
        jitter_params: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 0.2),
        gray_p: float = 0.2,
        blur_kernel: int | None = None,
    ) -> None:
        self.size = size
        self.jitter_params = jitter_params
        self.gray_p = gray_p
        self.blur_kernel = blur_kernel or max(3, int(size * 0.1) | 1)

    def _sample_jitter_params(
        self, brightness: float, contrast: float, saturation: float, hue: float
    ) -> tuple[float, float, float, float]:
        b = random.uniform(max(0, 1 - brightness), 1 + brightness)
        c = random.uniform(max(0, 1 - contrast), 1 + contrast)
        s = random.uniform(max(0, 1 - saturation), 1 + saturation)
        h = random.uniform(-hue, hue)
        return b, c, s, h

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        i, j, h, w = RandomResizedCrop(
            size=self.size, ratio=(3.0 / 4.0, 4.0 / 3.0)
        ).get_params(video[0], scale=[0.08, 1.0], ratio=[3.0 / 4.0, 4.0 / 3.0])
        video = F.resized_crop(video, i, j, h, w, [self.size, self.size])

        if random.random() < 0.5:
            video = F.hflip(video)

        b, c, s, h = self._sample_jitter_params(*self.jitter_params)
        video = F.adjust_brightness(video, b)
        video = F.adjust_contrast(video, c)
        video = F.adjust_saturation(video, s)
        video = F.adjust_hue(video, h)

        if random.random() < self.gray_p:
            video = F.rgb_to_grayscale(video, num_output_channels=3)

        if random.random() < 0.5:
            video = F.gaussian_blur(
                video, kernel_size=[self.blur_kernel, self.blur_kernel]
            )

        return video
