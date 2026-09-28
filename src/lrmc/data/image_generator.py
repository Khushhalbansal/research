"""Image Generator (diagram box: "Image Generator").

Turns a raw byte stream (from :class:`lrmc.data.preprocessor.FilePreprocessor`)
into a fixed-size grayscale image, using the Nataraj et al. (2011) file-size-to-width
table, then produces byte-image-safe two-view augmentations for SupCon training.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from lrmc.config import AugmentationConfig

# Nataraj et al. (2011) file-size -> image-width lookup table.
_NATARAJ_TABLE: list[tuple[int, int]] = [
    (10 * 1024, 32),
    (30 * 1024, 64),
    (60 * 1024, 128),
    (100 * 1024, 256),
    (200 * 1024, 384),
    (500 * 1024, 512),
    (1000 * 1024, 768),
]
_NATARAJ_DEFAULT_WIDTH = 1024


def nataraj_width(file_size_bytes: int) -> int:
    """Width in pixels for a file of the given size, per the Nataraj table."""
    for threshold, width in _NATARAJ_TABLE:
        if file_size_bytes < threshold:
            return width
    return _NATARAJ_DEFAULT_WIDTH


class ImageGenerator:
    """Byte stream -> fixed-size grayscale image, plus two-view augmentation."""

    def __init__(self, image_size: int = 64, augmentation: AugmentationConfig | None = None):
        self.image_size = image_size
        self.augmentation = augmentation or AugmentationConfig()

    def bytes_to_image(self, byte_stream: np.ndarray) -> np.ndarray:
        """Render a raw byte stream into a 2D uint8 array via the Nataraj table."""
        n = byte_stream.size
        if n == 0:
            return np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        width = nataraj_width(n)
        height = math.ceil(n / width)
        padded = np.zeros(width * height, dtype=np.uint8)
        padded[:n] = byte_stream
        grid = padded.reshape(height, width)
        return grid

    def to_model_input(self, grid: np.ndarray) -> torch.Tensor:
        """Resize an arbitrary-size 2D array to (1, image_size, image_size) in [0, 1]."""
        t = torch.from_numpy(grid.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(
            t, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )
        return resized.squeeze(0)  # (1, H, W)

    def image_array_to_model_input(self, arr: np.ndarray) -> torch.Tensor:
        """For pre-rendered images (e.g. Malimg), resize directly (no byte reshape)."""
        return self.to_model_input(arr)

    # ---- augmentation --------------------------------------------------

    def _truncate_or_pad_bytes(self, byte_stream: np.ndarray) -> np.ndarray:
        cfg = self.augmentation
        if np.random.rand() > cfg.byte_truncate_pad_prob or byte_stream.size == 0:
            return byte_stream
        frac = np.random.uniform(0, cfg.byte_truncate_pad_frac)
        n = byte_stream.size
        if np.random.rand() < 0.5:
            # truncate from the tail
            keep = max(1, int(n * (1 - frac)))
            return byte_stream[:keep]
        else:
            # pad with zeros at the tail
            pad_len = int(n * frac)
            return np.concatenate([byte_stream, np.zeros(pad_len, dtype=byte_stream.dtype)])

    def _pixel_augment(self, image: torch.Tensor) -> torch.Tensor:
        """Domain-appropriate augmentations. Deliberately NO horizontal flip:
        byte layout is directional (header -> code -> data), so mirroring an
        image destroys structure that is meaningful to the model."""
        cfg = self.augmentation
        c, h, w = image.shape

        # Random resized crop (mild scale), no flip, aspect ratio preserved.
        scale_lo, scale_hi = cfg.random_resized_crop_scale
        scale = np.random.uniform(scale_lo, scale_hi)
        crop_h = max(1, int(h * math.sqrt(scale)))
        crop_w = max(1, int(w * math.sqrt(scale)))
        top = np.random.randint(0, max(1, h - crop_h + 1))
        left = np.random.randint(0, max(1, w - crop_w + 1))
        cropped = image[:, top : top + crop_h, left : left + crop_w]
        image = F.interpolate(
            cropped.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
        ).squeeze(0)

        # Small translation via padding + crop.
        max_t = int(cfg.max_translate_frac * h)
        if max_t > 0:
            dy = np.random.randint(-max_t, max_t + 1)
            dx = np.random.randint(-max_t, max_t + 1)
            image = torch.roll(image, shifts=(dy, dx), dims=(1, 2))

        # Gaussian noise.
        if cfg.gaussian_noise_std > 0:
            image = image + torch.randn_like(image) * cfg.gaussian_noise_std
            image = image.clamp(0.0, 1.0)

        # Random erasing (pixel dropout patch).
        if np.random.rand() < cfg.random_erasing_prob:
            scale_lo_e, scale_hi_e = cfg.random_erasing_scale
            erase_area = np.random.uniform(scale_lo_e, scale_hi_e) * h * w
            aspect = np.random.uniform(0.3, 3.3)
            eh = min(h, max(1, int(math.sqrt(erase_area * aspect))))
            ew = min(w, max(1, int(math.sqrt(erase_area / aspect))))
            ey = np.random.randint(0, max(1, h - eh + 1))
            ex = np.random.randint(0, max(1, w - ew + 1))
            image[:, ey : ey + eh, ex : ex + ew] = 0.0

        return image

    def generate_view(self, byte_stream: np.ndarray, is_pre_rendered_image: bool) -> torch.Tensor:
        """Produce one augmented view starting from raw bytes or a pre-rendered image."""
        if is_pre_rendered_image:
            assert byte_stream.ndim == 2, "pre-rendered images must keep their 2D shape"
            grid = byte_stream
        else:
            stream = self._truncate_or_pad_bytes(byte_stream)
            grid = self.bytes_to_image(stream)
        image = self.to_model_input(grid)
        return self._pixel_augment(image)

    def two_views(
        self, byte_stream: np.ndarray, is_pre_rendered_image: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Two independent augmented views of the same sample, for SupCon."""
        v1 = self.generate_view(byte_stream, is_pre_rendered_image)
        v2 = self.generate_view(byte_stream, is_pre_rendered_image)
        return v1, v2

    def eval_view(self, byte_stream: np.ndarray, is_pre_rendered_image: bool) -> torch.Tensor:
        """Deterministic (no augmentation) view for validation/test/inference."""
        if is_pre_rendered_image:
            assert byte_stream.ndim == 2, "pre-rendered images must keep their 2D shape"
            grid = byte_stream
        else:
            grid = self.bytes_to_image(byte_stream)
        return self.to_model_input(grid)
