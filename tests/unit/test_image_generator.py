import numpy as np
import torch

from lrmc.config import AugmentationConfig
from lrmc.data.image_generator import ImageGenerator, nataraj_width


def test_nataraj_width_table_boundaries():
    assert nataraj_width(5 * 1024) == 32
    assert nataraj_width(10 * 1024) == 64  # exactly at threshold -> next bucket
    assert nataraj_width(29 * 1024) == 64
    assert nataraj_width(31 * 1024) == 128
    assert nataraj_width(2_000_000) == 1024


def test_bytes_to_image_shape_matches_width_table():
    gen = ImageGenerator(image_size=64)
    stream = (np.arange(5000) % 256).astype(np.uint8)
    grid = gen.bytes_to_image(stream)
    width = nataraj_width(stream.size)
    assert grid.shape[1] == width
    assert grid.size >= stream.size


def test_bytes_to_image_empty_stream_returns_zero_image():
    gen = ImageGenerator(image_size=32)
    grid = gen.bytes_to_image(np.array([], dtype=np.uint8))
    assert grid.shape == (32, 32)
    assert grid.sum() == 0


def test_to_model_input_range_and_shape():
    gen = ImageGenerator(image_size=48)
    grid = np.full((100, 100), 255, dtype=np.uint8)
    t = gen.to_model_input(grid)
    assert t.shape == (1, 48, 48)
    assert t.max().item() <= 1.0 + 1e-6
    assert t.min().item() >= 0.0 - 1e-6


def test_two_views_are_stochastic_and_correct_shape():
    aug = AugmentationConfig(byte_truncate_pad_prob=1.0)
    gen = ImageGenerator(image_size=32, augmentation=aug)
    stream = np.random.randint(0, 256, size=3000, dtype=np.uint8)
    torch.manual_seed(0)
    np.random.seed(0)
    v1, v2 = gen.two_views(stream)
    assert v1.shape == (1, 32, 32)
    assert v2.shape == (1, 32, 32)
    assert not torch.allclose(v1, v2)


def test_eval_view_deterministic_for_same_input():
    gen = ImageGenerator(image_size=32)
    stream = np.random.randint(0, 256, size=3000, dtype=np.uint8)
    v1 = gen.eval_view(stream, is_pre_rendered_image=False)
    v2 = gen.eval_view(stream, is_pre_rendered_image=False)
    assert torch.allclose(v1, v2)


def test_eval_view_pre_rendered_image_uses_true_shape():
    gen = ImageGenerator(image_size=16)
    grid = np.random.randint(0, 256, size=(40, 30), dtype=np.uint8)
    v = gen.eval_view(grid, is_pre_rendered_image=True)
    assert v.shape == (1, 16, 16)


def test_no_horizontal_flip_in_augmentation_config():
    # structural guarantee: AugmentationConfig has no flip-related field at all,
    # so a flip can never be introduced accidentally via config.
    fields = AugmentationConfig.__dataclass_fields__.keys()
    assert not any("flip" in f.lower() for f in fields)
