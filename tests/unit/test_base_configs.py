import pytest

from lrmc.config import load_config

CONFIG_FILES = [
    "configs/base_malimg.yaml",
    "configs/base_big2015.yaml",
    "configs/rehearsal_mock_malimg.yaml",
    "configs/rehearsal_mock_big2015.yaml",
]


@pytest.mark.parametrize("path", CONFIG_FILES)
def test_config_file_loads(path):
    cfg = load_config(path)
    assert cfg.run_name


def test_rehearsal_configs_are_marked_synthetic():
    cfg = load_config("configs/rehearsal_mock_malimg.yaml")
    assert cfg.is_synthetic is True
    cfg2 = load_config("configs/rehearsal_mock_big2015.yaml")
    assert cfg2.is_synthetic is True


def test_base_configs_are_not_marked_synthetic():
    cfg = load_config("configs/base_malimg.yaml")
    assert cfg.is_synthetic is False
