from lrmc.config import Config
from lrmc.data.factory import build_fold, build_loader
from lrmc.data.splits import assert_fold_disjoint


def test_build_loader_and_fold_mock_malimg(tmp_path):
    cfg = Config()
    cfg.data.dataset = "mock_malimg"
    cfg.data.root = str(tmp_path / "mock_malimg")
    cfg.split.protocol = "random_k_unknown"
    cfg.split.n_unknown = 2
    loader = build_loader(cfg)
    fold = build_fold(cfg, loader)
    assert_fold_disjoint(fold, loader.records)
    assert len(fold.unknown_families) == 2


def test_build_loader_and_fold_mock_big2015(tmp_path):
    cfg = Config()
    cfg.data.dataset = "mock_big2015"
    cfg.data.root = str(tmp_path / "mock_big2015")
    cfg.data.cache_dir = str(tmp_path / "cache_big2015")
    cfg.split.protocol = "random_k_unknown"
    cfg.split.n_unknown = 1
    loader = build_loader(cfg)
    fold = build_fold(cfg, loader)
    assert_fold_disjoint(fold, loader.records)


def test_build_fold_leave_one_family_out_respects_fold_index(tmp_path):
    cfg = Config()
    cfg.data.dataset = "mock_malimg"
    cfg.data.root = str(tmp_path / "mock_malimg")
    cfg.split.protocol = "leave_one_family_out"
    cfg.split.fold_index = 2
    loader = build_loader(cfg)
    fold = build_fold(cfg, loader)
    assert len(fold.unknown_families) == 1
