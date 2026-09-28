from pathlib import Path

from lrmc.config import config_hash, load_config

ABLATION_DIR = Path("configs/ablation")


def test_ablation_configs_exist():
    assert ABLATION_DIR.exists()
    files = list(ABLATION_DIR.glob("*.yaml"))
    assert len(files) >= 20


def test_all_ablation_configs_load_and_are_distinct():
    files = sorted(ABLATION_DIR.glob("*.yaml"))
    hashes = set()
    for f in files:
        cfg = load_config(f)
        hashes.add(config_hash(cfg))
    assert len(hashes) == len(files), "two ablation configs resolved to the identical config"


def test_no_l_in_actually_zeroes_alpha():
    cfg = load_config(ABLATION_DIR / "no_l_in.yaml")
    assert cfg.loss.alpha == 0.0
    assert cfg.loss.beta != 0.0


def test_gamma_sweep_values_present():
    values = set()
    for f in ABLATION_DIR.glob("gamma_*.yaml"):
        cfg = load_config(f)
        values.add(cfg.loss.gamma)
    assert {0.05, 0.2, 0.3, 0.5}.issubset(values)


def test_split_naive_holdout_differs_from_base_group_aware():
    base = load_config("configs/base_malimg.yaml")
    naive = load_config(ABLATION_DIR / "split_naive_holdout.yaml")
    assert base.split.group_aware is True
    assert naive.split.group_aware is False
