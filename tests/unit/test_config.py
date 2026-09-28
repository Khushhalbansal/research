import copy

from lrmc.config import Config, config_hash, load_config, merge_overrides, save_config


def test_default_config_builds():
    cfg = Config()
    assert cfg.backbone.name == "tiny_test"
    assert cfg.loss.alpha == 1.0


def test_config_hash_deterministic_and_sensitive():
    cfg1 = Config()
    cfg2 = copy.deepcopy(cfg1)
    assert config_hash(cfg1) == config_hash(cfg2)

    cfg3 = merge_overrides(cfg1, {"optim.lr": 0.5})
    assert config_hash(cfg1) != config_hash(cfg3)


def test_merge_overrides_nested():
    cfg = Config()
    cfg2 = merge_overrides(cfg, {"loss.gamma": 0.2, "backbone.depth": 4})
    assert cfg2.loss.gamma == 0.2
    assert cfg2.backbone.depth == 4
    # original untouched
    assert cfg.loss.gamma == 0.1
    assert cfg.backbone.depth == 2


def test_save_and_load_roundtrip(tmp_path):
    cfg = Config(run_name="roundtrip_test")
    cfg.loss.gamma = 0.33
    path = tmp_path / "cfg.yaml"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.run_name == "roundtrip_test"
    assert loaded.loss.gamma == 0.33
    assert config_hash(cfg) == config_hash(loaded)


def test_unknown_key_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("not_a_real_field: 1\n")
    try:
        load_config(path)
        raised = False
    except ValueError:
        raised = True
    assert raised
