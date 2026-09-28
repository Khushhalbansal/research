from lrmc.data.splits import load_family_groups


def test_malimg_groups_config_loads():
    mapping = load_family_groups("configs/data/malimg_groups.yaml")
    assert mapping["Allaple.A"] == "Allaple"
    assert mapping["Allaple.L"] == "Allaple"
    assert mapping["Lolyda.AA1"] == "Lolyda.AA"
    assert mapping["Lolyda.AA3"] == "Lolyda.AA"
    assert "Fakerean" not in mapping  # singleton, not listed
