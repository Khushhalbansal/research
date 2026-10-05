import numpy as np
from PIL import Image

from lrmc.data.malimg import MalimgLoader


def _png(path, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.default_rng(seed).integers(0, 255, (16, 16), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_presplit_train_val_test_layout_is_pooled(tmp_path):
    n = 0
    for split in ("train", "val", "test"):
        for fam in ("FamA", "FamB"):
            for i in range(3):
                n += 1
                _png(tmp_path / "malimg_dataset" / split / fam / f"img{i}.png", n)

    loader = MalimgLoader(tmp_path)

    assert loader.report.n_images == 18
    assert loader.report.n_families == 2
    assert loader.report.family_counts == {"FamA": 9, "FamB": 9}
    assert len(loader.records) == 18
    assert len({r.sample_id for r in loader.records}) == 18
    for r in loader.records:
        loader.get_array(r.sample_id)
