import json

from lrmc.data.big2015 import Big2015Loader
from lrmc.data.mock import generate_mock_big2015


def test_big2015_loader_builds_cache_from_mock(tmp_path):
    root = tmp_path / "mock_big2015"
    report = generate_mock_big2015(root, n_per_family=4, n_families=3, seed=0)
    cache_dir = tmp_path / "cache"
    loader = Big2015Loader(root, cache_dir, intermediate_size=32, n_workers=1)
    result = loader.build_cache()
    assert result.n_labeled == report["total_files"]
    assert result.n_families == 3


def test_big2015_loader_flags_mismatch_against_real_counts(tmp_path):
    root = tmp_path / "mock_big2015"
    generate_mock_big2015(root, n_per_family=4, n_families=3, seed=0)
    cache_dir = tmp_path / "cache"
    loader = Big2015Loader(root, cache_dir, intermediate_size=32)
    result = loader.build_cache()
    assert any("10868" in m for m in result.mismatches)
    assert any("9" in m for m in result.mismatches)


def test_big2015_get_array_shape(tmp_path):
    root = tmp_path / "mock_big2015"
    generate_mock_big2015(root, n_per_family=3, n_families=2, seed=1)
    cache_dir = tmp_path / "cache"
    loader = Big2015Loader(root, cache_dir, intermediate_size=48)
    loader.build_cache()
    sample_id = loader.records[0].sample_id
    arr, is_image = loader.get_array(sample_id)
    assert arr.shape == (48, 48)
    assert is_image is True


def test_big2015_cache_is_resumable_after_interruption(tmp_path):
    root = tmp_path / "mock_big2015"
    generate_mock_big2015(root, n_per_family=5, n_families=3, seed=2)
    cache_dir = tmp_path / "cache"

    # First (uninterrupted) full build, to know the ground-truth manifest.
    full_loader = Big2015Loader(root, cache_dir, intermediate_size=32)
    full_result = full_loader.build_cache()
    full_manifest_path = cache_dir / "big2015_manifest.json"
    with open(full_manifest_path) as fh:
        full_manifest = json.load(fh)

    # Simulate an interruption: truncate the manifest to pretend only half
    # the samples were cached before the process died (memmap rows for the
    # "lost" entries are still on disk from the full build, but resumption
    # must not rely on that -- it should reprocess anything not yet in the
    # manifest and overwrite those rows).
    cache_dir2 = tmp_path / "cache2"
    cache_dir2.mkdir()
    import shutil

    shutil.copy(cache_dir / "big2015_cache.dat", cache_dir2 / "big2015_cache.dat")
    half = len(full_manifest) // 2
    with open(cache_dir2 / "big2015_manifest.json", "w") as fh:
        json.dump(full_manifest[:half], fh)

    resumed_loader = Big2015Loader(root, cache_dir2, intermediate_size=32)
    resumed_result = resumed_loader.build_cache(resume=True)

    assert resumed_result.n_labeled == full_result.n_labeled
    resumed_ids = {e["sample_id"] for e in json.load(open(cache_dir2 / "big2015_manifest.json"))}
    full_ids = {e["sample_id"] for e in full_manifest}
    assert resumed_ids == full_ids


def test_big2015_handles_unknown_byte_tokens_end_to_end(tmp_path):
    root = tmp_path / "mock_big2015"
    generate_mock_big2015(root, n_per_family=3, n_families=2, seed=3, unknown_byte_frac=0.2)
    cache_dir = tmp_path / "cache"
    loader = Big2015Loader(root, cache_dir, intermediate_size=32)
    loader.build_cache()
    with open(cache_dir / "big2015_manifest.json") as fh:
        manifest = json.load(fh)
    assert any(e["unknown_byte_frac"] > 0 for e in manifest)
