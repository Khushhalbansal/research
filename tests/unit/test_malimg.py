from lrmc.data.malimg import MalimgLoader
from lrmc.data.mock import generate_mock_malimg


def test_malimg_loader_scans_mock_dataset(tmp_path):
    root = tmp_path / "mock_malimg"
    report = generate_mock_malimg(root, n_per_family=5, n_duplicates_per_family=2, seed=0)
    loader = MalimgLoader(root)
    assert loader.report.n_images == report["total_images"]
    assert loader.report.n_families == len(report["families"])


def test_malimg_loader_reports_mismatch_against_real_counts(tmp_path):
    root = tmp_path / "mock_malimg"
    generate_mock_malimg(root, n_per_family=5, seed=0)
    loader = MalimgLoader(root)
    # mock dataset is intentionally much smaller than real Malimg (25 fam / 9339 img)
    assert any("9339" in m for m in loader.report.mismatches)


def test_malimg_loader_detects_duplicates(tmp_path):
    root = tmp_path / "mock_malimg"
    generate_mock_malimg(root, n_per_family=5, n_duplicates_per_family=3, seed=1)
    loader = MalimgLoader(root)
    assert loader.report.n_duplicate_files > 0
    assert loader.report.n_duplicate_hashes > 0


def test_malimg_get_array_returns_2d_image(tmp_path):
    root = tmp_path / "mock_malimg"
    generate_mock_malimg(root, n_per_family=3, image_size=40, seed=2)
    loader = MalimgLoader(root)
    sample_id = loader.records[0].sample_id
    arr, is_image = loader.get_array(sample_id)
    assert is_image is True
    assert arr.ndim == 2
