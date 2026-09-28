import torch

from lrmc.models.radii import LearnableRadiiParameters


def test_radii_init_matches_r_init():
    radii = LearnableRadiiParameters(num_classes=5, r_init=0.5)
    r = radii.radii
    assert torch.allclose(r, torch.full((5,), 0.5), atol=1e-5)


def test_radii_always_positive_even_with_negative_rho():
    radii = LearnableRadiiParameters(num_classes=3, r_init=0.5)
    with torch.no_grad():
        radii.rho.fill_(-100.0)
    assert (radii.radii > 0).all()
    assert (radii.radii < 1e-3).all()


def test_radii_gather_by_class_id():
    radii = LearnableRadiiParameters(num_classes=4, r_init=0.3)
    ids = torch.tensor([0, 2, 2, 3])
    gathered = radii(ids)
    assert gathered.shape == (4,)
    assert torch.allclose(gathered, radii.radii[ids])
