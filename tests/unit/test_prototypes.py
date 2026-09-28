import torch
import torch.nn.functional as F

from lrmc.models.prototypes import ClassPrototypeEstimator


def test_lazy_init_from_first_batch():
    proto = ClassPrototypeEstimator(num_classes=3, embed_dim=4, mode="ema", momentum=0.9)
    z = F.normalize(torch.randn(5, 4), dim=1)
    y = torch.tensor([0, 0, 1, 1, 1])
    proto.update(z, y)
    assert bool(proto.initialized[0]) and bool(proto.initialized[1])
    assert not bool(proto.initialized[2])
    expected_0 = F.normalize(z[y == 0].mean(dim=0), dim=0)
    assert torch.allclose(proto.prototypes[0], expected_0, atol=1e-6)


def test_ema_update_matches_formula():
    proto = ClassPrototypeEstimator(num_classes=2, embed_dim=4, mode="ema", momentum=0.8)
    z1 = F.normalize(torch.randn(4, 4), dim=1)
    y1 = torch.tensor([0, 0, 0, 0])
    proto.update(z1, y1)
    p_before = proto.prototypes[0].clone()

    z2 = F.normalize(torch.randn(4, 4), dim=1)
    y2 = torch.tensor([0, 0, 0, 0])
    proto.update(z2, y2)
    batch_mean_2 = F.normalize(z2.mean(dim=0), dim=0)
    expected = F.normalize(0.8 * p_before + 0.2 * batch_mean_2, dim=0)
    assert torch.allclose(proto.prototypes[0], expected, atol=1e-6)


def test_missing_class_in_batch_is_untouched():
    proto = ClassPrototypeEstimator(num_classes=3, embed_dim=4, mode="ema", momentum=0.9)
    z1 = F.normalize(torch.randn(6, 4), dim=1)
    y1 = torch.tensor([0, 1, 2, 0, 1, 2])
    proto.update(z1, y1)
    snapshot = proto.prototypes.clone()

    # batch 2 has NO samples of class 2
    z2 = F.normalize(torch.randn(4, 4), dim=1)
    y2 = torch.tensor([0, 0, 1, 1])
    proto.update(z2, y2)

    assert torch.allclose(proto.prototypes[2], snapshot[2])  # untouched
    assert not torch.allclose(proto.prototypes[0], snapshot[0])  # updated


def test_batch_mean_mode_no_memory_across_batches():
    proto = ClassPrototypeEstimator(num_classes=1, embed_dim=4, mode="batch_mean")
    z1 = F.normalize(torch.randn(4, 4), dim=1)
    y1 = torch.zeros(4, dtype=torch.long)
    proto.update(z1, y1)
    assert torch.allclose(proto.prototypes[0], F.normalize(z1.mean(dim=0), dim=0), atol=1e-6)

    z2 = F.normalize(torch.randn(4, 4), dim=1)
    y2 = torch.zeros(4, dtype=torch.long)
    proto.update(z2, y2)
    # batch_mean has no memory: prototype should equal batch 2's mean exactly,
    # NOT a blend with batch 1.
    assert torch.allclose(proto.prototypes[0], F.normalize(z2.mean(dim=0), dim=0), atol=1e-6)


def test_negative_labels_are_ignored():
    proto = ClassPrototypeEstimator(num_classes=2, embed_dim=4, mode="ema")
    z = F.normalize(torch.randn(4, 4), dim=1)
    y = torch.tensor([-1, 0, -1, 0])
    proto.update(z, y)
    assert bool(proto.initialized[0])
    assert not bool(proto.initialized[1])


def test_prototypes_are_no_grad_buffer_not_parameter():
    proto = ClassPrototypeEstimator(num_classes=2, embed_dim=4, mode="ema")
    assert not any(p is proto.prototypes for p in proto.parameters())
    assert "prototypes" in dict(proto.named_buffers())
