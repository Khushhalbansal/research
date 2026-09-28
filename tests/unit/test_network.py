import torch

from lrmc.config import BackboneConfig, ProjectionHeadConfig
from lrmc.models.network import FeatureExtractionNetwork


def test_tiny_test_backbone_forward_shape_and_unit_norm():
    backbone_cfg = BackboneConfig(
        name="tiny_test",
        img_size=32,
        patch_size=8,
        depth=2,
        embed_dim=16,
        num_heads=2,
        mlp_ratio=2.0,
        in_chans_mode="replicate",
    )
    head_cfg = ProjectionHeadConfig(hidden_dim=32, out_dim=8)
    net = FeatureExtractionNetwork(backbone_cfg, head_cfg)
    net.eval()
    x = torch.rand(4, 1, 32, 32)
    z = net(x)
    assert z.shape == (4, 8)
    norms = z.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-4)


def test_adapt_patch_embed_mode_uses_single_channel():
    backbone_cfg = BackboneConfig(
        name="tiny_test",
        img_size=32,
        patch_size=8,
        depth=2,
        embed_dim=16,
        num_heads=2,
        mlp_ratio=2.0,
        in_chans_mode="adapt_patch_embed",
    )
    head_cfg = ProjectionHeadConfig(hidden_dim=32, out_dim=8)
    net = FeatureExtractionNetwork(backbone_cfg, head_cfg)
    assert net.backbone.in_chans == 1
    net.eval()
    x = torch.rand(2, 1, 32, 32)
    z = net(x)
    assert z.shape == (2, 8)


def test_pretrained_source_is_random_init_for_tiny_test():
    backbone_cfg = BackboneConfig(
        name="tiny_test", img_size=32, patch_size=8, depth=2, embed_dim=16
    )
    head_cfg = ProjectionHeadConfig(hidden_dim=16, out_dim=8)
    net = FeatureExtractionNetwork(backbone_cfg, head_cfg)
    assert net.pretrained_source == "random_init"


def test_resnet18_backbone_offline_falls_back_to_random_init():
    backbone_cfg = BackboneConfig(name="resnet18", pretrained=True, in_chans_mode="replicate")
    head_cfg = ProjectionHeadConfig(hidden_dim=32, out_dim=8)
    net = FeatureExtractionNetwork(backbone_cfg, head_cfg)
    # No local_weights_path configured and (in this sandboxed test env) no
    # network access -> must fall back to random init rather than crash.
    assert net.pretrained_source in ("random_init", "timm_pretrained")
    net.eval()
    x = torch.rand(2, 1, 64, 64)
    z = net(x)
    assert z.shape == (2, 8)
