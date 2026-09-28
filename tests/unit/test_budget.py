from lrmc.config import BackboneConfig, ProjectionHeadConfig
from lrmc.orchestration.budget import estimate_gpu_minutes, measure_cpu_step_seconds


def test_measure_cpu_step_seconds_is_positive():
    backbone_cfg = BackboneConfig(
        name="tiny_test", img_size=32, patch_size=8, depth=2, embed_dim=16, num_heads=2, mlp_ratio=2.0
    )
    head_cfg = ProjectionHeadConfig(hidden_dim=16, out_dim=8)
    sec_per_sample = measure_cpu_step_seconds(backbone_cfg, head_cfg, batch_size=4, n_steps=2)
    assert sec_per_sample > 0


def test_estimate_gpu_minutes_scales_with_epochs_and_samples():
    result_small = estimate_gpu_minutes(
        n_epochs=1, n_train_samples=1000, cpu_sec_per_sample_tiny=0.01, flops_ratio=5.0
    )
    result_large = estimate_gpu_minutes(
        n_epochs=2, n_train_samples=1000, cpu_sec_per_sample_tiny=0.01, flops_ratio=5.0
    )
    assert result_small["estimated"] is True
    assert result_large["estimated_gpu_minutes"] == 2 * result_small["estimated_gpu_minutes"]
    assert "assumptions" in result_small
