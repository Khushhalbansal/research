import torch
import torch.nn as nn

from lrmc.eval.efficiency import approx_flops, count_parameters, efficiency_report


def test_count_parameters_matches_manual_sum():
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    result = count_parameters(model)
    expected = sum(p.numel() for p in model.parameters())
    assert result["total_params"] == expected
    assert result["trainable_params"] == expected


def test_approx_flops_linear_matches_hand_calc():
    model = nn.Linear(10, 20, bias=False)
    x = torch.randn(4, 10)
    flops = approx_flops(model, x)
    assert flops == 2 * 4 * 10 * 20


def test_efficiency_report_has_expected_keys():
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    x = torch.randn(2, 8)
    report = efficiency_report(model, x, torch.device("cpu"))
    for key in (
        "total_params",
        "approx_flops",
        "mean_latency_s_per_sample",
        "throughput_samples_per_s",
        "peak_memory_bytes",
        "device",
    ):
        assert key in report
    assert report["throughput_samples_per_s"] > 0
