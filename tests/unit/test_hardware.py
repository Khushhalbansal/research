import torch

from lrmc.utils.hardware import (
    auto_num_workers,
    free_disk_bytes,
    gpu_status_list,
    pick_least_loaded_gpu,
    resolve_device,
    system_summary,
)


def test_resolve_device_auto_without_cuda_returns_cpu():
    if torch.cuda.is_available():
        return  # this environment actually has a GPU; skip the no-GPU assertion
    assert resolve_device("auto") == torch.device("cpu")
    assert resolve_device("cuda") == torch.device("cpu")


def test_resolve_device_explicit_cpu():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_cuda_falls_back_when_unavailable():
    if torch.cuda.is_available():
        return
    assert resolve_device("cuda:0") == torch.device("cpu")


def test_auto_num_workers_is_non_negative_and_bounded():
    n = auto_num_workers(cap=8)
    assert 0 <= n <= 8


def test_free_disk_bytes_positive():
    assert free_disk_bytes(".") > 0


def test_gpu_status_list_returns_list_even_without_nvidia_smi():
    result = gpu_status_list()
    assert isinstance(result, list)


def test_pick_least_loaded_gpu_none_when_no_multi_gpu():
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return
    assert pick_least_loaded_gpu() is None or isinstance(pick_least_loaded_gpu(), int)


def test_system_summary_has_expected_keys():
    summary = system_summary()
    for key in (
        "python_version",
        "os",
        "cpu_count",
        "torch_version",
        "cuda_available",
        "gpu_count",
        "gpus",
        "free_disk_gb",
    ):
        assert key in summary
