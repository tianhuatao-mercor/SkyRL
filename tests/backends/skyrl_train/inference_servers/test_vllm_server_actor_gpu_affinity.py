import os

from skyrl.backends.skyrl_train.inference_servers.vllm_server_actor import VLLMServerActor


def test_uni_backend_receives_reserved_bundle_gpu() -> None:
    kwargs = VLLMServerActor.prepare_server_kwargs(
        pg=None,
        start_bundle_idx=7,
        num_gpus_per_server=1,
        _gpu_ids=[6],
        distributed_executor_backend="uni",
    )

    assert kwargs["direct_cuda_visible_devices"] == "6"


def test_mp_backend_receives_all_reserved_bundle_gpus() -> None:
    kwargs = VLLMServerActor.prepare_server_kwargs(
        pg=None,
        start_bundle_idx=2,
        num_gpus_per_server=2,
        _gpu_ids=[2, 3],
        distributed_executor_backend="mp",
    )

    assert kwargs["direct_cuda_visible_devices"] == "2,3"


def test_ray_backend_does_not_override_worker_visibility() -> None:
    kwargs = VLLMServerActor.prepare_server_kwargs(
        pg=None,
        start_bundle_idx=4,
        num_gpus_per_server=1,
        _gpu_ids=[4],
        distributed_executor_backend="ray",
    )

    assert "direct_cuda_visible_devices" not in kwargs


def test_direct_backend_visibility_is_exported_for_engine_children(monkeypatch) -> None:
    actor = object.__new__(VLLMServerActor)
    actor._server_idx = 3
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "stale")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "stale")

    actor._setup_direct_gpu_visibility("5")

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "5"
    assert "ROCR_VISIBLE_DEVICES" not in os.environ
    assert "HIP_VISIBLE_DEVICES" not in os.environ
