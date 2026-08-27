from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import prepare_runtime_environment


def test_qualification_runtime_environment_is_forwarded(monkeypatch):
    values = {
        "SKYRL_INFERENCE_BIND_HOST": "127.0.0.1",
        "SKYRL_INFERENCE_ADVERTISE_HOST": "127.0.0.1",
        "SKYRL_WEIGHT_SYNC_MASTER_ADDR": "127.0.0.1",
        "SKYRL_QUAL_RESULT_DIR": "/tmp/qualification-results",
        "SKYRL_QUAL_REPEAT_EVAL_STEP": "2",
        "SKYRL_QUAL_ROLLOUT_RESOURCE": "b300_lifecycle_rollout",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr("skyrl.train.utils.utils.peer_access_supported", lambda **kwargs: True)

    env_vars = prepare_runtime_environment(SkyRLTrainConfig())

    for name, value in values.items():
        assert env_vars[name] == value
