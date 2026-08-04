from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.entrypoints.main_tinker import (
    FullyAsyncTinkerExp,
    experiment_class,
)


def test_direct_entrypoint_selects_sync_and_async_schedulers() -> None:
    cfg = SkyRLTrainConfig()
    assert experiment_class(cfg) is BasePPOExp

    cfg.trainer.fully_async.enabled = True
    assert experiment_class(cfg) is FullyAsyncTinkerExp
