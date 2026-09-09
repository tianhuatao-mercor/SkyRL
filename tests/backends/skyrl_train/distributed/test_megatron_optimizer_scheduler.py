"""CPU wiring tests for Megatron LR decay styles."""

from unittest.mock import MagicMock, patch

import pytest

from skyrl.backends.skyrl_train.distributed.megatron.optimizer import (
    get_megatron_optimizer_param_scheduler,
)
from skyrl.train.config import OptimizerConfig


@pytest.mark.parametrize(
    ("scheduler", "expected_style"),
    [
        ("constant_with_warmup", "constant"),
        ("linear", "linear"),
        ("cosine", "cosine"),
    ],
)
def test_scheduler_name_maps_to_megatron_decay_style(scheduler, expected_style):
    config = OptimizerConfig(
        lr=5e-7,
        min_lr=5e-8,
        num_warmup_steps=13,
        scheduler=scheduler,
        weight_decay=0.0,
    )
    optimizer = MagicMock()
    with patch(
        "skyrl.backends.skyrl_train.distributed.megatron.optimizer.OptimizerParamScheduler"
    ) as constructor:
        get_megatron_optimizer_param_scheduler(optimizer, config, num_training_steps=447)

    kwargs = constructor.call_args.kwargs
    assert kwargs["init_lr"] == 0.0
    assert kwargs["max_lr"] == 5e-7
    assert kwargs["min_lr"] == 5e-8
    assert kwargs["lr_warmup_steps"] == 13
    assert kwargs["lr_decay_steps"] == 447
    assert kwargs["lr_decay_style"] == expected_style


def test_unknown_megatron_scheduler_rejected():
    config = OptimizerConfig(scheduler="polynomial")
    with pytest.raises(ValueError, match="Megatron scheduler"):
        get_megatron_optimizer_param_scheduler(MagicMock(), config, num_training_steps=10)
