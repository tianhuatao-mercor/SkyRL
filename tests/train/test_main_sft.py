from unittest.mock import MagicMock, patch

import pytest

from skyrl.train.main_sft import sft_entrypoint


def test_sft_entrypoint_always_shuts_down_after_success():
    trainer = MagicMock()
    with patch("skyrl.train.main_sft.SFTTrainer", return_value=trainer):
        sft_entrypoint._function(MagicMock(), MagicMock())

    trainer.setup.assert_called_once_with()
    trainer.train.assert_called_once_with()
    trainer.shutdown.assert_called_once_with()


def test_sft_entrypoint_logs_failure_and_still_shuts_down():
    trainer = MagicMock()
    failure = RuntimeError("training failed")
    trainer.train.side_effect = failure
    trainer.global_step = 7

    with patch("skyrl.train.main_sft.SFTTrainer", return_value=trainer), pytest.raises(RuntimeError, match="training failed"):
        sft_entrypoint._function(MagicMock(), MagicMock())

    trainer.tracker.log_exception.assert_called_once_with(failure, step=7)
    trainer.shutdown.assert_called_once_with()
