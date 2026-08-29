"""Tests for Megatron backend correctness fixes.

Tests that require megatron-core (GPU dependency) are skipped when it is not
installed.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _fft_dispatch_cfg(weight_sync_backend: str = "nccl") -> SimpleNamespace:
    """Build the minimal ``self.cfg`` view that ``save_weights_for_sampler``
    inspects on the non-colocated path. Defaults to FFT (lora.rank=0) so
    the pause/resume branch is taken.

    ``weight_sync_backend`` defaults to ``"nccl"`` so the caller-pauses branch is
    exercised; pass ``"delta"`` for the branch where the sender pauses internally.
    """
    return SimpleNamespace(
        trainer=SimpleNamespace(
            strategy="fsdp",
            policy=SimpleNamespace(
                model=SimpleNamespace(lora=SimpleNamespace(rank=0)),
                megatron_config=SimpleNamespace(lora_config=SimpleNamespace(merge_lora=False)),
            ),
        ),
        generator=SimpleNamespace(
            inference_engine=SimpleNamespace(offload_kv_for_weight_sync=False, weight_sync_backend=weight_sync_backend)
        ),
    )


_has_megatron = "megatron" in sys.modules or __import__("importlib").util.find_spec("megatron") is not None


# ---------------------------------------------------------------------------
# C1: grad_scale_func fix
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_megatron, reason="megatron-core not installed")
class TestGradScaleFunc:
    """Verify MegatronModelWrapper sets grad_scale_func when optimizer is provided."""

    def test_grad_scale_func_set_with_optimizer(self):
        """When optimizer is provided, grad_scale_func should be set."""
        from skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper import (
            MegatronModelWrapper,
        )

        mock_module = MagicMock()
        mock_config_obj = MagicMock()
        mock_config_obj.finalize_model_grads_func = None
        mock_config_obj.grad_scale_func = None

        mock_optimizer = MagicMock()
        mock_optimizer.scale_loss = MagicMock(return_value=1.0)

        with patch(
            "skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper.get_model_config",
            return_value=mock_config_obj,
        ):
            mock_skyrl_config = MagicMock()
            mock_skyrl_config.trainer.remove_microbatch_padding = False

            MegatronModelWrapper(
                config=mock_skyrl_config,
                actor_module=[mock_module],
                actor_optimizer=mock_optimizer,
            )

        assert mock_config_obj.grad_scale_func is mock_optimizer.scale_loss

    def test_grad_scale_func_not_set_without_optimizer(self):
        """When optimizer is None (ref model), grad_scale_func stays None."""
        from skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper import (
            MegatronModelWrapper,
        )

        mock_module = MagicMock()
        mock_config_obj = MagicMock()
        mock_config_obj.finalize_model_grads_func = None
        mock_config_obj.grad_scale_func = None

        with patch(
            "skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper.get_model_config",
            return_value=mock_config_obj,
        ):
            mock_skyrl_config = MagicMock()
            mock_skyrl_config.trainer.remove_microbatch_padding = False

            MegatronModelWrapper(
                config=mock_skyrl_config,
                actor_module=[mock_module],
                actor_optimizer=None,
            )

        assert mock_config_obj.grad_scale_func is None


@pytest.mark.skipif(not _has_megatron, reason="megatron-core not installed")
class TestDeferredGradSync:
    """Verify the deferred finalizer dispatches only missing overlap handles."""

    @staticmethod
    def _bucket(*, overlap=True, finished=False, handle=None):
        bucket = MagicMock()
        bucket.ddp_config = SimpleNamespace(overlap_grad_reduce=overlap)
        bucket.grad_reduce_finished = finished
        bucket.grad_reduce_handle = handle
        return bucket

    def test_dispatches_missing_regular_and_expert_buckets_before_finalize(self):
        from skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper import (
            MegatronModelWrapper,
        )

        missing = self._bucket()
        expert_missing = self._bucket()
        already_dispatched = self._bucket(handle=object())
        already_finished = self._bucket(finished=True)
        non_overlap = self._bucket(overlap=False)
        model_chunk = SimpleNamespace(
            bucket_groups=[missing, already_dispatched, already_finished, non_overlap],
            expert_parallel_bucket_groups=[expert_missing],
        )
        wrapper = MegatronModelWrapper.__new__(MegatronModelWrapper)
        wrapper.actor_module = [model_chunk]
        wrapper._pending_grad_sync = {"num_tokens": 123}

        with patch(
            "skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper.finalize_model_grads"
        ) as finalize:
            wrapper.run_pending_grad_sync()

        missing.start_grad_sync.assert_called_once_with()
        expert_missing.start_grad_sync.assert_called_once_with()
        already_dispatched.start_grad_sync.assert_not_called()
        already_finished.start_grad_sync.assert_not_called()
        non_overlap.start_grad_sync.assert_not_called()
        finalize.assert_called_once_with([model_chunk], 123)
        assert wrapper._pending_grad_sync is None

    def test_rank_without_pending_schedule_still_dispatches_and_finalizes(self):
        from skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper import (
            MegatronModelWrapper,
        )

        missing = self._bucket()
        model_chunk = SimpleNamespace(bucket_groups=[missing], expert_parallel_bucket_groups=[])
        wrapper = MegatronModelWrapper.__new__(MegatronModelWrapper)
        wrapper.actor_module = [model_chunk]
        wrapper._pending_grad_sync = None

        with patch(
            "skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper.finalize_model_grads"
        ) as finalize:
            wrapper.run_pending_grad_sync()

        missing.start_grad_sync.assert_called_once_with()
        finalize.assert_called_once_with([model_chunk], None)


# ---------------------------------------------------------------------------
# C4: Seed variation by PP rank
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_megatron, reason="megatron-core not installed")
class TestSeedVariation:
    """Verify set_seed varies the seed by PP rank."""

    @pytest.mark.parametrize(
        "pp_rank, expected_seed",
        [
            (0, 42),  # PP=1: seed unchanged
            (1, 142),  # 42 + 100*1
            (3, 342),  # 42 + 100*3
        ],
    )
    def test_seed_offset_by_pp_rank(self, pp_rank, expected_seed):
        from skyrl.backends.skyrl_train.distributed.megatron.megatron_strategy import (
            MegatronStrategy,
        )
        from skyrl.train.config.config import MegatronConfig

        strategy = MegatronStrategy(megatron_config=MegatronConfig(), seed=42)

        with patch("skyrl.backends.skyrl_train.distributed.megatron.megatron_strategy.mpu") as mock_mpu:
            mock_mpu.get_pipeline_model_parallel_rank.return_value = pp_rank
            captured = []
            with patch("random.seed", side_effect=lambda s: captured.append(s)):
                strategy.set_seed(42)
            assert captured[0] == expected_seed
