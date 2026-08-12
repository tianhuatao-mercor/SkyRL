"""Regression tests for completed-step checkpoint labels.

The fully-async loop uses ``global_step`` as a next-step cursor at epoch
exhaustion. Checkpoints, however, must identify the last optimizer step that is
actually present in the saved model and optimizer state.
"""

import asyncio
import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import skyrl.train.trainer as trainer_module
from skyrl.train.fully_async_trainer import (
    FullyAsyncRayPPOTrainer,
    _needs_final_checkpoint,
)
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils.trainer_utils import ResumeMode


class _RecordingDispatch:
    def __init__(self):
        self.checkpoint_calls = []
        self.hf_calls = []

    def save_checkpoint(self, model, path, tokenizer):
        self.checkpoint_calls.append((model, path, tokenizer))

    def save_hf_model(self, model, path, tokenizer):
        self.hf_calls.append((model, path, tokenizer))


class _StatefulLoader:
    def state_dict(self):
        return {"position": 7}


def _make_base_trainer(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer_module, "asdict", lambda _cfg: {"config": "sentinel"})
    trainer = object.__new__(RayPPOTrainer)
    trainer.cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            ckpt_path=str(tmp_path / "checkpoints"),
            export_path=str(tmp_path / "exports"),
            critic=SimpleNamespace(model=SimpleNamespace(path=None)),
        )
    )
    trainer.global_step = 13
    trainer.dispatch = _RecordingDispatch()
    trainer.tokenizer = object()
    trainer.train_dataloader = _StatefulLoader()
    trainer.all_timings = {}
    trainer._cleanup_old_checkpoints = lambda: None
    return trainer


def test_base_checkpoint_uses_explicit_completed_step_without_moving_cursor(tmp_path, monkeypatch):
    trainer = _make_base_trainer(tmp_path, monkeypatch)

    checkpoint_dir = trainer.save_checkpoints(checkpoint_step=12)

    assert trainer.global_step == 13
    assert checkpoint_dir == str(tmp_path / "checkpoints" / "global_step_12")
    assert trainer.dispatch.checkpoint_calls[0][1] == os.path.join(checkpoint_dir, "policy")
    assert not (tmp_path / "checkpoints" / "global_step_13").exists()
    assert (tmp_path / "checkpoints" / "latest_ckpt_global_step.txt").read_text() == "12"
    state = torch.load(os.path.join(checkpoint_dir, "trainer_state.pt"), weights_only=False)
    assert state["global_step"] == 12


def test_base_checkpoint_can_defer_latest_publication(tmp_path, monkeypatch):
    trainer = _make_base_trainer(tmp_path, monkeypatch)

    checkpoint_dir = trainer.save_checkpoints(checkpoint_step=12, publish=False)

    latest = tmp_path / "checkpoints" / "latest_ckpt_global_step.txt"
    assert not latest.exists()
    trainer._publish_checkpoint(12, checkpoint_dir)
    assert latest.read_text() == "12"


def test_async_state_failure_does_not_publish_incomplete_checkpoint(
    tmp_path, monkeypatch
):
    trainer = object.__new__(FullyAsyncRayPPOTrainer)
    base = _make_base_trainer(tmp_path, monkeypatch)
    trainer.__dict__.update(base.__dict__)
    trainer.epoch = 3
    trainer.async_train_dataloader = SimpleNamespace(
        get_consumed_uids_list=lambda: ["0"],
        get_filtered_uids_list=lambda: [],
        get_epoch_start_dataloader_state=lambda: {"epoch": 3},
    )
    latest = tmp_path / "checkpoints" / "latest_ckpt_global_step.txt"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text("11")
    real_open = trainer_module.io.open_file

    def fail_async_state(path, mode):
        if str(path).endswith("fully_async_state.pt"):
            raise OSError("injected async-state failure")
        return real_open(path, mode)

    monkeypatch.setattr(trainer_module.io, "open_file", fail_async_state)

    with pytest.raises(OSError, match="injected async-state failure"):
        trainer.save_checkpoints(checkpoint_step=12)

    assert latest.read_text() == "11"


def test_hf_export_uses_explicit_completed_step_without_moving_cursor(tmp_path, monkeypatch):
    trainer = _make_base_trainer(tmp_path, monkeypatch)

    trainer.save_models(checkpoint_step=12)

    assert trainer.global_step == 13
    assert trainer.dispatch.hf_calls[0][1] == str(tmp_path / "exports" / "global_step_12" / "policy")


def test_completed_step_checkpoint_guard_skips_step_zero_and_duplicates():
    assert _needs_final_checkpoint(10, completed_step=0, last_checkpoint_step=None) is False
    assert _needs_final_checkpoint(10, completed_step=12, last_checkpoint_step=12) is False
    assert _needs_final_checkpoint(10, completed_step=12, last_checkpoint_step=10) is True


class _AsyncLoader:
    def __init__(self):
        self.consumed = set()
        self.filtered = set()

    def num_trained(self):
        return len(self.consumed - self.filtered)

    async def mark_consumed_uids(self, uids):
        self.consumed.update(uids)

    async def mark_filtered_uids(self, uids):
        self.consumed.update(uids)
        self.filtered.update(uids)

    async def reset_at_epoch_end(self):
        self.consumed.clear()
        self.filtered.clear()

    def load_state_from_checkpoint(self, consumed, filtered, epoch_start_state=None):
        self.consumed = set(consumed or ())
        self.filtered = set(filtered or ())


class _StalenessManager:
    def __init__(self):
        self.loaded_cursor = None

    def load_state_from_checkpoint(self, cursor):
        self.loaded_cursor = cursor

    async def notify_capacity_change(self, _cursor):
        return None

    async def validate_state_at_epoch_end(self, _cursor):
        return None

    async def on_rollout_filtered(self):
        return None


class _AsyncDispatch:
    async def save_weights_for_sampler(self):
        return None

    def finalize_pending_saves(self, _model):
        return None


class _Tracker:
    def log(self, *_args, **_kwargs):
        return None

    def finish(self):
        return None


def _make_async_loop(*, resumed: bool):
    trainer = object.__new__(FullyAsyncRayPPOTrainer)
    trainer.cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            epochs=1,
            max_training_steps=None,
            eval_interval=-1,
            eval_before_train=False,
            ckpt_interval=10,
            hf_save_interval=-1,
            update_ref_every_epoch=False,
            critic=SimpleNamespace(model=SimpleNamespace(path=None)),
        )
    )
    trainer.resume_mode = ResumeMode.LATEST if resumed else ResumeMode.NONE
    trainer.total_training_steps = 2
    trainer.num_steps_per_epoch = 2
    trainer.mini_batch_size = 1
    trainer._gen_buffer_maxsize = 2
    trainer.sample_full_batch = False
    trainer.async_train_dataloader = _AsyncLoader()
    trainer._staleness_manager = _StalenessManager()
    trainer.dispatch = _AsyncDispatch()
    trainer.tracker = _Tracker()
    trainer.ref_model = None
    trainer._ray_gpu_monitor = None
    trainer._vllm_metrics_scraper = None
    trainer.all_metrics = {}
    trainer.all_timings = {}
    trainer._phase_gauge = SimpleNamespace(timed_phase=lambda *_args, **_kwargs: nullcontext())
    trainer._loop_gauges = SimpleNamespace(set=lambda *_args, **_kwargs: None)
    trainer.init_weight_sync_state = lambda: None
    trainer._profiler_start = lambda: None
    trainer._profiler_step = lambda: None
    trainer._profiler_stop = lambda: None

    async def generation_supervisor(_buffer):
        await asyncio.sleep(0)

    trainer._run_generation_task_group = generation_supervisor
    trainer.convert_generation_group_mini_batch_to_training_input = lambda *_args: object()

    async def run_training(_training_input):
        return {"loss": 1.0}

    trainer._run_training = run_training
    trainer.save_calls = []

    def save_checkpoint(*, checkpoint_step=None):
        trainer.save_calls.append((checkpoint_step, trainer.global_step))
        return f"global_step_{checkpoint_step}"

    trainer.save_checkpoints = save_checkpoint
    if resumed:
        trainer.load_checkpoints = lambda: (
            1,
            "global_step_1",
            {"old"},
            set(),
            0,
            None,
        )

    return trainer


@pytest.mark.asyncio
async def test_epoch_exhaustion_checkpoints_completed_step_not_next_cursor():
    trainer = _make_async_loop(resumed=False)
    calls = 0

    async def collect(_buffer, _supervisor):
        nonlocal calls
        calls += 1
        if calls == 1:
            return [SimpleNamespace(uid="new")], [], False
        return [], [], True

    trainer._collect_generation_mini_batch = collect

    await trainer.train()

    # Step 1 completed, then the async cursor advanced to step 2 before the
    # dataloader exhausted. The saved state must still be labeled step 1.
    assert trainer.save_calls == [(1, 2)]


@pytest.mark.asyncio
async def test_resume_trains_and_checkpoints_the_immediately_following_step():
    trainer = _make_async_loop(resumed=True)

    async def collect(_buffer, _supervisor):
        return [SimpleNamespace(uid="new")], [], False

    trainer._collect_generation_mini_batch = collect

    await trainer.train()

    assert trainer._staleness_manager.loaded_cursor == 2
    assert trainer.save_calls == [(2, 2)]
