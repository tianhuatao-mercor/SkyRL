"""
CPU unit tests for fully-async trainer building blocks that back `sample_full_batch`:
the staleness manager's filtered-rollout accounting, the dataloader's trained-vs-filtered
UID tracking, and the consumer's exhaustion-aware buffer drain.
"""

import asyncio
import threading
from types import SimpleNamespace

import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

import skyrl.train.fully_async_trainer as fully_async_trainer_module
import skyrl.train.trainer as trainer_module
from skyrl.train.fully_async_trainer import (
    FullyAsyncRayPPOTrainer,
    GeneratedOutputGroup,
    _advance_global_step_after_training,
    _AsyncDataloader,
    _AsyncStalenessManager,
    _needs_final_checkpoint,
)
from skyrl.train.utils.callbacks import CallbackHandler, TrainingCallback, TrainingControl


class _StatusRecorder(TrainingCallback):
    def __init__(self):
        self.events = []
        self.thread_ids = []

    def on_status_change(self, trainer, callback_input, control):
        self.events.append(callback_input)
        self.thread_ids.append(threading.get_ident())


def test_fully_async_trainer_supports_status_callbacks(monkeypatch):
    """Status observers receive the same scalar transition written for the dashboard."""

    progress_events = []
    monkeypatch.setattr(
        trainer_module,
        "emit_progress_event",
        lambda event, **payload: progress_events.append((event, payload)),
    )

    recorder = _StatusRecorder()
    trainer = FullyAsyncRayPPOTrainer.__new__(FullyAsyncRayPPOTrainer)
    trainer.global_step = 7
    trainer.total_training_steps = 241
    trainer._current_epoch = 0
    trainer.train_dataloader = []
    trainer.cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            policy=SimpleNamespace(model=SimpleNamespace(path="Qwen/Qwen3.6-35B-A3B"))
        )
    )
    trainer._callback_handler = CallbackHandler()
    trainer._training_control = TrainingControl()
    trainer.add_callback(recorder)

    trainer._report_trainer_status(
        "forward_backward",
        detail="provider schedule in flight; internal microbatch progress unavailable",
        model_role="policy",
        update_epoch=1,
        update_epochs_total=1,
        mini_batch=2,
        mini_batches_total=4,
        progress_granularity="optimizer_minibatch",
        policy_version=6,
    )

    assert len(recorder.events) == 1
    callback_input = recorder.events[0]
    assert callback_input.status == "forward_backward"
    assert callback_input.mini_batch == 2
    assert callback_input.mini_batches_total == 4
    assert callback_input.progress_granularity == "optimizer_minibatch"
    assert progress_events == [
        (
            "trainer_status_changed",
            {
                "global_step": 7,
                "total_steps": 241,
                "status": "forward_backward",
                "status_detail": "provider schedule in flight; internal microbatch progress unavailable",
                "model_name": "Qwen/Qwen3.6-35B-A3B",
                "model_role": "policy",
                "update_epoch": 1,
                "update_epochs_total": 1,
                "mini_batch": 2,
                "mini_batches_total": 4,
                "progress_granularity": "optimizer_minibatch",
                "policy_version": 6,
            },
        )
    ]


def test_status_callback_from_provider_thread_is_marshaled_to_owner_loop(monkeypatch):
    monkeypatch.setattr(trainer_module, "emit_progress_event", lambda *_args, **_kwargs: None)
    recorder = _StatusRecorder()
    trainer = FullyAsyncRayPPOTrainer.__new__(FullyAsyncRayPPOTrainer)
    trainer.global_step = 3
    trainer.total_training_steps = 10
    trainer._current_epoch = 0
    trainer.train_dataloader = []
    trainer.cfg = SimpleNamespace(
        trainer=SimpleNamespace(policy=SimpleNamespace(model=SimpleNamespace(path="test-model")))
    )
    trainer._callback_handler = CallbackHandler([recorder])
    trainer._training_control = TrainingControl()

    async def run_from_provider_thread():
        owner_thread = threading.get_ident()
        trainer._trainer_status_callback_loop = asyncio.get_running_loop()
        trainer._trainer_status_callback_thread_id = owner_thread
        await asyncio.to_thread(
            trainer._report_trainer_status,
            "forward_backward",
            model_role="policy",
            mini_batch=1,
            mini_batches_total=2,
        )
        return owner_thread

    owner_thread = asyncio.run(run_from_provider_thread())
    assert recorder.thread_ids == [owner_thread]


def test_train_wrapper_reports_failures_outside_the_main_step_loop(monkeypatch):
    progress_events = []
    monkeypatch.setattr(
        fully_async_trainer_module,
        "emit_progress_event",
        lambda event, **payload: progress_events.append((event, payload)),
    )
    monkeypatch.setattr(
        trainer_module,
        "emit_progress_event",
        lambda event, **payload: progress_events.append((event, payload)),
    )

    trainer = FullyAsyncRayPPOTrainer.__new__(FullyAsyncRayPPOTrainer)
    trainer.global_step = 0
    trainer.total_training_steps = 241
    trainer._current_epoch = 0
    trainer.train_dataloader = []
    trainer.cfg = SimpleNamespace(
        trainer=SimpleNamespace(policy=SimpleNamespace(model=SimpleNamespace(path="test-model")))
    )
    trainer.inference_engine_client = SimpleNamespace(weight_version=0)
    trainer._callback_handler = CallbackHandler()
    trainer._training_control = TrainingControl()
    trainer._ray_gpu_monitor = None

    async def fail_during_initialization():
        raise RuntimeError("initial synchronization failed")

    trainer._train_impl = fail_during_initialization

    with pytest.raises(RuntimeError, match="initial synchronization failed"):
        asyncio.run(trainer.train())

    assert [event for event, _ in progress_events] == [
        "training_run_failed",
        "trainer_status_changed",
    ]
    assert progress_events[-1][1]["status"] == "failed"
    assert progress_events[-1][1]["status_detail"] == "RuntimeError: training aborted"


def _make_async_dataloader(
    num_prompts: int,
    mini_batch_size: int,
    *,
    shuffle: bool = False,
) -> _AsyncDataloader:
    """Build an _AsyncDataloader over a trivial dataset of `num_prompts` single-prompt batches."""
    dataset = [[{"uid": str(i)}] for i in range(num_prompts)]
    # batch_size=1 (one prompt per draw) and identity collate so each batch is a list with one dict.
    generator = torch.Generator().manual_seed(42)
    loader = StatefulDataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        generator=generator,
        collate_fn=lambda batch: batch[0],
    )
    return _AsyncDataloader(loader, mini_batch_size)


def test_max_training_step_keeps_final_checkpoint_cursor_on_completed_step():
    assert _advance_global_step_after_training(3, 3) == (3, True)
    assert _advance_global_step_after_training(3, 60) == (4, False)
    assert _advance_global_step_after_training(3, None) == (4, False)


def test_final_checkpoint_is_not_duplicated_at_same_step():
    assert _needs_final_checkpoint(10, 60, 60) is False
    assert _needs_final_checkpoint(10, 60, 50) is True
    assert _needs_final_checkpoint(10, 60, None) is True
    assert _needs_final_checkpoint(-1, 60, None) is False


# --------------------------------------------------------------------------------------
# _AsyncStalenessManager
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_staleness_manager_filter_restores_capacity():
    """Dropping an accepted group via on_rollout_filtered must give producer capacity back.

    This is the deadlock regression: without reclassifying accepted -> filtered, dropped groups
    keep `accepted` climbing against a fixed staleness ceiling and starve the producers.
    """
    mgr = _AsyncStalenessManager(max_concurrent_generation_groups=4, mini_batch_size=2, max_staleness_steps=1)
    # consumer_capacity = (max_staleness_steps + current_global_step) * mini_batch = (1 + 1) * 2 = 4.
    for _ in range(4):
        await mgr.acquire_submission_slot()
    for _ in range(4):
        await mgr.on_rollout_accepted()

    # At the staleness ceiling: accepted == 4 == ceiling, so no producer capacity remains.
    assert mgr._compute_capacity_unlocked() == 0

    await mgr.on_rollout_filtered()

    # Capacity restored by exactly one slot, and accounting is consistent.
    assert mgr._compute_capacity_unlocked() == 1
    assert mgr._stat.accepted == 3
    assert mgr._stat.filtered == 1
    assert mgr._stat.running == 0
    assert mgr._stat.submitted == 4


@pytest.mark.asyncio
async def test_staleness_manager_validate_epoch_end_with_filtered():
    """At epoch end, submitted == accepted + filtered and accepted == trained steps * mini_batch."""
    mgr = _AsyncStalenessManager(max_concurrent_generation_groups=4, mini_batch_size=2, max_staleness_steps=1)
    # Submit and finish 4 groups; drop 2 of them, train on the remaining 2 (one step).
    for _ in range(4):
        await mgr.acquire_submission_slot()
    for _ in range(4):
        await mgr.on_rollout_accepted()
    for _ in range(2):
        await mgr.on_rollout_filtered()

    # One training step completed -> we are now working on global_step 2.
    await mgr.notify_capacity_change(2)
    await mgr.validate_state_at_epoch_end(global_step=2)  # must not raise


# --------------------------------------------------------------------------------------
# _AsyncDataloader
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_dataloader_filtered_uids_tracking():
    adl = _make_async_dataloader(num_prompts=6, mini_batch_size=2)

    await adl.mark_consumed_uids(["0", "1"])
    await adl.mark_filtered_uids(["2"])

    assert adl.num_trained() == 2
    assert set(adl.get_filtered_uids_list()) == {"2"}
    assert set(adl.get_consumed_uids_list()) == {"0", "1", "2"}


@pytest.mark.asyncio
async def test_async_dataloader_skips_filtered_uids():
    adl = _make_async_dataloader(num_prompts=6, mini_batch_size=2)
    await adl.mark_consumed_uids(["0", "1"])
    await adl.mark_filtered_uids(["2"])

    seen = []
    while True:
        prompts = await adl.get_next_non_consumed_data()
        if prompts is None:
            break
        seen.append(prompts[0]["uid"])

    # Trained (0, 1) and filtered (2) are all skipped; only the rest are drawn.
    assert seen == ["3", "4", "5"]


@pytest.mark.asyncio
async def test_async_dataloader_load_state_restores_filtered():
    adl = _make_async_dataloader(num_prompts=6, mini_batch_size=2)
    adl.load_state_from_checkpoint({"0", "1", "2"}, {"2"})

    assert adl.num_trained() == 2
    assert set(adl.get_filtered_uids_list()) == {"2"}

    seen = []
    while True:
        prompts = await adl.get_next_non_consumed_data()
        if prompts is None:
            break
        seen.append(prompts[0]["uid"])
    assert seen == ["3", "4", "5"]


@pytest.mark.asyncio
async def test_async_dataloader_load_state_without_filtered_is_backward_compatible():
    adl = _make_async_dataloader(num_prompts=6, mini_batch_size=2)
    # Old checkpoints have no filtered set; default treats everything consumed as trained.
    adl.load_state_from_checkpoint({"0", "1"})
    assert adl.num_trained() == 2
    assert adl.get_filtered_uids_list() == []


@pytest.mark.asyncio
async def test_async_dataloader_epoch_reset_advances_shuffle_order():
    adl = _make_async_dataloader(num_prompts=8, mini_batch_size=2, shuffle=True)

    first_epoch = []
    while (prompts := await adl.get_next_non_consumed_data()) is not None:
        first_epoch.append(prompts[0]["uid"])

    await adl.reset_at_epoch_end()
    second_epoch = []
    while (prompts := await adl.get_next_non_consumed_data()) is not None:
        second_epoch.append(prompts[0]["uid"])

    assert sorted(first_epoch) == sorted(second_epoch)
    assert first_epoch != second_epoch


@pytest.mark.asyncio
async def test_async_dataloader_resume_restores_current_epoch_shuffle_order():
    adl = _make_async_dataloader(num_prompts=8, mini_batch_size=2, shuffle=True)

    while await adl.get_next_non_consumed_data() is not None:
        pass
    await adl.reset_at_epoch_end()
    epoch_start_state = adl.get_epoch_start_dataloader_state()

    consumed = []
    for _ in range(3):
        prompts = await adl.get_next_non_consumed_data()
        assert prompts is not None
        consumed.append(prompts[0]["uid"])
    await adl.mark_consumed_uids(consumed)

    expected_remaining = []
    while (prompts := await adl.get_next_non_consumed_data()) is not None:
        expected_remaining.append(prompts[0]["uid"])

    resumed = _make_async_dataloader(num_prompts=8, mini_batch_size=2, shuffle=True)
    resumed.load_state_from_checkpoint(
        set(consumed),
        set(),
        epoch_start_state,
    )
    actual_remaining = []
    while (prompts := await resumed.get_next_non_consumed_data()) is not None:
        actual_remaining.append(prompts[0]["uid"])

    assert actual_remaining == expected_remaining


# --------------------------------------------------------------------------------------
# _drain_next_group
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drain_next_group_returns_buffered_items_then_exhaustion():
    buffer: asyncio.Queue = asyncio.Queue()
    supervisor = asyncio.create_task(asyncio.sleep(0))
    await supervisor
    buffer.put_nowait("a")
    buffer.put_nowait("b")

    drain = FullyAsyncRayPPOTrainer._drain_next_group
    dummy = object()

    assert await drain(dummy, buffer, supervisor) == "a"
    assert await drain(dummy, buffer, supervisor) == "b"

    # Buffer empty and the generation TaskGroup completed successfully -> exhausted.
    assert await drain(dummy, buffer, supervisor) is None


@pytest.mark.asyncio
async def test_drain_next_group_drains_remaining_before_exhaustion():
    """If generators finish while items remain, those items are returned before None."""
    buffer: asyncio.Queue = asyncio.Queue()
    supervisor = asyncio.create_task(asyncio.sleep(0))
    await supervisor
    buffer.put_nowait("a")

    drain = FullyAsyncRayPPOTrainer._drain_next_group
    dummy = object()
    assert await drain(dummy, buffer, supervisor) == "a"
    assert await drain(dummy, buffer, supervisor) is None


@pytest.mark.asyncio
async def test_drain_next_group_blocks_until_item_arrives():
    buffer: asyncio.Queue = asyncio.Queue()
    supervisor = asyncio.create_task(asyncio.Event().wait())
    drain = FullyAsyncRayPPOTrainer._drain_next_group
    dummy = object()

    async def delayed_put():
        await asyncio.sleep(0.05)
        buffer.put_nowait("x")

    producer = asyncio.create_task(delayed_put())
    try:
        assert await drain(dummy, buffer, supervisor) == "x"
        await producer
    finally:
        supervisor.cancel()
        await asyncio.gather(supervisor, return_exceptions=True)


class _ExpectedGenerationError(RuntimeError):
    pass


@pytest.mark.asyncio
async def test_generation_task_group_failure_wakes_blocked_buffer_consumer():
    """A worker failure propagates through TaskGroup instead of hanging or killing the process."""
    buffer: asyncio.Queue = asyncio.Queue()

    async def fail_generation(_buffer):
        raise _ExpectedGenerationError("generation failed")

    trainer = SimpleNamespace(
        num_parallel_generation_workers=1,
        _run_generate_for_a_group_loop=fail_generation,
    )
    supervisor = asyncio.create_task(FullyAsyncRayPPOTrainer._run_generation_task_group(trainer, buffer))

    with pytest.raises(ExceptionGroup) as exc_info:
        await asyncio.wait_for(
            FullyAsyncRayPPOTrainer._drain_next_group(trainer, buffer, supervisor),
            timeout=1,
        )

    assert exc_info.value.subgroup(_ExpectedGenerationError) is not None
    assert supervisor.done()
    assert not supervisor.cancelled()
    # Reaching this assertion is the regression check that a worker error no longer exits the process.
    assert asyncio.current_task() is not None


@pytest.mark.asyncio
async def test_generation_worker_failure_releases_staleness_slot(monkeypatch):
    """A failed provider call returns its reserved generation capacity before propagating."""

    progress_events = []

    class OnePromptDataloader:
        def __init__(self):
            self.returned = False

        async def get_next_non_consumed_data(self):
            if self.returned:
                return None
            self.returned = True
            return [{"uid": "uid-1"}]

    class FailingGenerator:
        async def generate(self, _generator_input):
            raise _ExpectedGenerationError("provider failed")

    monkeypatch.setattr(
        fully_async_trainer_module,
        "prepare_generator_input",
        lambda *_args, **_kwargs: ({"prompts": ["prompt"]}, ["uid-1"]),
    )
    monkeypatch.setattr(
        fully_async_trainer_module,
        "get_sampling_params_for_backend",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        fully_async_trainer_module,
        "emit_progress_event",
        lambda event, **payload: progress_events.append((event, payload)),
    )

    staleness_manager = _AsyncStalenessManager(
        max_concurrent_generation_groups=1,
        mini_batch_size=1,
        max_staleness_steps=0,
    )
    trainer = SimpleNamespace(
        async_train_dataloader=OnePromptDataloader(),
        cfg=SimpleNamespace(
            generator=SimpleNamespace(
                # The mocked helper deliberately returns one UID below. Progress must still report
                # the configured trajectory group size rather than coupling itself to helper output.
                n_samples_per_prompt=4,
                inference_engine=SimpleNamespace(backend="test"),
                sampling_params=object(),
            ),
            environment=SimpleNamespace(env_class="test"),
        ),
        _staleness_manager=staleness_manager,
        generator=FailingGenerator(),
        global_step=1,
    )

    with pytest.raises(_ExpectedGenerationError, match="provider failed"):
        await FullyAsyncRayPPOTrainer._run_generate_for_a_group_loop(trainer, asyncio.Queue(maxsize=1))

    assert staleness_manager._stat.submitted == 1
    assert staleness_manager._stat.running == 0
    assert staleness_manager._stat.accepted == 0
    assert staleness_manager._compute_capacity_unlocked() == 1
    assert progress_events == [
        (
            "group_scheduled",
            {"global_step": 1, "group_uid": "uid-1", "group_size": 4},
        ),
        (
            "group_failed",
            {"global_step": 1, "group_uid": "uid-1"},
        ),
    ]


# --------------------------------------------------------------------------------------
# _should_keep_group
# --------------------------------------------------------------------------------------


def _trainer_with_tol(tol: float):
    """A stand-in for `self` exposing just the cfg field _should_keep_group reads."""
    return SimpleNamespace(
        cfg=SimpleNamespace(trainer=SimpleNamespace(algorithm=SimpleNamespace(zero_variance_filter_tol=tol)))
    )


def _group(rewards, loss_masks, uid="u"):
    return GeneratedOutputGroup(
        generator_output={"rewards": rewards, "loss_masks": loss_masks},
        uid=uid,
        global_step_when_scheduled=0,
    )


def test_should_keep_group():
    keep = FullyAsyncRayPPOTrainer._should_keep_group

    # Zero-variance group -> drop.
    assert keep(_trainer_with_tol(0.0), _group([1.0, 1.0], [[1], [1]])) is False
    # Group with reward spread -> keep.
    assert keep(_trainer_with_tol(0.0), _group([1.0, 0.0], [[1], [1]])) is True
    # Singleton and fully masked groups have no group-relative signal -> drop.
    assert keep(_trainer_with_tol(0.0), _group([1.0], [[1]])) is False
    assert keep(_trainer_with_tol(0.0), _group([0.0, 0.0], [[0], [0]])) is False
    # Masked trajectories are ignored: two equal live rewards + one masked -> still zero-variance.
    assert keep(_trainer_with_tol(0.0), _group([1.0, 1.0, 0.0], [[1], [1], [0]])) is False
    # Near-equal float rewards within tol -> drop.
    assert keep(_trainer_with_tol(1e-6), _group([0.6667, 0.66670001], [[1], [1]])) is False


def test_reprefix_metrics():
    """generate/X -> generate_<suffix>/X, preserving the leading namespace for tracker grouping."""
    reprefix = FullyAsyncRayPPOTrainer._reprefix_metrics
    out = reprefix(
        {"generate/avg_num_tokens": 10.0, "environment/score": 0.5, "bare": 1},
        "dropped",
    )
    assert out == {
        "generate_dropped/avg_num_tokens": 10.0,
        "environment_dropped/score": 0.5,
        "dropped/bare": 1,
    }


def test_exact_sampler_version_staleness_takes_precedence_over_schedule_step():
    trainer = SimpleNamespace(
        global_step=20,
        inference_engine_client=SimpleNamespace(weight_version=5),
    )
    group = GeneratedOutputGroup(
        generator_output={"response_ids": [[1], [2]], "sampler_versions": [3, 4]},
        uid="u",
        global_step_when_scheduled=19,
    )

    staleness, versions = FullyAsyncRayPPOTrainer._get_group_staleness(trainer, group)

    assert staleness == 2
    assert versions == [3, 4]


def test_sampler_staleness_falls_back_to_schedule_step():
    trainer = SimpleNamespace(
        global_step=20,
        inference_engine_client=SimpleNamespace(weight_version=5),
    )
    group = GeneratedOutputGroup(
        generator_output={"response_ids": [[1]], "sampler_versions": None},
        uid="u",
        global_step_when_scheduled=18,
    )

    staleness, versions = FullyAsyncRayPPOTrainer._get_group_staleness(trainer, group)

    assert staleness == 2
    assert versions == []


def test_should_keep_group_token_level_rewards():
    """Token-level rewards are collapsed to per-trajectory sequence rewards for the variance check."""
    keep = FullyAsyncRayPPOTrainer._should_keep_group

    # Two trajectories, both summing to 1.0 -> zero variance -> drop.
    assert (
        keep(
            _trainer_with_tol(0.0),
            _group([[0.0, 1.0], [1.0, 0.0]], [[1, 1], [1, 1]]),
        )
        is False
    )
    # One trajectory sums to 1.0, the other to 0.0 -> variance -> keep.
    assert (
        keep(
            _trainer_with_tol(0.0),
            _group([[0.0, 1.0], [0.0, 0.0]], [[1, 1], [1, 1]]),
        )
        is True
    )
