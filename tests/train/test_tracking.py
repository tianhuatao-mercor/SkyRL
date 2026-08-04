"""
uv run --isolated --extra dev --extra skyrl-train pytest -s tests/train/test_tracking.py
"""

from unittest.mock import MagicMock, patch

from skyrl.train.utils.tracking import Tracking


def test_wandb_init_receives_tags():
    """Tags passed to Tracking are forwarded to wandb.init."""
    with patch.dict("sys.modules", {"wandb": MagicMock()}) as mocked:
        wandb_mock = mocked["wandb"]
        Tracking(
            project_name="proj",
            experiment_name="exp",
            backend="wandb",
            config={},
            tags=["foo", "bar"],
        )

        wandb_mock.init.assert_called_once()
        kwargs = wandb_mock.init.call_args.kwargs
        assert kwargs["tags"] == ["foo", "bar"]
        assert kwargs["project"] == "proj"
        assert kwargs["name"] == "exp"


def test_wandb_init_tags_default_none():
    """When tags are not provided, wandb.init receives tags=None."""
    with patch.dict("sys.modules", {"wandb": MagicMock()}) as mocked:
        wandb_mock = mocked["wandb"]
        Tracking(
            project_name="proj",
            experiment_name="exp",
            backend="wandb",
            config={},
        )

        wandb_mock.init.assert_called_once()
        assert wandb_mock.init.call_args.kwargs["tags"] is None


def test_external_metrics_are_logged_and_written_to_wandb_summary():
    with patch.dict("sys.modules", {"wandb": MagicMock()}) as mocked:
        wandb_mock = mocked["wandb"]
        tracker = Tracking(
            project_name="proj",
            experiment_name="exp",
            backend="wandb",
            config={},
        )
        tracker.set_metrics_provider(
            lambda: {"usage/tokens": 123},
            summary_provider=lambda: {"usage/tokens": 123, "usage/run_id": "run-1"},
        )

        tracker.log({"reward": 1.0}, step=2, commit=True)
        tracker.finish()

        wandb_mock.log.assert_called_once_with(
            data={"reward": 1.0, "usage/tokens": 123},
            step=2,
            commit=True,
        )
        wandb_mock.run.summary.update.assert_called_once_with({"usage/tokens": 123, "usage/run_id": "run-1"})
