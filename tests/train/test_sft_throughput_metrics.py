"""Unit coverage for SFT throughput and sequence accounting."""

import torch

from skyrl.backends.skyrl_train.training_batch import TensorList, TrainingInputBatch
from skyrl.train.sft_trainer import _model_throughput_metrics, _training_sequence_stats


def test_training_sequence_stats_unpacked_include_executed_rows():
    batch = TrainingInputBatch(
        {
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 0, 0],
                ]
            )
        }
    )
    batch.metadata = {"pad_size": 1}

    # The zero-loss duplicate row is excluded from useful TPS, but it still
    # executes model FLOPs and therefore belongs in model-throughput metrics.
    assert _training_sequence_stats(batch) == (3, 11, 43)


def test_training_sequence_stats_packed_respect_trajectory_boundaries():
    batch = TrainingInputBatch(
        {
            "attention_mask": torch.ones(2, 12, dtype=torch.long),
            "sub_seq_lengths": TensorList(
                [
                    torch.tensor([5, 3]),
                    torch.tensor([7, 2]),
                ]
            ),
        }
    )
    batch.metadata = {
        "num_real_examples": 3,
        "num_padding_examples": 1,
        "num_padding_tokens": 2,
    }

    # Squaring each independent trajectory gives 25+9+49+4=87. Squaring
    # packed row lengths would overstate attention work as 8**2+9**2=145.
    assert _training_sequence_stats(batch) == (4, 17, 87)


def test_model_throughput_metrics_make_peak_explicit():
    metrics = _model_throughput_metrics(
        model_flops=144e15,
        step_seconds=1.0,
        num_gpus=64,
        peak_tflops_per_gpu=2250,
    )

    assert metrics["throughput/tflops/device"] == 2250
    assert metrics["throughput/mfu"] == 1.0
    assert metrics["throughput/mfu_percent"] == 100.0
    assert metrics["throughput/peak_tflops/device"] == 2250
