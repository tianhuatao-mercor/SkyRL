"""Direct entrypoint for hosted Tinker training.

Tinker owns policy and sampling compute, so the SkyRL orchestration loop can
run in this process without creating local Ray model workers.
"""

from __future__ import annotations

import sys

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.utils import validate_cfg


class FullyAsyncTinkerExp(BasePPOExp):
    """Use SkyRL's fully-async scheduler while Tinker owns model compute."""

    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        generator,
        colocate_pg,
    ):
        from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer

        return FullyAsyncRayPPOTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )


def experiment_class(cfg: SkyRLTrainConfig):
    """Select the synchronous or fully-async SkyRL scheduling loop."""

    return FullyAsyncTinkerExp if cfg.trainer.fully_async.enabled else BasePPOExp


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    if cfg.trainer.strategy != "tinker":
        raise ValueError("main_tinker requires trainer.strategy='tinker'")
    experiment_class(cfg)(cfg).run()


if __name__ == "__main__":
    main()
