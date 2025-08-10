from collections.abc import Iterator
from typing import Any, cast

import torch.distributed as dist

from models.latent_fm import FMDataContext
from utils.config import create_component


class DataStreamer:
    """Base class for data streamers."""

    def __init__(
        self, config: dict[str, Any], data_seed: int = 42, data_process_group: dist.ProcessGroup | None = None
    ):
        self.config = config

        self.data_module = create_component(
            config["module"], config["params"], data_seed=data_seed, data_process_group=data_process_group
        )

    def train_dataloader(self) -> Iterator[FMDataContext]:
        """Return the training dataloader with a precise type for static analysis."""
        return cast(Iterator[FMDataContext], self.data_module.train_dataloader())

    def val_dataloader(self) -> Iterator[FMDataContext]:
        """Return the validation dataloader with a precise type for static analysis."""
        return cast(Iterator[FMDataContext], self.data_module.val_dataloader())
