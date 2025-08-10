"""
ImageNet Data Module

This module provides a complete data loading pipeline for ImageNet dataset with
aspect ratio bucketing and efficient batching. It includes dataset classes,
data loaders, and samplers optimized for training vision models.

Classes:
    ImagenetDataModuleParams: Configuration parameters for the data module
    ImagenetDataModule: Main data module class that manages datasets and loaders
    ImagenetDataset: Dataset class for ImageNet with aspect ratio bucketing
    ImagenetDataSampler: Custom sampler for efficient batch sampling
"""

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from einops import rearrange
from PIL import Image
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from models.latent_fm import FMDataContext
from utils.config import BaseParams, ConfigurableModule
from utils.log import get_logger
from utils.misc import load_pt_data_from_path

logger = get_logger(__name__)


class ImagenetWrapper:
    def __init__(self, dataloader: Any, infinite: bool = True) -> None:
        self.dataloader = dataloader
        self._iterator: Iterator[Any] | None = None
        self.infinite = infinite

    def __iter__(self) -> Iterator[FMDataContext]:
        """
        Returns an iterator for the wrapped dataloader. A new iterator is
        created for each epoch, making the wrapper reusable.
        """
        self._iterator = iter(self.dataloader)
        return self

    def __next__(self) -> FMDataContext:
        """
        Returns the next item from the iterator. This makes the wrapper
        itself an iterator, as expected by some frameworks.
        """
        if self._iterator is None:
            # This handles cases where next() is called without a for loop.
            self._iterator = iter(self.dataloader)

        data_raw = None
        try:
            data_raw = next(self._iterator)
        except StopIteration:
            # Reset and re-raise to signal the end of an epoch.
            if self.infinite:
                self._iterator = iter(self.dataloader)
                data_raw = next(self._iterator)
            else:
                self._iterator = None
                raise StopIteration from None

        return FMDataContext(
            raw_texts=data_raw["raw_texts"],
            raw_images=data_raw["raw_images"],
        )


@dataclass
class ImagenetDataModuleParams(BaseParams):
    """
    Configuration parameters for the ImageNet data module.
    """

    batch_size: int = 32
    resolution: int = 256
    num_workers: int = 8
    data_seed: int = 42
    p_horizon_flip: float = 0.5

    # The default values are actually not valid, please overwrite them with the correct values
    data_root_dir: Path = Path("<MINFM_DATA_DIR>/imagenet")
    image_metas_path: str = "<MINFM_DATA_DIR>/imagenet/ilsvrc2012_meta.pt::image_metas"
    label_to_txt_path: str = "<MINFM_DATA_DIR>/imagenet/ilsvrc2012_meta.pt::label_to_txt"


def imagenet_collate_fn(batch: list[dict]) -> dict:
    return {
        "raw_images": rearrange(torch.stack([item["raw_images"] for item in batch]), "b c h w -> b c 1 h w"),
        "raw_texts": [item["raw_texts"] for item in batch],
    }


class ImagenetDataModule(ConfigurableModule[ImagenetDataModuleParams]):
    """
    Main data module for ImageNet dataset with aspect ratio bucketing.

    This class manages the creation and configuration of ImageNet datasets
    and data loaders for both training and validation splits. It implements
    aspect ratio bucketing to improve training efficiency by grouping images
    with similar aspect ratios together.

    Attributes:
        params: Configuration parameters
        datasets: Dictionary containing train and validation datasets
        samplers: Dictionary containing train and validation samplers
    """

    def __init__(
        self,
        params: ImagenetDataModuleParams,
        data_seed: int = 42,
        data_process_group: dist.ProcessGroup | None = None,
    ):
        """
        Initialize the ImageNet data module.

        Args:
            params: Configuration parameters for the data module
        """
        self.params = deepcopy(params)

        # Sync the data seed across the data group.
        if dist.is_initialized():
            data_seed_tensor: torch.Tensor = torch.tensor(data_seed, dtype=torch.int64).cuda()
            dist.all_reduce(data_seed_tensor, op=ReduceOp.MIN, group=data_process_group)
            data_seed = int(data_seed_tensor.cpu())

        logger.info(f"Imagenet set unique data seed to: {data_seed}")
        self.params.data_seed = data_seed

        self.datasets = {
            split: ImagenetDataset(
                data_root_dir=params.data_root_dir,
                image_metas_path=params.image_metas_path,
                label_to_txt_path=params.label_to_txt_path,
                split=split,
            )
            for split in ["train", "val"]
        }

    def train_dataloader(self) -> Any:
        """
        Get the training data loader.

        Returns:
            DataLoader configured for training with shuffling enabled
        """

        sampler = None
        dataloader_shuffle = None
        if dist.is_initialized():
            logger.info("Using DistributedSampler for train dataloader")
            sampler = DistributedSampler(
                self.datasets["train"],
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                seed=self.params.data_seed,
                shuffle=True,
            )
        else:
            logger.info("Using regular DataLoader for train dataloader")
            dataloader_shuffle = True

        dataloader = DataLoader(
            self.datasets["train"],
            sampler=sampler,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers,
            collate_fn=imagenet_collate_fn,
            shuffle=dataloader_shuffle,
        )
        # In training we create an infinite wrapper to allow for multiple epochs.
        return ImagenetWrapper(dataloader, infinite=True)

    def val_dataloader(self) -> Any:
        """
        Get the validation data loader.

        Returns:
            DataLoader configured for validation without shuffling
        """
        sampler = None
        dataloader_shuffle = None
        if dist.is_initialized():
            logger.info("Using DistributedSampler for val dataloader")
            sampler = DistributedSampler(
                self.datasets["val"],
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                seed=self.params.data_seed,
                shuffle=False,
            )
        else:
            logger.info("Using regular DataLoader for val dataloader")
            dataloader_shuffle = False

        dataloader = DataLoader(
            self.datasets["val"],
            sampler=sampler,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers,
            collate_fn=imagenet_collate_fn,
            shuffle=dataloader_shuffle,
        )
        return ImagenetWrapper(dataloader, infinite=False)

    @classmethod
    def get_default_params(cls) -> ImagenetDataModuleParams:
        """
        Return the default parameters for ImagenetDataModule.

        Returns:
            Default configuration parameters
        """
        return ImagenetDataModuleParams()


class ImagenetDataset(Dataset):
    """
    ImageNet dataset with aspect ratio bucketing.

    This dataset loads ImageNet images and groups them into aspect ratio buckets.
    Images are resized and cropped to fit within their assigned buckets while maintaining aspect ratios.

    Attributes:
        split: Dataset split ('train' or 'val')
        data_root_dir: Root directory containing image data
        data_meta: Metadata for all images in the split
        label_to_txt: Mapping from label to text
        resolution: Resolution of the image
        p_horizon_flip: Probability of horizontal flip
    """

    def __init__(
        self,
        data_root_dir: Path,
        image_metas_path: str,
        label_to_txt_path: str,
        resolution: int = 256,
        p_horizon_flip: float = 0.5,
        split: str = "train",
    ) -> None:
        """
        Initialize the ImageNet dataset.

        Args:
            data_root_dir: Root directory containing image data
            image_metas_path: Path to metadata pytorch pt file
            label_to_txt_path: Path to label to text pytorch pt file
            split: Dataset split ('train' or 'val')

        """
        super().__init__()
        self.resolution = resolution
        self.split = split
        self.data_root_dir = data_root_dir
        self.data_meta = load_pt_data_from_path(image_metas_path)[split]
        self.label_to_txt = load_pt_data_from_path(label_to_txt_path)

        def crop_to_square(image: Image.Image) -> Image.Image:
            width, height = image.size
            min_dim = min(width, height)
            left, top, right, bottom = (
                (width - min_dim) // 2,
                (height - min_dim) // 2,
                (width + min_dim) // 2,
                (height + min_dim) // 2,
            )
            return image.crop((left, top, right, bottom))

        self.image_transforms = transforms.Compose(
            [
                transforms.Lambda(crop_to_square),
                transforms.Resize((resolution, resolution), Image.Resampling.LANCZOS),
                transforms.RandomHorizontalFlip(p=p_horizon_flip),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single image and its metadata.

        Args:
            idx: Index of the image to retrieve

        Returns:
            Dictionary containing:
                - raw_images: Tensor of shape (3, H, W) with pixel values in [-1, 1]
                - raw_texts: String label for the image

        The image is loaded, cropped to fit its aspect ratio bucket, and resized
        to the bucket dimensions while maintaining the aspect ratio.
        """
        data_meta = self.data_meta[idx]
        img_path = Path(self.data_root_dir) / self.split / data_meta["img_path"]
        label = int(data_meta["label"])
        txt = self.label_to_txt[label]

        if not img_path.exists():
            # In case download from huggingface, the image path does not have subfolders
            img_path = (
                Path(self.data_root_dir)
                / "extracted"
                / self.split
                / f"{img_path.stem}_{img_path.parent.name}{img_path.suffix}"
            )

        image = Image.open(img_path).convert("RGB")
        image = self.image_transforms(image)

        return {
            "raw_images": image.clip(-1.0, 1.0),
            "raw_texts": txt,
        }

    def __len__(self) -> int:
        """
        Get the total number of images in the dataset.

        Returns:
            Number of images in the dataset
        """
        return len(self.data_meta)
