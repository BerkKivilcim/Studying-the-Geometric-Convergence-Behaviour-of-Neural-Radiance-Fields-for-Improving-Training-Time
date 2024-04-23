"""
Template DataManager
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union

import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.data.pixel_samplers import (
    PixelSampler,
    PixelSamplerConfig,
    PatchPixelSamplerConfig,
    PatchPixelSampler
)
from nerfstudio.configs.base_config import (
    InstantiateConfig,
)


@dataclass
class reconstructionDataManagerConfig(VanillaDataManagerConfig):
    """Template DataManager Config

    Add your custom datamanager config parameters here.
    """

    _target: Type = field(default_factory=lambda: reconstructionDataManager)
    implement_masking: bool= False
class reconstructionDataManager(VanillaDataManager):
    """Template DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: reconstructionDataManagerConfig

    def __init__(
        self,
        config: reconstructionDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )
    
    def next_train(self, step: int, mask) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)

        if self.config.implement_masking is True:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            mask = torch.tensor(mask, device=device, dtype=torch.int)
            image_batch["mask"] = mask.unsqueeze(-1)

        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)

        return ray_bundle, batch
