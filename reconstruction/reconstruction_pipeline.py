
"""
Nerfstudio Template Pipeline
"""
from os import path
import sys
sys.path.append('E:/Berk/code/reconstruction')
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import typing
from dataclasses import dataclass, field
from typing import Literal, Optional, Type

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from reconstruction.reconstruction_datamanager import reconstructionDataManagerConfig
from reconstruction.reconstruction_model import reconstructionModel, reconstructionModelConfig

from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)


from typing import Dict, List, Literal, Optional, Tuple, Type, cast

from nerfstudio.exporter.exporter_utils import generate_point_cloud,  render_trajectory
import open3d as o3d
from pathlib import Path
from nerfstudio.cameras.cameras import Cameras

from nerfstudio.engine.trainer import TrainerConfig
import random
import os

import torch
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler, Sampler
from nerfstudio.engine.trainer import Trainer
import numpy as np

@dataclass
class reconstructionPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: reconstructionPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = reconstructionDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = reconstructionModelConfig()
    """specifies the model config"""

    mask_pixel = None

    sample_percentage_train_image_rendering: float = field(default_factory=lambda: 0.05)
    early_stop: bool = True
    decrease_coarse_sampling: bool = False
    export_rendered_image_per_sample: bool = False
    export_point_cloud_per_sample: bool = False
    export_voxel_arrays_per_sample: bool = False
    output_export_dir: Path  = "  "


class reconstructionPipeline(VanillaPipeline):
    """Template Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """
    #field: NerfactoField
    def __init__(
        self,
        config: reconstructionPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
        *args,
        **kwargs,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, *args, **kwargs
        )

        self.datamanager.to(device)
          
        assert self.datamanager.train_dataset is not None, "Missing input dataset"
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                reconstructionModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True)
            )
            dist.barrier(device_ids=[local_rank])
        
        #The below code part is added because in (256,96),48 sampling, the coarse sampling decrease gives gpu cuda memory error.
        #However by decreasing the values by 1 which means (255,95),47 does not give any error
        self.initial_num_samples = self.model.config.num_proposal_samples_per_ray[0] -1
        self.first_pdf = self.model.config.num_proposal_samples_per_ray[1] -1
        self.rest_pdf = self.model.config.num_nerf_samples_per_ray -1

        #Important Parameters for early stop or decrease coarse decissions
        self.prev_voxel_density = None
        self.stop_training = False
######################################################################################################
    def train_step(self, step: int, samples):
        if step in samples:
            new_num_prop_samples, new_num_nerf_samples = self.calculate_new_num_samples(step)
            self.model.update_sampler_config(new_num_prop_samples, new_num_nerf_samples)

    def calculate_new_num_samples(self, step:int) ->int:
        new_initial_num_samples = round(self.initial_num_samples * np.log(1 + 100 * self.mean) / np.log(1 + 100)) + 1
        #new_first_pdf = round(self.first_pdf * np.log(1 + 100 * self.mean) / np.log(1 + 100)) + 1
        #new_rest_pdf = round(self.rest_pdf * np.log(1 + 100 * self.mean) / np.log(1 + 100)) + 1
        return (new_initial_num_samples, 96), 48
####################################################################################################
    def incremental_sampling(self, first_step, num_partitions, last_step):
        steps_to_distribute = last_step - first_step
        # divide the data range logaritmicaly
        remaining_samples = np.logspace(np.log10(first_step), np.log10(steps_to_distribute), num=num_partitions-1, endpoint=False, base=10.0) #base=10 means in the base of log10.
        remaining_samples = np.round(remaining_samples).astype(int)
        # makes every step unique
        for i in range(1, len(remaining_samples)):
            if remaining_samples[i] <= remaining_samples[i-1]:
                remaining_samples[i] = remaining_samples[i-1] + 1
        # add last step
        samples = np.append(remaining_samples, last_step-1)
        return samples

########################################################################################################
    def generate_and_save_point_cloud(self, step: int, output_directory: Path, samples):
        if step  in samples:
            if self.config.export_point_cloud_per_sample is True:
                pcd = generate_point_cloud(
                    pipeline=self,
                    num_points=1000000,  
                    remove_outliers=True,  
                    estimate_normals=True,  
                    reorient_normals=True,  
                    rgb_output_name="rgb",  
                    depth_output_name="depth",   #depth, expected_depth, custom_depth_render
                    normal_output_name=None,  
                    use_bounding_box=True,  
                    bounding_box_min=(-1, -1, -1.5),  
                    bounding_box_max=(1, 1, 1.5),  
                    crop_obb=None,  
                    std_ratio=10.0,  # Outlier removal threshold
                )
                #output_path = output_directory / f"point_cloud_{step}.ply"
                output_path = Path(self.config.output_export_dir / f"pointclouds")
                output_path_full = output_path / f"point_cloud_{step}.ply"
                os.makedirs(output_path, exist_ok=True)
                o3d.io.write_point_cloud(str(output_path_full), pcd)
        ####################################################################################################
            if self.config.export_rendered_image_per_sample is True:
                total_images = len(self.datamanager.train_dataset.cameras)
                sample_percentage = self.config.sample_percentage_train_image_rendering #uniform sampling rate for selecting the training images for rendering
                num_samples = int(total_images * sample_percentage)
                step_size = total_images // num_samples
                selected_indices = list(range(0, total_images, step_size))

                for i in selected_indices:
                    rgb_images, depth_images = render_trajectory(
                        pipeline=self,
                        cameras=self.datamanager.train_dataset.cameras[i], 
                        rgb_output_name="rgb",
                        depth_output_name="depth", #depth, expected_depth, custom_depth_render
                        rendered_resolution_scaling_factor=1.0,
                        disable_distortion=True,
                        return_rgba_images=True
                    )

                    for j, (rgb_img, depth_img) in enumerate(zip(rgb_images, depth_images)): #checking each validation (or training, depends on the cameras) images
                        #output_directory_rgb = Path("E:/Berk/code/reconstruction/outputs/outputs_rgb/"f"image_{i}")
                        output_directory_rgb = Path(self.config.output_export_dir / f"outputs_rgb/image_{i}")
                        #output_directory_depth = Path("E:/Berk/code/reconstruction/outputs/outputs_depth/"f"image_{i}")
                        output_directory_depth = Path(self.config.output_export_dir / f"outputs_depth/image_{i}")  
                        ###################################################################################################
                            #creates the folders
                        os.makedirs(output_directory_rgb, exist_ok=True)
                        os.makedirs(output_directory_depth, exist_ok=True)

                        rgb_img = (rgb_img * 255).astype(np.uint8) #converting to uint8. Without them the results looks bad. Probably previous format is float32

                        log_data = np.log10(depth_img)
                        log_min, log_max = log_data.min(), log_data.max()
                        normalized_data = (log_data - log_min) / (log_max - log_min)
                        depth_final_img = (normalized_data *255).astype(np.uint8)

                        o3d_rgb = o3d.geometry.Image(rgb_img) #without them it does not worko3d
                        o3d_depth = o3d.geometry.Image(depth_final_img)

                        rgb_path = output_directory_rgb / f"rgb_{step}_{i}.png"
                        depth_path = output_directory_depth / f"depth_{step}_{i}.png"

                        o3d.io.write_image(str(rgb_path), o3d_rgb)
                        o3d.io.write_image(str(depth_path), o3d_depth)
            #####################################################################################################
            voxel_density = reconstructionModel.voxel_density_values(self._model)
            if self.config.export_voxel_arrays_per_sample is True:
                #output_directory = Path("E:/Berk/code/reconstruction/outputs/outputs_density_voxels") 
                output_directory = Path(self.config.output_export_dir / f"outputs_density_voxels")
                output_directory.mkdir(parents=True, exist_ok=True)
                output_path = output_directory / f"density_values_{step}.npy"
                np.save(output_path, voxel_density)
            self.calculate_voxel_differences(voxel_density)
############################################################################################################
    def calculate_voxel_differences(self, current_voxel_density):
        #SET HIGHEST TOP %1 VALUES TO NEXT HIGHEST VALUES
        top_percent_value_current = np.percentile(current_voxel_density, 99) 
        current_voxel_density[current_voxel_density > top_percent_value_current] = top_percent_value_current
        #Normalize the voxels between 0-1
        current_voxel_density_normalized = (current_voxel_density - np.min(current_voxel_density)) / (np.max(current_voxel_density) - np.min(current_voxel_density))

        if self.prev_voxel_density is not None:
            diff = np.abs(current_voxel_density_normalized - self.prev_voxel_density) #Absolute differences between previous voxel and recent voxel
            median = np.median(diff)
            self.mean = np.mean(diff) #Used in coarse sampling decreasing function
            self.model.config.difference_voxel = diff #UPDATES DIFFERENCE VOXEL INSIDE OUR MODEL
            if median < self.model.config.converge_threshold: 
                self.stop_training = True #Updates parameter for early stop
        else:
            median = np.inf
            self.mean = 1
        self.prev_voxel_density = current_voxel_density_normalized
    
############################################################################################################
    def get_train_loss_dict(self, step: int):
        #Creates samples such as [100, 135,.....,22150, 29999], Only compiles one time at the start of the training.
        if step == 0:
            first_step = self.model.config.first_step #first step 
            num_partitions= self.model.config.num_partitions #number of partitions
            last_step = self.model.config.max_num_iterations_in_pipeline 
            self.samples = self.incremental_sampling(first_step, num_partitions, last_step)
            #################################################################################
            num_images = len(self.datamanager.train_dataset.cameras)  # Number of images in training dataset
            height = self.datamanager.train_dataset.cameras[0].height  #Height of first image (assumes all images have same height)
            width = self.datamanager.train_dataset.cameras[0].width # Width of first image (assumes all images have same width)
            self.config.mask_pixel = torch.ones((num_images, height, width), dtype=torch.int)

        # Get the next dataset from training datas
        if self.world_size > 1 and step:
            assert self.datamanager.train_sampler is not None
            self.datamanager.train_sampler.set_epoch(step)

        ray_bundle, batch = self.datamanager.next_train(step, self.config.mask_pixel)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        output_directory = Path("E:/Berk/code/reconstruction/outputs/outputs_pointcloud")  
        self.generate_and_save_point_cloud(step, output_directory, self.samples) #IN THE END WE ALSO UPDATE THE DIFFERENCE VOXEL IN OUR MODEL, SO THIS FUNCTION IS IMPORTANT

        if step > self.samples[1] + 1 and self.datamanager.config.implement_masking is True:
            self.config.mask_pixel = reconstructionModel.map_rays_to_voxels(self._model, model_outputs, ray_bundle, self.config.mask_pixel)
            total_zeros = (self.config.mask_pixel == 0).sum().item()

        if self.config.decrease_coarse_sampling is True:
            #logaritmically decreases coarse sampling according to voxel means
            #Decreasing rate is significantly increased if voxel mean getting closer to zero, but decreasing rate is still remains low if mean value is higher than 0.1
            self.train_step(step, self.samples) 

        return model_outputs, loss_dict, metrics_dict