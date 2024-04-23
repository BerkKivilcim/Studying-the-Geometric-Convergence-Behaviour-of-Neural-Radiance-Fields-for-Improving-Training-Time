"""
Template Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
from dataclasses import dataclass, field
from typing import Type

from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig  # for subclassing Nerfacto model
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model

from nerfstudio.model_components.renderers import DepthRenderer
import torch
from nerfstudio.cameras.rays import RaySamples, RayBundle

from nerfstudio.fields.nerfacto_field import NerfactoField
import numpy as np
from pathlib import Path
from nerfstudio.field_components.spatial_distortions import SceneContraction
from torch.nn import Parameter
import nerfacc
from nerfstudio.model_components.ray_samplers import VolumetricSampler
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler, Sampler, UniformLinDispPiecewiseSampler
from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples

##############################################################################################
@dataclass
class reconstructionModelConfig(NerfactoModelConfig):
    """Template Model Configuration.

    Add your custom model config parameters here.
    """
    converge_threshold: float = field(default_factory=lambda: 0.0001)

    num_proposal_samples_per_ray: (256, 96)
    num_nerf_samples_per_ray: int = 48

    voxel_resolution: int = field(default_factory=lambda:300)
    difference_voxel = None
    camera_pixels_for_voxel = []

    #Below 3 parameters used in pipeline
    first_step: int = field(default_factory=lambda: 100)
    num_partitions: int = field(default_factory=lambda: 20)
    max_num_iterations_in_pipeline: int = field(default_factory=lambda: 30000)

    _target: Type = field(default_factory=lambda: reconstructionModel)
    depth_method: str = "median"
    depth_output_name: str = "custom_depth_render"
###########################################################################################
class CustomDepthRenderer(DepthRenderer):
    def __init__(self, method="median"):
        super().__init__(method)
        
    def forward(self, weights, ray_samples:RaySamples):
        if self.method == "custom_depth_render":
            return self.calculate_alpha_depth(weights, ray_samples)
        else:
            return super().forward(weights, ray_samples)

    @staticmethod
    def calculate_alpha_depth(weights, ray_samples:RaySamples):
        """Hesapla ve dön alpha değeri en yüksek olan noktanın derinliğini"""
        alpha_weights = weights[..., -1]  # gets alphas
        max_alpha_indices = torch.argmax(alpha_weights, dim=-2)  # finds largest alpha 
        steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
        alpha_depth = torch.gather(steps, dim=-2, index=max_alpha_indices)  # finds depth
        return alpha_depth
##########################################################################################

class reconstructionModel(NerfactoModel):
    """Template Model."""
    config: reconstructionModelConfig
    def populate_modules(self):
        super().populate_modules()
        self.renderer_depth = CustomDepthRenderer(method=self.config.depth_method) #Initial Value
        self.create_samplers() #When the training starts, defines the Sampler with initial parameters

    def create_samplers(self): #Updates or Initialize the Sampler with Parameters
        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=lambda step: 1,
            initial_sampler=None,
        )
        
    def update_sampler_config(self, new_num_prop_samples, new_num_nerf_samples): #Updates Sampler Parameters
        self.config.num_proposal_samples_per_ray = new_num_prop_samples
        self.config.num_nerf_samples_per_ray = new_num_nerf_samples
        self.create_samplers() #Calls the sampler again with updated parameters

    #####################################################################################################
    ###############  CREATES DENSITY VOXELS  ##############
    #####################################################################################################
    def voxel_density_values(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        reso = self.config.voxel_resolution + 1 # Adjust for voxel center calculation
        scene_aabb = self.scene_box.aabb.flatten()
        aabb_min, aabb_max = torch.tensor(scene_aabb[:3], device=device, dtype=torch.float32), torch.tensor(scene_aabb[3:], device=device, dtype=torch.float32)

        # Adjusts for voxel center calculation. Creates coordinates for each voxel's center in scene box (-1,1)
        x = torch.linspace(aabb_min[0], aabb_max[0], steps=reso, device=device)[:-1] + (aabb_max[0] - aabb_min[0]) / (reso * 2)
        y = torch.linspace(aabb_min[1], aabb_max[1], steps=reso, device=device)[:-1] + (aabb_max[1] - aabb_min[1]) / (reso * 2)
        z = torch.linspace(aabb_min[2], aabb_max[2], steps=reso, device=device)[:-1] + (aabb_max[2] - aabb_min[2]) / (reso * 2)
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")

        voxel_centers = torch.stack((grid_x, grid_y, grid_z), dim=-1).reshape(-1, 3)  # Voxel centers

        # Puts the voxel center points into model and finds density values for each voxel center
        densities_tensor = self.field.density_fn(voxel_centers).reshape(reso-1, reso-1, reso-1)

        # Convert densities tensor to numpy array
        densities = densities_tensor.detach().cpu().numpy()

        return densities

    ###############NOT HIGHEST DENSITY BUT MEDIAN DEPTH######################
    def map_rays_to_voxels(self, outputs, ray_bundle, mask_pixel):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        voxel_resolution = self.config.voxel_resolution
        #Find the x,y,z coordinates (in the coordinate system of scene_box) of each sample on a ray. Index of [-1] is used because of considering the final iteration samples from the proposal network
        sample_coordinates = outputs['ray_samples_list'][-1].frustums.get_positions() #Size of (4096,48,3) >> (Batch size(number of rays), Number of samples on a ray, xyz coordinates)
        sample_weights = outputs['weights_list'][-1] #Weights are created with using density values. It is a normalized version of density values.
        
        cumulative_weights = torch.cumsum(sample_weights, dim=1)
        mask = cumulative_weights >= 0.5
        int_mask = mask.long()
        first_pass_indices = torch.argmax(int_mask, dim=1, keepdim=True)
        selected_positions = torch.gather(sample_coordinates, 1, first_pass_indices.expand(-1, -1, 3))
        
        #FIND VOXEL COORDINATES OF EACH SAMPLE ON THE RAYS (4096,1,3)
        scene_aabb = self.scene_box.aabb.flatten()
        aabb_min, aabb_max = torch.tensor(scene_aabb[:3], device=device, dtype=torch.float32), torch.tensor(scene_aabb[3:], device=device, dtype=torch.float32)
        voxel_size = (aabb_max - aabb_min) / voxel_resolution  # The range is [-1, 1], so total should be length is 2
        voxel_indices = ((selected_positions + 1.0) / voxel_size).long()  # Normalize to [0, voxel_resolution]
        voxel_indices = torch.clamp(voxel_indices, 0, voxel_resolution - 1) 
        voxel_indices.shape

        #FIND THE CAMERA INDEX AND PIXEL COORDINATES OF EACH RAY THAT VOXEL VALUES SMALLER THAN A THRESHOLD(4096,1,3)
        # Extract voxel XYZ coordinates separately 
        voxel_x, voxel_y, voxel_z = voxel_indices[:, 0, :].T        
        camera_indices = ray_bundle.camera_indices
        pixel_coords = ray_bundle.coords.floor().long()

        #Move the difference voxels into GPU for fast implementation and also voxel_x, voxel_y, voxel_z values are in GPU too. 
        difference_voxel_tensor = torch.tensor(self.config.difference_voxel, device=device, dtype=torch.float32)

        mask = difference_voxel_tensor[voxel_x, voxel_y, voxel_z] < self.config.converge_threshold #find the voxel coordinates where the value of voxel below the threshhold
        
        #Finds the pixel informations (camera index and pixel_x, pixel_y) which is not see any voxel region with above a threshhold
        filtered_camera_indices = camera_indices[mask] 
        filtered_pixel_coords = pixel_coords[mask]
        result = torch.cat([filtered_camera_indices, filtered_pixel_coords], dim=1) #(x,3) >> (x is number of pixels, 3 for (camera index, pixel_x, pixel_y))

        num_images, height, width = mask_pixel.shape
        offsets = torch.tensor([[0],[0]], device=device) #Defines patches for 1x1, 3x3...
        y_offsets, x_offsets = torch.meshgrid(offsets[0], offsets[1], indexing='ij')
        y_offsets, x_offsets = y_offsets.flatten(), x_offsets.flatten()
        result_y, result_x = result[:, 1], result[:, 2]
        result_y, result_x = result_y[:, None], result_x[:, None]
        neighbor_y_indices = result_y + y_offsets
        neighbor_x_indices = result_x + x_offsets

        repeated_camera_indices = result[:,0].repeat_interleave(1) #Do not forget to fix here, 9 element for 3x3 patch
        flat_neighbor_y_indices = neighbor_y_indices.view(-1)
        flat_neighbor_x_indices = neighbor_x_indices.view(-1)
        # Ensure indices are within the image dimensions and clamp if necessary
        neighbor_y_indices = torch.clamp(flat_neighbor_y_indices, 0, height-1)
        neighbor_x_indices = torch.clamp(flat_neighbor_x_indices, 0, width-1)

        mask_pixel = torch.tensor(mask_pixel, device=device)
        mask_pixel[repeated_camera_indices, neighbor_y_indices, neighbor_x_indices] = 0
        return mask_pixel
    #############################################################################################################           
    def get_outputs(self, ray_bundle:RayBundle):
        outputs = super().get_outputs(ray_bundle)
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)

        #For CUSTOM DEPTH. It is UNNECESSARY FOR NOW.
        custom_depth = self.renderer_depth(weights=weights_list[-1], ray_samples=ray_samples)
        outputs[self.config.depth_output_name] = custom_depth

        return outputs
    # TODO: Override any potential functions/methods to implement your own method
    # or subclass from "Model" and define all mandatory fields.
