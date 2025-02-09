
from dataclasses import dataclass

import torch


@dataclass
class Camera:

    camera_to_world: torch.Tensor
    """Camera to World matrix"""
    proj_mat: torch.Tensor
    """Projection matrix"""
    cam_center: torch.Tensor
    """Camera center in world coordinates"""

    f_x: float
    """Focal length in x direction"""
    f_y: float
    """Focal length in y direction"""
    c_x: float
    """Principal point in x direction"""
    c_y: float
    """Principal point in y direction"""
    fov_x: float
    """Field of view in x direction"""
    fov_y: float
    """Field of view in y direction"""
    near: float
    """Near plane distance"""
    far: float
    """Far plane distance"""
    image_width: int
    """Image width"""
    image_height: int
    """Image height"""
