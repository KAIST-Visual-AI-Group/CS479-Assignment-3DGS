
from dataclasses import dataclass

from jaxtyping import Shaped
import torch


@dataclass
class Scene:

    mean_3d: Shaped[torch.Tensor, "N 3"]
    """Mean 3D points of splats."""
    shs: Shaped[torch.Tensor, "N K 3"]
    """SH coefficients of splats."""
    opacities: Shaped[torch.Tensor, "N 1"]
    """Opacity of splats."""
    scales: Shaped[torch.Tensor, "N 3"]
    """Scale of splats."""
    rotations: Shaped[torch.Tensor, "N 4"]
    """Rotation of splats represented as quaternions."""
