"""
render.py

A script for rendering Gaussian Splats.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import imageio.v2 as imageio
import numpy as np
from plyfile import PlyData
import torch
import torchvision.utils as tvu
from tqdm import tqdm
import tyro

from src.camera import Camera
from src.renderer import GSRasterizer
from src.scene import Scene


@dataclass
class Args:
    
    scene_type: Literal["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"] = "lego"
    """Type of scene to render."""
    device_type: Literal["cpu", "cuda"] = "cuda"
    """Device to use for rendering."""

    out_root: Path = Path("./outputs")
    """Root directory for saving outputs."""

def main(args: Args):

    out_dir = args.out_root / args.scene_type
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to: {str(out_dir)}")

    device = torch.device(args.device_type)
    print(f"Using device: {device}")

    print(f"Loading Scene: {args.scene_type}")
    scene = load_scene(args.scene_type, device)
    print("Loaded Scene.")

    # Load camera data
    (
        c2ws, proj_mat, fov, focal, near, far, img_width, img_height
    ) = load_camera(device)
    print("Loaded Camera Data.")

    # Initialize renderer
    renderer = GSRasterizer(active_sh_degree=3)
    print("Initialized Renderer.")

    # Render images
    w = imageio.get_writer(
        out_dir / "video.mp4",
        format="FFMPEG",
        mode="I",
        fps=24,
        macro_block_size=1,
    )

    for view_idx, c2w in tqdm(enumerate(c2ws)):

        # Setup camera
        c2w_ = torch.from_numpy(c2w).float().to(device)
        proj_mat_ = proj_mat.float().to(device)
        cam = Camera(
            camera_to_world=c2w_, proj_mat=proj_mat_, cam_center=c2w_[:3, 3],
            fov_x=fov, fov_y=fov, near=near, far=far, image_width=img_width, image_height=img_height,
            f_x=focal, f_y=focal,
        )

        # Render
        img, _, _ = renderer.render_scene(scene, cam)
        img = img.reshape(img_height, img_width, 3)
        img = torch.clamp(img, 0.0, 1.0)

        # Record images for video
        w.append_data((img.cpu().numpy() * 255).astype(np.uint8))

        # Save individual frame
        out_path = out_dir / f"img_{view_idx:03d}.png"
        img = img.permute(2, 0, 1)
        tvu.save_image(img, out_path)

def load_camera(device):
    camera_poses = np.load("./data/cam_data.npz")
    c2ws = camera_poses["poses"]
    img_height, img_width = int(camera_poses["height"]), int(camera_poses["width"])
    focal = torch.from_numpy(camera_poses["focal"]).float().to(device)
    img_height *= 2
    img_width *= 2
    focal *= 2
    near = 1e-2
    far = 10.0
    fov = focal2fov(focal, img_width)
    proj_mat = compute_proj_mat(near, far, fov, fov)
    return c2ws, proj_mat, fov, focal, near, far, img_width, img_height

def load_scene(scene_type, device):
    ply_path = Path(f"./data/{scene_type}.ply")
    assert ply_path.exists(), f"Path {ply_path} does not exist."

    # Load splats from ply file
    mean_3d, shs, opacities, scales, rotations = load_ply(ply_path)
    mean_3d = mean_3d.to(device)
    shs = shs.to(device)
    opacities = opacities.to(device)
    scales = scales.to(device)
    rotations = rotations.to(device)
    assert torch.all(opacities >= 0) and torch.all(opacities <= 1)
    assert torch.all(scales >= 0), f"Scale has negative values: {scales.min()}"
    assert torch.allclose(torch.norm(rotations, dim=1), torch.ones_like(torch.norm(rotations, dim=1)))

    scene = Scene(
        mean_3d=mean_3d,
        shs=shs,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
    )

    return scene

def load_ply(path):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    # features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
    features_extra = features_extra.reshape((features_extra.shape[0], 3, -1))

    features_dc = features_dc.transpose(0, 2, 1)
    features_extra = features_extra.transpose(0, 2, 1)
    shs = np.concatenate([features_dc, features_extra], axis=1)

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # Apply activations
    xyz = torch.from_numpy(xyz).float()
    shs = torch.from_numpy(shs).float()
    opacities = torch.sigmoid(torch.tensor(opacities, dtype=torch.float))
    scales = torch.exp(torch.tensor(scales, dtype=torch.float))
    rots = torch.nn.functional.normalize(torch.tensor(rots, dtype=torch.float))

    return xyz, shs, opacities, scales, rots

def focal2fov(focal, pixels):
    return 2.0 * torch.atan(pixels/(2*focal))

def fov2focal(fov, pixels):
    return pixels / (2.0 * torch.tan(fov / 2.0))

def compute_inverse_pose(pose):
    R = pose[:3, :3]
    t = pose[:3, 3:4]

    inv_R = R.T
    inv_t = -inv_R @ t
    inv_pose = np.concatenate([inv_R, inv_t], axis=1)
    assert inv_pose.shape == pose.shape, f"Inverse pose has wrong shape {inv_pose.shape}. Expected {pose.shape}."
    return inv_pose

def compute_proj_mat(near, far, fov_x, fov_y):
    tanHalfFovY = torch.tan((fov_y / 2))
    tanHalfFovX = torch.tan((fov_x / 2))

    top = tanHalfFovY * near
    bottom = -top
    right = tanHalfFovX * near
    left = -right

    proj_mat = torch.zeros(4, 4).to(fov_x.device)

    z_sign = 1.0

    proj_mat[0, 0] = 2.0 * near / (right - left)
    proj_mat[1, 1] = 2.0 * near / (top - bottom)
    proj_mat[0, 2] = (right + left) / (right - left)
    proj_mat[1, 2] = (top + bottom) / (top - bottom)
    proj_mat[3, 2] = z_sign
    proj_mat[2, 2] = z_sign * far / (far - near)
    proj_mat[2, 3] = -(far * near) / (far - near)
    
    return proj_mat


if __name__ == "__main__":
    main(
        tyro.cli(Args, )
    )
