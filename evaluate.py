"""
evaluate.py

A script for evaluating the quality of rendered images.
"""

import csv
from pathlib import Path

from tqdm import tqdm

from src.rgb_metrics import (
    compute_lpips_between_directories,
    compute_psnr_between_directories,
    compute_ssim_between_directories,
)

def main() -> None:

    # Directory containing rendered images
    ref_root = Path("data/nerf_synthetic")
    out_root = Path("outputs")
    scene_types = ["chair", "lego", "materials", "drums"]
    
    # Evaluate
    metrics_list = []

    lpips_avg = 0.0
    psnr_avg = 0.0
    ssim_avg = 0.0

    for scene_type in tqdm(scene_types):
        metrics = {}

        ref_dir = ref_root / scene_type / "test"
        out_dir = out_root / scene_type
        assert ref_dir.exists(), f"Scene {scene_type} not found."
        assert out_dir.exists(), f"Scene {scene_type} not found."
        print(f"Evaluating scene: {scene_type}")

        metrics["scene"] = scene_type
        metrics["lpips"] = compute_lpips_between_directories(out_dir, ref_dir)
        metrics["psnr"] = compute_psnr_between_directories(out_dir, ref_dir)
        metrics["ssim"] = compute_ssim_between_directories(out_dir, ref_dir)

        lpips_avg += metrics["lpips"]
        psnr_avg += metrics["psnr"]
        ssim_avg += metrics["ssim"]

        metrics_list.append(metrics)

    # Compute average
    lpips_avg /= len(scene_types)
    psnr_avg /= len(scene_types)
    ssim_avg /= len(scene_types)
    metrics_list.append({
        "scene": "average",
        "lpips": lpips_avg,
        "psnr": psnr_avg,
        "ssim": ssim_avg
    })

    # Save metrics to CSV
    csv_file = "./metrics.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["scene", "lpips", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(metrics_list)

    print(f"Metrics saved to {csv_file}")



if __name__ == "__main__":
    main()
