"""
evaluate.py

A script for evaluating the quality of rendered images.
"""

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
        #metrics["ssim"] = compute_ssim_between_directories(out_dir, ref_dir)

        lpips_avg += metrics["lpips"]
        psnr_avg += metrics["psnr"]
        #ssim_avg += metrics["ssim"]

        metrics_list.append(metrics)

    # Compute average
    lpips_avg /= len(scene_types)
    psnr_avg /= len(scene_types)
    #ssim_avg /= len(scene_types)
    metrics_list.append({
        "scene": "average",
        "lpips": lpips_avg,
        "psnr": psnr_avg,
    })

    # Save metrics to CSV
    print(f"LPIPS: {lpips_avg:.4f}")
    print(f"PSNR: {psnr_avg:.4f}")
    # print(f"SSIM: {ssim_avg:.4f}")

    # Save per-scene metrics and averages to evaluation.txt
    with open("evaluation.txt", "w") as f:
        for metrics in metrics_list:
            f.write(
                f"scene: {metrics['scene']}, "
                f"lpips: {metrics['lpips']:.4f}, "
                f"psnr: {metrics['psnr']:.4f}\n"
            )
    print("Done.")



if __name__ == "__main__":
    main()
