"""Score a COLMAP point cloud on the same held-out views as the NeRF/3DGS runs (PSNR / SSIM / LPIPS).

Run with the nerfstudio environment (envs/nerfstudio). The point cloud is rendered with the z-buffered point
splatter from render_colmap_flythrough.py at the nerfstudio dataset resolution, from every --eval-interval'th
registered image (sorted by name, index % 8 == 0, the views nerfstudio holds out), and compared against the
dataset images composited onto the same background, with the metrics nerfstudio's Nerfacto uses (torchmetrics
PSNR, SSIM and LPIPS with normalize=True).

Caveat: COLMAP's dense fusion used every image, including these held-out ones, so this is not a true novel-view
test for COLMAP. Re-run patch_match_stereo / stereo_fusion without them for a strictly fair number.

Writes <output>/metrics.json (same layout as ns-eval) and <output>/comparisons/eval_img_XXXX.png (ground truth |
render).

Example:
    python scripts/eval_colmap_views.py --model colmap/bear-mask/dense/0/sparse \
        --ply colmap/bear-mask/dense/0/fused.ply --images data/nerfstudio/bear-mask/images \
        --output results/colmap/bear-mask/eval
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from PIL import Image
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.data.utils.colmap_parsing_utils import qvec2rotmat, read_images_binary
from render_colmap_flythrough import PointRenderer, intrinsics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="Undistorted COLMAP model dir (cameras.bin, images.bin)")
    parser.add_argument("--ply", type=Path, required=True, help="Point cloud in the same frame, e.g. dense fused.ply")
    parser.add_argument("--images", type=Path, required=True, help="Nerfstudio dataset images (ground truth)")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--max-size", type=int, default=1600, help="Longest side of the nerfstudio dataset (default: 1600)")
    parser.add_argument("--eval-interval", type=int, default=8, help="As nerfstudio-data --eval-interval (default: 8)")
    parser.add_argument("--point-size", type=float, default=2.0, help="Rendered point size in pixels (default: 2)")
    parser.add_argument("--background", type=float, nargs=3, default=[1, 1, 1], help="RGB in 0-1 (default: white)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    K, w, h = intrinsics(args.model, args.max_size, downscale=1)
    pcd = o3d.io.read_point_cloud(str(args.ply))
    renderer = PointRenderer(np.asarray(pcd.points), np.asarray(pcd.colors), K, w, h, args.point_size,
                             args.background, device)

    images = sorted(read_images_binary(args.model / "images.bin").values(), key=lambda im: im.name)
    eval_images = [im for i, im in enumerate(images) if i % args.eval_interval == 0]
    gt_files = {p.stem: p for p in args.images.iterdir()}

    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
    lpips_fn = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
    bg = np.array(args.background, dtype=np.float32)
    (args.output / "comparisons").mkdir(parents=True, exist_ok=True)

    per_view, times = [], []
    for idx, im in enumerate(eval_images):
        R = qvec2rotmat(im.qvec)
        c2w = np.hstack([R.T, (-R.T @ im.tvec)[:, None]])
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = renderer.render(c2w)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

        gt = Image.open(gt_files[Path(im.name).stem])
        assert gt.size == (w, h), f"{im.name}: ground truth {gt.size} != render {(w, h)}"
        gt = np.asarray(gt.convert("RGBA"), dtype=np.float32) / 255
        gt = gt[..., :3] * gt[..., 3:] + bg * (1 - gt[..., 3:])  # composite onto the background, as nerfstudio
        pred_f = pred.astype(np.float32) / 255

        g = torch.tensor(gt, device=device).permute(2, 0, 1)[None]
        p = torch.tensor(pred_f, device=device).permute(2, 0, 1)[None]
        per_view.append({
            "image": im.name,
            "psnr": float(psnr_fn(g, p)),
            "ssim": float(structural_similarity_index_measure(g, p)),
            "lpips": float(lpips_fn(g, p)),
        })
        Image.fromarray(np.hstack([np.round(gt * 255).astype(np.uint8), pred])).save(
            args.output / "comparisons" / f"eval_img_{idx:04d}.png")
        print(f"{im.name}: PSNR {per_view[-1]['psnr']:.2f}  SSIM {per_view[-1]['ssim']:.3f}  LPIPS {per_view[-1]['lpips']:.3f}")

    results = {}
    for k in ("psnr", "ssim", "lpips"):
        v = np.array([r[k] for r in per_view])
        results[k], results[f"{k}_std"] = float(v.mean()), float(v.std())
    fps = 1 / np.array(times[1:] or times)  # skip the first render (CUDA warm-up)
    results["fps"], results["fps_std"] = float(fps.mean()), float(fps.std())
    out = {
        "experiment_name": args.ply.parent.name,
        "method_name": "colmap-points",
        "point_cloud": str(args.ply),
        "num_points": len(pcd.points),
        "resolution": [w, h],
        "point_size": args.point_size,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "results": results,
        "per_view": per_view,
    }
    (args.output / "metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
