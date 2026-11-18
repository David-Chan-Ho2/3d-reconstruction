"""Render a fly-through video of a COLMAP point cloud along the same camera path as `ns-render interpolate`.

Run with the nerfstudio environment (envs/nerfstudio). The NeRF/3DGS notebooks render with

    ns-render interpolate --pose-source train --order-poses True --interpolation-steps 3 --frame-rate 24
                          --downscale-factor 2

on a dataset made by scripts/prepare_nerfstudio.py (longest side 1600, every 8th image held out). This script
rebuilds that path from the COLMAP model itself:

  1. training views = registered images sorted by name, dropping every --eval-interval'th (index % 8 == 0),
  2. ordered greedily by nearest camera centre, starting from the first one,
  3. 3 poses per consecutive pair (rotation slerp + linear position), using nerfstudio's own functions.

Nerfstudio only applies one global rotation, uniform scale and translation to the poses, and every step above is
unaffected by that, so the path is the same as the NeRF/3DGS videos, expressed in COLMAP's coordinate frame,
which is the frame the dense point cloud (fused.ply) is in. Frames are rendered at the same resolution and
intrinsics, so the three videos line up frame for frame.

Example:
    python scripts/render_colmap_flythrough.py --model colmap/bear-mask/dense/0/sparse \
        --ply colmap/bear-mask/dense/0/fused.ply --output results/colmap/bear-mask/flythrough.mp4
"""

import argparse
from pathlib import Path

import mediapy as media
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

from nerfstudio.cameras.camera_utils import get_interpolated_poses_many
from nerfstudio.data.utils.colmap_parsing_utils import qvec2rotmat, read_cameras_binary, read_images_binary


def camera_path(model: Path, eval_interval: int, steps: int):
    """Interpolated camera-to-world poses (N, 3, 4) in COLMAP's frame (OpenCV axes), plus the number of train views."""
    images = sorted(read_images_binary(model / "images.bin").values(), key=lambda im: im.name)
    train = [im for i, im in enumerate(images) if i % eval_interval != 0]
    c2w = []
    for im in train:
        R = qvec2rotmat(im.qvec)  # world -> camera
        c2w.append(np.hstack([R.T, (-R.T @ im.tvec)[:, None]]))
    poses = torch.tensor(np.stack(c2w), dtype=torch.float32)
    Ks = torch.eye(3).expand(len(train), 3, 3)  # single shared camera; intrinsics are handled separately
    path, _ = get_interpolated_poses_many(poses, Ks, steps_per_transition=steps, order_poses=True)
    return path.numpy().astype(np.float64), len(train)


def intrinsics(model: Path, max_size: int, downscale: float):
    """Intrinsics at the resolution nerfstudio renders: prepare_nerfstudio.py resize, then --downscale-factor."""
    cams = read_cameras_binary(model / "cameras.bin")
    assert len(cams) == 1, "expected a single shared camera"
    cam = next(iter(cams.values()))
    assert cam.model == "PINHOLE", f"expected an undistorted (PINHOLE) model, got {cam.model}"
    fx, fy, cx, cy = cam.params
    # prepare_nerfstudio.py: longest side -> max_size, rounded, intrinsics scaled per axis.
    s = min(1.0, max_size / max(cam.width, cam.height))
    w, h = round(cam.width * s), round(cam.height * s)
    sx, sy = w / cam.width, h / cam.height
    # Cameras.rescale_output_resolution: sizes truncated, intrinsics scaled.
    r = 1.0 / downscale
    w, h = int(w * r), int(h * r)
    K = np.array([[fx * sx * r, 0, cx * sx * r], [0, fy * sy * r, cy * sy * r], [0, 0, 1]])
    return K, w, h


class PointRenderer:
    """Z-buffered point splatting with a pinhole camera (OpenCV/COLMAP convention), in PyTorch.

    Each point is drawn as a point_size x point_size pixel square and the nearest point wins each pixel, the
    same look as a GL point cloud viewer. (Open3D's offscreen renderer was tried first, but on Windows/Vulkan it
    returns empty frames after a few renders.)
    """

    def __init__(self, points, colors, K, width, height, point_size, background, device):
        self.xyz = torch.tensor(points, dtype=torch.float32, device=device)
        rgb = colors if len(colors) else np.full_like(points, 0.5)
        self.rgb = torch.tensor(np.round(rgb * 255), dtype=torch.uint8, device=device)
        self.fx, self.fy, self.cx, self.cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        self.w, self.h = width, height
        s = max(1, round(point_size))
        r = torch.arange(s, device=device) - (s - 1) // 2
        self.offsets = torch.cartesian_prod(r, r)  # (s*s, 2) pixel offsets (dx, dy)
        self.bg = torch.tensor(np.round(np.array(background) * 255), dtype=torch.uint8, device=device)

    @torch.no_grad()
    def render(self, c2w):
        R = torch.tensor(c2w[:, :3], dtype=torch.float32, device=self.xyz.device)
        t = torch.tensor(c2w[:, 3], dtype=torch.float32, device=self.xyz.device)
        cam = (self.xyz - t) @ R  # world -> camera: R^T (x - t)
        z = cam[:, 2]
        front = z > 1e-6
        cam, z, rgb = cam[front], z[front], self.rgb[front]
        # COLMAP pixel centres sit at integer + 0.5, so pixel index = floor(projection).
        u = torch.floor(self.fx * cam[:, 0] / z + self.cx).long()
        v = torch.floor(self.fy * cam[:, 1] / z + self.cy).long()

        n = self.w * self.h
        pix, depth, col = [], [], []
        for dx, dy in self.offsets.tolist():
            uu, vv = u + dx, v + dy
            ok = (uu >= 0) & (uu < self.w) & (vv >= 0) & (vv < self.h)
            pix.append(vv[ok] * self.w + uu[ok])
            depth.append(z[ok])
            col.append(rgb[ok])
        pix, depth, col = torch.cat(pix), torch.cat(depth), torch.cat(col)

        zbuf = torch.full((n,), float("inf"), device=z.device).scatter_reduce_(0, pix, depth, "amin")
        nearest = depth <= zbuf[pix]
        img = self.bg.expand(n, 3).clone()
        img[pix[nearest]] = col[nearest]
        return img.view(self.h, self.w, 3).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="Undistorted COLMAP model dir (cameras.bin, images.bin)")
    parser.add_argument("--ply", type=Path, required=True, help="Point cloud in the same frame, e.g. dense fused.ply")
    parser.add_argument("--output", type=Path, required=True, help="Output .mp4 (or .png with --preview)")
    parser.add_argument("--max-size", type=int, default=1600, help="Longest side of the nerfstudio dataset (default: 1600)")
    parser.add_argument("--downscale-factor", type=float, default=2, help="As ns-render --downscale-factor (default: 2)")
    parser.add_argument("--interpolation-steps", type=int, default=3, help="As ns-render (default: 3)")
    parser.add_argument("--frame-rate", type=int, default=24, help="As ns-render (default: 24)")
    parser.add_argument("--eval-interval", type=int, default=8, help="As nerfstudio-data --eval-interval (default: 8)")
    parser.add_argument("--point-size", type=float, default=2.0, help="Rendered point size in pixels (default: 2)")
    parser.add_argument("--background", type=float, nargs=3, default=[1, 1, 1], help="RGB in 0-1 (default: white, as the notebooks)")
    parser.add_argument("--preview", type=int, metavar="FRAME", help="Only render this frame index to a PNG")
    args = parser.parse_args()

    path, n_train = camera_path(args.model, args.eval_interval, args.interpolation_steps)
    K, w, h = intrinsics(args.model, args.max_size, args.downscale_factor)
    pcd = o3d.io.read_point_cloud(str(args.ply))
    assert pcd.has_points(), f"no points in {args.ply}"
    print(f"{len(pcd.points)} points | {n_train} train views -> {len(path)} frames at {w}x{h}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    renderer = PointRenderer(np.asarray(pcd.points), np.asarray(pcd.colors), K, w, h, args.point_size,
                             args.background, device)
    render = renderer.render

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.preview is not None:
        media.write_image(args.output, render(path[args.preview]))
        print("wrote", args.output)
        return

    # ns-render sets fps = frames / (steps * train_views / frame_rate), so match it for identical timing.
    fps = len(path) / (args.interpolation_steps * n_train / args.frame_rate)
    # yuv420p (plays in browsers) needs even dimensions: drop the odd last row/column if there is one.
    eh, ew = h - h % 2, w - w % 2
    with media.VideoWriter(args.output, shape=(eh, ew), fps=fps, encoded_format="yuv420p") as writer:
        for c2w in tqdm(path, desc="rendering"):
            writer.add_image(render(c2w)[:eh, :ew])
    print(f"wrote {args.output} ({len(path)} frames, {fps:.2f} fps, {ew}x{eh})")


if __name__ == "__main__":
    main()
