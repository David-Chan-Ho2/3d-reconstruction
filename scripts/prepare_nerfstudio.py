"""Convert an undistorted COLMAP model into a Nerfstudio dataset (transforms.json + images).

Run with the nerfstudio environment (envs/nerfstudio). Source images and the COLMAP
model are only read; everything is written to --output.

Optionally attaches foreground masks as an alpha channel (RGBA PNG). Masks are expected
in the *distorted* raw-image frame (as produced by scripts/create_masks.py), so they are
warped through the raw SIMPLE_RADIAL camera into the undistorted PINHOLE camera.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from nerfstudio.data.utils.colmap_parsing_utils import read_cameras_binary
from nerfstudio.process_data.colmap_utils import colmap_to_json


def undistort_mask(mask, raw_cam, dense_cam):
    """Warp a mask from the raw (SIMPLE_RADIAL) frame into the undistorted (PINHOLE) frame."""
    assert raw_cam.model == "SIMPLE_RADIAL" and dense_cam.model == "PINHOLE"
    f, cx0, cy0, k = raw_cam.params
    fx, fy, cx, cy = dense_cam.params
    # Pixel centres sit at integer + 0.5 in COLMAP's continuous image coordinates.
    u, v = np.meshgrid(np.arange(dense_cam.width) + 0.5, np.arange(dense_cam.height) + 0.5)
    x, y = (u - cx) / fx, (v - cy) / fy
    d = 1 + k * (x * x + y * y)
    map_x = (f * x * d + cx0 - 0.5).astype(np.float32)
    map_y = (f * y * d + cy0 - 0.5).astype(np.float32)
    return cv2.remap(mask, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="Undistorted COLMAP model dir (cameras.bin, images.bin, points3D.bin)")
    parser.add_argument("--images", type=Path, required=True, help="Undistorted images matching --model")
    parser.add_argument("--output", type=Path, required=True, help="Output dataset directory")
    parser.add_argument("--max-size", type=int, default=1600, help="Longest image side after downscaling (default: 1600)")
    parser.add_argument("--masks", type=Path, help="Masks in the raw frame, named <image name>.png")
    parser.add_argument("--raw-model", type=Path, help="Distorted COLMAP model the masks belong to (required with --masks)")
    args = parser.parse_args()
    if args.masks and not args.raw_model:
        parser.error("--masks requires --raw-model")

    out_images = args.output / "images"
    out_images.mkdir(parents=True, exist_ok=False)  # refuse to mix into an existing dataset

    dense_cam = read_cameras_binary(args.model / "cameras.bin")[1]
    raw_cam = read_cameras_binary(args.raw_model / "cameras.bin")[1] if args.masks else None

    names = sorted(p.name for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    rename = {n: (Path(n).stem + ".png" if args.masks else n) for n in names}

    # transforms.json + sparse_pc.ply at the model's native resolution.
    n_registered = colmap_to_json(args.model, args.output, image_rename_map=rename)
    meta_path = args.output / "transforms.json"
    meta = json.loads(meta_path.read_text())

    w, h = meta["w"], meta["h"]
    scale = min(1.0, args.max_size / max(w, h))
    new_w, new_h = round(w * scale), round(h * scale)
    sx, sy = new_w / w, new_h / h
    meta.update(w=new_w, h=new_h, fl_x=meta["fl_x"] * sx, fl_y=meta["fl_y"] * sy, cx=meta["cx"] * sx, cy=meta["cy"] * sy)
    meta_path.write_text(json.dumps(meta, indent=4))

    for name in names:
        img = Image.open(args.images / name).convert("RGB")
        assert img.size == (w, h), f"{name}: {img.size} != model size {(w, h)}"
        if args.masks:
            mask = np.array(Image.open(args.masks / f"{name}.png").convert("L"))
            assert mask.shape == (raw_cam.height, raw_cam.width), f"{name}: mask size does not match raw camera"
            alpha = Image.fromarray(undistort_mask(mask, raw_cam, dense_cam))
            img = img.resize((new_w, new_h), Image.LANCZOS)
            alpha = alpha.resize((new_w, new_h), Image.BILINEAR)
            img.putalpha(alpha)
            img.save(out_images / rename[name])
        else:
            img.resize((new_w, new_h), Image.LANCZOS).save(out_images / rename[name], quality=95)

    print(f"{n_registered} registered frames, {len(names)} images -> {args.output} at {new_w}x{new_h}")


if __name__ == "__main__":
    main()
