import argparse
import glob
import json
import os
import shutil

import numpy as np
import torch
import trimesh

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def parse_args():
    parser = argparse.ArgumentParser(description="Export VGGT predictions as a colored PLY point cloud.")
    parser.add_argument("--image_folder", required=True, help="Folder containing input images.")
    parser.add_argument("--output", required=True, help="Output .ply path.")
    parser.add_argument(
        "--conf_percentile",
        type=float,
        default=25.0,
        help="Drop this percentage of lowest-confidence points.",
    )
    parser.add_argument(
        "--use_point_map",
        action="store_true",
        help="Use VGGT point-map branch instead of depth unprojection.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=0,
        help="Optionally cap exported points by random sampling. 0 means no cap.",
    )
    parser.add_argument("--camera_json", help="Optional output path for camera intrinsics/extrinsics JSON.")
    parser.add_argument("--camera_npz", help="Optional output path for camera intrinsics/extrinsics NPZ.")
    parser.add_argument("--copy_images_dir", help="Optional directory to copy original input images into.")
    return parser.parse_args()


def main():
    args = parse_args()
    image_paths = sorted(glob.glob(os.path.join(args.image_folder, "*")))
    if not image_paths:
        raise ValueError(f"No images found in {args.image_folder}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    print(f"Using device: {device}")
    print(f"Found {len(image_paths)} images")

    model = VGGT()
    url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval().to(device)

    images = load_and_preprocess_images(image_paths).to(device)
    print(f"Preprocessed images shape: {tuple(images.shape)}")

    with torch.no_grad():
        if device == "cuda":
            with torch.amp.autocast("cuda", dtype=dtype):
                predictions = model(images)
        else:
            predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    extrinsic_np = extrinsic.cpu().numpy().squeeze(0)
    intrinsic_np = intrinsic.cpu().numpy().squeeze(0)

    images_np = images.cpu().numpy()
    colors = (images_np.transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)

    if args.copy_images_dir:
        os.makedirs(args.copy_images_dir, exist_ok=True)
        for image_path in image_paths:
            shutil.copy2(image_path, os.path.join(args.copy_images_dir, os.path.basename(image_path)))
        print(f"Copied {len(image_paths)} original images to {args.copy_images_dir}")

    if args.camera_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.camera_json)), exist_ok=True)
        camera_to_world = np.linalg.inv(
            np.concatenate(
                [
                    extrinsic_np,
                    np.broadcast_to(np.array([0, 0, 0, 1], dtype=extrinsic_np.dtype), (len(extrinsic_np), 1, 4)),
                ],
                axis=1,
            )
        )
        camera_data = {
            "image_folder": os.path.abspath(args.image_folder),
            "preprocessed_size": list(images.shape[-2:]),
            "coordinate_convention": "extrinsic is OpenCV camera-from-world; camera_to_world is its inverse.",
            "cameras": [
                {
                    "image_name": os.path.basename(image_path),
                    "image_path": os.path.abspath(image_path),
                    "intrinsic_3x3": intrinsic_np[i].tolist(),
                    "extrinsic_3x4_camera_from_world": extrinsic_np[i].tolist(),
                    "camera_to_world_4x4": camera_to_world[i].tolist(),
                }
                for i, image_path in enumerate(image_paths)
            ],
        }
        with open(args.camera_json, "w", encoding="utf-8") as f:
            json.dump(camera_data, f, indent=2, ensure_ascii=False)
        print(f"Exported camera JSON to {args.camera_json}")

    if args.camera_npz:
        os.makedirs(os.path.dirname(os.path.abspath(args.camera_npz)), exist_ok=True)
        np.savez(
            args.camera_npz,
            image_names=np.array([os.path.basename(path) for path in image_paths]),
            extrinsic=extrinsic_np,
            intrinsic=intrinsic_np,
        )
        print(f"Exported camera NPZ to {args.camera_npz}")

    if args.use_point_map:
        points_map = predictions["world_points"].cpu().numpy().squeeze(0)
        conf_map = predictions["world_points_conf"].cpu().numpy().squeeze(0)
    else:
        depth = predictions["depth"].cpu().numpy().squeeze(0)
        points_map = unproject_depth_map_to_point_map(depth, extrinsic_np, intrinsic_np)
        conf_map = predictions["depth_conf"].cpu().numpy().squeeze(0)

    points = points_map.reshape(-1, 3)
    conf = conf_map.reshape(-1)

    threshold = np.percentile(conf, args.conf_percentile) if args.conf_percentile > 0 else 0.0
    mask = (conf >= threshold) & (conf > 1e-5) & np.isfinite(points).all(axis=1)
    points = points[mask]
    colors = colors[mask]

    if args.max_points > 0 and len(points) > args.max_points:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(points), size=args.max_points, replace=False)
        points = points[indices]
        colors = colors[indices]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    trimesh.PointCloud(points, colors=colors).export(args.output)
    print(f"Exported {len(points)} points to {args.output}")


if __name__ == "__main__":
    main()
