import argparse
import glob
import json
import os
import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F

from vggt.dependency.track_predict import predict_tracks
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def parse_args():
    parser = argparse.ArgumentParser(description="Export VGGT tracks, camera, point, color, and source image data.")
    parser.add_argument("--image_folder", required=True, help="Folder containing input images.")
    parser.add_argument("--output_dir", required=True, help="Directory where export files are written.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vggt_resolution", type=int, default=518)
    parser.add_argument("--track_image_resolution", type=int, default=1024)
    parser.add_argument("--query_frame_num", type=int, default=5)
    parser.add_argument("--max_query_pts", type=int, default=4096)
    parser.add_argument("--max_points_num", type=int, default=163840)
    parser.add_argument("--keypoint_extractor", default="aliked+sp")
    parser.add_argument("--no_fine_tracking", action="store_true")
    parser.add_argument("--no_complete_non_vis", action="store_true")
    return parser.parse_args()


def run_vggt(model, images, dtype, resolution):
    images_518 = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype, enabled=images_518.is_cuda):
            images_batch = images_518[None]
            aggregated_tokens_list, ps_idx = model.aggregator(images_batch)

        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_batch.shape[-2:])
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images_batch, ps_idx)

    return (
        extrinsic.squeeze(0).cpu().numpy(),
        intrinsic.squeeze(0).cpu().numpy(),
        depth_map.squeeze(0).cpu().numpy(),
        depth_conf.squeeze(0).cpu().numpy(),
    )


def main():
    args = parse_args()
    image_paths = sorted(glob.glob(os.path.join(args.image_folder, "*")))
    if not image_paths:
        raise ValueError(f"No images found in {args.image_folder}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Track export requires CUDA for the VGGSfM tracker path.")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    os.makedirs(args.output_dir, exist_ok=True)
    images_out_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_out_dir, exist_ok=True)
    for image_path in image_paths:
        shutil.copy2(image_path, os.path.join(images_out_dir, os.path.basename(image_path)))

    print(f"Using device: {device}")
    print(f"Found {len(image_paths)} images")

    images, original_coords = load_and_preprocess_images_square(image_paths, args.track_image_resolution)
    images = images.to(device)
    original_coords_np = original_coords.cpu().numpy()
    print(f"Track images shape: {tuple(images.shape)}")

    model = VGGT()
    url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(url, map_location="cpu"))
    model.eval().to(device)

    extrinsic, intrinsic_518, depth_map, depth_conf = run_vggt(model, images, dtype, args.vggt_resolution)
    points_3d_dense = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic_518)

    scale = args.track_image_resolution / args.vggt_resolution
    intrinsic_track = intrinsic_518.copy()
    intrinsic_track[:, :2, :] *= scale

    print("Predicting tracks...")
    with torch.amp.autocast("cuda", dtype=dtype):
        pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
            images,
            conf=depth_conf,
            points_3d=points_3d_dense,
            masks=None,
            max_query_pts=args.max_query_pts,
            query_frame_num=args.query_frame_num,
            keypoint_extractor=args.keypoint_extractor,
            max_points_num=args.max_points_num,
            fine_tracking=not args.no_fine_tracking,
            complete_non_vis=not args.no_complete_non_vis,
        )

    image_names = np.array([os.path.basename(path) for path in image_paths])
    npz_path = os.path.join(args.output_dir, "vggt_tracks_bundle.npz")
    np.savez_compressed(
        npz_path,
        image_names=image_names,
        original_image_paths=np.array([os.path.abspath(path) for path in image_paths]),
        original_coords=original_coords_np,
        pred_tracks=pred_tracks,
        pred_vis_scores=pred_vis_scores,
        pred_confs=pred_confs,
        points_3d=points_3d,
        points_rgb=points_rgb,
        intrinsic=intrinsic_track,
        intrinsic_518=intrinsic_518,
        extrinsic=extrinsic,
    )

    json_path = os.path.join(args.output_dir, "metadata.json")
    metadata = {
        "image_folder": os.path.abspath(args.image_folder),
        "image_names": image_names.tolist(),
        "track_image_resolution": args.track_image_resolution,
        "vggt_resolution": args.vggt_resolution,
        "coordinate_notes": {
            "pred_tracks": "2D tracks in padded square track-image coordinates.",
            "original_coords": "[x1, y1, x2, y2, original_width, original_height] in padded square coordinates.",
            "intrinsic": f"Intrinsics scaled to {args.track_image_resolution}x{args.track_image_resolution}.",
            "intrinsic_518": f"Intrinsics at VGGT inference resolution {args.vggt_resolution}x{args.vggt_resolution}.",
            "extrinsic": "OpenCV camera-from-world 3x4 matrices.",
            "points_3d": "Tracked 3D points corresponding to the exported track points.",
            "points_rgb": "RGB colors for points_3d, uint8 in 0..255.",
        },
        "shapes": {
            "pred_tracks": list(pred_tracks.shape),
            "pred_vis_scores": list(pred_vis_scores.shape),
            "pred_confs": None if pred_confs is None else list(pred_confs.shape),
            "points_3d": None if points_3d is None else list(points_3d.shape),
            "points_rgb": None if points_rgb is None else list(points_rgb.shape),
            "intrinsic": list(intrinsic_track.shape),
            "extrinsic": list(extrinsic.shape),
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Exported bundle: {npz_path}")
    print(f"Exported metadata: {json_path}")
    print(f"Copied original images: {images_out_dir}")
    print(f"pred_tracks shape: {pred_tracks.shape}")
    print(f"pred_vis_scores shape: {pred_vis_scores.shape}")
    print(f"points_3d shape: {None if points_3d is None else points_3d.shape}")
    print(f"points_rgb shape: {None if points_rgb is None else points_rgb.shape}")


if __name__ == "__main__":
    main()
