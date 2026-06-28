import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, binary_dilation, label

try:
    from skimage.morphology import skeletonize as skimage_skeletonize
except ImportError:
    skimage_skeletonize = None


def load_skeleton_mask(ocm_dir: Path, coords_name: str, height: int, width: int, point_radius: int) -> np.ndarray:
    coords = np.load(ocm_dir / coords_name)
    mask = np.zeros((height, width), dtype=bool)
    radius = max(0, int(point_radius))
    for r, c in coords.astype(np.int32):
        if r < 0 or r >= height or c < 0 or c >= width:
            continue
        r0, r1 = max(0, r - radius), min(height, r + radius + 1)
        c0, c1 = max(0, c - radius), min(width, c + radius + 1)
        mask[r0:r1, c0:c1] = True
    return mask


def rebuild_clean_ocm(
    ocm_dir: Path,
    height: int,
    width: int,
    fill_length: int,
    color_aggregation: str,
) -> np.ndarray:
    coords_rc = np.load(ocm_dir / "coords_rc.npy")
    colors = np.load(ocm_dir / "ocm_colors.npy")
    if color_aggregation not in ("last", "mean"):
        raise ValueError("color_aggregation must be 'last' or 'mean'")
    img = np.zeros((height, width, 3), dtype=np.uint8)
    radius = max(0, int(fill_length) // 2)
    if color_aggregation == "mean":
        point_colors = aggregate_pixel_colors(coords_rc, colors, height, width)
    else:
        point_colors = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
    for i, (r, c) in enumerate(coords_rc.astype(np.int32)):
        if r < 0 or r >= height or c < 0 or c >= width:
            continue
        r0, r1 = max(0, r - radius), min(height, r + radius + 1)
        c0, c1 = max(0, c - radius), min(width, c + radius + 1)
        img[r0:r1, c0:c1] = point_colors[i]
    return img


def aggregate_pixel_colors(coords_rc: np.ndarray, colors: np.ndarray, height: int, width: int) -> np.ndarray:
    valid = (
        (coords_rc[:, 0] >= 0) & (coords_rc[:, 0] < height) &
        (coords_rc[:, 1] >= 0) & (coords_rc[:, 1] < width)
    )
    out = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
    if not np.any(valid):
        return out
    flat = coords_rc[valid, 0].astype(np.int64) * int(width) + coords_rc[valid, 1].astype(np.int64)
    sums = np.zeros((height * width, 3), dtype=np.float64)
    counts = np.zeros(height * width, dtype=np.int64)
    np.add.at(sums, flat, colors[valid])
    np.add.at(counts, flat, 1)
    occupied = counts > 0
    means = np.zeros_like(sums)
    means[occupied] = sums[occupied] / counts[occupied, None]
    out[valid] = np.clip(means[flat] * 255, 0, 255).astype(np.uint8)
    return out


def keep_large_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    if min_pixels <= 1:
        return mask
    labels, count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    keep = sizes >= int(min_pixels)
    keep[0] = False
    return keep[labels]


def zhang_suen_thin(mask: np.ndarray, max_iter: int = 100) -> np.ndarray:
    img = mask.astype(np.uint8).copy()
    if img.size == 0:
        return img.astype(bool)

    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1, mode="constant")
            p2 = p[:-2, 1:-1]
            p3 = p[:-2, 2:]
            p4 = p[1:-1, 2:]
            p5 = p[2:, 2:]
            p6 = p[2:, 1:-1]
            p7 = p[2:, :-2]
            p8 = p[1:-1, :-2]
            p9 = p[:-2, :-2]
            neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
            neighbor_count = sum(neighbors)
            transitions = sum((neighbors[i] == 0) & (neighbors[(i + 1) % 8] == 1) for i in range(8))

            if step == 0:
                removable = (
                    (img == 1) &
                    (neighbor_count >= 2) & (neighbor_count <= 6) &
                    (transitions == 1) &
                    ((p2 * p4 * p6) == 0) &
                    ((p4 * p6 * p8) == 0)
                )
            else:
                removable = (
                    (img == 1) &
                    (neighbor_count >= 2) & (neighbor_count <= 6) &
                    (transitions == 1) &
                    ((p2 * p4 * p8) == 0) &
                    ((p2 * p6 * p8) == 0)
                )
            if np.any(removable):
                img[removable] = 0
                changed = True
        if not changed:
            break
    return img.astype(bool)


def skeletonize_mask(mask: np.ndarray, method: str) -> np.ndarray:
    if method == "skimage":
        if skimage_skeletonize is None:
            raise RuntimeError(
                "scikit-image is required for --thin_method skimage. "
                "Install requirements.txt or use --thin_method zhang_suen."
            )
        return skimage_skeletonize(mask).astype(bool)
    if method == "zhang_suen":
        return zhang_suen_thin(mask)
    raise ValueError("thin_method must be 'skimage' or 'zhang_suen'")


def prune_spurs(mask: np.ndarray, iterations: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(max(0, int(iterations))):
        p = np.pad(out.astype(np.uint8), 1, mode="constant")
        neighbor_count = (
            p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
            p[1:-1, :-2] + p[1:-1, 2:] +
            p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
        )
        endpoints = out & (neighbor_count <= 1)
        if not np.any(endpoints):
            break
        out[endpoints] = False
    return out


def paint_overlay(base_rgb: np.ndarray, line_mask: np.ndarray, line_width: int) -> np.ndarray:
    out = base_rgb.copy()
    if line_width > 1:
        radius = max(0, int(line_width) // 2)
        line_mask = binary_dilation(
            line_mask,
            structure=np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool),
        )
    out[line_mask] = 0
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Convert projected OCM skeleton points into a cleaned 2D centerline overlay."
    )
    parser.add_argument("--ocm_dir", required=True, help="Existing OCM output directory")
    parser.add_argument(
        "--base_mode",
        choices=["clean", "image"],
        default="clean",
        help="Use a rebuilt clean OCM base or the existing image file.",
    )
    parser.add_argument("--image_name", default="ocm_image.png", help="Base OCM image filename")
    parser.add_argument(
        "--color_aggregation",
        choices=["auto", "last", "mean"],
        default="auto",
        help="Color aggregation used when rebuilding a clean base image.",
    )
    parser.add_argument("--coords_name", default="skeleton_coords_rc.npy", help="Projected skeleton coordinate file")
    parser.add_argument("--out_prefix", default="skeleton_centerline", help="Output file prefix")
    parser.add_argument("--point_radius", type=int, default=1, help="Raster radius for each skeleton point")
    parser.add_argument("--close_radius", type=int, default=1, help="Binary closing radius before thinning")
    parser.add_argument("--min_component_pixels", type=int, default=20, help="Remove small connected components")
    parser.add_argument("--prune_iterations", type=int, default=4, help="Endpoint pruning iterations after thinning")
    parser.add_argument("--thin_method", choices=["skimage", "zhang_suen"], default="skimage")
    parser.add_argument("--line_width", type=int, default=1, help="Overlay line width in pixels")
    args = parser.parse_args()

    ocm_dir = Path(args.ocm_dir)
    meta = json.loads((ocm_dir / "ocm_metadata.json").read_text(encoding="utf-8"))
    height = int(meta["image_height"])
    width = int(meta["image_width"])
    fill_length = int(meta["fill_length_FL"])
    if args.color_aggregation == "auto":
        color_aggregation = meta.get("config", {}).get("color_aggregation") or "last"
    else:
        color_aggregation = args.color_aggregation
    if args.base_mode == "clean":
        base_rgb = rebuild_clean_ocm(ocm_dir, height, width, fill_length, color_aggregation)
        base_image = "rebuilt_clean_ocm"
    else:
        base_rgb = np.array(Image.open(ocm_dir / args.image_name).convert("RGB"))
        base_image = str(ocm_dir / args.image_name)

    mask = load_skeleton_mask(ocm_dir, args.coords_name, height, width, args.point_radius)
    component_mask = keep_large_components(mask, args.min_component_pixels)
    if args.close_radius > 0:
        radius = int(args.close_radius)
        component_mask = binary_closing(
            component_mask,
            structure=np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool),
        )
    centerline = skeletonize_mask(component_mask, args.thin_method)
    centerline = prune_spurs(centerline, args.prune_iterations)
    centerline = keep_large_components(centerline, max(1, args.min_component_pixels // 4))

    mask_img = np.zeros((height, width), dtype=np.uint8)
    mask_img[centerline] = 255
    overlay = paint_overlay(base_rgb, centerline, args.line_width)

    mask_path = ocm_dir / f"{args.out_prefix}_mask.png"
    overlay_path = ocm_dir / f"{args.out_prefix}_overlay.png"
    meta_path = ocm_dir / f"{args.out_prefix}_metadata.json"
    Image.fromarray(mask_img).save(mask_path)
    Image.fromarray(overlay).save(overlay_path)

    out_meta = {
        "source_ocm_dir": str(ocm_dir),
        "base_mode": args.base_mode,
        "base_image": base_image,
        "color_aggregation": color_aggregation,
        "coords_name": args.coords_name,
        "mask_image": str(mask_path),
        "overlay_image": str(overlay_path),
        "point_radius": int(args.point_radius),
        "close_radius": int(args.close_radius),
        "min_component_pixels": int(args.min_component_pixels),
        "prune_iterations": int(args.prune_iterations),
        "thin_method": args.thin_method,
        "line_width": int(args.line_width),
        "input_pixels": int(mask.sum()),
        "component_pixels": int(component_mask.sum()),
        "centerline_pixels": int(centerline.sum()),
    }
    meta_path.write_text(json.dumps(out_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved centerline mask: {mask_path}")
    print(f"saved centerline overlay: {overlay_path}")
    print(
        f"input_pixels={out_meta['input_pixels']}, "
        f"component_pixels={out_meta['component_pixels']}, "
        f"centerline_pixels={out_meta['centerline_pixels']}"
    )


if __name__ == "__main__":
    main()
