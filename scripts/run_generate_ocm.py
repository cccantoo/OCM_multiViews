import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ocm_rock.config import OCMConfig
from ocm_rock.pipeline import generate_multi_view_ocm_from_pointcloud


def parse_views(value: str):
    views = []
    for item in value.split(","):
        item = item.strip()
        if item:
            views.append(float(item))
    if not views:
        raise ValueError("--views must contain at least one angle")
    return views


def parse_optional_float(value: str):
    if value.lower() in {"none", "null", "off"}:
        return None
    return float(value)


UNSET = object()


def main():
    parser = argparse.ArgumentParser(description="Generate OCM image(s) and NPW-OC skeleton lines")
    parser.add_argument("--point_cloud", required=True, help="Input point cloud: txt/xyz/csv/ply/pcd")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--knn", type=int, default=20)
    parser.add_argument("--remove_outlier", action="store_true")
    parser.add_argument("--views", default="0", help="Comma-separated view angles in degrees, e.g. 0,30,60,90")
    parser.add_argument("--color_mapping", choices=["physical", "adaptive_pca"], default=None)
    parser.add_argument("--color_aggregation", choices=["last", "mean"], default=None)
    parser.add_argument("--normal_smoothing_iter", type=int, default=None)
    parser.add_argument("--adaptive_color_percentile", type=float, default=None)
    parser.add_argument("--adaptive_color_gain", type=float, default=None)
    parser.add_argument("--sharp_threshold_mode", choices=["paper_mean", "mean_std", "percentile"], default=None)
    parser.add_argument("--sharp_percentile", type=float, default=None)
    parser.add_argument("--sharp_mean_std_alpha", type=float, default=None)
    parser.add_argument("--sharp_min_angle_deg", type=float, default=None)
    parser.add_argument("--sharp_max_ratio", type=parse_optional_float, default=UNSET)
    parser.add_argument(
        "--draw_sharp_points",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Overlay raw sharp points during OCM filling, matching the paper-style visual cue.",
    )
    parser.add_argument(
        "--draw_skeleton",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Overlay NPW-OC skeleton lines. Defaults to OCMConfig.draw_skeleton.",
    )
    parser.add_argument("--skeleton_line_width", type=int, default=None)
    parser.add_argument("--skeleton_link_neighbors", type=int, default=None)
    parser.add_argument("--skeleton_max_link_px", type=float, default=None)
    parser.add_argument("--skeleton_filter_mode", choices=["color_contrast", "none"], default=None)
    parser.add_argument("--skeleton_min_color_contrast_thresh", type=parse_optional_float, default=UNSET)
    args = parser.parse_args()

    cfg = OCMConfig(knn=args.knn)
    if args.color_mapping is not None:
        cfg.color_mapping = args.color_mapping
    if args.color_aggregation is not None:
        cfg.color_aggregation = args.color_aggregation
    if args.normal_smoothing_iter is not None:
        cfg.normal_smoothing_iter = args.normal_smoothing_iter
    if args.adaptive_color_percentile is not None:
        cfg.adaptive_color_percentile = args.adaptive_color_percentile
    if args.adaptive_color_gain is not None:
        cfg.adaptive_color_gain = args.adaptive_color_gain
    if args.sharp_threshold_mode is not None:
        cfg.sharp_threshold_mode = args.sharp_threshold_mode
    if args.sharp_percentile is not None:
        cfg.sharp_percentile = args.sharp_percentile
    if args.sharp_mean_std_alpha is not None:
        cfg.sharp_mean_std_alpha = args.sharp_mean_std_alpha
    if args.sharp_min_angle_deg is not None:
        cfg.sharp_min_angle_deg = args.sharp_min_angle_deg
    if args.sharp_max_ratio is not UNSET:
        cfg.sharp_max_ratio = args.sharp_max_ratio
    if args.draw_sharp_points is not None:
        cfg.draw_sharp_points = args.draw_sharp_points
    if args.draw_skeleton is not None:
        cfg.draw_skeleton = args.draw_skeleton
    if args.skeleton_line_width is not None:
        cfg.skeleton_line_width = args.skeleton_line_width
    if args.skeleton_link_neighbors is not None:
        cfg.skeleton_link_neighbors = args.skeleton_link_neighbors
    if args.skeleton_max_link_px is not None:
        cfg.skeleton_max_link_px = args.skeleton_max_link_px
    if args.skeleton_filter_mode is not None:
        cfg.skeleton_filter_mode = None if args.skeleton_filter_mode == "none" else args.skeleton_filter_mode
    if args.skeleton_min_color_contrast_thresh is not UNSET:
        cfg.skeleton_min_color_contrast_thresh = args.skeleton_min_color_contrast_thresh

    infos = generate_multi_view_ocm_from_pointcloud(
        args.point_cloud,
        args.out,
        cfg,
        parse_views(args.views),
        args.remove_outlier,
    )
    print(f"OCM generation completed: {len(infos)} view(s)")
    for info in infos:
        print(f"view={info['view_angle_deg']}, image={info['ocm_image']}, void_ratio={info['void_ratio']:.6f}")


if __name__ == "__main__":
    main()
