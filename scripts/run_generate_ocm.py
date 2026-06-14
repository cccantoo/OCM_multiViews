import argparse

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


def main():
    parser = argparse.ArgumentParser(description="Generate OCM image(s) and NPW-OC skeleton lines")
    parser.add_argument("--point_cloud", required=True, help="Input point cloud: txt/xyz/csv/ply/pcd")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--knn", type=int, default=20)
    parser.add_argument("--remove_outlier", action="store_true")
    parser.add_argument("--views", default="0", help="Comma-separated view angles in degrees, e.g. 0,30,60,90")
    parser.add_argument("--color_mapping", choices=["physical", "adaptive_pca"], default=None)
    parser.add_argument("--adaptive_color_percentile", type=float, default=None)
    parser.add_argument("--adaptive_color_gain", type=float, default=None)
    parser.add_argument("--color_mapping", choices=["physical", "adaptive_pca"], default=None)
    parser.add_argument("--adaptive_color_percentile", type=float, default=None)
    parser.add_argument("--adaptive_color_gain", type=float, default=None)
    parser.add_argument(
        "--draw_skeleton",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Overlay NPW-OC skeleton lines. Defaults to OCMConfig.draw_skeleton.",
    )
    parser.add_argument("--skeleton_line_width", type=int, default=None)
    parser.add_argument("--skeleton_link_neighbors", type=int, default=None)
    parser.add_argument("--skeleton_max_link_px", type=float, default=None)
    args = parser.parse_args()

    cfg = OCMConfig(knn=args.knn)
    if args.color_mapping is not None:
        cfg.color_mapping = args.color_mapping
    if args.adaptive_color_percentile is not None:
        cfg.adaptive_color_percentile = args.adaptive_color_percentile
    if args.adaptive_color_gain is not None:
        cfg.adaptive_color_gain = args.adaptive_color_gain
    if args.color_mapping is not None:
        cfg.color_mapping = args.color_mapping
    if args.adaptive_color_percentile is not None:
        cfg.adaptive_color_percentile = args.adaptive_color_percentile
    if args.adaptive_color_gain is not None:
        cfg.adaptive_color_gain = args.adaptive_color_gain
    if args.draw_skeleton is not None:
        cfg.draw_skeleton = args.draw_skeleton
    if args.skeleton_line_width is not None:
        cfg.skeleton_line_width = args.skeleton_line_width
    if args.skeleton_link_neighbors is not None:
        cfg.skeleton_link_neighbors = args.skeleton_link_neighbors
    if args.skeleton_max_link_px is not None:
        cfg.skeleton_max_link_px = args.skeleton_max_link_px

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
