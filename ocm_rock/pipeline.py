from pathlib import Path
from typing import Dict, Optional, Sequence
import numpy as np

from .config import OCMConfig
from .io_utils import load_point_cloud, write_json
from .preprocess import statistical_outlier_filter, pca_normals
from .skeleton import detect_sharp_points, npw_oriented_contraction
from .ocm_mapping import (
    generate_cdps, optimal_normal_rotation, normals_to_rgb, calibrate_direction,
    normals_to_rgb_adaptive_pca, normalize_xz_to_image, project_xz_to_image, fill_ocm_image,
    draw_skeleton_lines, save_image
)


def generate_ocm_from_pointcloud(point_cloud_path: str, out_dir: str, cfg: OCMConfig, remove_outlier: bool = False) -> Dict:
    """论文 Step 1-3 的一键实现：点云 -> NPW-OC骨架 -> OCM图像。"""
    infos = generate_multi_view_ocm_from_pointcloud(point_cloud_path, out_dir, cfg, [0.0], remove_outlier)
    return infos[0]


def generate_multi_view_ocm_from_pointcloud(
    point_cloud_path: str,
    out_dir: str,
    cfg: OCMConfig,
    view_angles: Sequence[float],
    remove_outlier: bool = False,
) -> Sequence[Dict]:
    """Generate one or more OCM images from the same point cloud with shared preprocessing."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    points, raw_colors = load_point_cloud(point_cloud_path)

    if remove_outlier:
        keep = statistical_outlier_filter(points, cfg.statistical_nb_neighbors, cfg.statistical_std_ratio)
        points = points[keep]
        raw_colors = raw_colors[keep] if raw_colors is not None else None

    normals, eigvals, eigvecs, nn_idx = pca_normals(points, cfg.knn)
    # sharp_mask, delta = detect_sharp_points(normals, nn_idx) # 论文原始设置的尖点阈值
    sharp_mask, delta = detect_sharp_points(
        normals=normals,
        nn_idx=nn_idx,
        threshold_mode="percentile",
        percentile=95.0,
        min_angle_deg=8.0,
        max_sharp_ratio=0.05,
    )
    # 修改为可控阈值
    skeleton = npw_oriented_contraction(points, sharp_mask, eigvals, eigvecs, cfg.knn, cfg.contraction_iter)

    cdps = generate_cdps(cfg.cdp_subdivision)
    normals_rot, R_ocm, sumang = optimal_normal_rotation(normals, cdps, cfg.boundary_dip_angle)
    if cfg.color_mapping == "physical":
        ocm_colors = normals_to_rgb(normals_rot)
    elif cfg.color_mapping == "adaptive_pca":
        ocm_colors = normals_to_rgb_adaptive_pca(
            normals_rot,
            percentile=cfg.adaptive_color_percentile,
            gain=cfg.adaptive_color_gain,
        )
    else:
        raise ValueError(f"Unknown color_mapping: {cfg.color_mapping}")

    infos = []
    multi_view = len(view_angles) > 1
    for view_angle in view_angles:
        view_out = out / f"view_{format_view_angle(view_angle)}" if multi_view else out
        view_out.mkdir(parents=True, exist_ok=True)
        info = write_ocm_view(
            view_out,
            point_cloud_path,
            points,
            normals,
            sharp_mask,
            skeleton,
            ocm_colors,
            R_ocm,
            cfg,
            float(view_angle),
        )
        infos.append(info)
    return infos


def write_ocm_view(
    out: Path,
    point_cloud_path: str,
    points: np.ndarray,
    normals: np.ndarray,
    sharp_mask: np.ndarray,
    skeleton: np.ndarray,
    ocm_colors: np.ndarray,
    R_ocm: np.ndarray,
    cfg: OCMConfig,
    view_angle: float,
) -> Dict:
    points_view, R_view = calibrate_direction(points, view_angle)
    coords_rc, H, W, meta = normalize_xz_to_image(points_view, cfg.image_length)
    img, fl, ratio = fill_ocm_image(
        coords_rc, ocm_colors, None, H, W,
        cfg.target_void_ratio, cfg.max_fill_length
    )
    if cfg.draw_skeleton and len(skeleton) > 0:
        skeleton_view = skeleton @ R_view.T
        skeleton_rc = project_xz_to_image(skeleton_view, meta)
        img = draw_skeleton_lines(
            img,
            skeleton_rc,
            line_width=cfg.skeleton_line_width,
            link_neighbors=cfg.skeleton_link_neighbors,
            max_link_px=cfg.skeleton_max_link_px,
        )
    else:
        skeleton_rc = np.empty((0, 2), dtype=np.int32)

    save_image(str(out / "ocm_image.png"), img)
    np.save(out / "points.npy", points)
    np.save(out / "points_view.npy", points_view)
    np.save(out / "normals.npy", normals)
    np.save(out / "sharp_mask.npy", sharp_mask)
    np.save(out / "sharp_skeleton.npy", skeleton)
    np.save(out / "skeleton_coords_rc.npy", skeleton_rc)
    np.save(out / "coords_rc.npy", coords_rc)
    np.save(out / "ocm_colors.npy", ocm_colors)
    np.save(out / "R_ocm.npy", R_ocm)
    np.save(out / "R_view.npy", R_view)

    info = {
        "point_cloud": str(point_cloud_path),
        "point_count": int(len(points)),
        "sharp_point_count": int(sharp_mask.sum()),
        "skeleton_point_count": int(len(skeleton)),
        "ocm_image": str(out / "ocm_image.png"),
        "image_height": int(H),
        "image_width": int(W),
        "fill_length_FL": int(fl),
        "void_ratio": float(ratio),
        "view_angle_deg": float(view_angle),
        "config": cfg.__dict__,
        "image_meta": meta,
    }
    write_json(str(out / "ocm_metadata.json"), info)
    return info


def format_view_angle(angle: float) -> str:
    rounded = int(round(angle))
    if abs(angle - rounded) < 1e-6:
        return f"{rounded % 360:03d}"
    safe = f"{angle % 360:.2f}".replace(".", "p")
    return safe
