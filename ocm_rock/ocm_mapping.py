from typing import Dict, Tuple, Optional
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .preprocess import global_plane_normal, hemispherize


def icosahedron_vertices_faces() -> Tuple[np.ndarray, np.ndarray]:
    """生成正二十面体顶点和面。"""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [0, -1,  phi], [0,  1,  phi], [0, -1, -phi], [0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1],
    ], dtype=np.float64)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array([
        [0,11,5], [0,5,1], [0,1,7], [0,7,10], [0,10,11],
        [1,5,9], [5,11,4], [11,10,2], [10,7,6], [7,1,8],
        [3,9,4], [3,4,2], [3,2,6], [3,6,8], [3,8,9],
        [4,9,5], [2,4,11], [6,2,10], [8,6,7], [9,8,1],
    ], dtype=np.int32)
    return verts, faces


def generate_cdps(subdivision: int = 5) -> np.ndarray:
    """Step 2.2.1：正二十面体细分生成 CDPs，筛选上半球。"""
    verts, faces = icosahedron_vertices_faces()
    vert_list = [tuple(v) for v in verts]
    for _ in range(subdivision):
        midpoint_cache = {}
        def midpoint(i, j):
            key = tuple(sorted((int(i), int(j))))
            if key in midpoint_cache:
                return midpoint_cache[key]
            v = (np.array(vert_list[i]) + np.array(vert_list[j])) / 2.0
            v = v / (np.linalg.norm(v) + 1e-12)
            vert_list.append(tuple(v))
            idx = len(vert_list) - 1
            midpoint_cache[key] = idx
            return idx

        new_faces = []
        for tri in faces:
            a, b, c = tri
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = np.array(new_faces, dtype=np.int32)
    verts = np.unique(np.round(np.array(vert_list), 12), axis=0)
    cdps = verts[verts[:, 2] >= -1e-12]
    cdps = cdps / (np.linalg.norm(cdps, axis=1, keepdims=True) + 1e-12)
    return cdps


def rotation_to_z(v: np.ndarray) -> np.ndarray:
    """ 将候选方向 v 旋转到 [0,0,1]。"""
    v = v / (np.linalg.norm(v) + 1e-12)
    target = np.array([0.0, 0.0, 1.0])
    cross = np.cross(v, target)
    dot = np.clip(np.dot(v, target), -1.0, 1.0)
    if np.linalg.norm(cross) < 1e-12:
        return np.eye(3) if dot > 0 else Rotation.from_euler("x", 180, degrees=True).as_matrix()
    rotvec = cross / np.linalg.norm(cross) * np.arccos(dot)
    return Rotation.from_rotvec(rotvec).as_matrix()


def dip_angles(normals: np.ndarray) -> np.ndarray:
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
    return np.degrees(np.arccos(np.clip(normals[:, 2], -1.0, 1.0)))


def optimal_normal_rotation(normals: np.ndarray, cdps: np.ndarray, boundary_dip_angle: float = 85.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Step 2.2.2：MBDA 最小边界倾角和，选择最优旋转方向。"""
    normals = hemispherize(normals)
    if len(normals) == 0:
        return normals, np.eye(3), np.zeros(len(cdps), dtype=np.float64)

    sums = np.zeros(len(cdps), dtype=np.float64)
    rotations = []
    for i, cdp in enumerate(cdps):
        R = rotation_to_z(cdp)
        rotations.append(R)
        rotated_z = normals @ R.T
        rotated_z = rotated_z[:, 2]
        dips = np.degrees(np.arccos(np.clip(np.abs(rotated_z), -1.0, 1.0)))
        boundary = dips[dips > boundary_dip_angle]
        sums[i] = float(np.sum(boundary))
    best = int(np.argmin(sums))
    Rbest = rotations[best]
    rotated_all = hemispherize(normals @ Rbest.T)
    return rotated_all, Rbest, sums


def normals_to_rgb(normals: np.ndarray) -> np.ndarray:
    """ Eq.(9)-(10)：将上半球法向量映射到 HSV 的 H/S，V=1，再转 RGB。"""
    n = hemispherize(normals)
    x, y, z = n[:, 0], n[:, 1], n[:, 2]
    r_xy = np.sqrt(x * x + y * y) + 1e-12
    angle = np.degrees(np.arccos(np.clip(x / r_xy, -1.0, 1.0))) / 360.0
    H = np.where(y > 0, angle, 1.0 - angle)
    S = np.clip(np.sqrt(x * x + y * y), 0.0, 1.0)
    V = np.ones_like(S)
    return hsv_to_rgb(H, S, V)


def normals_to_rgb_adaptive_pca(
    normals: np.ndarray,
    percentile: float = 98.0,
    gain: float = 1.0,
) -> np.ndarray:
    n = hemispherize(normals)
    center = np.median(n, axis=0)
    centered = n - center
    cov = centered.T @ centered / max(1, len(centered))
    w, v = np.linalg.eigh(cov)
    basis = v[:, np.argsort(w)[::-1][:2]]
    uv = centered @ basis

    q = np.percentile(np.abs(uv), percentile, axis=0)
    q = np.maximum(q, 1e-6)
    uv = np.clip((uv / q) * gain, -1.0, 1.0)

    radius = np.sqrt(np.sum(uv * uv, axis=1))
    H = (np.arctan2(uv[:, 1], uv[:, 0]) / (2.0 * np.pi) + 1.0) % 1.0
    S = np.clip(radius, 0.0, 1.0)
    V = np.ones_like(S)
    return hsv_to_rgb(H, S, V)


def hsv_to_rgb(H: np.ndarray, S: np.ndarray, V: np.ndarray) -> np.ndarray:
    H6 = (H % 1.0) * 6.0
    I = np.floor(H6).astype(int)
    F = H6 - I
    M = V * (1 - S)
    N = V * (1 - S * F)
    K = V * (1 - S * (1 - F))
    rgb = np.zeros((len(H), 3), dtype=np.float64)
    cases = [
        np.stack([V, K, M], axis=1),
        np.stack([N, V, M], axis=1),
        np.stack([M, V, K], axis=1),
        np.stack([M, N, V], axis=1),
        np.stack([K, M, V], axis=1),
        np.stack([V, M, N], axis=1),
    ]
    for k in range(6):
        rgb[I == k] = cases[k][I == k]
    return np.clip(rgb, 0.0, 1.0)


def calibrate_direction(points: np.ndarray, view_angle_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """ Step 3.1：绕 z 轴旋转，使整体拟合面法向量平行负 y 轴，并从 xoz 视角成像。"""
    normal = global_plane_normal(points)
    target = np.array([0.0, -1.0, 0.0])
    # 只考虑 xy 投影，绕 z 轴旋转。
    a = np.arctan2(normal[1], normal[0])
    b = np.arctan2(target[1], target[0])
    theta = b - a + np.deg2rad(view_angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return points @ Rz.T, Rz


def normalize_xz_to_image(points_rot: np.ndarray, image_length: int = 800) -> Tuple[np.ndarray, int, int, Dict[str, float]]:
    """将旋转后的 x,z 坐标映射为图像列/行。"""
    x = points_rot[:, 0]
    z = points_rot[:, 2]
    x0 = x - x.min()
    z0 = z - z.min()
    scale = image_length / (max(x0.max(), z0.max()) + 1e-12)
    col = np.round(x0 * scale).astype(np.int32)
    row = np.round((z0.max() - z0) * scale).astype(np.int32)  # 图像 y 向下，z 向上，故反转。
    W = int(col.max() + 1)
    H = int(row.max() + 1)
    meta = {"x_min": float(x.min()), "z_min": float(z.min()), "z_max0": float(z0.max()), "scale": float(scale)}
    return np.stack([row, col], axis=1), H, W, meta


def project_xz_to_image(points_rot: np.ndarray, meta: Dict[str, float]) -> np.ndarray:
    """Project rotated 3D points into an existing OCM x-z image frame."""
    x0 = points_rot[:, 0] - meta["x_min"]
    z0 = points_rot[:, 2] - meta["z_min"]
    scale = meta["scale"]
    col = np.round(x0 * scale).astype(np.int32)
    row = np.round((meta["z_max0"] - z0) * scale).astype(np.int32)
    return np.stack([row, col], axis=1)


def _fill_ocm_image_rectangles(
    coords_rc: np.ndarray,
    colors: np.ndarray,
    sharp_mask: Optional[np.ndarray],
    H: int,
    W: int,
    target_void_ratio: float = 0.10,
    min_fill_length: int = 1,
    max_fill_length: int = 15,
    color_aggregation: str = "last",
) -> Tuple[np.ndarray, int, float]:
    """Step 3.2: fill OCM colors with FL rectangles and keep coverage separate from overlays."""
    if color_aggregation not in ("last", "mean"):
        raise ValueError("color_aggregation must be 'last' or 'mean'")
    if color_aggregation == "mean":
        point_colors = aggregate_pixel_colors(coords_rc, colors, H, W)
    else:
        point_colors = (colors * 255).astype(np.uint8)

    min_fill_length = max(1, int(min_fill_length))
    max_fill_length = max(min_fill_length, int(max_fill_length))
    if min_fill_length % 2 == 0:
        min_fill_length += 1
    if max_fill_length % 2 == 0:
        max_fill_length -= 1

    best_img, best_ratio, best_fl = None, 1e9, min_fill_length
    for fl in range(min_fill_length, max_fill_length + 1, 2):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        coverage = np.zeros((H, W), dtype=bool)
        rad = fl // 2
        for i, (r, c) in enumerate(coords_rc):
            if r < 0 or r >= H or c < 0 or c >= W:
                continue
            color = point_colors[i]
            r0, r1 = max(0, r - rad), min(H, r + rad + 1)
            c0, c1 = max(0, c - rad), min(W, c + rad + 1)
            img[r0:r1, c0:c1] = color
            coverage[r0:r1, c0:c1] = True
        ratio = void_ratio_from_coverage(coverage)
        best_img, best_ratio, best_fl = img, ratio, fl
        if ratio <= target_void_ratio:
            break
    if sharp_mask is not None:
        draw_sharp_point_overlay(best_img, coords_rc, sharp_mask)
    return best_img, best_fl, best_ratio


def fill_ocm_image(
    coords_rc: np.ndarray,
    colors: np.ndarray,
    sharp_mask: Optional[np.ndarray],
    H: int,
    W: int,
    target_void_ratio: float = 0.10,
    min_fill_length: int = 1,
    max_fill_length: int = 15,
    color_aggregation: str = "last",
) -> Tuple[np.ndarray, int, float]:
    """Step 3.2: fill each point into an FL x FL rectangle as described in the paper."""
    return _fill_ocm_image_rectangles(
        coords_rc,
        colors,
        sharp_mask,
        H,
        W,
        target_void_ratio,
        min_fill_length,
        max_fill_length,
        color_aggregation,
    )


def aggregate_pixel_colors(coords_rc: np.ndarray, colors: np.ndarray, H: int, W: int) -> np.ndarray:
    valid = (
        (coords_rc[:, 0] >= 0) & (coords_rc[:, 0] < H) &
        (coords_rc[:, 1] >= 0) & (coords_rc[:, 1] < W)
    )
    out = (colors * 255).astype(np.uint8)
    if not np.any(valid):
        return out

    flat = coords_rc[valid, 0].astype(np.int64) * int(W) + coords_rc[valid, 1].astype(np.int64)
    sums = np.zeros((H * W, 3), dtype=np.float64)
    counts = np.zeros(H * W, dtype=np.int64)
    np.add.at(sums, flat, colors[valid])
    np.add.at(counts, flat, 1)
    means = np.zeros_like(sums)
    occupied = counts > 0
    means[occupied] = sums[occupied] / counts[occupied, None]
    out[valid] = np.clip(means[flat] * 255, 0, 255).astype(np.uint8)
    return out


def draw_skeleton_lines(
    img: np.ndarray,
    skeleton_rc: np.ndarray,
    line_width: int = 2,
    link_neighbors: int = 3,
    max_link_px: float = 8.0,
) -> np.ndarray:
    """Rasterize NPW-OC skeleton points as connected black linework on an OCM image."""
    out = img.copy()
    if len(skeleton_rc) == 0:
        return out

    H, W = out.shape[:2]
    valid = (
        (skeleton_rc[:, 0] >= 0) & (skeleton_rc[:, 0] < H) &
        (skeleton_rc[:, 1] >= 0) & (skeleton_rc[:, 1] < W)
    )
    pts = np.unique(skeleton_rc[valid].astype(np.int32), axis=0)
    if len(pts) == 0:
        return out

    radius = max(0, int(line_width) // 2)
    for r, c in pts:
        paint_disk(out, int(r), int(c), radius)

    if len(pts) < 2 or link_neighbors <= 0:
        return out

    tree = cKDTree(pts.astype(np.float64))
    k = min(len(pts), int(link_neighbors) + 1)
    dists, nn_idx = tree.query(pts, k=k)
    if k == 1:
        return out
    if nn_idx.ndim == 1:
        nn_idx = nn_idx[:, None]
        dists = dists[:, None]

    for i in range(len(pts)):
        for dist, j in zip(dists[i, 1:], nn_idx[i, 1:]):
            if i >= j or dist > max_link_px:
                continue
            draw_line(out, pts[i], pts[j], radius)
    return out


def filter_skeleton_by_color_contrast(
    img: np.ndarray,
    skeleton_rc: np.ndarray,
    radius: int = 3,
    contrast_thresh: Optional[float] = None,
    side_offset: int = 4,
    contrast_percentile: float = 50.0,
    min_contrast_thresh: float = 60.0,
) -> np.ndarray:
    """Keep skeleton points whose two sides have enough OCM color contrast."""
    if len(skeleton_rc) == 0:
        return skeleton_rc

    H, W = img.shape[:2]
    radius = max(1, int(radius))
    side_offset = max(radius + 1, int(side_offset))

    valid = (
        (skeleton_rc[:, 0] >= 0) & (skeleton_rc[:, 0] < H) &
        (skeleton_rc[:, 1] >= 0) & (skeleton_rc[:, 1] < W)
    )
    valid_ids = np.flatnonzero(valid)
    pts = skeleton_rc[valid].astype(np.float64)
    if len(pts) < 2:
        return skeleton_rc[valid]

    tree = cKDTree(pts)
    contrasts = np.full(len(pts), np.nan, dtype=np.float64)
    k = min(9, len(pts))
    for local_i, point in enumerate(pts):
        _, nn_idx = tree.query(point, k=k)
        neighbors = pts[np.atleast_1d(nn_idx)]
        tangent = local_tangent(neighbors)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        left = sample_patch_mean(img, point + side_offset * normal, radius)
        right = sample_patch_mean(img, point - side_offset * normal, radius)
        if left is None or right is None:
            continue
        contrasts[local_i] = np.linalg.norm(left - right)
    finite = np.isfinite(contrasts)
    if not np.any(finite):
        return np.empty((0, 2), dtype=skeleton_rc.dtype)
    if contrast_thresh is None:
        threshold = float(np.percentile(contrasts[finite], contrast_percentile))
        threshold = max(threshold, float(min_contrast_thresh))
    else:
        threshold = float(contrast_thresh)
    keep_valid = finite & (contrasts >= threshold)
    return skeleton_rc[valid_ids[keep_valid]]


def local_tangent(points_rc: np.ndarray) -> np.ndarray:
    centered = points_rc - points_rc.mean(axis=0)
    cov = centered.T @ centered / max(1, len(points_rc))
    w, v = np.linalg.eigh(cov)
    tangent = v[:, int(np.argmax(w))]
    norm = np.linalg.norm(tangent)
    if norm < 1e-12:
        return np.array([1.0, 0.0], dtype=np.float64)
    return tangent / norm


def sample_patch_mean(img: np.ndarray, center_rc: np.ndarray, radius: int) -> Optional[np.ndarray]:
    H, W = img.shape[:2]
    r, c = np.round(center_rc).astype(np.int32)
    if r < 0 or r >= H or c < 0 or c >= W:
        return None
    r0, r1 = max(0, r - radius), min(H, r + radius + 1)
    c0, c1 = max(0, c - radius), min(W, c + radius + 1)
    patch = img[r0:r1, c0:c1].reshape(-1, 3).astype(np.float32)
    patch = patch[np.any(patch > 0, axis=1)]
    if len(patch) == 0:
        return None
    return patch.mean(axis=0)


def paint_disk(img: np.ndarray, r: int, c: int, radius: int) -> None:
    H, W = img.shape[:2]
    r0, r1 = max(0, r - radius), min(H, r + radius + 1)
    c0, c1 = max(0, c - radius), min(W, c + radius + 1)
    img[r0:r1, c0:c1] = 0


def draw_line(img: np.ndarray, p0: np.ndarray, p1: np.ndarray, radius: int) -> None:
    length = int(np.ceil(np.linalg.norm(p1 - p0)))
    if length <= 0:
        paint_disk(img, int(p0[0]), int(p0[1]), radius)
        return
    rows = np.round(np.linspace(p0[0], p1[0], length + 1)).astype(np.int32)
    cols = np.round(np.linspace(p0[1], p1[1], length + 1)).astype(np.int32)
    for r, c in zip(rows, cols):
        paint_disk(img, int(r), int(c), radius)


def void_ratio(img: np.ndarray) -> float:
    nonblack = np.any(img > 0, axis=2)
    return void_ratio_from_coverage(nonblack)


def void_ratio_from_coverage(coverage: np.ndarray) -> float:
    nonblack = coverage.astype(bool)
    if nonblack.sum() == 0:
        return 1.0
    # Black pixels in the 8-neighborhood of covered pixels are void pixels.
    from scipy.ndimage import binary_dilation
    neigh = binary_dilation(nonblack, structure=np.ones((3, 3), dtype=bool))
    void = (~nonblack) & neigh
    return float(void.sum() / (nonblack.sum() + 1e-12))


def draw_sharp_point_overlay(img: np.ndarray, coords_rc: np.ndarray, sharp_mask: np.ndarray) -> None:
    H, W = img.shape[:2]
    sharp_coords = coords_rc[sharp_mask]
    valid = (
        (sharp_coords[:, 0] >= 0) & (sharp_coords[:, 0] < H) &
        (sharp_coords[:, 1] >= 0) & (sharp_coords[:, 1] < W)
    )
    if np.any(valid):
        rows = sharp_coords[valid, 0].astype(np.int32)
        cols = sharp_coords[valid, 1].astype(np.int32)
        img[rows, cols] = 0


def save_image(path: str, img: np.ndarray) -> None:
    Image.fromarray(img).save(path)
