import json
from pathlib import Path
from typing import Tuple, Optional

import numpy as np


def load_point_cloud(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """读取点云。

    支持：
    - txt/xyz/csv: 至少 3 列 x y z，可附带 r g b；
    - ply/pcd 等 Open3D 支持格式。
    返回 points(N,3), colors(N,3)；colors 若不存在则为 None，若存在则归一到 [0,1]。
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in [".txt", ".xyz", ".csv"]:
        # 自动判断分隔符：空格、逗号、制表符均可。
        try:
            arr = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
        except Exception:
            arr = np.genfromtxt(path, delimiter=",", invalid_raise=False)
        arr = arr[~np.isnan(arr).any(axis=1)]
        if arr.shape[1] < 3:
            raise ValueError(f"点云文件至少需要 x y z 三列: {path}")
        pts = arr[:, :3].astype(np.float64)
        colors = None
        if arr.shape[1] >= 6:
            colors = arr[:, 3:6].astype(np.float64)
            if colors.max() > 1.0:
                colors = colors / 255.0
            colors = np.clip(colors, 0.0, 1.0)
        return pts, colors
    else:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(path))
        pts = np.asarray(pcd.points, dtype=np.float64)
        colors = np.asarray(pcd.colors, dtype=np.float64) if pcd.has_colors() else None
        return pts, colors


def save_point_cloud_ply(path: str, points: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        if colors.max() > 1:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1).astype(np.float64))
    o3d.io.write_point_cloud(str(path), pcd)


def write_json(path: str, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
