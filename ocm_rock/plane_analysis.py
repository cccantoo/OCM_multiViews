from typing import Dict, List, Tuple
import csv
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .preprocess import global_plane_normal, hemispherize


def normal_to_orientation(normal: np.ndarray) -> Tuple[float, float]:
    """论文 Eq.(24)-(25)：由结构面法向量计算倾向 DD 与倾角 DA。"""
    n = normal / (np.linalg.norm(normal) + 1e-12)
    if n[2] < 0:
        n = -n
    xp, yp, zp = n
    denom = np.sqrt(xp * xp + yp * yp) + 1e-12
    # 与论文公式一致：以 y 方向为参考，xp 判断象限。
    dd = np.degrees(np.arccos(np.clip(yp / denom, -1.0, 1.0))) if xp > 0 else 360.0 - np.degrees(np.arccos(np.clip(yp / denom, -1.0, 1.0)))
    da = np.degrees(np.arccos(np.clip(zp / (np.linalg.norm(n) + 1e-12), -1.0, 1.0)))
    return float(dd % 360.0), float(da)


def analyze_planes(points: np.ndarray, labels: np.ndarray, min_points: int = 50) -> List[Dict]:
    """对每个识别结构面拟合法向量、计算倾向倾角、中心点和三维迹长。"""
    results = []
    for lab in sorted(set(labels.tolist())):
        if lab <= 0:
            continue
        ids = np.where(labels == lab)[0]
        if len(ids) < min_points:
            continue
        pts = points[ids]
        normal = global_plane_normal(pts)
        dd, da = normal_to_orientation(normal)
        # 三维迹长：论文应用部分采用结构面点集中最远两点距离。
        trace_len = approximate_trace_length(pts)
        results.append({
            "plane_id": int(lab),
            "num_points": int(len(ids)),
            "normal": [float(x) for x in normal],
            "dip_direction_DD": dd,
            "dip_angle_DA": da,
            "center": [float(x) for x in pts.mean(axis=0)],
            "trace_length": float(trace_len),
        })
    return results


def approximate_trace_length(pts: np.ndarray, sample: int = 3000) -> float:
    if len(pts) < 2:
        return 0.0
    if len(pts) > sample:
        rng = np.random.default_rng(42)
        pts = pts[rng.choice(len(pts), sample, replace=False)]
    # 用包围盒对角近似筛选，再做两次 farthest point 加速。
    p0 = pts[0]
    i = int(np.argmax(np.linalg.norm(pts - p0, axis=1)))
    j = int(np.argmax(np.linalg.norm(pts - pts[i], axis=1)))
    return float(np.linalg.norm(pts[i] - pts[j]))


def best_orientation_grouping(normals: np.ndarray, k_min: int = 2, k_max: int = 6) -> Tuple[np.ndarray, int, float]:
    """论文应用部分：KMeans k=2..6，用 Silhouette 选最优组数。"""
    normals = hemispherize(normals)
    best_score = -1.0
    best_labels = None
    best_k = k_min
    for k in range(k_min, min(k_max, len(normals) - 1) + 1):
        labels = KMeans(n_clusters=k, n_init="auto", random_state=42).fit_predict(normals)
        score = silhouette_score(normals, labels)
        if score > best_score:
            best_score, best_labels, best_k = score, labels, k
    return best_labels, best_k, float(best_score)


def write_planes_csv(path: str, planes: List[Dict]) -> None:
    keys = ["plane_id", "num_points", "dip_direction_DD", "dip_angle_DA", "trace_length", "normal", "center"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for p in planes:
            row = p.copy()
            row["normal"] = str(row["normal"])
            row["center"] = str(row["center"])
            writer.writerow(row)
