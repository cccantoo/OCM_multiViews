from typing import Tuple, Optional
import numpy as np
from scipy.spatial import cKDTree


def statistical_outlier_filter(points: np.ndarray, nb_neighbors: int = 20, std_ratio: float = 2.0) -> np.ndarray:
    """统计离群点滤波。核心不是滤波，但真实岩体扫描常含漂浮点，工程复现需要打开。"""
    if len(points) <= nb_neighbors:
        return np.ones(len(points), dtype=bool)
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=nb_neighbors + 1)
    mean_d = dists[:, 1:].mean(axis=1)
    threshold = mean_d.mean() + std_ratio * mean_d.std()
    return mean_d <= threshold


# // 调整knn参数
def pca_normals(points: np.ndarray, knn: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """通过邻域 PCA 估计法向量。

    返回：
    - normals: 第三特征向量 e3，即局部面法向量；
    - eigvals: λ1>=λ2>=λ3；
    - eigvecs: 对应特征向量，eigvecs[i,:,0] 是 e1；
    - nn_idx: 每个点的 knn 邻域索引。
    """
    n = len(points)
    if n < knn + 1:
        raise ValueError(f"点数 {n} 过少，无法使用 knn={knn} 计算 PCA 法向量")
    tree = cKDTree(points)
    _, nn_idx = tree.query(points, k=knn + 1)
    nn_idx = nn_idx[:, 1:]

    normals = np.zeros((n, 3), dtype=np.float64)
    eigvals = np.zeros((n, 3), dtype=np.float64)
    eigvecs = np.zeros((n, 3, 3), dtype=np.float64)
    for i in range(n):
        neigh = points[nn_idx[i]]
        centered = neigh - points[i]
        cov = (centered.T @ centered) / knn
        w, v = np.linalg.eigh(cov)  # 升序
        order = np.argsort(w)[::-1]  # 降序 λ1>=λ2>=λ3
        w = w[order]
        v = v[:, order]
        eigvals[i] = w
        eigvecs[i] = v
        normals[i] = v[:, 2]

    normals = hemispherize(normals)
    return normals, eigvals, eigvecs, nn_idx


def hemispherize(normals: np.ndarray) -> np.ndarray:
    """半球化：z<0 的法向量反向，使其落到上半球。"""
    normals = normals.copy()
    norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    normals = normals / norms
    flip = normals[:, 2] < 0
    normals[flip] *= -1.0
    return normals


def global_plane_normal(points: np.ndarray) -> np.ndarray:
    """整体拟合面法向量，用于 Step 3.1 方向校准。"""
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = centered.T @ centered / len(points)
    w, v = np.linalg.eigh(cov)
    normal = v[:, np.argmin(w)]
    if normal[2] < 0:
        normal = -normal
    return normal / (np.linalg.norm(normal) + 1e-12)
