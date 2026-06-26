from typing import Tuple
import numpy as np
from scipy.spatial import cKDTree


def acute_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """论文 Eq.(2)：法向量夹角距离，取绝对点积形成锐角。"""
    dot = np.abs(np.sum(a * b, axis=-1))
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    cosv = np.clip(dot / (na * nb + 1e-12), -1.0, 1.0)
    return np.degrees(np.arccos(cosv))


# def detect_sharp_points(normals: np.ndarray, nn_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     """论文 Step 1.2：邻域法向量平均角变异 δ_i，大于全局均值则为 sharp point。"""
#     n = len(normals)
#     delta = np.zeros(n, dtype=np.float64)
#     for i in range(n):
#         nb_normals = normals[nn_idx[i]]
#         delta[i] = acute_angle_deg(normals[i][None, :], nb_normals).mean()
#     sharp_mask = delta > delta.mean()
#     return sharp_mask, delta

def detect_sharp_points(
    normals: np.ndarray,
    nn_idx: np.ndarray,
    threshold_mode: str = "percentile",
    percentile: float = 95.0,
    mean_std_alpha: float = 1.0,
    min_angle_deg: float = 8.0,
    max_sharp_ratio: float = 0.05,
    exclude_self: bool = True,
    chunk_size: int = 50000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    鲁棒版 sharp point 检测。

    论文原始逻辑：
        1. 计算每个点与邻域点法向量的平均锐角变化 delta_i；
        2. delta_i > mean(delta) 的点判定为 sharp point。

    你的 Lato 点云中，论文默认 mean 阈值会导致 sharp point 过多。
    因此这里保留 delta 计算方式，但把阈值改成更稳健的方式。

    参数说明：
    normals:
        点云法向量，shape = [N, 3]

    nn_idx:
        每个点的邻域索引，shape = [N, K]
        注意：如果 nn_idx 第一列包含自身点，会自动排除。

    threshold_mode:
        "paper_mean"  : 完全复现论文，delta > mean(delta)
        "mean_std"    : delta > mean(delta) + alpha * std(delta)
        "percentile"  : delta > percentile(delta)，推荐 Lato 使用

    percentile:
        分位数阈值。
        95 表示只保留角度变化最大的 5% 点。
        推荐先用 95；如果仍然过多，改成 97。

    mean_std_alpha:
        threshold_mode="mean_std" 时使用。
        推荐 1.0 或 1.5。

    min_angle_deg:
        最小角度阈值。
        防止一些很小的法向量波动也被判定为 sharp point。
        推荐 8~12。

    max_sharp_ratio:
        最大 sharp point 比例。
        推荐 0.03~0.06。
        你的原始结果是 0.333，明显过高。

    exclude_self:
        如果邻域索引中包含自身点，则排除自身点，避免 0° 角拉低 delta。

    chunk_size:
        分块计算，避免一次性占用过多内存。

    返回：
    sharp_mask:
        bool 数组，True 表示 sharp point。

    delta:
        每个点的邻域法向量平均角变异。
    """
    n = len(normals)

    if nn_idx.ndim != 2:
        raise ValueError("nn_idx 必须是二维数组，shape = [N, K]")

    # 如果 knn 结果第一列是自身点，则去掉。
    idx = nn_idx
    if exclude_self and idx.shape[1] > 1:
        row_ids = np.arange(n)
        if np.all(idx[:, 0] == row_ids):
            idx = idx[:, 1:]

    delta = np.zeros(n, dtype=np.float64)

    # 分块计算 delta，避免 76 万点一次性展开内存过大。
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        center_normals = normals[start:end, None, :]      # [B, 1, 3]
        neighbor_normals = normals[idx[start:end]]        # [B, K, 3]

        dot = np.abs(np.sum(center_normals * neighbor_normals, axis=-1))
        na = np.linalg.norm(center_normals, axis=-1)
        nb = np.linalg.norm(neighbor_normals, axis=-1)

        cosv = np.clip(dot / (na * nb + 1e-12), -1.0, 1.0)
        angle = np.degrees(np.arccos(cosv))

        delta[start:end] = angle.mean(axis=1)

    # 1. 计算基础阈值
    if threshold_mode == "paper_mean":
        threshold = float(delta.mean())

    elif threshold_mode == "mean_std":
        threshold = float(delta.mean() + mean_std_alpha * delta.std())

    elif threshold_mode == "percentile":
        threshold = float(np.percentile(delta, percentile))

    else:
        raise ValueError(
            "threshold_mode 只能是 'paper_mean'、'mean_std' 或 'percentile'"
        )

    # 2. 加一个最小角度阈值，避免噪声波动过小也被识别为 sharp
    threshold = max(threshold, float(min_angle_deg))

    sharp_mask = delta > threshold

    # 3. 强制限制 sharp point 最大比例
    #    你的数据原来是 33%，这里建议压到 5% 左右。
    if max_sharp_ratio is not None and max_sharp_ratio > 0:
        current_ratio = sharp_mask.mean()

        if current_ratio > max_sharp_ratio:
            ratio_threshold = float(
                np.percentile(delta, 100.0 * (1.0 - max_sharp_ratio))
            )
            threshold = max(threshold, ratio_threshold)
            sharp_mask = delta > threshold

    print("========== Sharp Point 检测统计 ==========")
    print(f"point_count        : {n}")
    print(f"delta_mean         : {delta.mean():.4f}")
    print(f"delta_std          : {delta.std():.4f}")
    print(f"delta_min          : {delta.min():.4f}")
    print(f"delta_max          : {delta.max():.4f}")
    print(f"threshold_mode     : {threshold_mode}")
    print(f"final_threshold    : {threshold:.4f}")
    print(f"sharp_point_count  : {int(sharp_mask.sum())}")
    print(f"sharp_ratio        : {sharp_mask.mean() * 100:.2f}%")
    print("==========================================")

    return sharp_mask, delta


def npw_oriented_contraction(
    points: np.ndarray,
    sharp_mask: np.ndarray,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    knn: int = 20,
    iterations: int = 2,
) -> np.ndarray:
    """论文 Step 1.3：NPW-OC 邻域 PCA 加权定向收缩。

    实现对应 Eq.(5)-(8)：
    - u1=λ1/(λ1+λ2+λ3)，表示局部线性显著性；
    - wc=u1^2，用较大权重吸引骨架附近点；
    - 先做加权邻域收缩，再把位移投影到 e2，使点沿垂直切向方向收缩。
    """
    sharp_idx = np.where(sharp_mask)[0]
    if len(sharp_idx) == 0:
        return np.empty((0, 3), dtype=np.float64)

    if eigvals.shape[0] != len(points) or eigvecs.shape[0] != len(points):
        raise ValueError("eigvals/eigvecs must correspond to the full input point cloud")

    shp0 = points[sharp_idx]
    shp = shp0.copy()
    sharp_eigvals = eigvals[sharp_idx]
    sharp_eigvecs = eigvecs[sharp_idx]
    u1 = sharp_eigvals[:, 0] / (sharp_eigvals.sum(axis=1) + 1e-12)
    wc = u1 ** 2
    e2 = sharp_eigvecs[:, :, 1]

    tree = cKDTree(shp0)
    k = min(knn + 1, len(shp0))
    _, local_idx = tree.query(shp0, k=k)
    if local_idx.ndim == 1:
        local_idx = local_idx[:, None]
    local_idx = local_idx[:, 1:] if local_idx.shape[1] > 1 else local_idx

    for _ in range(iterations):
        new_shp = shp.copy()
        for i in range(len(shp)):
            ids = local_idx[i]
            weights = wc[ids]
            if weights.sum() < 1e-12:
                continue
            disp = (weights[:, None] * (shp[ids] - shp[i])).sum(axis=0) / weights.sum()
            # 定向校准：将位移投影到 e2，保持沿骨架方向的连续性。
            disp_norm = np.linalg.norm(disp)
            if disp_norm < 1e-12:
                continue
            proj = np.dot(disp, e2[i]) * e2[i]
            proj_norm = np.linalg.norm(proj)
            if proj_norm < 1e-12:
                new_shp[i] = shp[i] + disp
            else:
                new_shp[i] = shp[i] + proj / proj_norm * disp_norm
        shp = new_shp
    return shp
