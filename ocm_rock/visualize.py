from typing import List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt

from .io_utils import save_point_cloud_ply


def plot_stereonet(planes: List[Dict], out_path: str) -> None:
    """绘制极射赤平投影的极点图。

    这里使用极坐标近似：theta=倾向 DD，r=tan(DA/2)，对应等角极射投影。
    """
    fig = plt.figure(figsize=(6, 6), dpi=160)
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 1.0)
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.set_title("Stereographic projection of discontinuity poles", pad=18)
    for p in planes:
        theta = np.deg2rad(p["dip_direction_DD"])
        r = np.tan(np.deg2rad(p["dip_angle_DA"]) / 2.0)
        ax.scatter(theta, r, s=24)
        ax.text(theta, min(r + 0.04, 1.0), str(p["plane_id"]), fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def colorize_by_labels(points: np.ndarray, labels: np.ndarray, out_path: str) -> None:
    rng = np.random.default_rng(2024)
    colors = np.zeros((len(points), 3), dtype=np.float64)
    unique = [x for x in sorted(set(labels.tolist())) if x > 0]
    table = {lab: rng.random(3) * 0.75 + 0.25 for lab in unique}
    for lab, color in table.items():
        colors[labels == lab] = color
    save_point_cloud_ply(out_path, points, colors)
