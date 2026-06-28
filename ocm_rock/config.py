from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class OCMConfig:
    # 论文默认 knn=20：小 knn 噪声大，大 knn 会过度平滑局部曲率。
    knn: int = 20
    # 论文默认 NPW-OC 两次收缩，在精度与效率间折中。
    contraction_iter: int = 2
    # 论文默认 CDP 细分 ndiv=5，对应 1321 个候选方向、平均邻近角 3.87°。
    cdp_subdivision: int = 5
    # ISRM 方向人工测量误差约 5°，论文选 dip>85° 作为边界法向量。
    boundary_dip_angle: float = 85.0
    # physical 是论文式绝对法向映射；adaptive_pca 会拉伸当前点云内部的法向差异。
    color_mapping: str = "physical"
    adaptive_color_percentile: float = 98.0
    adaptive_color_gain: float = 1.0
    # OCM 参考图像边长 Limg=800。
    image_length: int = 800
    # 论文填充空洞阈值 ratiovd<=0.1，FL 按 1,3,5... 增加。
    target_void_ratio: float = 0.10
    min_fill_length: int = 1
    max_fill_length: int = 15
    normal_smoothing_iter: int = 0
    color_aggregation: str = "last"
    sharp_threshold_mode: str = "percentile"
    sharp_percentile: float = 95.0
    sharp_mean_std_alpha: float = 1.0
    sharp_min_angle_deg: float = 8.0
    sharp_max_ratio: Optional[float] = 0.05
    draw_sharp_points: bool = False
    # 将 NPW-OC 收缩后的 sharp skeleton 作为连续黑色交线叠加到 OCM 图像上。
    draw_skeleton: bool = True
    skeleton_line_width: int = 1
    skeleton_link_neighbors: int = 3
    skeleton_max_link_px: float = 8.0
    skeleton_filter_mode: str = "color_contrast"
    skeleton_filter_radius: int = 3
    skeleton_filter_side_offset: int = 4
    skeleton_color_contrast_thresh: Optional[float] = None
    skeleton_color_contrast_percentile: float = 50.0
    skeleton_min_color_contrast_thresh: float = 60.0
    # 统计滤波，可根据点云噪声调节；论文核心流程未强调去噪，这里作为工程输入前处理。
    statistical_nb_neighbors: int = 20
    statistical_std_ratio: float = 2.0
    # 最小实例点数，过滤过小 Mask 反投影噪声。
    min_plane_points: int = 50
    # Mask R-CNN 训练参数，对齐论文。
    epochs: int = 260
    batch_size: int = 4
    lr: float = 1e-5
    nms_thresh: float = 0.7
    score_thresh: float = 0.05
    detections_per_img: int = 512


DEFAULT_CONFIG = OCMConfig()
