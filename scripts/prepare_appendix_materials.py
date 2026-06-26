import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "appendix_materials"
CODE_OUT = OUT / "01_core_code_screenshots"
RESULT_OUT = OUT / "02_experiment_result_screenshots"
TABLE_OUT = OUT / "03_quantitative_results"


CODE_SNIPPETS = [
    {
        "name": "01_ocm_pipeline_core.png",
        "title": "OCM生成主流程：点云 -> 法向量 -> OCM图像",
        "path": ROOT / "ocm_rock" / "pipeline.py",
        "ranges": [(16, 69), (85, 146)],
    },
    {
        "name": "02_optimal_color_mapping.png",
        "title": "最优色彩映射：CDP搜索、法向旋转、HSV/RGB映射",
        "path": ROOT / "ocm_rock" / "ocm_mapping.py",
        "ranges": [(28, 116)],
    },
    {
        "name": "03_sharp_skeleton_npw_oc.png",
        "title": "Sharp Point检测与NPW-OC骨架收缩",
        "path": ROOT / "ocm_rock" / "skeleton.py",
        "ranges": [(25, 65), (168, 232)],
    },
    {
        "name": "04_sam_mask_to_labelme.png",
        "title": "SAM二值mask转LabelMe polygon标注",
        "path": ROOT / "scripts" / "sam_masks_to_labelme.py",
        "ranges": [(11, 73)],
    },
    {
        "name": "05_maskrcnn_dataset.png",
        "title": "Mask R-CNN数据适配：LabelMe polygon转mask/box/label",
        "path": ROOT / "ocm_rock" / "dataset.py",
        "ranges": [(11, 71)],
    },
    {
        "name": "06_maskrcnn_model_training.png",
        "title": "Mask R-CNN模型构建与训练入口",
        "path": ROOT / "ocm_rock" / "model.py",
        "ranges": [(8, 45)],
        "extra": ROOT / "scripts" / "train_maskrcnn.py",
        "extra_ranges": [(35, 75)],
    },
    {
        "name": "07_infer_backprojection.png",
        "title": "二维mask推理与三维反投影",
        "path": ROOT / "scripts" / "run_infer.py",
        "ranges": [(15, 48), (51, 88)],
    },
    {
        "name": "08_multiview_fusion.png",
        "title": "多视角结构面候选融合",
        "path": ROOT / "scripts" / "fuse_multiview_planes.py",
        "ranges": [(190, 259), (337, 394)],
    },
]


RESULT_FILES = [
    (
        ROOT / "outputs" / "rockbench_mbda_fixed_full" / "ocm_image.png",
        RESULT_OUT / "rockbench_01_ocm_image.png",
    ),
    (
        ROOT / "outputs" / "infer_rockbench_view000" / "mask_pred.png",
        RESULT_OUT / "rockbench_02_mask_pred_view000.png",
    ),
    (
        ROOT / "outputs" / "infer_rockbench_view000" / "stereonet.png",
        RESULT_OUT / "rockbench_03_stereonet_view000.png",
    ),
    (
        ROOT / "outputs" / "fused_rockbench_geometric" / "fused_stereonet.png",
        RESULT_OUT / "rockbench_04_fused_stereonet.png",
    ),
    (
        ROOT / "outputs" / "jiangmula_mbda_skeleton_default" / "ocm_image.png",
        RESULT_OUT / "jiangmula_01_ocm_image.png",
    ),
    (
        ROOT / "outputs" / "infer_jiangmula_multiview" / "view_075" / "mask_pred.png",
        RESULT_OUT / "jiangmula_02_mask_pred_view075.png",
    ),
    (
        ROOT / "outputs" / "infer_jiangmula_multiview" / "view_075" / "stereonet.png",
        RESULT_OUT / "jiangmula_03_stereonet_view075.png",
    ),
    (
        ROOT / "data" / "sam" / "ocm_image_075_nosk" / "ocm_image_075_nosk_J001_overlay.png",
        RESULT_OUT / "sam_01_overlay_example.png",
    ),
    (
        ROOT / "data" / "sam" / "ocm_image_075_nosk" / "ocm_image_075_nosk_J001_mask.png",
        RESULT_OUT / "sam_02_binary_mask_example.png",
    ),
]


def load_font(size: int, bold: bool = False):
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def read_ranges(path: Path, ranges):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    selected = []
    for start, end in ranges:
        for lineno in range(start, min(end, len(lines)) + 1):
            text = lines[lineno - 1]
            if '"""' in text or "'''" in text:
                continue
            if text.strip().startswith("#") and any(ord(ch) > 127 for ch in text):
                continue
            selected.append((lineno, text.rstrip()))
        selected.append((None, ""))
    return selected


def render_code_image(item):
    title_font = load_font(30, bold=True)
    meta_font = load_font(18)
    code_font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 18)

    rows = read_ranges(item["path"], item["ranges"])
    if item.get("extra"):
        rows.append((None, ""))
        rows.append((None, f"# {item['extra'].relative_to(ROOT)}"))
        rows.extend(read_ranges(item["extra"], item["extra_ranges"]))

    max_chars = max(len(text) for _, text in rows) if rows else 80
    line_h = 25
    pad = 32
    gutter = 82
    width = min(1900, max(1200, pad * 2 + gutter + max_chars * 11))
    height = pad * 2 + 68 + max(1, len(rows)) * line_h
    img = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, width, 78], fill="#1f2937")
    draw.text((pad, 18), item["title"], fill="white", font=title_font)
    source = str(item["path"].relative_to(ROOT)).replace("\\", "/")
    draw.text((pad, 88), f"source: {source}", fill="#4b5563", font=meta_font)

    y = 120
    for lineno, text in rows:
        if lineno is None and text == "":
            y += line_h // 2
            continue
        if lineno is None:
            draw.text((pad + gutter, y), text, fill="#6b7280", font=code_font)
            y += line_h
            continue
        if y // line_h % 2 == 0:
            draw.rectangle([pad - 10, y - 2, width - pad + 10, y + line_h - 2], fill="#ffffff")
        draw.text((pad, y), f"{lineno:>4}", fill="#9ca3af", font=code_font)
        color = "#111827"
        stripped = text.strip()
        if stripped.startswith("def ") or stripped.startswith("class "):
            color = "#0f766e"
        elif stripped.startswith("return ") or stripped.startswith("if ") or stripped.startswith("for "):
            color = "#7c3aed"
        draw.text((pad + gutter, y), text.replace("\t", "    "), fill=color, font=code_font)
        y += line_h

    img.save(CODE_OUT / item["name"])


def copy_result_images():
    for src, dst in RESULT_FILES:
        if src.exists():
            shutil.copy2(src, dst)


def plot_labeled_projection(points_path: Path, labels_path: Path, out_path: Path, title: str):
    points = np.load(points_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")
    rng = np.random.default_rng(42)
    n = len(points)
    sample_size = min(140000, n)
    ids = rng.choice(n, sample_size, replace=False)
    pts = np.asarray(points[ids])
    labs = np.asarray(labels[ids])

    fig, ax = plt.subplots(figsize=(10, 7), dpi=180)
    bg = labs <= 0
    ax.scatter(pts[bg, 0], pts[bg, 2], s=0.08, c="#d1d5db", alpha=0.25, linewidths=0)
    fg = labs > 0
    ax.scatter(pts[fg, 0], pts[fg, 2], s=0.22, c=labs[fg], cmap="tab20", alpha=0.85, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def create_quantitative_tables():
    rock_meta = json.loads((ROOT / "outputs" / "rockbench_mbda_fixed_full" / "ocm_metadata.json").read_text(encoding="utf-8"))
    jiang_meta = json.loads((ROOT / "outputs" / "jiangmula_mbda_skeleton_default" / "ocm_metadata.json").read_text(encoding="utf-8"))
    fusion = json.loads((ROOT / "outputs" / "fused_rockbench_geometric" / "fusion_summary.json").read_text(encoding="utf-8"))
    jiang_mv = json.loads((ROOT / "outputs" / "infer_jiangmula_multiview" / "multi_view_summary.json").read_text(encoding="utf-8"))

    rows = [
        ["Rockbench原始点云规模", rock_meta["point_count"], "点"],
        ["Rockbench OCM图像尺寸", f'{rock_meta["image_height"]} x {rock_meta["image_width"]}', "像素"],
        ["Rockbench OCM空洞率", f'{rock_meta["void_ratio"]:.4f}', "目标<=0.1"],
        ["Rockbench sharp point数量", rock_meta["sharp_point_count"], "点"],
        ["Rockbench融合前候选实例", fusion["candidate_total"], "个"],
        ["Rockbench三视角实例数", "43 / 26 / 29", "个"],
        ["Rockbench几何匹配边数", fusion["matched_edges"], "条"],
        ["Rockbench融合后结构面", fusion["fused_planes"], "个"],
        ["Rockbench融合后标记点数", fusion["fused_labeled_points"], "点"],
        ["Jiangmula原始点云规模", jiang_meta["point_count"], "点"],
        ["Jiangmula OCM图像尺寸", f'{jiang_meta["image_height"]} x {jiang_meta["image_width"]}', "像素"],
        ["Jiangmula OCM空洞率", f'{jiang_meta["void_ratio"]:.4f}', "目标<=0.1"],
        ["Jiangmula多视角测试", "0-180, interval 15", "度"],
        ["Jiangmula单视角实例数范围", f'{min(x["mask_instances"] for x in jiang_mv)} - {max(x["mask_instances"] for x in jiang_mv)}', "个"],
        ["Jiangmula单视角结构面数范围", f'{min(x["plane_count"] for x in jiang_mv)} - {max(x["plane_count"] for x in jiang_mv)}', "个"],
    ]

    with open(TABLE_OUT / "quantitative_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["指标", "数值", "说明"])
        writer.writerows(rows)

    render_table_image(rows, TABLE_OUT / "quantitative_results.png")


def render_table_image(rows, out_path: Path):
    title_font = load_font(30, bold=True)
    cell_font = load_font(21)
    header_font = load_font(22, bold=True)
    width = 1380
    row_h = 44
    height = 100 + row_h * (len(rows) + 1) + 30
    img = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width, 72], fill="#1f2937")
    draw.text((28, 18), "实验量化结果汇总", fill="white", font=title_font)
    y = 92
    col_x = [32, 620, 910]
    headers = ["指标", "数值", "说明"]
    draw.rectangle([24, y - 8, width - 24, y + row_h - 8], fill="#e5e7eb")
    for x, h in zip(col_x, headers):
        draw.text((x, y), h, fill="#111827", font=header_font)
    y += row_h
    for i, row in enumerate(rows):
        fill = "#f9fafb" if i % 2 == 0 else "#ffffff"
        draw.rectangle([24, y - 8, width - 24, y + row_h - 8], fill=fill)
        for x, value in zip(col_x, row):
            draw.text((x, y), str(value), fill="#111827", font=cell_font)
        y += row_h
    img.save(out_path)


def write_readme():
    text = """# 附件成果材料说明

本目录用于录屏、汇报或提交附件，按材料类型分为三部分。

## 01_core_code_screenshots

- `01_ocm_pipeline_core.png`：OCM生成主流程，展示点云读取、PCA法向量、骨架、色彩映射和OCM图像输出。
- `02_optimal_color_mapping.png`：最优色彩映射核心代码，展示CDP候选方向、最优旋转和HSV/RGB映射。
- `03_sharp_skeleton_npw_oc.png`：sharp point检测与NPW-OC骨架收缩。
- `04_sam_mask_to_labelme.png`：SAM输出mask转换为LabelMe JSON标注。
- `05_maskrcnn_dataset.png`：LabelMe polygon转换为Mask R-CNN训练所需masks、boxes、labels。
- `06_maskrcnn_model_training.png`：Mask R-CNN模型构建、预测头替换和训练入口。
- `07_infer_backprojection.png`：二维mask推理和三维点云反投影。
- `08_multiview_fusion.png`：多视角结构面候选的几何一致性融合。

## 02_experiment_result_screenshots

- `rockbench_01_ocm_image.png`：Rockbench点云生成的OCM图像。
- `rockbench_02_mask_pred_view000.png`：Rockbench单视角结构面实例分割结果。
- `rockbench_03_stereonet_view000.png`：Rockbench单视角姿态极射赤平投影。
- `rockbench_04_fused_stereonet.png`：Rockbench多视角融合后的姿态极射赤平投影。
- `rockbench_05_fused_3d_projection.png`：Rockbench融合结构面反投影到三维点云后的X-Z投影预览。
- `jiangmula_01_ocm_image.png`：Jiangmula真实隧道点云生成的OCM图像。
- `jiangmula_02_mask_pred_view075.png`：Jiangmula view_075阶段性实例分割结果。
- `jiangmula_03_stereonet_view075.png`：Jiangmula view_075姿态结果。
- `jiangmula_04_view075_3d_projection.png`：Jiangmula view_075反投影结果X-Z投影预览。
- `sam_01_overlay_example.png`：SAM辅助标注overlay示例。
- `sam_02_binary_mask_example.png`：SAM输出二值mask示例。

## 03_quantitative_results

- `quantitative_results.csv`：量化结果表。
- `quantitative_results.png`：可直接放入PPT或附件的量化结果截图。

## 结果讲法建议

Rockbench作为流程完整验证结果展示，重点说明OCM图像生成、实例分割、三维反投影、多视角融合和姿态参数输出已经形成闭环。

Jiangmula作为真实场景迁移验证结果展示，重点说明流程已跑通，但受真实点云噪声、视角、结构尺度和训练数据分布影响，当前用于阶段性验证和后续优化依据。
"""
    (OUT / "README_附件说明.md").write_text(text, encoding="utf-8")


def main():
    CODE_OUT.mkdir(parents=True, exist_ok=True)
    RESULT_OUT.mkdir(parents=True, exist_ok=True)
    TABLE_OUT.mkdir(parents=True, exist_ok=True)

    for item in CODE_SNIPPETS:
        render_code_image(item)

    copy_result_images()
    plot_labeled_projection(
        ROOT / "outputs" / "fused_rockbench_geometric" / "points.npy",
        ROOT / "outputs" / "fused_rockbench_geometric" / "fused_labels3d.npy",
        RESULT_OUT / "rockbench_05_fused_3d_projection.png",
        "Rockbench fused 3D labels projection",
    )
    plot_labeled_projection(
        ROOT / "outputs" / "infer_jiangmula_multiview" / "view_075" / "points.npy",
        ROOT / "outputs" / "infer_jiangmula_multiview" / "view_075" / "labels3d.npy",
        RESULT_OUT / "jiangmula_04_view075_3d_projection.png",
        "Jiangmula view_075 3D labels projection",
    )
    create_quantitative_tables()
    write_readme()
    print(f"appendix materials written to: {OUT}")


if __name__ == "__main__":
    main()
