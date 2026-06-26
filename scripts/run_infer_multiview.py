import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ocm_rock.config import OCMConfig
from ocm_rock.io_utils import write_json
from ocm_rock.model import load_model
from ocm_rock.pipeline import generate_multi_view_ocm_from_pointcloud, format_view_angle
from ocm_rock.plane_analysis import analyze_planes, write_planes_csv
from ocm_rock.visualize import colorize_by_labels, plot_stereonet


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
    parser = argparse.ArgumentParser(description="Multi-view OCM + Mask R-CNN inference")
    parser.add_argument("--point_cloud", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--views", default="0,15,30,45,60,75,90,105,120,135,150,165,180")
    parser.add_argument("--score", type=float, default=0.5)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--min_size", type=int, default=800)
    parser.add_argument("--max_size", type=int, default=1333)
    parser.add_argument("--no_draw_skeleton", action="store_true")
    parser.add_argument("--skip_ply", action="store_true", help="Skip colored_planes.ply to save disk/time")
    args = parser.parse_args()

    cfg = OCMConfig()
    cfg.draw_skeleton = not args.no_draw_skeleton
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    views = parse_views(args.views)
    generate_multi_view_ocm_from_pointcloud(args.point_cloud, args.out, cfg, views)

    device = choose_device(args.device)
    model, device = load_model(
        args.weights,
        device=device,
        score_thresh=cfg.score_thresh,
        nms_thresh=cfg.nms_thresh,
        detections_per_img=cfg.detections_per_img,
        min_size=args.min_size,
        max_size=args.max_size,
    )

    summary = []
    for angle in views:
        view_name = f"view_{format_view_angle(angle)}"
        view_dir = out / view_name
        row = infer_one_view(view_dir, model, device, cfg, args.score, skip_ply=args.skip_ply)
        row["view_angle_deg"] = float(angle)
        row["view"] = view_name
        summary.append(row)
        print(
            f"{view_name}: masks={row['mask_instances']}, planes={row['plane_count']}, "
            f"labeled_points={row['labeled_points']}"
        )

    write_summary(out / "multi_view_summary.csv", summary)
    write_json(str(out / "multi_view_summary.json"), summary)
    best = max(summary, key=lambda r: (r["plane_count"], r["labeled_points"])) if summary else None
    if best:
        print(f"Best by plane_count/labeled_points: {best['view']} ({best['view_angle_deg']} deg)")


def choose_device(requested: str):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return requested


@torch.no_grad()
def infer_one_view(view_dir: Path, model, device: str, cfg: OCMConfig, score: float, skip_ply: bool = False):
    img = Image.open(view_dir / "ocm_image.png").convert("RGB")
    image_tensor = torch.as_tensor(np.array(img).transpose(2, 0, 1) / 255.0, dtype=torch.float32)
    pred = model([image_tensor.to(device)])[0]

    mask_label = build_instance_label_map(pred, img.size[::-1], score_thr=score)
    Image.fromarray(color_mask(mask_label)).save(view_dir / "mask_pred.png")
    np.save(view_dir / "mask_label.npy", mask_label)

    points = np.load(view_dir / "points.npy")
    coords_rc = np.load(view_dir / "coords_rc.npy")
    labels3d = map_mask_to_points(mask_label, coords_rc)
    np.save(view_dir / "labels3d.npy", labels3d)

    planes = analyze_planes(points, labels3d, cfg.min_plane_points)
    write_json(str(view_dir / "planes.json"), planes)
    write_planes_csv(str(view_dir / "planes.csv"), planes)
    plot_stereonet(planes, str(view_dir / "stereonet.png"))
    if not skip_ply:
        colorize_by_labels(points, labels3d, str(view_dir / "colored_planes.ply"))

    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "mask_instances": int(mask_label.max()),
        "plane_count": int(len(planes)),
        "labeled_points": int((labels3d > 0).sum()),
        "image_height": int(mask_label.shape[0]),
        "image_width": int(mask_label.shape[1]),
        "out_dir": str(view_dir),
    }


def build_instance_label_map(pred, hw, score_thr=0.5):
    H, W = hw
    label = np.zeros((H, W), dtype=np.uint16)
    if "masks" not in pred:
        return label
    scores = pred["scores"].detach().cpu().numpy()
    masks = pred["masks"].detach().cpu().numpy()[:, 0]
    order = np.argsort(scores)
    cur = 1
    for idx in order:
        if scores[idx] < score_thr:
            continue
        m = masks[idx] > 0.5
        label[m] = cur
        cur += 1
    return label


def color_mask(label):
    rng = np.random.default_rng(123)
    rgb = np.zeros((*label.shape, 3), dtype=np.uint8)
    for lab in sorted(set(label.ravel().tolist())):
        if lab == 0:
            continue
        rgb[label == lab] = (rng.random(3) * 220 + 30).astype(np.uint8)
    return rgb


def map_mask_to_points(mask_label, coords_rc):
    H, W = mask_label.shape
    labels = np.zeros(len(coords_rc), dtype=np.int32)
    for i, (r, c) in enumerate(coords_rc):
        if 0 <= r < H and 0 <= c < W:
            labels[i] = int(mask_label[r, c])
    return labels


def write_summary(path: Path, rows):
    if not rows:
        return
    keys = [
        "view",
        "view_angle_deg",
        "mask_instances",
        "plane_count",
        "labeled_points",
        "image_height",
        "image_width",
        "out_dir",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


if __name__ == "__main__":
    main()
