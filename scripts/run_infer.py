import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from ocm_rock.config import OCMConfig
from ocm_rock.pipeline import generate_ocm_from_pointcloud
from ocm_rock.model import load_model
from ocm_rock.plane_analysis import analyze_planes, write_planes_csv
from ocm_rock.visualize import plot_stereonet, colorize_by_labels
from ocm_rock.io_utils import write_json


def main():
    parser = argparse.ArgumentParser(description="OCM + Mask R-CNN 推理 + 3D 反投影")
    parser.add_argument("--point_cloud", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--score", type=float, default=0.5, help="实例置信度阈值")
    args = parser.parse_args()

    cfg = OCMConfig()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    generate_ocm_from_pointcloud(args.point_cloud, args.out, cfg)
    img_path = out / "ocm_image.png"
    img = Image.open(img_path).convert("RGB")
    image_tensor = torch.as_tensor(np.array(img).transpose(2,0,1) / 255.0, dtype=torch.float32)

    model, device = load_model(args.weights, score_thresh=cfg.score_thresh, nms_thresh=cfg.nms_thresh, detections_per_img=cfg.detections_per_img)
    with torch.no_grad():
        pred = model([image_tensor.to(device)])[0]

    mask_label = build_instance_label_map(pred, img.size[::-1], score_thr=args.score)
    Image.fromarray(color_mask(mask_label)).save(out / "mask_pred.png")
    np.save(out / "mask_label.npy", mask_label)

    points = np.load(out / "points.npy")
    coords_rc = np.load(out / "coords_rc.npy")
    labels3d = map_mask_to_points(mask_label, coords_rc)
    np.save(out / "labels3d.npy", labels3d)

    planes = analyze_planes(points, labels3d, cfg.min_plane_points)
    write_json(str(out / "planes.json"), planes)
    write_planes_csv(str(out / "planes.csv"), planes)
    plot_stereonet(planes, str(out / "stereonet.png"))
    colorize_by_labels(points, labels3d, str(out / "colored_planes.ply"))
    print(f"识别完成，结构面数量：{len(planes)}，输出目录：{out}")


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


if __name__ == "__main__":
    main()
