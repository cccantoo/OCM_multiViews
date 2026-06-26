import argparse
import contextlib
import io
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ocm_rock.config import OCMConfig
from ocm_rock.dataset import LabelMeMaskDataset, collate_fn
from ocm_rock.model import build_maskrcnn


def main():
    parser = argparse.ArgumentParser(description="训练 OCM Mask R-CNN")
    parser.add_argument("--dataset", required=True, help="数据集根目录，包含 train/val")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument("--epochs", type=int, default=260)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--no_pretrained", action="store_true", help="Do not download/use COCO pretrained weights")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Training device")
    parser.add_argument("--min_size", type=int, default=800, help="Torchvision detection resize min_size")
    parser.add_argument("--max_size", type=int, default=1333, help="Torchvision detection resize max_size")
    parser.add_argument("--eval_ap_every", type=int, default=1, help="Evaluate COCO mask AP every N epochs; 0 disables AP evaluation")
    args = parser.parse_args()

    cfg = OCMConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build and check the NVIDIA driver."
            )
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    train_ds = LabelMeMaskDataset(args.dataset, "train")
    val_ds = LabelMeMaskDataset(args.dataset, "val")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    model = build_maskrcnn(
        score_thresh=cfg.score_thresh,
        nms_thresh=cfg.nms_thresh,
        detections_per_img=cfg.detections_per_img,
        pretrained=not args.no_pretrained,
        min_size=args.min_size,
        max_size=args.max_size,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=cfg.lr * 0.01)

    log_path = out / "train_log.txt"
    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        for images, targets in tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.epochs}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
        if epoch < 150:
            scheduler.step()

        val_loss = evaluate_loss(model, val_loader, device)
        ap_metrics = {"AP": float("nan"), "AP50": float("nan"), "AP75": float("nan")}
        if args.eval_ap_every > 0 and ((epoch + 1) % args.eval_ap_every == 0 or epoch + 1 == cfg.epochs):
            ap_metrics = evaluate_coco_mask_ap(model, val_loader, device)
        line = (
            f"epoch={epoch+1}, train_loss={total/max(1,len(train_loader)):.6f}, "
            f"val_loss={val_loss:.6f}, val_AP={ap_metrics['AP']:.6f}, "
            f"val_AP50={ap_metrics['AP50']:.6f}, val_AP75={ap_metrics['AP75']:.6f}, "
            f"lr={optimizer.param_groups[0]['lr']:.8g}\n"
        )
        print(line.strip())
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)
        torch.save({"model": model.state_dict(), "epoch": epoch + 1, "config": cfg.__dict__}, out / "model_last.pth")


@torch.no_grad()
def evaluate_loss(model, loader, device):
    # torchvision detection 模型在 eval 下返回预测，在 train 下返回 loss；这里暂时切 train 且 no_grad。
    model.train()
    total = 0.0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        total += float(sum(loss_dict.values()).detach().cpu())
    model.eval()
    return total / max(1, len(loader))


@torch.no_grad()
def evaluate_coco_mask_ap(model, loader, device):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    model.eval()
    images_info = []
    annotations = []
    detections = []
    ann_id = 1
    image_ids = []

    for images, targets in loader:
        images_on_device = [img.to(device) for img in images]
        preds = model(images_on_device)
        for image, target, pred in zip(images, targets, preds):
            image_id = int(target["image_id"].item())
            H, W = int(image.shape[1]), int(image.shape[2])
            image_ids.append(image_id)
            images_info.append({"id": image_id, "height": H, "width": W})

            gt_masks = target["masks"].detach().cpu().numpy()
            for mask in gt_masks:
                mask = mask.astype(np.uint8)
                if mask.sum() == 0:
                    continue
                ys, xs = np.where(mask > 0)
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "segmentation": encode_mask(mask),
                        "area": float(mask.sum()),
                        "bbox": [
                            float(xs.min()),
                            float(ys.min()),
                            float(xs.max() - xs.min() + 1),
                            float(ys.max() - ys.min() + 1),
                        ],
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

            pred_masks = pred.get("masks")
            if pred_masks is None:
                continue
            pred_masks = pred_masks.detach().cpu().numpy()[:, 0]
            pred_scores = pred["scores"].detach().cpu().numpy()
            pred_boxes = pred["boxes"].detach().cpu().numpy()
            for mask_prob, score, box in zip(pred_masks, pred_scores, pred_boxes):
                mask = (mask_prob > 0.5).astype(np.uint8)
                if mask.sum() == 0:
                    continue
                x1, y1, x2, y2 = box.tolist()
                detections.append(
                    {
                        "image_id": image_id,
                        "category_id": 1,
                        "segmentation": encode_mask(mask),
                        "score": float(score),
                        "bbox": [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))],
                    }
                )

    if not annotations:
        return {"AP": float("nan"), "AP50": float("nan"), "AP75": float("nan")}
    if not detections:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}

    coco_gt = COCO()
    coco_gt.dataset = {
        "info": {},
        "licenses": [],
        "images": images_info,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "discontinuity"}],
    }
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(detections)
        evaluator = COCOeval(coco_gt, coco_dt, iouType="segm")
        evaluator.params.imgIds = image_ids
        evaluator.params.catIds = [1]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    return {
        "AP": float(evaluator.stats[0]),
        "AP50": float(evaluator.stats[1]),
        "AP75": float(evaluator.stats[2]),
    }


def encode_mask(mask: np.ndarray):
    from pycocotools import mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


if __name__ == "__main__":
    main()
