import argparse
from pathlib import Path
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

        # 简化版验证：记录验证集 loss；COCO AP 可接 pycocotools 扩展。
        val_loss = evaluate_loss(model, val_loader, device)
        line = f"epoch={epoch+1}, train_loss={total/max(1,len(train_loader)):.6f}, val_loss={val_loss:.6f}, lr={optimizer.param_groups[0]['lr']:.8g}\n"
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


if __name__ == "__main__":
    main()
